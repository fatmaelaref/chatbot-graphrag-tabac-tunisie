"""
============================================================
evaluate.py -- EVALUATION DU PIPELINE GRAPHRAG COMPLET
PROJET TABAC TUNISIE
============================================================
Remplace et etend 4_evaluate_chatbot.py (deplace ici). Fait tourner le
pipeline LangGraph complet de chatbot_pipeline.py (routage -> reecriture ->
recherche hybride -> grading -> generation -> verification groundedness) sur
gold_eval_set.json et calcule, par question puis agrege par categorie/type :

  - route_match           : le routage securite/domaine est-il celui attendu ?
  - behavior_match        : le comportement (repondre/abstention/refus-
                             redirection) correspond-il a expected_behavior ?
  - elements_recall       : fraction des faits attendus (expected_elements)
                             retrouves dans la reponse (match deterministe
                             normalise, rapide et reproductible).
  - faithfulness (0-10)   : juge LLM (CONTROL_MODEL, prompt dedie) -- chaque
                             affirmation de la reponse est-elle supportee par
                             les preuves retenues par le pipeline lui-meme ?
  - correctness (1-5)     : juge LLM (CONTROL_MODEL -- PAS GENERATION_MODEL,
                             pour eviter qu'un modele se note lui-meme) --
                             la reponse correspond-elle a reference_answer ?
  - semantic_similarity   : cosinus embedding (nomic-embed-text) reponse vs
                             reference -- signal secondaire sans biais de juge.
  - hit/rank retrieval    : le chunk attendu (expected_snippet) est-il
                             remonte, par quel canal (vecteur/graphe/hybride),
                             a quel rang -- calcule le hit-rate et le MRR.
  - n_llm_calls, latence  : cout et temps de reponse.

Ce script NE MODIFIE RIEN dans Neo4j (lecture seule). Rejouable en une
commande : `python evaluate.py` depuis le dossier evaluation/.
"""
import importlib.util
import json
import math
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).parent
PROJECT_DIR = EVAL_DIR.parent
GOLD_SET_PATH = EVAL_DIR / "gold_eval_set.json"
CHATBOT_MODULE_PATH = PROJECT_DIR / "chatbot_pipeline.py"

sys.path.insert(0, str(EVAL_DIR))
from common import (normalize, element_recall, retrieval_hit, infer_behavior,
                     avg, mrr, median, aggregate_by, write_markdown_report,
                     plot_category_scores)

print("=" * 60)
print("EVALUATION DU PIPELINE GRAPHRAG -- jeu de test gold")
print("=" * 60)

# ============================================================
# 0. CHARGEMENT PARESSEUX DE chatbot_pipeline.py (vit dans le dossier
#    parent, pas dans evaluation/ -- pas un package, donc pas importable
#    normalement)
# ============================================================
# Importer chatbot_pipeline.py execute son code de niveau module, qui
# construit un VectorCypherRetriever -- celui-ci interroge REELLEMENT Neo4j
# a la construction (SHOW VECTOR INDEXES). `--finalize-only` et `--from` ne
# font que reagreger des resultats deja notes (aucune question relancee) :
# differer ce chargement jusqu'au premier `run_question()` reel leur evite
# d'exiger Neo4j/Groq pour rien.
chatbot = None
app = None
_llm_call_count = {"n": 0}


def _ensure_pipeline_loaded():
    global chatbot, app
    if chatbot is not None:
        return

    spec = importlib.util.spec_from_file_location("chatbot_ultime", CHATBOT_MODULE_PATH)
    chatbot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chatbot)
    app = chatbot.build_graph()

    # Compteur d'appels LLM : wrappe le wrapper retry deja pose dans
    # chatbot_pipeline.py (le retry reste actif, on ajoute juste un comptage).
    already_wrapped_create = chatbot.groq_client.chat.completions.create

    def _counting_create(*args, **kwargs):
        _llm_call_count["n"] += 1
        return already_wrapped_create(*args, **kwargs)

    chatbot.groq_client.chat.completions.create = _counting_create


# ============================================================
# 1. JUGES LLM (prompts dedies, JSON strict, temperature 0)
# ============================================================
FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score"],
}

FAITHFULNESS_SYSTEM = """Tu evalues si une reponse de chatbot medical ne contient QUE des
affirmations supportees litteralement par les preuves fournies (aucun chiffre, loi, dosage,
nom ou fait qui n'apparait pas dans les preuves, et aucune generalisation au-dela de ce
qu'elles disent). Donne un score de 0 a 10 : 10 = toutes les affirmations sont strictement
supportees, 5 = certaines affirmations depassent les preuves sans etre fausses pour autant,
0 = la reponse est largement inventee ou hors des preuves. Liste les affirmations non
supportees si besoin. Si la reponse est un aveu explicite d'absence d'information (pas
d'affirmation factuelle a verifier), donne le score 10.
Reponds UNIQUEMENT en JSON."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "justification": {"type": "string"},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "incorrect_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "justification"],
}

CORRECTNESS_SYSTEM = """Tu compares la reponse d'un chatbot medical a une reponse de
reference ecrite par un humain a partir des documents sources. Note la reponse du chatbot
sur une echelle de 1 a 5 :
5 = correspond entierement a la reference, aucune erreur ni omission significative.
4 = correspond a la reference mais omet un detail secondaire.
3 = partiellement correcte : couvre une partie de la reference, ou une imprecision mineure.
2 = largement incorrecte ou incomplete par rapport a la reference.
1 = contredit la reference ou invente des faits qui ne s'y trouvent pas.
Liste les points manquants et les points incorrects separement. Reponds UNIQUEMENT en JSON."""


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(text_a: str, text_b: str):
    try:
        emb_a = chatbot.embedder.embed_query(text_a)
        emb_b = chatbot.embedder.embed_query(text_b)
        return round(_cosine(emb_a, emb_b), 4)
    except Exception as e:
        print(f"   [!] Echec calcul similarite embedding : {e}")
        return None


def judge_faithfulness(final_answer: str, evidence_items: list) -> dict:
    if not evidence_items:
        return {"score": None, "unsupported_claims": []}
    evidence_text = "\n\n".join(
        chatbot.clean_chunk(e["text"])[:chatbot.GROUNDEDNESS_TRUNCATE_CHARS] for e in evidence_items
    )
    user_prompt = f"Preuves :\n{evidence_text}\n\nReponse a evaluer :\n{final_answer}"
    result = chatbot.call_llm_json(
        chatbot.CONTROL_MODEL, FAITHFULNESS_SYSTEM, user_prompt, FAITHFULNESS_SCHEMA
    )
    if result is None:
        return {"score": None, "unsupported_claims": [], "error": "juge indisponible"}
    return result


def judge_correctness(final_answer: str, reference_answer: str, question: str) -> dict:
    # CONTROL_MODEL (pas GENERATION_MODEL) : le juge doit etre independant du
    # modele qui a produit la reponse a noter (corrige le 2026-10-07 -- l'ancien
    # 4_evaluate_chatbot.py utilisait GENERATION_MODEL pour les deux, un biais
    # potentiel explicitement signale dans la mission).
    user_prompt = (
        f"Question : {question}\n\n"
        f"Reponse de reference :\n{reference_answer}\n\n"
        f"Reponse du chatbot a evaluer :\n{final_answer}"
    )
    result = chatbot.call_llm_json(
        chatbot.CONTROL_MODEL, CORRECTNESS_SYSTEM, user_prompt, CORRECTNESS_SCHEMA
    )
    if result is None:
        return {"score": None, "justification": "juge indisponible"}
    return result


def is_abstention(final_answer: str) -> bool:
    norm = normalize(final_answer or "")
    markers = [
        "ne peux pas evaluer une situation medicale urgente",
        "ne peux pas te donner un diagnostic",
        "sort du champ de ce chatbot",
        "n'ai pas trouve, dans les documents analyses",
        "je ne suis pas certain de pouvoir repondre",
    ]
    if any(normalize(m) in norm for m in markers):
        return True
    return bool(chatbot._NO_INFO_PATTERNS.search(final_answer or ""))


# ============================================================
# 2. BOUCLE D'EVALUATION
# ============================================================
def run_question(item: dict) -> dict:
    _ensure_pipeline_loaded()
    _llm_call_count["n"] = 0
    start = time.time()

    initial_state = {
        "question": item["question"],
        "retry_count": 0,
        "rewrite_history": [],
        "groundedness_fix_count": 0,
    }
    final_state = app.invoke(initial_state)

    latency_s = round(time.time() - start, 2)

    route = final_state.get("route")
    final_answer = final_state.get("final_answer", "")
    sources = final_state.get("sources", [])
    graded_evidence = final_state.get("graded_evidence", [])
    vector_candidates = final_state.get("vector_candidates", [])
    graph_candidates = final_state.get("graph_candidates", [])

    behavior = infer_behavior(route, final_answer, is_abstention)

    result = {
        "id": item["id"],
        "category": item.get("category"),
        "type": item["type"],
        "question": item["question"],
        "route": route,
        "expected_route": item.get("expected_route"),
        "route_match": (route == item["expected_route"]) if item.get("expected_route") else None,
        "behavior": behavior,
        "expected_behavior": item.get("expected_behavior"),
        "behavior_match": (behavior == item["expected_behavior"]) if item.get("expected_behavior") else None,
        "n_evidence_retained": len(graded_evidence),
        "n_vector_candidates": len(vector_candidates),
        "n_graph_candidates": len(graph_candidates),
        "sources": sources,
        "final_answer": final_answer,
        "latency_s": latency_s,
        "n_llm_calls": _llm_call_count["n"],
    }

    # --- Retrieval : le chunk attendu est-il remonte, par quel canal ? ---
    snippet = item.get("expected_snippet")
    hit_v, rank_v = retrieval_hit(vector_candidates, snippet)
    hit_g, rank_g = retrieval_hit(graph_candidates, snippet)
    hit_hybrid = (hit_v or hit_g) if (hit_v is not None or hit_g is not None) else None
    hit_graded, rank_graded = retrieval_hit(graded_evidence, snippet)
    result.update({
        "hit_vector": hit_v, "rank_vector": rank_v,
        "hit_graph": hit_g, "rank_graph": rank_g,
        "hit_hybrid": hit_hybrid,
        "hit_graded": hit_graded, "rank_graded": rank_graded,
    })

    # --- Rappel deterministe des faits attendus ---
    result["elements_recall"] = element_recall(final_answer, item.get("expected_elements"))

    # --- Juges LLM + similarite (uniquement si une reponse de contenu est attendue) ---
    reference = item.get("reference_answer")
    should_score_content = bool(reference) and item.get("expected_behavior") == "repondre"
    if should_score_content:
        faith = judge_faithfulness(final_answer, graded_evidence)
        correct = judge_correctness(final_answer, reference, item["question"])
        sim = semantic_similarity(final_answer, reference)
        result["faithfulness_score"] = faith.get("score")
        result["faithfulness_unsupported_claims"] = faith.get("unsupported_claims", [])
        result["correctness_score"] = correct.get("score")
        result["correctness_justification"] = correct.get("justification")
        result["correctness_missing_points"] = correct.get("missing_points", [])
        result["correctness_incorrect_points"] = correct.get("incorrect_points", [])
        result["semantic_similarity"] = sim
    else:
        result["faithfulness_score"] = None
        result["correctness_score"] = None
        result["semantic_similarity"] = None

    return result


PARTIAL_PATH = EVAL_DIR / "resultats_partial.json"


def _load_partial():
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_partial(results):
    with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-questions", type=int, default=None,
                         help="Nombre maximum de NOUVELLES questions a traiter dans ce "
                              "process avant de sauvegarder et sortir (permet de relancer "
                              "`python evaluate.py` plusieurs fois de suite en processus "
                              "separes pour limiter le pic de RAM -- chaque relance reprend "
                              "ou le precedent run s'est arrete, sans retraiter les questions "
                              "deja faites). Omis = traite tout ce qui manque en un seul run.")
    parser.add_argument("--finalize-only", action="store_true",
                         help="Ne relance aucune question -- regenere juste le rapport/CSV/"
                              "graphique a partir de resultats_partial.json existant.")
    parser.add_argument("--from", dest="from_file", default=None, metavar="FICHIER.json",
                         help="Regenere UNIQUEMENT le rapport (+ graphique) a partir de ce "
                              "fichier resultats_*.json officiel deja complet (ex. "
                              "resultats_20261011.json), au lieu de resultats_partial.json -- "
                              "qui est local/non versionne et donc absent d'un clone frais du "
                              "depot. Implique --finalize-only : aucune question n'est "
                              "relancee, et contrairement a --finalize-only seul, n'ecrit PAS "
                              "de nouveaux resultats_AAAAMMJJ.json/.csv (la source en est deja "
                              "un officiel).")
    args = parser.parse_args()

    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        gold_set = json.load(f)

    if args.from_file:
        from_path = Path(args.from_file)
        if not from_path.exists():
            from_path = EVAL_DIR / args.from_file
        if not from_path.exists():
            print(f"❌ Fichier introuvable : {args.from_file}")
            sys.exit(1)

        with open(from_path, encoding="utf-8") as f:
            results = json.load(f)

        # La date du rapport doit refleter celle du run officiel source (son
        # nom de fichier), pas la date de cette regeneration -- sinon le
        # rapport regenere ne serait jamais identique a l'original, meme a
        # donnees inchangees.
        date_match = re.search(r"(\d{8})", from_path.stem)
        today = date_match.group(1) if date_match else date.today().strftime("%Y%m%d")

        print(f"\n[i] Regeneration du rapport a partir de {from_path} "
              f"({len(results)} questions, date={today}) -- resultats_partial.json ignore, "
              f"aucun resultats_{{date}}.json/.csv reecrit.")

        report_path = EVAL_DIR / "rapport_evaluation.md"
        chart_path = EVAL_DIR / "scores_par_categorie.png"

        print_summary(results)
        baseline_results = load_latest_baseline()
        write_report(results, report_path, chart_path, today, baseline_results)

        print(f"\nRapport regenere depuis {from_path.name}.")
        print(f"Rapport : {report_path}")
        print(f"Graphique : {chart_path}")
        return

    results = _load_partial()
    done_ids = {r["id"] for r in results}
    todo = [item for item in gold_set if item["id"] not in done_ids]

    if args.finalize_only:
        todo = []
    elif args.max_questions is not None:
        todo = todo[: args.max_questions]

    if todo:
        print(f"\n{len(done_ids)}/{len(gold_set)} déjà traitées (résultats_partial.json) -- "
              f"{len(todo)} à traiter dans ce run.")

    for i, item in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)} de ce run -- {len(results)+1}/{len(gold_set)} au total] "
              f"{item['id']} ({item['category']}/{item['type']}) : {item['question']}")
        try:
            r = run_question(item)
            results.append(r)
            print(f"   route={r['route']} (attendu={r['expected_route']}) "
                  f"comportement={r['behavior']} (attendu={r['expected_behavior']}) "
                  f"correctness={r['correctness_score']} faithfulness={r['faithfulness_score']} "
                  f"hit_hybrid={r['hit_hybrid']} latence={r['latency_s']}s appels_llm={r['n_llm_calls']}")
        except Exception as e:
            print(f"   [!] ECHEC sur cette question : {e}")
            results.append({"id": item["id"], "category": item.get("category"), "type": item["type"],
                             "question": item["question"], "error": str(e)})
        _save_partial(results)  # sauvegarde APRES CHAQUE question : resumable a tout instant
        time.sleep(1)  # marge anti rate-limit Groq (en plus du retry/backoff)

    remaining = len(gold_set) - len(results)
    if remaining > 0:
        print(f"\n{len(results)}/{len(gold_set)} traitées jusqu'ici -- {remaining} restantes.")
        print("Relance `python evaluate.py` (même commande) pour continuer là où ça s'est arrêté.")
        if chatbot is not None:
            chatbot.driver.close()
        return

    today = date.today().strftime("%Y%m%d")
    results_json_path = EVAL_DIR / f"resultats_{today}.json"
    results_csv_path = EVAL_DIR / f"resultats_{today}.csv"
    report_path = EVAL_DIR / "rapport_evaluation.md"
    chart_path = EVAL_DIR / "scores_par_categorie.png"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_csv(results, results_csv_path)
    print_summary(results)

    baseline_results = load_latest_baseline()
    write_report(results, report_path, chart_path, today, baseline_results)

    if chatbot is not None:
        chatbot.driver.close()
    print(f"\nToutes les {len(gold_set)} questions sont traitées.")
    print(f"Détail complet : {results_json_path}")
    print(f"CSV : {results_csv_path}")
    print(f"Rapport : {report_path}")
    print(f"Graphique : {chart_path}")


def write_csv(results, path):
    import csv
    cols = ["id", "category", "type", "question", "route", "expected_route", "route_match",
            "behavior", "expected_behavior", "behavior_match", "elements_recall",
            "faithfulness_score", "correctness_score", "semantic_similarity",
            "hit_vector", "rank_vector", "hit_graph", "rank_graph", "hit_hybrid",
            "n_evidence_retained", "n_llm_calls", "latency_s", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)


def print_summary(results):
    print("\n" + "=" * 60)
    print("RESUME GLOBAL")
    print("=" * 60)
    routed = [r for r in results if r.get("expected_route") is not None]
    behaved = [r for r in results if r.get("expected_behavior") is not None]
    print(f"Questions évaluées        : {len(results)}")
    print(f"Précision du routage      : {avg([1.0 if r.get('route_match') else 0.0 for r in routed])} (n={len(routed)})")
    print(f"Précision du comportement : {avg([1.0 if r.get('behavior_match') else 0.0 for r in behaved])} (n={len(behaved)})")
    print(f"Rappel éléments attendus  : {avg([r.get('elements_recall') for r in results])}")
    print(f"Faithfulness moyenne (/10): {avg([r.get('faithfulness_score') for r in results])}")
    print(f"Correctness moyenne (/5)  : {avg([r.get('correctness_score') for r in results])}")
    print(f"Hit-rate hybride          : {avg([1.0 if r.get('hit_hybrid') else 0.0 for r in results if r.get('hit_hybrid') is not None])}")
    print(f"MRR vecteur / graphe      : {mrr([r.get('rank_vector') for r in results if r.get('rank_vector') is not None or r.get('hit_vector') is False])} / "
          f"{mrr([r.get('rank_graph') for r in results if r.get('rank_graph') is not None or r.get('hit_graph') is False])}")
    print(f"Latence moy./médiane (s)  : {avg([r.get('latency_s') for r in results])} / {median([r.get('latency_s') for r in results])}")
    print(f"Appels LLM moyens/question: {avg([r.get('n_llm_calls') for r in results])}")


def load_latest_baseline():
    """Charge le run baseline.py le plus recent (evaluation/resultats_baseline_*.json),
    s'il existe, pour construire la section de comparaison du rapport. Ne plante
    jamais si aucun run baseline n'a encore ete fait -- la comparaison est alors
    simplement absente du rapport (signale comme limite)."""
    candidates = sorted(EVAL_DIR.glob("resultats_baseline_*.json"))
    if not candidates:
        print("\n[i] Aucun résultat baseline trouvé (lance baseline.py pour la comparaison) -- "
              "le rapport sera généré sans la section de comparaison.")
        return None
    latest = candidates[-1]
    with open(latest, encoding="utf-8") as f:
        print(f"\n[i] Résultats baseline chargés pour comparaison : {latest.name}")
        return json.load(f)


def build_comparison(pipeline_results, baseline_results):
    if not baseline_results:
        return None

    def behavior_acc(results):
        beh = [r for r in results if r.get("expected_behavior") is not None]
        return avg([1.0 if r.get("behavior_match") else 0.0 for r in beh])

    return {
        "Correctness moyenne (/5)": (
            avg([r.get("correctness_score") for r in pipeline_results]),
            avg([r.get("correctness_score") for r in baseline_results]),
        ),
        "Faithfulness moyenne (/10)": (
            avg([r.get("faithfulness_score") for r in pipeline_results]),
            avg([r.get("faithfulness_score") for r in baseline_results]),
        ),
        "Rappel éléments attendus": (
            avg([r.get("elements_recall") for r in pipeline_results]),
            avg([r.get("elements_recall") for r in baseline_results]),
        ),
        "Précision du comportement (sécurité/abstention)": (
            behavior_acc(pipeline_results),
            behavior_acc(baseline_results),
        ),
        "Hit-rate retrieval": (
            avg([1.0 if r.get("hit_hybrid") else 0.0 for r in pipeline_results if r.get("hit_hybrid") is not None]),
            avg([1.0 if r.get("hit_vector") else 0.0 for r in baseline_results if r.get("hit_vector") is not None]),
        ),
        "Latence moyenne (s)": (
            avg([r.get("latency_s") for r in pipeline_results]),
            avg([r.get("latency_s") for r in baseline_results]),
        ),
        "Appels LLM moyens / question": (
            avg([r.get("n_llm_calls") for r in pipeline_results]),
            avg([r.get("n_llm_calls") for r in baseline_results]),
        ),
    }


def write_report(results, report_path, chart_path, today, baseline_results=None):
    by_category = aggregate_by(results, "category")
    by_type = aggregate_by(results, "type")

    plot_category_scores(by_category, str(chart_path), "Scores du pipeline GraphRAG par catégorie")

    flagged = []
    for r in results:
        if r.get("error"):
            flagged.append({**r, "cause_probable": f"Exception pendant l'exécution : {r['error']}"})
        elif r.get("route_match") is False:
            flagged.append({**r, "cause_probable": f"Routage incorrect (obtenu={r.get('route')}, attendu={r.get('expected_route')})"})
        elif r.get("behavior_match") is False:
            flagged.append({**r, "cause_probable": f"Comportement incorrect (obtenu={r.get('behavior')}, attendu={r.get('expected_behavior')})"})
        elif r.get("hit_hybrid") is False:
            flagged.append({**r, "cause_probable": "Le chunk attendu n'a été retrouvé par aucun canal (ni vecteur ni graphe) -- probable lacune de retrieval."})
        elif (r.get("correctness_score") or 5) <= 2:
            flagged.append({**r, "cause_probable": "Correctness jugée faible par le juge LLM -- voir correctness_justification dans le JSON détaillé."})
        elif r.get("faithfulness_score") is not None and r.get("faithfulness_score") <= 4:
            flagged.append({**r, "cause_probable": "Faithfulness jugée faible -- la réponse contient probablement des affirmations non supportées par les preuves."})

    global_stats = {
        "Questions évaluées": len(results),
        "Précision du routage": avg([1.0 if r.get("route_match") else 0.0 for r in results if r.get("route_match") is not None]),
        "Précision du comportement": avg([1.0 if r.get("behavior_match") else 0.0 for r in results if r.get("behavior_match") is not None]),
        "Rappel éléments attendus": avg([r.get("elements_recall") for r in results]),
        "Faithfulness moyenne (/10)": avg([r.get("faithfulness_score") for r in results]),
        "Correctness moyenne (/5)": avg([r.get("correctness_score") for r in results]),
        "Similarité sémantique moyenne": avg([r.get("semantic_similarity") for r in results]),
        "Hit-rate retrieval (hybride)": avg([1.0 if r.get("hit_hybrid") else 0.0 for r in results if r.get("hit_hybrid") is not None]),
        "Hit-rate retrieval (vecteur seul)": avg([1.0 if r.get("hit_vector") else 0.0 for r in results if r.get("hit_vector") is not None]),
        "Hit-rate retrieval (graphe seul)": avg([1.0 if r.get("hit_graph") else 0.0 for r in results if r.get("hit_graph") is not None]),
        "MRR vecteur": mrr([r.get("rank_vector") for r in results if r.get("rank_vector") is not None or r.get("hit_vector") is False]),
        "MRR graphe": mrr([r.get("rank_graph") for r in results if r.get("rank_graph") is not None or r.get("hit_graph") is False]),
        "Latence moyenne (s)": avg([r.get("latency_s") for r in results]),
        "Latence médiane (s)": median([r.get("latency_s") for r in results]),
        "Appels LLM moyens / question": avg([r.get("n_llm_calls") for r in results]),
    }

    limitations = [
        f"Jeu de test de {len(results)} questions, dérivé du corpus de 10 documents -- taille modeste pour des intervalles de confiance serrés, en particulier par catégorie (certaines catégories n'ont que 7-9 questions).",
        "Le juge LLM (gpt-oss-20b via Groq, CONTROL_MODEL) peut se tromper ou halluciner son propre score -- `correctness_score` et `faithfulness_score` sont des signaux probabilistes, pas une vérité absolue. `elements_recall` (match déterministe) sert de garde-fou complémentaire.",
        "`expected_snippet` est un extrait court choisi manuellement par question -- un hit-rate bas peut signaler un snippet mal choisi autant qu'un vrai échec de retrieval ; vérifier les cas flagués individuellement avant de conclure.",
        "Les questions de sécurité (urgence/demande_personnelle/hors_domaine) ne sont volontairement pas notées en correctness/faithfulness (pas de 'bonne réponse' textuelle attendue) -- seul le routage et le comportement comptent pour elles.",
    ]

    comparison = build_comparison(results, baseline_results)
    if baseline_results is None:
        limitations.append(
            "Aucune comparaison pipeline vs. baseline dans ce rapport -- lance "
            "`python baseline.py` puis relance `python evaluate.py` pour l'inclure."
        )

    write_markdown_report(
        report_path,
        title=f"Rapport d'évaluation du chatbot GraphRAG Tabac Tunisie -- {today}",
        meta={
            "Date d'exécution": today,
            "Jeu de test": f"evaluation/gold_eval_set.json ({len(results)} questions)",
            "Pipeline évalué": "chatbot_pipeline.py (build_graph(), pipeline complet)",
            "Juge LLM": "openai/gpt-oss-20b (Groq, CONTROL_MODEL) -- différent de GENERATION_MODEL (gpt-oss-120b) qui génère les réponses",
        },
        global_stats=global_stats,
        by_category=by_category,
        by_type=by_type,
        flagged=flagged,
        comparison=comparison,
        limitations=limitations,
        ablation_csv_path=EVAL_DIR / "resultats_ablation.csv",
    )


if __name__ == "__main__":
    main()
