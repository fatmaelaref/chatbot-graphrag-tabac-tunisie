"""
============================================================
baseline.py -- RAG SIMPLE DE COMPARAISON (sans routage/grading/groundedness)
PROJET TABAC TUNISIE
============================================================
Chiffre l'apport reel du pipeline GraphRAG controle (3_chatbot_ULTIME.py) en
le comparant a la version "RAG naif" la plus courante : recherche vectorielle
seule (meme index Neo4j, meme embedder), UN SEUL appel LLM de generation,
AUCUN routage de securite, AUCUN grading des preuves, AUCUNE verification de
groundedness. Tourne sur le meme gold_eval_set.json que evaluate.py.

Reutilise volontairement les memes composants bas niveau que le pipeline
(driver Neo4j, vector_retriever, embedder, GENERATION_MODEL) -- ce qui change
n'est PAS le corpus ni le modele, seulement l'ABSENCE des etapes de controle.
C'est cette difference-la, et uniquement elle, que la comparaison doit
mesurer.

Rejouable en une commande : `python baseline.py` depuis evaluation/.
"""
import importlib.util
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).parent
PROJECT_DIR = EVAL_DIR.parent
GOLD_SET_PATH = EVAL_DIR / "gold_eval_set.json"
CHATBOT_MODULE_PATH = PROJECT_DIR / "3_chatbot_ULTIME.py"

sys.path.insert(0, str(EVAL_DIR))
from common import normalize, element_recall, retrieval_hit, infer_behavior, avg, median

spec = importlib.util.spec_from_file_location("chatbot_ultime", CHATBOT_MODULE_PATH)
chatbot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chatbot)

print("=" * 60)
print("EVALUATION BASELINE -- RAG simple (vecteur seul, 1 appel LLM)")
print("=" * 60)

_llm_call_count = {"n": 0}
_already_wrapped_create = chatbot.groq_client.chat.completions.create


def _counting_create(*args, **kwargs):
    _llm_call_count["n"] += 1
    return _already_wrapped_create(*args, **kwargs)


chatbot.groq_client.chat.completions.create = _counting_create

# Prompt volontairement minimal -- pas de many-shot, pas d'instructions de
# prudence elaborees, pas de distinction de routes : exactement ce qu'un RAG
# "boite noire" de base ferait.
NAIVE_SYSTEM = """Tu es un assistant qui répond à des questions sur le tabagisme en te basant
sur les extraits de documents fournis. Réponds à la question en utilisant les extraits
ci-dessous. Si l'information ne s'y trouve pas, dis que tu ne sais pas."""

# Juges identiques a evaluate.py (CONTROL_MODEL, independant du modele de
# generation) -- pour que les scores pipeline/baseline restent comparables.
FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["score"],
}
FAITHFULNESS_SYSTEM = """Tu evalues si une reponse de chatbot medical ne contient QUE des
affirmations supportees litteralement par les preuves fournies. Donne un score de 0 a 10 :
10 = toutes les affirmations sont strictement supportees, 5 = certaines affirmations
depassent les preuves sans etre fausses, 0 = largement invente ou hors des preuves. Si la
reponse est un aveu explicite d'absence d'information, donne le score 10.
Reponds UNIQUEMENT en JSON."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"}, "justification": {"type": "string"},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "incorrect_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "justification"],
}
CORRECTNESS_SYSTEM = """Tu compares la reponse d'un chatbot medical a une reponse de reference.
Note de 1 a 5 : 5 = correspond entierement, 4 = omet un detail secondaire, 3 = partiellement
correcte, 2 = largement incorrecte/incomplete, 1 = contredit la reference ou invente des faits.
Reponds UNIQUEMENT en JSON."""


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_similarity(a, b):
    try:
        return round(_cosine(chatbot.embedder.embed_query(a), chatbot.embedder.embed_query(b)), 4)
    except Exception:
        return None


def judge_faithfulness(answer, evidence_items):
    if not evidence_items:
        return {"score": None}
    evidence_text = "\n\n".join(chatbot.clean_chunk(c["text"])[:700] for c in evidence_items)
    result = chatbot.call_ollama_json(chatbot.CONTROL_MODEL, FAITHFULNESS_SYSTEM,
                                       f"Preuves :\n{evidence_text}\n\nReponse a evaluer :\n{answer}",
                                       FAITHFULNESS_SCHEMA)
    return result or {"score": None}


def judge_correctness(answer, reference, question):
    result = chatbot.call_ollama_json(
        chatbot.CONTROL_MODEL, CORRECTNESS_SYSTEM,
        f"Question : {question}\n\nReponse de reference :\n{reference}\n\nReponse a evaluer :\n{answer}",
        CORRECTNESS_SCHEMA,
    )
    return result or {"score": None}


def is_abstention(answer):
    norm = normalize(answer or "")
    return "je ne sais pas" in norm or "ne sais pas" in norm or bool(chatbot._NO_INFO_PATTERNS.search(answer or ""))


def run_question(item):
    _llm_call_count["n"] = 0
    start = time.time()

    vector_candidates = []
    try:
        result = chatbot.vector_retriever.search(query_text=item["question"], top_k=chatbot.TOP_K_VECTOR)
        for it in result.items:
            content = it.content if isinstance(it.content, str) else str(it.content)
            meta = getattr(it, "metadata", None) or {}
            vector_candidates.append({"text": content, "source": chatbot.short_source(meta.get("source", ""))})
    except Exception as e:
        print(f"   [!] Echec recherche vectorielle : {e}")

    evidence_text = "\n\n".join(chatbot.clean_chunk(c["text"])[:700] for c in vector_candidates)
    user_prompt = f"Extraits :\n{evidence_text}\n\nQuestion : {item['question']}"

    final_answer = ""
    try:
        response = chatbot.groq_client.chat.completions.create(
            model=chatbot.GENERATION_MODEL,
            messages=[{"role": "system", "content": NAIVE_SYSTEM}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        final_answer = response.choices[0].message.content or ""
    except Exception as e:
        final_answer = f"[ERREUR LLM : {e}]"

    latency_s = round(time.time() - start, 2)
    behavior = infer_behavior(None, final_answer, is_abstention)  # pas de routage -> jamais "refus_redirection"

    result_row = {
        "id": item["id"], "category": item.get("category"), "type": item["type"],
        "question": item["question"], "final_answer": final_answer,
        "behavior": behavior, "expected_behavior": item.get("expected_behavior"),
        "behavior_match": (behavior == item["expected_behavior"]) if item.get("expected_behavior") else None,
        "latency_s": latency_s, "n_llm_calls": _llm_call_count["n"],
    }

    hit_v, rank_v = retrieval_hit(vector_candidates, item.get("expected_snippet"))
    result_row.update({"hit_vector": hit_v, "rank_vector": rank_v})
    result_row["elements_recall"] = element_recall(final_answer, item.get("expected_elements"))

    reference = item.get("reference_answer")
    if reference and item.get("expected_behavior") == "repondre":
        faith = judge_faithfulness(final_answer, vector_candidates)
        correct = judge_correctness(final_answer, reference, item["question"])
        result_row["faithfulness_score"] = faith.get("score")
        result_row["correctness_score"] = correct.get("score")
        result_row["semantic_similarity"] = semantic_similarity(final_answer, reference)
    else:
        result_row["faithfulness_score"] = None
        result_row["correctness_score"] = None
        result_row["semantic_similarity"] = None

    return result_row


def main():
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        gold_set = json.load(f)

    results = []
    for i, item in enumerate(gold_set, 1):
        print(f"[{i}/{len(gold_set)}] {item['id']} : {item['question']}")
        try:
            r = run_question(item)
            results.append(r)
            print(f"   comportement={r['behavior']} (attendu={r['expected_behavior']}) "
                  f"correctness={r['correctness_score']} hit_vector={r['hit_vector']} latence={r['latency_s']}s")
        except Exception as e:
            print(f"   [!] ECHEC : {e}")
            results.append({"id": item["id"], "category": item.get("category"), "type": item["type"],
                             "question": item["question"], "error": str(e)})
        time.sleep(1)

    today = date.today().strftime("%Y%m%d")
    out_json = EVAL_DIR / f"resultats_baseline_{today}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESUME BASELINE")
    print("=" * 60)
    print(f"Correctness moyenne (/5)  : {avg([r.get('correctness_score') for r in results])}")
    print(f"Faithfulness moyenne (/10): {avg([r.get('faithfulness_score') for r in results])}")
    print(f"Rappel éléments attendus  : {avg([r.get('elements_recall') for r in results])}")
    behaved = [r for r in results if r.get("expected_behavior") is not None]
    print(f"Précision comportement    : {avg([1.0 if r.get('behavior_match') else 0.0 for r in behaved])} (n={len(behaved)})")
    print(f"Hit-rate vecteur          : {avg([1.0 if r.get('hit_vector') else 0.0 for r in results if r.get('hit_vector') is not None])}")
    print(f"Latence moy./médiane (s)  : {avg([r.get('latency_s') for r in results])} / {median([r.get('latency_s') for r in results])}")
    print(f"Appels LLM moyens/question: {avg([r.get('n_llm_calls') for r in results])}")
    print(f"\nRésultats écrits dans : {out_json}")
    print("Relance `python evaluate.py` pour inclure la comparaison dans le rapport.")

    chatbot.driver.close()


if __name__ == "__main__":
    main()
