"""
============================================================
ABLATION_GRAPHE.PY -- mesure l'apport reel du canal graphe (Cypher)
============================================================
Compare 3 configurations sur le MEME jeu de questions, TOUTES avec le code
ACTUEL de chatbot_pipeline.py (donc comparables entre elles -- pas de biais
du type "une config a le correctif du bug des 700 caracteres, l'autre non") :
  A = RAG simple (vecteur seul, 1 appel LLM, sans notation ni verification)
      -> reutilise le run baseline deja fait (resultats_baseline_*.json) --
         non affecte par le bug de troncature (baseline.py ne fait pas de
         grading multi-candidats), donc reutilisable sans biais.
  B = pipeline complet SANS le canal graphe (USE_GRAPH=0)
  C = pipeline complet AVEC le canal graphe (USE_GRAPH=1, comportement par defaut)

B et C sont CHACUN relances avec le code actuel (jamais reutilises d'un
ancien fichier), pour garantir une comparaison equitable.

Ne modifie PAS le comportement par defaut du chatbot (USE_GRAPH=1 par defaut).

Usage :
  python ablation_graphe.py --config B [--categories legislation,epidemiologie]
  python ablation_graphe.py --config C [--categories legislation,epidemiologie]
  python ablation_graphe.py --finalize   # combine A + B + C et ecrit le rapport

Checkpoint apres chaque question (resultats_ablation_B_partial.json /
resultats_ablation_C_partial.json) : rejouable sans refaire le travail deja fait.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).parent
PROJECT_DIR = EVAL_DIR.parent
GOLD_SET_PATH = EVAL_DIR / "gold_eval_set.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", choices=["B", "C"], help="Configuration a lancer/reprendre.")
parser.add_argument("--finalize", action="store_true", help="Combine A+B+C et ecrit le rapport final.")
parser.add_argument("--categories", default="legislation,epidemiologie",
                     help="Categories a tester (separees par des virgules), ou 'all'. "
                          "Defaut : legislation,epidemiologie.")
args = parser.parse_args()

if not args.config and not args.finalize:
    print("Precise --config B, --config C, ou --finalize.")
    sys.exit(1)

with open(GOLD_SET_PATH, encoding="utf-8") as f:
    gold_set = json.load(f)

if args.categories == "all":
    subset = gold_set
else:
    cats = {c.strip() for c in args.categories.split(",")}
    subset = [g for g in gold_set if g["category"] in cats]
subset_ids = {g["id"] for g in subset}

print(f"Sous-ensemble : {len(subset)} question(s) -- categories : "
      f"{sorted({g['category'] for g in subset})}")


def run_config(letter, use_graph):
    os.environ["USE_GRAPH"] = "1" if use_graph else "0"
    sys.path.insert(0, str(EVAL_DIR))
    import evaluate  # noqa: E402  (import tardif volontaire, apres le set env)

    assert evaluate.chatbot.USE_GRAPH is use_graph, \
        f"USE_GRAPH={use_graph} n'a pas ete pris en compte -- abandon."
    print(f"✅ Config {letter} confirmee : USE_GRAPH={use_graph} dans ce process.")

    partial_path = EVAL_DIR / f"resultats_ablation_{letter}_partial.json"
    results = json.load(open(partial_path, encoding="utf-8")) if partial_path.exists() else []
    done_ids = {r["id"] for r in results}
    todo = [item for item in subset if item["id"] not in done_ids]

    if done_ids:
        print(f"{len(done_ids)} question(s) deja faites (config {letter}, reprise) -- "
              f"{len(todo)} restantes.")

    for i, item in enumerate(todo, 1):
        print(f"\n[{letter} {i}/{len(todo)}] {item['id']} ({item['category']}) : {item['question']}")
        try:
            r = evaluate.run_question(item)
            results.append(r)
            print(f"   route={r['route']} comportement={r['behavior']} "
                  f"correctness={r['correctness_score']} rappel={r['elements_recall']} "
                  f"latence={r['latency_s']}s")
        except Exception as e:
            print(f"   [!] ECHEC : {e}")
            results.append({"id": item["id"], "category": item.get("category"),
                             "type": item["type"], "question": item["question"], "error": str(e)})
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        time.sleep(1)

    remaining = len(subset) - len(results)
    evaluate.chatbot.driver.close()
    if remaining > 0:
        print(f"\n{len(results)}/{len(subset)} (config {letter}) traitees -- {remaining} restantes.")
        print(f"Relance `python ablation_graphe.py --config {letter}` pour continuer.")
        sys.exit(0)
    print(f"\n✅ Config {letter} terminee : {len(results)}/{len(subset)} questions.")


if args.config == "B":
    run_config("B", use_graph=False)
    sys.exit(0)
elif args.config == "C":
    run_config("C", use_graph=True)
    sys.exit(0)

# ============================================================
# --finalize : combine A (reutilise) + B + C (partials complets) et ecrit
# le rapport. N'importe PAS evaluate.py (pas besoin de Neo4j/Groq ici).
# ============================================================
sys.path.insert(0, str(EVAL_DIR))
from common import avg  # noqa: E402


def _load_latest(prefix, exclude=()):
    candidates = sorted(EVAL_DIR.glob(f"{prefix}_*.json"), reverse=True)
    skip = ("partial", "ablation") + tuple(exclude)
    candidates = [c for c in candidates if not any(s in c.name for s in skip)]
    if not candidates:
        return None, None
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f), candidates[0].name


results_a_full, fname_a = _load_latest("resultats_baseline")
if not results_a_full:
    print("⚠️ Aucun run baseline trouve (resultats_baseline_*.json) -- lance baseline.py d'abord. Abandon.")
    sys.exit(1)
print(f"Config A (baseline) reutilisee depuis {fname_a}")

b_path = EVAL_DIR / "resultats_ablation_B_partial.json"
c_path = EVAL_DIR / "resultats_ablation_C_partial.json"
if not b_path.exists() or not c_path.exists():
    print("⚠️ Il manque resultats_ablation_B_partial.json et/ou _C_partial.json -- "
          "lance `--config B` et `--config C` d'abord. Abandon.")
    sys.exit(1)

results_b_full = json.load(open(b_path, encoding="utf-8"))
results_c_full = json.load(open(c_path, encoding="utf-8"))

missing_b = subset_ids - {r["id"] for r in results_b_full}
missing_c = subset_ids - {r["id"] for r in results_c_full}
if missing_b:
    print(f"⚠️ Config B incomplete sur ce sous-ensemble : {len(missing_b)} question(s) manquante(s) "
          f"{sorted(missing_b)} -- relance `--config B`.")
if missing_c:
    print(f"⚠️ Config C incomplete sur ce sous-ensemble : {len(missing_c)} question(s) manquante(s) "
          f"{sorted(missing_c)} -- relance `--config C`.")
if missing_b or missing_c:
    sys.exit(1)

results_a = [r for r in results_a_full if r["id"] in subset_ids]
results_b = [r for r in results_b_full if r["id"] in subset_ids]
results_c = [r for r in results_c_full if r["id"] in subset_ids]

missing_a = subset_ids - {r["id"] for r in results_a}
if missing_a:
    print(f"⚠️ {len(missing_a)} question(s) du sous-ensemble absentes du run baseline : {sorted(missing_a)}")


def abstention_rate(results):
    vals = [1.0 if r.get("behavior") == "abstention" else 0.0 for r in results if r.get("behavior")]
    return avg(vals)


def route_precision(results):
    vals = [1.0 if r.get("route_match") else 0.0 for r in results if r.get("route_match") is not None]
    return avg(vals)


def metrics_for(results):
    return {
        "n": len(results),
        "precision_routage": route_precision(results),
        "rappel_elements": avg([r.get("elements_recall") for r in results]),
        "justesse_5": avg([r.get("correctness_score") for r in results]),
        "latence_moy_s": avg([r.get("latency_s") for r in results]),
        "taux_abstention": abstention_rate(results),
    }


def by_category(results):
    cats = {}
    for r in results:
        cats.setdefault(r.get("category"), []).append(r)
    return {c: metrics_for(rs) for c, rs in sorted(cats.items())}


configs = {"A_rag_simple": results_a, "B_sans_graphe": results_b, "C_avec_graphe": results_c}
global_metrics = {name: metrics_for(rs) for name, rs in configs.items()}
cat_metrics = {name: by_category(rs) for name, rs in configs.items()}

csv_path = EVAL_DIR / "resultats_ablation.csv"
cols = ["scope", "config", "n", "precision_routage", "rappel_elements",
        "justesse_5", "latence_moy_s", "taux_abstention"]

rows = []
all_cats = sorted({c for name in configs for c in cat_metrics[name]})
for cat in all_cats:
    for name in configs:
        m = cat_metrics[name].get(cat)
        if m:
            rows.append({"scope": cat, "config": name, **m})
for name in configs:
    rows.append({"scope": "GLOBAL", "config": name, **global_metrics[name]})

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    import csv as csv_module
    writer = csv_module.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\n✅ Resultats ecrits dans {csv_path}")

print("\n" + "=" * 70)
print("TABLEAU GLOBAL A / B / C")
print("=" * 70)
header = f"{'Metrique':<22}{'A (RAG simple)':>16}{'B (sans graphe)':>18}{'C (avec graphe)':>18}"
print(header)
for key, label in [("precision_routage", "Precision routage"),
                    ("rappel_elements", "Rappel elements"),
                    ("justesse_5", "Justesse (/5)"),
                    ("latence_moy_s", "Latence moy. (s)"),
                    ("taux_abstention", "Taux abstention")]:
    a = global_metrics["A_rag_simple"].get(key)
    b = global_metrics["B_sans_graphe"].get(key)
    c = global_metrics["C_avec_graphe"].get(key)
    print(f"{label:<22}{a!s:>16}{b!s:>18}{c!s:>18}")
print(f"\n(n = {global_metrics['A_rag_simple']['n']} / "
      f"{global_metrics['B_sans_graphe']['n']} / {global_metrics['C_avec_graphe']['n']})")

print("\n" + "=" * 70)
print("PAR CATEGORIE (B vs C -- l'apport du graphe toutes choses egales par ailleurs)")
print("=" * 70)
for cat in all_cats:
    b = cat_metrics["B_sans_graphe"].get(cat)
    c = cat_metrics["C_avec_graphe"].get(cat)
    if b and c:
        d_rappel = (c["rappel_elements"] or 0) - (b["rappel_elements"] or 0)
        d_justesse = (c["justesse_5"] or 0) - (b["justesse_5"] or 0)
        print(f"  {cat:<15} rappel B={b['rappel_elements']} -> C={c['rappel_elements']} "
              f"(delta {d_rappel:+.3f})  |  justesse B={b['justesse_5']} -> C={c['justesse_5']} "
              f"(delta {d_justesse:+.3f})")

print("\n" + "=" * 70)
print("INTERPRETATION (a completer/ajuster a la main si besoin)")
print("=" * 70)
d_rappel_glob = (global_metrics["C_avec_graphe"]["rappel_elements"] or 0) - \
                (global_metrics["B_sans_graphe"]["rappel_elements"] or 0)
d_justesse_glob = (global_metrics["C_avec_graphe"]["justesse_5"] or 0) - \
                   (global_metrics["B_sans_graphe"]["justesse_5"] or 0)
sens = "ameliore" if d_rappel_glob > 0.01 else ("degrade" if d_rappel_glob < -0.01 else "ne change quasiment pas")
print(f"1. Sur ce sous-ensemble ({', '.join(sorted({g['category'] for g in subset}))}), "
      f"ajouter le canal graphe {sens} le rappel des elements attendus "
      f"({global_metrics['B_sans_graphe']['rappel_elements']} -> {global_metrics['C_avec_graphe']['rappel_elements']}).")
print(f"2. La justesse (/5, juge LLM) passe de {global_metrics['B_sans_graphe']['justesse_5']} "
      f"(sans graphe) a {global_metrics['C_avec_graphe']['justesse_5']} (avec graphe) "
      f"-- {'un gain reel' if d_justesse_glob > 0.1 else 'un ecart marginal, pas de conclusion forte'}.")
weakest = min(all_cats, key=lambda c: (cat_metrics['C_avec_graphe'].get(c, {}).get('rappel_elements') or 0) -
                                       (cat_metrics['B_sans_graphe'].get(c, {}).get('rappel_elements') or 0),
              default=None)
if weakest:
    print(f"3. La categorie ou le graphe aide le moins (voire pas) sur ce sous-ensemble : "
          f"{weakest} -- a ne pas cacher, meme si l'hypothese de depart etait que le graphe y aiderait.")
