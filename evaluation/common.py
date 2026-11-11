"""
============================================================
common.py -- utilitaires partages par evaluate.py et baseline.py
============================================================
Rien ici ne doit dependre de chatbot_pipeline.py (pour que baseline.py reste
independant du pipeline qu'il sert a comparer) -- uniquement des fonctions
pures de scoring/normalisation et de generation de rapport.
"""
import unicodedata
import statistics as _stats


_TYPOGRAPHIC_EQUIVALENTS = {
    " ": " ", " ": " ", " ": " ",            # espaces insecables/fines
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",  # tirets typographiques
    "‘": "'", "’": "'", "“": '"', "”": '"',                 # apostrophes/guillemets courbes
}


def normalize(text: str) -> str:
    """Minuscule + accents retires + variantes typographiques ramenees a
    l'ASCII (espace insecable, tiret long, apostrophe courbe...) -- les
    modeles Groq (gpt-oss) utilisent couramment cette typographie francaise
    correcte (ex: "demi‑vie" avec un vrai tiret insecable), ce qui
    faisait manquer des correspondances pourtant correctes avec un simple
    strip d'accents (constate le 2026-10-07 sur Q56 : element_recall=0.0
    alors que la reponse etait exacte, juste a cause du caractere tiret)."""
    if not text:
        return ""
    for src, dst in _TYPOGRAPHIC_EQUIVALENTS.items():
        text = text.replace(src, dst)
    text = text.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def element_recall(answer_text: str, expected_elements: list) -> float:
    """Fraction des `expected_elements` retrouves (match substring normalise)
    dans la reponse generee. Match deterministe -- pas de juge LLM ici,
    volontairement : sert de signal rapide et reproductible en complement du
    juge LLM (qui, lui, capture la paraphrase)."""
    if not expected_elements:
        return None
    norm_answer = normalize(answer_text)
    found = sum(1 for e in expected_elements if normalize(e) in norm_answer)
    return round(found / len(expected_elements), 3)


def retrieval_hit(candidates: list, expected_snippet: str):
    """Cherche `expected_snippet` (normalise) dans le texte de chaque
    candidat, dans l'ordre fourni (= ordre de pertinence du canal). Retourne
    (hit: bool, rank: int|None) -- rank est la position 1-indexee du premier
    candidat qui contient le snippet, utile pour calculer un MRR."""
    if not expected_snippet or not candidates:
        return (None, None)
    norm_snippet = normalize(expected_snippet)
    for i, c in enumerate(candidates):
        if norm_snippet in normalize(c.get("text", "")):
            return (True, i + 1)
    return (False, None)


def infer_behavior(route: str, final_answer: str, is_abstention_fn) -> str:
    """Deduit le comportement effectif du chatbot a partir de la route prise
    et du contenu de la reponse -- utilise pour comparer a expected_behavior
    (repondre / abstention / refus_redirection)."""
    if route in ("hors_domaine", "urgence", "demande_personnelle"):
        return "refus_redirection"
    if is_abstention_fn(final_answer):
        return "abstention"
    return "repondre"


def avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def mrr(ranks):
    """Mean Reciprocal Rank a partir d'une liste de rangs (None = pas trouve)."""
    reciprocals = [1.0 / r if r else 0.0 for r in ranks]
    return round(sum(reciprocals) / len(reciprocals), 3) if reciprocals else None


def median(values):
    values = [v for v in values if v is not None]
    return round(_stats.median(values), 3) if values else None


def aggregate_by(results: list, key: str) -> dict:
    """Regroupe les resultats par la valeur de `key` (ex: 'category' ou
    'type') et calcule les moyennes des metriques numeriques usuelles."""
    groups = {}
    for r in results:
        groups.setdefault(r.get(key), []).append(r)

    out = {}
    for group_key, items in groups.items():
        out[group_key] = {
            "n": len(items),
            "route_accuracy": avg([1.0 if i.get("route_match") else 0.0
                                    for i in items if i.get("route_match") is not None]),
            "behavior_accuracy": avg([1.0 if i.get("behavior_match") else 0.0
                                       for i in items if i.get("behavior_match") is not None]),
            "elements_recall": avg([i.get("elements_recall") for i in items]),
            "faithfulness": avg([i.get("faithfulness_score") for i in items]),
            "correctness": avg([i.get("correctness_score") for i in items]),
            "semantic_similarity": avg([i.get("semantic_similarity") for i in items]),
            "hit_rate_hybrid": avg([1.0 if i.get("hit_hybrid") else 0.0
                                     for i in items if i.get("hit_hybrid") is not None]),
            "latency_s": avg([i.get("latency_s") for i in items]),
        }
    return out


def write_ablation_section(csv_path) -> list:
    """
    Lit resultats_ablation.csv (produit par `ablation_graphe.py --finalize`)
    et rend la section Markdown "Ablation du canal graphe". Calcule les
    deltas par categorie depuis les donnees plutot que de les copier a la
    main -- evite que cette section disparaisse ou devienne obsolete a
    chaque regeneration complete du rapport par evaluate.py (bug constate
    le 2026-10-11 : une section ajoutee a la main avait ete silencieusement
    effacee par le run suivant).
    """
    import csv as csv_module
    from pathlib import Path

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    rows = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            rows.setdefault(row["scope"], {})[row["config"]] = row

    def fmt(v):
        return v if v else "-"

    lines = ["## Ablation du canal graphe (Cypher)", "",
             "Comparaison sur le jeu de test complet entre trois configurations : "
             "**A** = RAG simple (recherche vectorielle seule, 1 appel LLM), "
             "**B** = pipeline complet avec `USE_GRAPH=0` (sans le canal graphe), "
             "**C** = pipeline complet par défaut (avec le canal graphe).", ""]

    g = rows.get("GLOBAL", {})
    if g:
        lines.append("| Métrique | A (RAG simple) | B (sans graphe) | C (avec graphe) |")
        lines.append("|---|---|---|---|")
        for key, label in [("precision_routage", "Précision routage"),
                            ("rappel_elements", "Rappel éléments"),
                            ("justesse_5", "Justesse (/5)"),
                            ("latence_moy_s", "Latence moy. (s)"),
                            ("taux_abstention", "Taux abstention")]:
            a = fmt(g.get("A_rag_simple", {}).get(key))
            b = fmt(g.get("B_sans_graphe", {}).get(key))
            c = fmt(g.get("C_avec_graphe", {}).get(key))
            lines.append(f"| {label} | {a} | {b} | {c} |")
        lines.append("")

    deltas = []
    for scope, configs in sorted(rows.items()):
        if scope == "GLOBAL":
            continue
        b, c = configs.get("B_sans_graphe"), configs.get("C_avec_graphe")
        if not b or not c or not b.get("rappel_elements") or not c.get("rappel_elements"):
            continue
        deltas.append((
            scope,
            float(c["rappel_elements"]) - float(b["rappel_elements"]),
            float(c["justesse_5"]) - float(b["justesse_5"]),
        ))

    if deltas:
        lines.append("Par catégorie (B -> C, apport du canal graphe) :")
        lines.append("")
        for scope, d_rappel, d_justesse in deltas:
            lines.append(f"- **{scope}** : rappel {d_rappel:+.3f} / justesse {d_justesse:+.3f}")
        lines.append("")

    return lines


def write_markdown_report(path, title, meta: dict, global_stats: dict,
                           by_category: dict, by_type: dict, flagged: list,
                           comparison: dict = None, limitations: list = None,
                           ablation_csv_path=None):
    lines = [f"# {title}", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}** : {v}")
    lines.append("")

    lines.append("## Résultats globaux")
    lines.append("")
    lines.append("| Métrique | Valeur |")
    lines.append("|---|---|")
    for k, v in global_stats.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Par catégorie")
    lines.append("")
    lines.append("| Catégorie | n | Précision routage | Précision comportement | Rappel éléments | Faithfulness /10 | Correctness /5 | Hit-rate retrieval | Latence moy. (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cat, s in sorted(by_category.items(), key=lambda x: str(x[0])):
        lines.append(f"| {cat} | {s['n']} | {s['route_accuracy']} | {s['behavior_accuracy']} | "
                      f"{s['elements_recall']} | {s['faithfulness']} | {s['correctness']} | "
                      f"{s['hit_rate_hybrid']} | {s['latency_s']} |")
    lines.append("")

    lines.append("## Par type de question")
    lines.append("")
    lines.append("| Type | n | Précision routage | Précision comportement | Rappel éléments | Faithfulness /10 | Correctness /5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for t, s in sorted(by_type.items(), key=lambda x: str(x[0])):
        lines.append(f"| {t} | {s['n']} | {s['route_accuracy']} | {s['behavior_accuracy']} | "
                      f"{s['elements_recall']} | {s['faithfulness']} | {s['correctness']} |")
    lines.append("")

    if comparison:
        lines.append("## Comparaison pipeline GraphRAG vs. baseline RAG simple")
        lines.append("")
        lines.append("| Métrique | Pipeline complet | Baseline (vecteur seul, 1 appel LLM) |")
        lines.append("|---|---|---|")
        for k, (pv, bv) in comparison.items():
            lines.append(f"| {k} | {pv} | {bv} |")
        lines.append("")

    if ablation_csv_path is not None:
        lines.extend(write_ablation_section(ablation_csv_path))

    if flagged:
        lines.append("## Échecs à regarder en priorité")
        lines.append("")
        for f in flagged:
            lines.append(f"- **{f['id']}** ({f.get('category')}/{f.get('type')}) : {f['question']}")
            lines.append(f"  - cause probable : {f.get('cause_probable', '(a investiguer)')}")
        lines.append("")

    if limitations:
        lines.append("## Limites de cette évaluation")
        lines.append("")
        for l in limitations:
            lines.append(f"- {l}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_category_scores(by_category: dict, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = sorted(by_category.keys(), key=lambda x: str(x))
    correctness = [(by_category[c]["correctness"] or 0) for c in cats]
    faithfulness = [(by_category[c]["faithfulness"] or 0) / 2 for c in cats]  # /10 -> /5 pour meme echelle
    recall = [(by_category[c]["elements_recall"] or 0) * 5 for c in cats]     # 0-1 -> 0-5 pour meme echelle

    x = range(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], correctness, width, label="Correctness (/5)", color="#4C72B0")
    ax.bar(list(x), faithfulness, width, label="Faithfulness (/10, mis à l'échelle /5)", color="#55A868")
    ax.bar([i + width for i in x], recall, width, label="Rappel éléments (0-1, mis à l'échelle /5)", color="#C44E52")

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Score (échelle commune /5)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
