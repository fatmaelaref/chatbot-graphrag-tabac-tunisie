"""
Sauvegarde complete du graphe Neo4j Aura (noeuds + relations) en JSON, avant
toute operation destructive (vidage/rebuild). Rejouable : cree un dossier
backup_AAAAMMJJ/ a la racine du projet (ecrase si deja present le meme jour).

Expose aussi dump_graph_to_json(), reutilisee par wipe_graph.py pour ne pas
dupliquer la logique de sauvegarde.

Usage (depuis la racine du projet) : python scripts/backup_graph.py
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(PROJECT_DIR / ".env")


def dump_graph_to_json(driver, out_dir: Path) -> tuple[int, int]:
    """Exporte tous les noeuds/relations du graphe en JSON dans out_dir
    (nodes.json / relationships.json). Retourne (nb_noeuds, nb_relations)."""
    out_dir.mkdir(exist_ok=True)

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
        ).data()
        rels = session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS start, elementId(b) AS end, "
            "type(r) AS type, properties(r) AS props"
        ).data()

    with open(out_dir / "nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(out_dir / "relationships.json", "w", encoding="utf-8") as f:
        json.dump(rels, f, ensure_ascii=False, indent=2)

    return len(nodes), len(rels)


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    today = date.today().strftime("%Y%m%d")
    out_dir = PROJECT_DIR / f"backup_{today}"
    n_nodes, n_rels = dump_graph_to_json(driver, out_dir)

    print(f"Sauvegarde terminee : {n_nodes} noeuds, {n_rels} relations -> {out_dir}/")
    driver.close()
