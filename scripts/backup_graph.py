"""
Sauvegarde complete du graphe Neo4j Aura (noeuds + relations) en JSON, avant
toute operation destructive (vidage/rebuild). Rejouable : cree un dossier
./backup_AAAAMMJJ/ (ecrase si deja present le meme jour).
"""
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

today = date.today().strftime("%Y%m%d")
out_dir = Path(f"./backup_{today}")
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

print(f"Sauvegarde terminee : {len(nodes)} noeuds, {len(rels)} relations -> {out_dir}/")
driver.close()
