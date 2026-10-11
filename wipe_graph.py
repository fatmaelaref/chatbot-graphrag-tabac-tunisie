"""
Sauvegarde le graphe Neo4j Aura (noeuds + relations en JSON, dans un dossier
horodate ./backup_wipe_AAAAMMJJ_HHMMSS/) puis le vide entierement
(MATCH (n) DETACH DELETE n). A lancer soi-meme avec `! python wipe_graph.py`
avant un rebuild complet, quand le graphe est dans un etat partiel
(rebuild precedent interrompu) qu'il faut repartir de zero.
"""
import json
import sys
from datetime import datetime
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

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f"./backup_wipe_{stamp}")
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
    print(f"Sauvegarde avant vidage : {len(nodes)} noeuds, {len(rels)} relations -> {out_dir}/")

    session.run("MATCH (n) DETACH DELETE n")
    remaining = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"Graphe vide. Noeuds restants : {remaining}")

driver.close()
