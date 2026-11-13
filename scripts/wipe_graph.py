"""
Sauvegarde le graphe Neo4j Aura (noeuds + relations en JSON, dans un dossier
horodate backup_wipe_AAAAMMJJ_HHMMSS/ a la racine du projet) puis le vide
entierement (MATCH (n) DETACH DELETE n). A lancer soi-meme avec
`! python scripts/wipe_graph.py` avant un rebuild complet, quand le graphe
est dans un etat partiel (rebuild precedent interrompu) qu'il faut repartir
de zero.

Reutilise dump_graph_to_json() de backup_graph.py pour la sauvegarde --
memes fichiers nodes.json/relationships.json, juste un dossier horodate
different.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Python ajoute automatiquement le dossier du script a sys.path -- backup_graph
# (meme dossier scripts/) est donc importable directement, sans manipulation.
from backup_graph import dump_graph_to_json

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from neo4j_connection import build_neo4j_driver

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(PROJECT_DIR / ".env")

driver = build_neo4j_driver(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USERNAME"),
    os.getenv("NEO4J_PASSWORD"),
)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = PROJECT_DIR / f"backup_wipe_{stamp}"
n_nodes, n_rels = dump_graph_to_json(driver, out_dir)
print(f"Sauvegarde avant vidage : {n_nodes} noeuds, {n_rels} relations -> {out_dir}/")

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    remaining = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"Graphe vide. Noeuds restants : {remaining}")

driver.close()
