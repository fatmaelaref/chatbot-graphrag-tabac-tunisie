"""
============================================================
AJOUTER MANUELLEMENT : noeud Programme "Projet FCTC 20-30"
============================================================
qwen2.5:7b n'a jamais extrait cette entite automatiquement malgre plusieurs
tentatives (le corpus ne la mentionne qu'une fois, entre parentheses, dans
une diapositive sur un tout autre sujet -- l'arrete de 2022 sur les
avertissements sanitaires). Script dedie a relancer APRES chaque rebuild
complet (2_build_kg_ULTIME.py vide le graphe, donc efface ce noeud).

Idempotent : utilise MERGE, donc rejouable sans creer de doublon.
Retrouve le chunk et la loi associee par le CONTENU du texte (pas par un
identifiant Neo4j, qui change a chaque rebuild).
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
driver.verify_connectivity()
print("✅ Connecté à Neo4j")

query = """
MATCH (a:Article)
WHERE a.text CONTAINS "arrêté du 21 janvier 2022" OR a.text CONTAINS "arrete du 21 janvier 2022"
WITH a LIMIT 1
MERGE (prog:Programme {name: "Projet FCTC 20-30"})
MERGE (prog)-[:DANS_ARTICLE]->(a)
WITH prog
MATCH (loi:Loi) WHERE toLower(loi.name) CONTAINS "21 janvier 2022"
MERGE (prog)-[:ASSOCIE_A]->(loi)
RETURN prog.name AS programme, loi.name AS loi_associee
"""

result = driver.execute_query(query)
if result.records:
    for r in result.records:
        print(f"✅ Ajouté/vérifié : {r['programme']} -- ASSOCIE_A --> {r['loi_associee']}")
else:
    print("⚠️ Aucun chunk contenant 'arrêté du 21 janvier 2022' trouvé -- "
          "le noeud Loi correspondant doit exister pour que ASSOCIE_A se crée. "
          "Vérifie manuellement dans Neo4j Browser.")

driver.close()
