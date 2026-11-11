"""
============================================================
AJOUTER MANUELLEMENT : noeud Programme "Projet FCTC 20-30"
============================================================
qwen2.5:7b n'a jamais extrait cette entite automatiquement malgre plusieurs
tentatives (le corpus ne la mentionne qu'une fois, entre parentheses, dans
une diapositive sur un tout autre sujet -- l'arrete de 2022 sur les
avertissements sanitaires). Script dedie a relancer APRES chaque rebuild
complet (build_knowledge_graph.py vide le graphe, donc efface ce noeud).

Idempotent : utilise MERGE, donc rejouable sans creer de doublon.
Retrouve le chunk et la loi associee par le CONTENU du texte (pas par un
identifiant Neo4j, qui change a chaque rebuild).

Usage (depuis la racine du projet) : python scripts/add_fctc_programme.py
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
driver.verify_connectivity()
print("✅ Connecté à Neo4j")

# Etape 1 : rattacher le noeud Programme a son chunk. Separee de l'etape 2
# (association a la Loi) pour pouvoir diagnostiquer precisement LAQUELLE des
# deux echoue si besoin -- avant cette separation, le message d'erreur
# parlait toujours de "chunk introuvable" meme quand le vrai probleme etait
# "Loi introuvable" (le noeud Programme etait alors deja cree silencieusement).
query_programme = """
MATCH (a:Article)
WHERE a.text CONTAINS "arrêté du 21 janvier 2022" OR a.text CONTAINS "arrete du 21 janvier 2022"
WITH a LIMIT 1
MERGE (prog:Programme {name: "Projet FCTC 20-30"})
MERGE (prog)-[:DANS_ARTICLE]->(a)
RETURN prog.name AS programme
"""

# Etape 2 : associer ce Programme a la Loi qui le mentionne (si elle existe
# deja dans le graphe -- depend de l'ordre d'extraction des fichiers).
query_link_loi = """
MATCH (prog:Programme {name: "Projet FCTC 20-30"})
MATCH (loi:Loi) WHERE toLower(loi.name) CONTAINS "21 janvier 2022"
MERGE (prog)-[:ASSOCIE_A]->(loi)
RETURN loi.name AS loi_associee
"""

result_programme = driver.execute_query(query_programme)
if not result_programme.records:
    print("⚠️ Aucun chunk contenant 'arrêté du 21 janvier 2022' trouvé -- "
          "le noeud Programme n'a pas pu être rattaché à un chunk. "
          "Vérifie manuellement dans Neo4j Browser.")
else:
    print(f"✅ Noeud Programme ajouté/vérifié : {result_programme.records[0]['programme']}")

    result_loi = driver.execute_query(query_link_loi)
    if result_loi.records:
        for r in result_loi.records:
            print(f"✅ Relation ASSOCIE_A créée/vérifiée vers la Loi : {r['loi_associee']}")
    else:
        print("⚠️ Aucun noeud Loi contenant '21 janvier 2022' trouvé -- le noeud Programme "
              "a bien été créé/rattaché à son chunk, mais PAS relié à une Loi (vérifie que "
              "le build/retraitement a bien extrait cette Loi). Vérifie manuellement dans "
              "Neo4j Browser.")

driver.close()
