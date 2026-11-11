"""
============================================================
RETRAITER PLUSIEURS FICHIERS PRECIS (ceux qui ont manque des entites)
============================================================
Reutilise exactement la meme configuration que build_knowledge_graph.py
(meme schema, meme prompt, meme modele) mais ne traite que les
fichiers listes ci-dessous, au lieu des 10 -- utile quand certaines
entites (ex: PNRT, Projet FCTC 20-30) n'ont pas ete extraites lors
du dernier run complet. Grace au MERGE, aucun risque de doublon :
tout ce qui a deja ete extrait de ces fichiers sera juste retrouve
et complete, pas duplique.
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_pipeline_common import (
    LLM_MODEL_NAME,
    build_llm_and_embedder,
    build_kg_pipeline,
    merge_case_insensitive_duplicates,
    relabel_source_without_path_as_citation,
)

# ============================================================
# MODIFIE CETTE LISTE SI TU DOIS RETRAITER D'AUTRES FICHIERS
# ============================================================
FICHIERS_A_RETRAITER = [
    "./textes/SOURCE_URL_SantePubliqueFrance_Consequences.txt",   # PNRT
    "./textes/CHATBOT 1 TABAC LEGISLATION ET DONNEES EPIDEMIO.txt",  # Projet FCTC 20-30
]
# Normalise en slash : build_knowledge_graph.py fait pareil depuis le 2026-10-07
# (glob.glob renvoie des antislash sur Windows) -- sans ca, ce script ne
# retrouve pas le noeud Source cree par build_knowledge_graph.py (chemins
# differents malgre le meme fichier) et en cree un deuxieme au lieu de le
# nettoyer avant retraitement (constate en pratique : 2 fichiers dupliques).
FICHIERS_A_RETRAITER = [f.replace("\\", "/") for f in FICHIERS_A_RETRAITER]

load_dotenv()

print("=" * 60)
print(f"RETRAITEMENT DE {len(FICHIERS_A_RETRAITER)} FICHIER(S) CIBLE(S)")
print("=" * 60)

for fichier in FICHIERS_A_RETRAITER:
    if not os.path.exists(fichier):
        print(f"❌ Fichier introuvable : {fichier}")
        print("   Verifie le chemin/nom exact (attention aux accents et espaces).")
        sys.exit(1)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()
print("✅ Connecté à Neo4j")

llm, embedder = build_llm_and_embedder()
print(f"✅ LLM ({LLM_MODEL_NAME}) et embedder chargés")

# perform_entity_resolution=False (voir kg_pipeline_common.build_kg_pipeline) :
# le resolveur integre de neo4j_graphrag appelle lui-meme apoc.refactor.mergeNodes
# en fin de traitement -- constate en pratique : plante avec "Node XXX not found"
# quand plusieurs groupes de doublons se chevauchent. On a deja notre propre
# fusion (plus bas), donc on desactive celui-ci.
kg_builder = build_kg_pipeline(neo4j_driver, llm, embedder)
print("✅ Pipeline configuré")


def _nettoyer_source_existante(fichier: str) -> None:
    """
    Supprime le noeud Source (et ses chunks Article) pour CE fichier precis,
    s'il existe deja -- constate en pratique : le writer de neo4j_graphrag
    fait un CREATE (pas un MERGE) sur le noeud Source a chaque run, donc
    relancer ce script une 2e fois sur un fichier deja traite plante
    systematiquement sur la contrainte d'unicite 'source_path'. On nettoie
    donc AVANT chaque traitement pour rendre ce script rejouable autant de
    fois que necessaire, sans jamais ce conflit -- les entites (Programme,
    Loi, etc.) deja extraites ne sont PAS touchees ici, seulement le noeud
    Source et ses chunks Article, qui seront recrees proprement juste apres.
    """
    neo4j_driver.execute_query(
        """
        MATCH (s:Source {path: $path})
        OPTIONAL MATCH (a:Article)-[:DANS_SOURCE]->(s)
        DETACH DELETE s, a
        RETURN count(s) AS nb
        """,
        path=fichier,
    )
    print(f"   🧹 Ancien noeud Source nettoyé pour ce fichier (si present)")


async def retraiter_tout():
    for fichier in FICHIERS_A_RETRAITER:
        print(f"\n🔄 Traitement : {os.path.basename(fichier)}")
        _nettoyer_source_existante(fichier)
        with open(fichier, 'r', encoding='utf-8') as f:
            file_text = f.read()
        try:
            await kg_builder.run_async(text=file_text, file_path=fichier)
            print(f"   ✅ Terminé")
        except Exception as e:
            print(f"   ❌ Échec sur ce fichier : {e}")
            print(f"   ⏭️ Passage au fichier suivant")


asyncio.run(retraiter_tout())

# Fusion des doublons + filet de securite sur TOUT le graphe -- memes requetes
# partagees que build_knowledge_graph.py (voir kg_pipeline_common.py). Comme
# le resolveur integre est desactive ci-dessus, c'est desormais la SEULE
# fusion qui s'applique -- deterministe, pas de plantage possible sur des
# noeuds entre-temps disparus (elle recalcule les groupes juste avant de
# fusionner, dans la meme requete).
print("\n" + "=" * 60)
print("FUSION DES DOUBLONS (casse/accents/type)")
print("=" * 60)
nb = merge_case_insensitive_duplicates(neo4j_driver)
if nb >= 0:
    print(f"   ✅ {nb} groupe(s) de doublons fusionnés (casse/accents/type)")

nb = relabel_source_without_path_as_citation(neo4j_driver)
if nb >= 0:
    print(f"   ✅ {nb} noeud(s) Source sans path relabelisé(s) en Citation")

# Verification rapide : les entites cherchees sont-elles bien la maintenant ?
# Recherche elargie (pas juste "pnrt"/"fctc 20-30") au cas ou le nom extrait
# varie legerement (ex: "PNRT (2016-2020)", "Projet FCTC 20-30 (2022)").
print("\n" + "=" * 60)
print("VÉRIFICATION")
print("=" * 60)
check_query = """
MATCH (e:__Entity__)
WHERE toLower(e.name) CONTAINS 'pnrt'
   OR toLower(e.name) CONTAINS 'fctc 20-30'
   OR toLower(e.name) CONTAINS 'fctc 20'
   OR toLower(e.name) CONTAINS 'programme national de reduction'
   OR toLower(e.name) CONTAINS 'programme national de lutte'
RETURN labels(e) AS type, e.name AS nom
"""
result = neo4j_driver.execute_query(check_query)
if result.records:
    for r in result.records:
        print(f"   ✅ Trouvé : {r['nom']} ({r['type']})")
else:
    print("   ⚠️ Toujours introuvable -- reessaie une nouvelle fois, ou verifie manuellement dans Neo4j Browser.")

neo4j_driver.close()
print("\n🎉 Retraitement terminé.")