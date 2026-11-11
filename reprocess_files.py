"""
============================================================
RETRAITER PLUSIEURS FICHIERS PRECIS (ceux qui ont manque des entites)
============================================================
Reutilise exactement la meme configuration que 2_build_kg_ULTIME.py
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

from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig

from kg_config import (
    text_splitter,
    NODE_TYPES,
    RELATIONSHIP_TYPES,
    prompt_template,
    FixedOllamaLLM,
    OLLAMA_JSON_SCHEMA,
)

# ============================================================
# MODIFIE CETTE LISTE SI TU DOIS RETRAITER D'AUTRES FICHIERS
# ============================================================
FICHIERS_A_RETRAITER = [
    "./textes/SOURCE_URL_SantePubliqueFrance_Consequences.txt",   # PNRT
    "./textes/CHATBOT 1 TABAC LEGISLATION ET DONNEES EPIDEMIO.txt",  # Projet FCTC 20-30
]
# Normalise en slash : 2_build_kg_ULTIME.py fait pareil depuis le 2026-10-07
# (glob.glob renvoie des antislash sur Windows) -- sans ca, ce script ne
# retrouve pas le noeud Source cree par 2_build_kg_ULTIME.py (chemins
# differents malgre le meme fichier) et en cree un deuxieme au lieu de le
# nettoyer avant retraitement (constate en pratique : 2 fichiers dupliques).
FICHIERS_A_RETRAITER = [f.replace("\\", "/") for f in FICHIERS_A_RETRAITER]

load_dotenv()

lexical_config = LexicalGraphConfig(
    chunk_node_label="Article",
    document_node_label="Source",
    chunk_to_document_relationship_type="DANS_SOURCE",
    next_chunk_relationship_type="ARTICLE_SUIVANT",
    node_to_chunk_relationship_type="DANS_ARTICLE",
    chunk_embedding_property="vecteur",
)

LLM_MODEL_NAME = "qwen2.5:7b"

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

llm = FixedOllamaLLM(
    model_name=LLM_MODEL_NAME,
    model_params={
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048},
        "format": OLLAMA_JSON_SCHEMA,
    },
)
embedder = OllamaEmbeddings(model="nomic-embed-text")
print(f"✅ LLM ({LLM_MODEL_NAME}) et embedder chargés")

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=None,
    embedder=embedder,
    from_pdf=False,
    lexical_graph_config=lexical_config,
    text_splitter=text_splitter,
    prompt_template=prompt_template,
    schema={"node_types": NODE_TYPES, "relationship_types": RELATIONSHIP_TYPES},
    perform_entity_resolution=False,  # le resolveur integre de neo4j_graphrag appelle
                                        # lui-meme apoc.refactor.mergeNodes en fin de
                                        # traitement -- constate en pratique : plante
                                        # avec "Node XXX not found" quand plusieurs
                                        # groupes de doublons se chevauchent. On a deja
                                        # notre propre fusion (plus bas, identique a celle
                                        # de 2_build_kg_ULTIME.py), donc on desactive celui-ci.
)
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
    result = neo4j_driver.execute_query(
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

# Fusion des doublons casse/accents/type sur TOUT le graphe -- identique a la
# Section 5bis de 2_build_kg_ULTIME.py (voir ce fichier pour le detail du
# correctif accents du 2026-10-07). Comme le resolveur integre est desactive
# ci-dessus, c'est desormais la SEULE fusion qui s'applique -- deterministe,
# pas de plantage possible sur des noeuds entre-temps disparus (elle recalcule
# les groupes juste avant de fusionner, dans la meme requete).
print("\n" + "=" * 60)
print("FUSION DES DOUBLONS (casse/accents/type)")
print("=" * 60)
merge_case_insensitive_query = """
MATCH (e:__Entity__)
WITH e, apoc.text.clean(e.name) AS cle,
     CASE WHEN e.name =~ '.*[àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ].*' THEN 1 ELSE 0 END AS quality
WHERE cle IS NOT NULL AND cle <> ''
ORDER BY cle, quality DESC
WITH cle, collect(e) AS noeuds
WHERE size(noeuds) > 1
CALL apoc.refactor.mergeNodes(noeuds, {properties: 'discard', mergeRels: true}) YIELD node
RETURN count(node) AS nb_fusionnes
"""
try:
    merge_result = neo4j_driver.execute_query(merge_case_insensitive_query)
    nb = merge_result.records[0]["nb_fusionnes"] if merge_result.records else 0
    print(f"   ✅ {nb} groupe(s) de doublons fusionnés (casse/type)")
except Exception as e:
    print(f"   ⚠️ Erreur pendant la fusion automatique : {e}")

# Filet de securite identique a celui de 2_build_kg_ULTIME.py (section 5ter) :
# un noeud Source sans propriete path est une collision de label (reference
# bibliographique extraite a tort comme Source au lieu de Citation).
try:
    relabel_result = neo4j_driver.execute_query("""
        MATCH (s:Source) WHERE s.path IS NULL
        REMOVE s:Source
        SET s:Citation
        RETURN count(s) AS n
    """)
    nb = relabel_result.records[0]["n"] if relabel_result.records else 0
    print(f"   ✅ {nb} noeud(s) Source sans path relabelisé(s) en Citation")
except Exception as e:
    print(f"   ⚠️ Erreur pendant le relabelling : {e}")

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