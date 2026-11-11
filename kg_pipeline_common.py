"""
============================================================
kg_pipeline_common.py -- briques PARTAGEES entre build_knowledge_graph.py
et reprocess_files.py
============================================================
Les deux scripts construisent le meme pipeline d'extraction (meme LLM/
embedder, meme LexicalGraphConfig, meme SimpleKGPipeline) et appliquent le
meme post-traitement deterministe (fusion des doublons de casse/accents,
filet de securite Source sans path -> Citation). Avant ce module, ce code
etait duplique a l'identique dans les deux fichiers -- facteur ici pour
qu'un futur correctif (ex. le bug de troncature deja rencontre, voir
docs/JOURNAL_TRAVAIL.md) n'ait besoin d'etre applique qu'a un seul endroit.
"""
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

LLM_MODEL_NAME = "qwen2.5:7b"

LEXICAL_CONFIG = LexicalGraphConfig(
    chunk_node_label="Article",
    document_node_label="Source",
    chunk_to_document_relationship_type="DANS_SOURCE",
    next_chunk_relationship_type="ARTICLE_SUIVANT",
    node_to_chunk_relationship_type="DANS_ARTICLE",
    chunk_embedding_property="vecteur",
)


def build_llm_and_embedder():
    """Cree le LLM Ollama (corrige + contraint JSON, voir kg_config.py) et
    l'embedder nomic-embed-text -- memes parametres partout ou le pipeline
    d'extraction est lance (build complet ou retraitement cible)."""
    llm = FixedOllamaLLM(
        model_name=LLM_MODEL_NAME,
        model_params={
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,       # fenetre de contexte suffisante pour le prompt many-shot
                "num_predict": 2048,   # plafond de generation par chunk (anti-derive)
            },
            "format": OLLAMA_JSON_SCHEMA,  # contraint reellement la sortie JSON d'Ollama
        },
    )
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embedder


def build_kg_pipeline(driver, llm, embedder) -> SimpleKGPipeline:
    """Cree le SimpleKGPipeline partage, avec perform_entity_resolution=False
    dans tous les cas : le resolveur integre de neo4j_graphrag appelle lui-meme
    apoc.refactor.mergeNodes en fin de traitement et plante ('Node XXX not
    found') quand plusieurs groupes de doublons se chevauchent (constate en
    pratique sur reprocess_files.py, voir docs/JOURNAL_TRAVAIL.md). On
    desactive ce resolveur integre partout et on applique a la place notre
    propre fusion deterministe (merge_case_insensitive_duplicates, plus bas),
    identique quel que soit le script qui lance ce pipeline."""
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        neo4j_database=None,
        embedder=embedder,
        from_pdf=False,
        lexical_graph_config=LEXICAL_CONFIG,
        text_splitter=text_splitter,
        prompt_template=prompt_template,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
        },
        perform_entity_resolution=False,
    )


# ============================================================
# FUSION AUTOMATIQUE PAR CASSE ET PAR ACCENTS (post-traitement deterministe)
# ============================================================
# Le resolver natif (SinglePropertyExactMatchResolver, desactive ci-dessus)
# compare les noms de facon SENSIBLE A LA CASSE : "Patch a la Nicotine" et
# "Patch a la nicotine" restent deux noeuds distincts a ses yeux, meme si le
# prompt demande un nom canonique -- un LLM ne peut pas garantir une casse
# identique a 100% sur des dizaines d'appels independants. Cette requete
# fusionne, APRES coup, tout groupe de noeuds (type + nom normalise par
# apoc.text.clean -- accents + casse + ponctuation) en un seul -- y compris a
# travers des types differents. Priorise le nom AVEC accents comme nom
# survivant (ORDER BY ... quality DESC avant le collect -- mergeNodes garde
# les proprietes du premier noeud de la liste avec properties:'discard').
# Historique complet du diagnostic : docs/JOURNAL_TRAVAIL.md, etape B.1.
MERGE_DUPLICATES_QUERY = """
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

# ============================================================
# FILET DE SECURITE : collision de label avec "Source"
# ============================================================
# Le JSON `format` impose a Ollama ne contraint QUE la structure
# (id/label/properties), pas les valeurs de `label` elles-memes (pas
# d'enum) -- qwen2.5:7b peut donc produire des noeuds `label: "Source"` pour
# des references bibliographiques CITEES dans le texte, en collision avec le
# noeud Source du framework lexical graph (qui, lui, porte toujours une
# propriete `path`). kg_config.py definit un label "Citation" dedie pour la
# prochaine extraction, mais ce filet corrige aussi, de facon deterministe,
# tout noeud `Source` sans `path` qui serait quand meme produit.
# Historique complet du diagnostic : docs/JOURNAL_TRAVAIL.md, etape B.2.
RELABEL_SOURCE_WITHOUT_PATH_QUERY = """
MATCH (s:Source) WHERE s.path IS NULL
REMOVE s:Source
SET s:Citation
RETURN count(s) AS n
"""


def merge_case_insensitive_duplicates(driver) -> int:
    """Execute MERGE_DUPLICATES_QUERY. Retourne le nombre de groupes
    fusionnes, ou -1 si la requete a echoue (erreur deja affichee)."""
    try:
        result = driver.execute_query(MERGE_DUPLICATES_QUERY)
        return result.records[0]["nb_fusionnes"] if result.records else 0
    except Exception as e:
        print(f"   ⚠️ Erreur pendant la fusion automatique : {e}")
        print("   Tu peux relancer manuellement cette requête dans Neo4j Browser si besoin.")
        return -1


def relabel_source_without_path_as_citation(driver) -> int:
    """Execute RELABEL_SOURCE_WITHOUT_PATH_QUERY. Retourne le nombre de
    noeuds relabellises, ou -1 si la requete a echoue (erreur deja
    affichee)."""
    try:
        result = driver.execute_query(RELABEL_SOURCE_WITHOUT_PATH_QUERY)
        return result.records[0]["n"] if result.records else 0
    except Exception as e:
        print(f"   ⚠️ Erreur pendant le relabelling : {e}")
        return -1
