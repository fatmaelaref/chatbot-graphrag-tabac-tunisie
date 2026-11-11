"""
============================================================
SCRIPT 2 : CONSTRUIRE LE KNOWLEDGE GRAPH - PROJET TABAC
============================================================
Le splitter, le schema et le prompt sont definis dans kg_config.py
(modifie-les la-bas, pas ici).
"""
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)

import os
import sys

# CORRECTIF ENCODAGE (Windows) : sans ca, rediriger la sortie vers un
# fichier (`> log.txt`) ou lancer depuis certains terminaux Windows fait
# planter le moindre print() contenant un emoji ou un accent, car Python
# bascule alors sur l'encodage par defaut du systeme (cp1252) au lieu
# d'UTF-8. Constate en pratique : crash immediat sur le tout premier
# print apres une connexion Neo4j pourtant reussie.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import csv
import asyncio
import glob
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

load_dotenv()

# ============================================================
# 1. LEXICAL GRAPH CONFIG
# ============================================================

lexical_config = LexicalGraphConfig(
    chunk_node_label="Article",
    document_node_label="Source",
    chunk_to_document_relationship_type="DANS_SOURCE",
    next_chunk_relationship_type="ARTICLE_SUIVANT",
    node_to_chunk_relationship_type="DANS_ARTICLE",
    chunk_embedding_property="vecteur",
)

# ============================================================
# 2. CHOIX DU MODELE
# ============================================================

LLM_MODEL_NAME = "qwen2.5:7b"

# ============================================================
# 3. CONNEXION AUX SERVICES
# ============================================================

print("=" * 60)
print("CONSTRUCTION DU KNOWLEDGE GRAPH - TABAGISME TUNISIE")
print(f"Modele LLM : {LLM_MODEL_NAME}")
print("=" * 60)

REQUIRED_ENV_VARS = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    print(f"❌ Variables manquantes dans .env : {', '.join(missing_vars)}")
    print("   Verifie que ton fichier .env contient bien ces 3 variables.")
    sys.exit(1)

try:
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    neo4j_driver.verify_connectivity()
    print("✅ Connecté à Neo4j")
except Exception as e:
    print(f"❌ Impossible de se connecter à Neo4j : {e}")
    print("   Verifie que Neo4j tourne bien et que les identifiants dans .env sont corrects.")
    sys.exit(1)

try:
    llm = FixedOllamaLLM(
        model_name=LLM_MODEL_NAME,
        model_params={
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,      # fenetre de contexte suffisante pour le prompt many-shot
                "num_predict": 2048,  # PLAFOND de generation par chunk : evite qu'un chunk
                                      # ambigu ne fasse partir le modele dans une reponse
                                      # interminable (protection contre le "chaos"/gaspillage
                                      # de tokens evoque plus tot dans la conversation)
            },
            "format": OLLAMA_JSON_SCHEMA,  # contraint reellement la sortie JSON d'Ollama
        },
    )
    print(f"✅ LLM Ollama ({LLM_MODEL_NAME}) chargé")

    embedder = OllamaEmbeddings(model="nomic-embed-text")
    print("✅ Embedder (nomic-embed-text) chargé")
except Exception as e:
    print(f"❌ Impossible de charger les modeles Ollama : {e}")
    print("   Verifie qu'Ollama tourne (commande 'ollama serve') et que les modeles")
    print(f"   sont bien telecharges ('ollama pull {LLM_MODEL_NAME}' et 'ollama pull nomic-embed-text').")
    neo4j_driver.close()
    sys.exit(1)

# ============================================================
# 4. CREER LE PIPELINE
# ============================================================

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=None,
    embedder=embedder,
    from_pdf=False,
    lexical_graph_config=lexical_config,
    text_splitter=text_splitter,
    prompt_template=prompt_template,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
    },
)
print("✅ Pipeline SimpleKG configuré")

# ============================================================
# 5. TRAITER TOUS LES DOCUMENTS (.txt deja extraits par script 1)
# ============================================================

async def process_all_documents():
    # .replace("\\", "/") : sur Windows, glob.glob() renvoie des chemins avec
    # antislash ("./textes\\Fichier.txt"). Ce chemin sert de CLE D'UNICITE pour
    # le noeud Source (contrainte source_path) -- si un autre script (ex.
    # retraiter_fichiers.py) ou un run ulterieur compare/ecrit ce meme fichier
    # avec des slashs normaux, il ne retrouve pas le noeud existant et en CREE
    # un deuxieme (constate en pratique le 2026-10-07 : 2 fichiers dupliques
    # en Source+Article apres un retraitement cible). On normalise ici une
    # fois pour toutes en slash avant de l'utiliser comme cle.
    text_files = sorted(p.replace("\\", "/") for p in glob.glob("./textes/*.txt"))

    if not text_files:
        print("\n❌ Aucun fichier .txt trouvé dans ./textes/")
        print("   Verifie que le dossier existe et que le script 1 a bien été lancé avant.")
        return 0, []

    print(f"\n📄 {len(text_files)} fichier(s) texte trouvé(s)")

    success_count = 0
    failed_files = []

    for txt_file in text_files:
        print(f"\n🔄 Traitement : {os.path.basename(txt_file)}")
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                file_text = f.read()

            if not file_text.strip():
                print(f"   ⚠️ Fichier vide, ignoré")
                continue

            await kg_builder.run_async(text=file_text, file_path=txt_file)
            print(f"   ✅ Terminé")
            success_count += 1

        except Exception as e:
            print(f"   ❌ Échec sur ce fichier : {e}")
            print(f"   ⏭️ Passage au fichier suivant (les autres ne sont pas affectés)")
            failed_files.append(os.path.basename(txt_file))

    return success_count, failed_files


success_count, failed_files = asyncio.run(process_all_documents())

if failed_files:
    print(f"\n⚠️ {len(failed_files)} fichier(s) ont échoué et n'ont PAS été ajoutés au graphe :")
    for f in failed_files:
        print(f"   - {f}")
    print("   Tu peux relancer le script : les fichiers déjà traités avec succès")
    print("   seront retraités aussi (pas de doublons grâce au MERGE, mais ça prend du temps).")

# ============================================================
# 5bis. FUSION AUTOMATIQUE PAR CASSE ET PAR ACCENTS (post-traitement deterministe)
# ============================================================
# Le resolver natif (SinglePropertyExactMatchResolver) compare les noms de
# facon SENSIBLE A LA CASSE : "Patch a la Nicotine" et "Patch a la nicotine"
# restent deux noeuds distincts a ses yeux, meme si le prompt demande un nom
# canonique -- un LLM ne peut pas garantir une casse identique a 100% sur
# des dizaines d'appels independants. Ce correctif fusionne, APRES coup,
# tout groupe de noeuds (type + nom en minuscules identiques) en un seul --
# y compris a travers des types differents (ex: "Diabete de type 2" en
# Pathologie ET en Concept se retrouvent fusionnes en un seul noeud portant
# les deux etiquettes). Decouvert et applique manuellement le 05/09/2026 ;
# desormais automatique a chaque reconstruction.
#
# Etendu le 2026-10-07 : la cle de regroupement utilisait toLower() seul, qui
# NE RETIRE PAS LES ACCENTS -- "Etats-unis" et "États-Unis" restaient deux
# noeuds Pays distincts (verifie sur le graphe live : 20 groupes de doublons
# purement accentuels, ex. Pays, Produit, Symptome, Concept, Loi). On bascule
# sur apoc.text.clean() (accents + casse + ponctuation normalises) pour la cle
# de regroupement, et on priorise le nom AVEC accents comme nom survivant
# (ORDER BY ... quality DESC avant le collect -- apoc.refactor.mergeNodes
# garde les proprietes du premier noeud de la liste avec properties:'discard').
print("\n" + "=" * 60)
print("FUSION AUTOMATIQUE DES DOUBLONS DE CASSE ET D'ACCENTS")
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
    print("   Tu peux relancer manuellement cette requête dans Neo4j Browser si besoin.")

# ============================================================
# 5ter. FILET DE SECURITE : collision de label avec "Source"
# ============================================================
# Constate en pratique le 2026-10-07 : le JSON `format` impose a Ollama ne
# contraint QUE la structure (id/label/properties), pas les valeurs de
# `label` elles-memes (pas d'enum) -- malgre un schema NODE_TYPES qui ne
# definit pas de label "Source" et malgre additional_node_types=False cote
# GraphSchema, qwen2.5:7b a quand meme produit des noeuds `label: "Source"`
# pour des references bibliographiques CITEES dans le texte (ex: "WHO Report
# on the Global Tobacco Epidemic 2008"), en collision avec le noeud Source
# du framework lexical graph (qui, lui, porte toujours une propriete `path`).
# On a ajoute un label "Citation" dedie dans kg_config.py pour la prochaine
# extraction, mais ce filet de securite corrige aussi, de façon deterministe,
# tout noeud `Source` sans `path` qui serait quand meme produit.
print("\n" + "=" * 60)
print("FILET DE SECURITE : Source sans path -> Citation")
print("=" * 60)
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

# ============================================================
# 6. ENRICHISSEMENT VIA sources.csv
# ============================================================

print("\n" + "=" * 60)
print("ENRICHISSEMENT STRUCTURÉ (via sources.csv)")
print("=" * 60)

theme_cypher = """
MATCH (s:Source)
WHERE s.path CONTAINS $stem
MERGE (t:Thematique {id: $theme})
MERGE (s)-[:TRAITE_DE]->(t)
RETURN s.path AS document
"""

if not os.path.exists("sources.csv"):
    print("⚠️ Fichier sources.csv introuvable — étape d'enrichissement ignorée.")
    print("   Pense à créer/mettre à jour ce fichier pour y inclure aussi les 6")
    print("   nouvelles fiches sources (guide Quit + OMS/CDC/NHS/Santé publique France).")
else:
    with open("sources.csv", encoding="utf-8", newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("⚠️ sources.csv est vide.")
    else:
        for row in rows:
            result = neo4j_driver.execute_query(
                theme_cypher,
                stem=row["stem"],
                theme=row["theme"],
            )
            if not result.records:
                print(f"   ⚠️ Aucun document ne correspond au stem '{row['stem']}' "
                      f"(vérifie l'orthographe exacte du nom de fichier dans sources.csv)")
            for record in result.records:
                print(f"   ✅ {record['document']} -> {row['theme']}")

# ============================================================
# 7. VERIFICATION FINALE
# ============================================================

print("\n" + "=" * 60)
print("VÉRIFICATION DU GRAPHE")
print("=" * 60)

try:
    stats_queries = {
        "Sources (documents)": "MATCH (s:Source) RETURN count(s) AS n",
        "Articles (chunks)": "MATCH (a:Article) RETURN count(a) AS n",
        "Entités totales": "MATCH (e:__Entity__) RETURN count(e) AS n",
    }

    for label, query in stats_queries.items():
        result = neo4j_driver.execute_query(query)
        count = result.records[0]["n"] if result.records else 0
        print(f"   📊 {label} : {count}")

    print("\n📋 Types d'entités extraits :")
    type_query = """
    MATCH (e:__Entity__)
    UNWIND labels(e) AS label
    WITH label, count(e) AS count
    WHERE label <> '__Entity__'
    RETURN label, count
    ORDER BY count DESC
    """
    type_result = neo4j_driver.execute_query(type_query)
    for record in type_result.records:
        print(f"   - {record['label']} : {record['count']}")

except Exception as e:
    print(f"⚠️ Erreur pendant la vérification finale : {e}")
    print("   Le graphe a probablement quand même été construit — vérifie directement dans Neo4j Browser.")

print(f"\n🎉 TRAITEMENT TERMINÉ : {success_count} document(s) ajouté(s) au graphe avec succès")
if failed_files:
    print(f"   ({len(failed_files)} document(s) en échec, voir détails ci-dessus)")
print("=" * 60)

# Fermeture propre
neo4j_driver.close()