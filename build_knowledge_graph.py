"""
============================================================
SCRIPT 2 : CONSTRUIRE LE KNOWLEDGE GRAPH - PROJET TABAC
============================================================
Le splitter, le schema et le prompt sont definis dans kg_config.py
(modifie-les la-bas, pas ici). Le LLM/embedder, le pipeline SimpleKGPipeline
et le post-traitement (fusion de doublons, filet Source->Citation) sont
partages avec reprocess_files.py via kg_pipeline_common.py.
"""
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)

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
from neo4j_connection import build_neo4j_driver

from kg_pipeline_common import (
    LLM_MODEL_NAME,
    build_llm_and_embedder,
    build_kg_pipeline,
    merge_case_insensitive_duplicates,
    relabel_source_without_path_as_citation,
)

load_dotenv()

# ============================================================
# CONNEXION AUX SERVICES
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
    neo4j_driver = build_neo4j_driver(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    )
    neo4j_driver.verify_connectivity()
    print("✅ Connecté à Neo4j")
except Exception as e:
    print(f"❌ Impossible de se connecter à Neo4j : {e}")
    print("   Verifie que Neo4j tourne bien et que les identifiants dans .env sont corrects.")
    sys.exit(1)

try:
    llm, embedder = build_llm_and_embedder()
    print(f"✅ LLM Ollama ({LLM_MODEL_NAME}) chargé")
    print("✅ Embedder (nomic-embed-text) chargé")
except Exception as e:
    print(f"❌ Impossible de charger les modeles Ollama : {e}")
    print("   Verifie qu'Ollama tourne (commande 'ollama serve') et que les modeles")
    print(f"   sont bien telecharges ('ollama pull {LLM_MODEL_NAME}' et 'ollama pull nomic-embed-text').")
    neo4j_driver.close()
    sys.exit(1)

# ============================================================
# CREER LE PIPELINE
# ============================================================

kg_builder = build_kg_pipeline(neo4j_driver, llm, embedder)
print("✅ Pipeline SimpleKG configuré")

# ============================================================
# TRAITER TOUS LES DOCUMENTS (.txt deja extraits par extract_documents.py)
# ============================================================

async def process_all_documents():
    # .replace("\\", "/") : sur Windows, glob.glob() renvoie des chemins avec
    # antislash ("./textes\\Fichier.txt"). Ce chemin sert de CLE D'UNICITE pour
    # le noeud Source (contrainte source_path) -- si un autre script (ex.
    # reprocess_files.py) ou un run ulterieur compare/ecrit ce meme fichier
    # avec des slashs normaux, il ne retrouve pas le noeud existant et en CREE
    # un deuxieme (constate en pratique le 2026-10-07 : 2 fichiers dupliques
    # en Source+Article apres un retraitement cible). On normalise ici une
    # fois pour toutes en slash avant de l'utiliser comme cle.
    text_files = sorted(p.replace("\\", "/") for p in glob.glob("./textes/*.txt"))

    if not text_files:
        print("\n❌ Aucun fichier .txt trouvé dans ./textes/")
        print("   Verifie que le dossier existe et que extract_documents.py a bien été lancé avant.")
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
    print("   ATTENTION avant de relancer ce script : le writer de neo4j_graphrag fait un")
    print("   CREATE (pas un MERGE) sur le noeud Source a chaque run -- relancer tel quel")
    print("   dupliquerait les fichiers déjà traités avec succès. Utilise plutot")
    print("   reprocess_files.py (qui nettoie l'ancien noeud Source avant de retraiter)")
    print("   pour ne cibler que les fichiers en échec, ou vide le graphe avant de relancer")
    print("   ce script en entier (voir scripts/wipe_graph.py).")

# ============================================================
# FUSION DES DOUBLONS + FILET DE SECURITE (post-traitement partage,
# voir kg_pipeline_common.py pour le detail de chaque requete)
# ============================================================
print("\n" + "=" * 60)
print("FUSION AUTOMATIQUE DES DOUBLONS DE CASSE ET D'ACCENTS")
print("=" * 60)
nb = merge_case_insensitive_duplicates(neo4j_driver)
if nb >= 0:
    print(f"   ✅ {nb} groupe(s) de doublons fusionnés (casse/accents/type)")

print("\n" + "=" * 60)
print("FILET DE SECURITE : Source sans path -> Citation")
print("=" * 60)
nb = relabel_source_without_path_as_citation(neo4j_driver)
if nb >= 0:
    print(f"   ✅ {nb} noeud(s) Source sans path relabelisé(s) en Citation")

# ============================================================
# ENRICHISSEMENT VIA sources.csv
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
# VERIFICATION FINALE
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