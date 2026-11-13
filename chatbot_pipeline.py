"""
============================================================
SCRIPT 3 (v2) : CHATBOT GROUNDED CONTROLE PAR LANGGRAPH
PROJET TABAC TUNISIE
============================================================
Ancienne version : un seul appel LLM (GraphRAG "boite noire"), zero-shot,
tous les chunks recuperes envoyes tels quels au LLM, aucun controle de
la qualite du retrieval avant de faire confiance a la reponse.

Cette version applique le pattern Corrective-RAG / Adaptive-RAG (celui
documente par LangChain/LangGraph pour les agents de retrieval, et
recommande par la litterature RAG medicale 2025-2026 : hybrid retrieval
+ reranking + prompting conservateur + verification de groundedness
avant de livrer une reponse -- voir sources discutees en chat) :

question utilisateur
   |
   v
[1] route_query          -- classification de securite/domaine, AVANT tout
   |                          retrieval (ne gaspille pas de recherche sur
   |                          une question hors-sujet ou dangereuse)
   |
   |-- hors_domaine / urgence / demande_personnelle --> reponse cadree, FIN
   |
   v (route == "normal")
[2] rewrite_query        -- reformule la requete AVANT qu'elle atteigne
   |                          l'index (vocabulaire du corpus, expansion des
   |                          synonymes/abreviations)
   v
[3] hybrid_retrieve       -- recherche vectorielle (index Neo4j) EN PARALLELE
   |                          d'une recherche structuree (Cypher direct sur
   |                          les entites du graphe) -- deux canaux, pas un
   |                          seul
   v
[4] grade_evidence        -- reranking : le LLM note CHAQUE passage candidat
   |                          (pertinent oui/non + raison). Fait office de
   |                          reranker cross-encoder en l'absence d'un tel
   |                          modele dans la stack locale (voir note plus
   |                          bas sur comment le remplacer par un vrai
   |                          cross-encoder si tu veux monter en precision).
   |                          Le LLM de generation en [6] ne verra JAMAIS
   |                          les candidats rejetes ici.
   v
[5] arete conditionnelle (PAS un noeud LLM -- juste un test sur l'etat)
   |
   |-- preuves insuffisantes ET retries restants --> retour a [2]
   |-- preuves insuffisantes ET plus de retries   --> reponse d'abstention, FIN
   |-- preuves suffisantes                        --> continue
   v
[6] generate_answer       -- prompt many-shot + chain-of-thought interne
   |                          (instructions explicites sur COMMENT raisonner,
   |                          pas juste sur le format de sortie). Contexte =
   |                          UNIQUEMENT les preuves retenues en [4].
   v
[7] validate_groundedness -- un second appel LLM (role de "critique")
   |                          verifie que chaque affirmation de la reponse
   |                          est bien supportee par les preuves fournies ;
   |                          une seule correction est tentee si besoin
   |                          (jamais de boucle infinie)
   v
[8] finalize              -- reponse + sources + avertissement medical

Controle des boucles : MAX_REWRITES et MAX_GROUNDEDNESS_FIXES sont des
constantes explicites en tete de fichier. Aucun noeud ne peut se rappeler
lui-meme indefiniment -- LangGraph rend cet etat inspectable a chaque etape
(state["retry_count"], state["groundedness_fix_count"], ...), ce qui est
le point demande : savoir exactement ce que fait l'agent a chaque etape,
plutot qu'un LLM en boucle non controlee qui brule des tokens.
"""

import os
import re
import copy
import json
import unicodedata
from pathlib import Path
from typing import TypedDict, Optional

from openai import OpenAI  # utilise aussi pour Groq -- meme librairie, juste une autre adresse
from dotenv import load_dotenv
from neo4j_connection import build_neo4j_driver
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================
# 0. CONSTANTES DE CONTROLE (pas de boucle LLM non bornee)
# ============================================================
# GROQ (API distante, gratuite) au lieu d'Ollama en local : supprime
# entierement le probleme de RAM/swap qui causait 20-30 min par question.
# Les modeles tournent sur les serveurs de Groq, plus sur cette machine.
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Retries avec backoff exponentiel sur les erreurs 429 (rate limit) du tier
# gratuit Groq (~8000 tokens/min) -- ajoute le 2026-10-07 : un script qui
# enchaine des dizaines de questions (evaluation) declenche regulierement des
# 429 sans ce correctif, ce qui faisait planter l'evaluation en cours de
# route. Monkeypatch au niveau du client, donc s'applique a TOUS les appels
# Groq du fichier (generation, routage, reecriture, grading, groundedness) et
# aux juges LLM de l'evaluation (qui reutilisent ce meme client).
import time as _time


def _groq_create_with_retry(_original_create, *args, **kwargs):
    max_attempts = 5
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return _original_create(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if not is_rate_limit or attempt == max_attempts:
                raise
            print(f"   ⏳ Groq 429 (tentative {attempt}/{max_attempts}) -- pause {delay:.0f}s")
            _time.sleep(delay)
            delay *= 2


_original_groq_chat_create = groq_client.chat.completions.create
groq_client.chat.completions.create = (
    lambda *a, **kw: _groq_create_with_retry(_original_groq_chat_create, *a, **kw)
)

GENERATION_MODEL = "openai/gpt-oss-120b"   # modele plus capable, pour la reponse finale
CONTROL_MODEL = "openai/gpt-oss-20b"       # modele plus leger/rapide, pour les taches de
                                            # controle (routage, reecriture, grading, critique)
GRADE_MODEL = CONTROL_MODEL

MAX_REWRITES = 1                # nb max de reformulations de requete
MAX_GROUNDEDNESS_FIXES = 1      # nb max de corrections post-generation
TOP_K_VECTOR = 8                # candidats bruts recherche vectorielle
MAX_GRAPH_CANDIDATES = 12       # plafond general du canal Cypher
USE_GRAPH = os.getenv("USE_GRAPH", "1") != "0"  # desactive UNIQUEMENT le canal
                                 # Cypher/graphe dans hybrid_retrieve (ablation) --
                                 # par defaut toujours actif (True), comportement
                                 # inchange pour tout le reste (routage, reecriture,
                                 # notation, generation, verification). Mettre
                                 # USE_GRAPH=0 dans l'environnement pour le desactiver.
MIN_RELEVANT_EVIDENCE = 1       # seuil minimal de preuves retenues pour repondre
MAX_EVIDENCE_TO_LLM = 4         # plafond de preuves envoyees au LLM de generation
                                 # (protection anti-"context dump" -- le LLM ne doit
                                 # pas voir toutes les reponses recuperees)

# Troncature des chunks montres aux LLM (caracteres). Avant le 2026-10-09, ces
# 3 valeurs etaient codees en dur a chaque site d'appel -- un vrai bug de justesse
# (pas seulement de style) est ne de cette duplication : GRADING_TRUNCATE_CHARS
# valait 700 partout, ce qui rendait invisible au LLM de grading un fait situe
# a la position 1407/1776 caracteres d'un chunk, cause d'une partie des
# abstentions injustifiees (Q42, Q61, Q65 -- voir JOURNAL_TRAVAIL.md). Releve a
# 1500 (couvre la quasi-totalite des chunks observes). Les 2 autres sites
# (generation, verification groundedness) ne portent que sur au plus
# MAX_EVIDENCE_TO_LLM preuves deja filtrees -- cout marginal, laisses a 2000.
GRADING_TRUNCATE_CHARS = 1500
GENERATION_TRUNCATE_CHARS = 2000
GROUNDEDNESS_TRUNCATE_CHARS = 2000

# ============================================================
# MODE STRICT (saturation Groq) -- desactive par defaut
# ============================================================
# False par defaut : AUCUN changement de comportement pour evaluate.py,
# baseline.py, ablation_graphe.py ni les tests -- ils continuent de recevoir
# les reponses de repli existantes (route='normal', grading conservateur a 3
# candidats, etc.), necessaires pour que leurs metriques restent definies et
# comparables dans le temps.
# app.py (l'interface utilisateur reelle) met ce flag a True explicitement
# apres avoir charge ce module : pour un vrai utilisateur, mieux vaut dire
# clairement "service sature, reessaie plus tard" que de livrer une reponse
# degradee (routage ou notation des preuves tombes en repli faute de quota
# Groq) en laissant croire qu'elle est fiable.
STRICT_LLM_FAILURE_MODE = False


def _sanitize_llm_error(exc: Exception) -> str:
    """Message affichable a un utilisateur final -- jamais le texte brut
    d'une exception API (peut contenir un identifiant d'organisation/projet
    Groq ou d'autres details internes). Ne distingue que le cas quota (429)
    du reste, qui reste generique."""
    text = str(exc)
    if "429" in text or "rate_limit" in text.lower():
        return "Quota Groq atteint (429)"
    return "Erreur du service de generation"


driver = build_neo4j_driver(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USERNAME"),
    os.getenv("NEO4J_PASSWORD"),
)
embedder = OllamaEmbeddings(model="nomic-embed-text")  # embeddings restent en local -- rapides,
                                                         # aucun souci de vitesse constate dessus


retrieval_query = """
MATCH (node)-[:DANS_SOURCE]->(d)
RETURN node.text AS text, score, d.path AS source
"""


def _format_vector_result(record) -> RetrieverResultItem:
    """
    Formatteur explicite pour VectorCypherRetriever.
    SANS CA, neo4j_graphrag met par defaut l'enregistrement Neo4j entier
    (texte + score + source concatenes) dans item.content, et item.metadata
    ne contient pas forcement "source" au bon endroit -- ce qui pollue le
    contexte envoye au LLM (constate en pratique : "source inconnue" dans
    les sorties, et des chiffres de produits differents fusionnes par le
    LLM parce qu'il recevait un texte mal decoupe). Ici on force : le texte
    brut de l'article va dans .content, tout le reste va dans .metadata.
    """
    return RetrieverResultItem(
        content=record.get("text", ""),
        metadata={"score": record.get("score"), "source": record.get("source")},
    )


vector_retriever = VectorCypherRetriever(
    driver,
    neo4j_database=None,
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
    result_formatter=_format_vector_result,
)


# ============================================================
# 1. UTILITAIRES
# ============================================================
def clean_chunk(text: str) -> str:
    """
    Nettoie les marqueurs de diapositive/page et les artefacts d'extraction.
    IMPORTANT : on garde les retours a la ligne (une ligne = un saut), on ne
    les remplace PAS par des espaces. Plusieurs chunks sont des tableaux
    markdown (ex: dosages de 5 produits TSN dans un seul tableau) -- aplatir
    tout en une seule ligne detruit la structure ligne/colonne et rend les
    chiffres ambigus pour un petit LLM local (constate en pratique : melange
    des dosages patch/gomme quand le tableau est aplati).
    """
    if not text:
        return ""
    text = re.sub(r'=== DIAPOSITIVE \d+ ===', '', text)
    text = re.sub(r'=== PAGE \d+ ===', '', text)
    text = re.sub(r'=== TABLEAU \d* ===', '', text)
    text = re.sub(r'=== FIN TABLEAU ===', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)   # au plus une ligne vide entre blocs
    text = re.sub(r'[ \t]+', ' ', text)       # espaces/tabs multiples -> un seul
    text = re.sub(r' *\n *', '\n', text)      # nettoie les espaces autour des \n
    return text.strip()


def short_source(path: str) -> str:
    if not path:
        return "source inconnue"
    return path.split("\\")[-1].split("/")[-1]


# ============================================================
# METADONNEES DE SOURCE (noms lisibles + liens cliquables)
# ============================================================
# sources_meta.json fait le lien entre le nom de fichier brut (short_source(),
# utilise partout comme cle d'identification interne -- state["sources"],
# evaluation/, tests/) et un affichage propre (nom lisible, organisme, URL
# officielle verifiee -- voir docs/JOURNAL_TRAVAIL.md pour la verification).
# Separe explicitement : l'IDENTITE d'une source (le nom de fichier) ne
# change jamais, seul son AFFICHAGE change ici.
def _load_sources_meta() -> dict:
    path = Path(__file__).resolve().parent / "sources_meta.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


SOURCES_META = _load_sources_meta()


def get_source_meta(filename: str) -> dict:
    """Metadonnees completes d'une source (display_name, organization, url)
    pour l'interface. Fallback sur le nom de fichier brut et des champs None
    si la source n'est pas (encore) documentee dans sources_meta.json --
    n'echoue jamais, pour ne jamais casser l'affichage sur un fichier non
    repertorie."""
    return SOURCES_META.get(filename) or {
        "display_name": filename, "organization": None, "url": None,
    }


def display_source(filename: str) -> str:
    """Nom lisible d'une source, pour les citations inline montrees au LLM
    de generation et pour la liste finale affichee a l'utilisateur."""
    return get_source_meta(filename)["display_name"]


FINAL_ANSWER_MARKER = "### REPONSE FINALE ###"
_FINAL_ANSWER_RE = re.compile(r'#{0,10}\s*R[EÉ]PONSE\s+FINALE\s*#{0,10}', re.IGNORECASE)


def extract_final_answer(raw_text: str) -> str:
    """
    Ne garde que ce qui suit le marqueur -- meme si le modele a quand meme
    ecrit son raisonnement avant (l'instruction seule ne suffit pas
    toujours a l'en empecher). Utilise une regex plutot qu'une recherche
    de saut de ligne : constate en pratique que le modele met parfois le
    marqueur ET la reponse sur la MEME ligne (ex: "5. ### REPONSE FINALE
    ### Le vapotage..."), ce qu'une simple recherche de "\\n" apres le
    marqueur ne coupe pas correctement. Si le marqueur est absent, on
    retombe sur le texte complet plutot que de renvoyer une chaine vide.
    """
    if not raw_text:
        return raw_text
    match = _FINAL_ANSWER_RE.search(raw_text)
    if match:
        after = raw_text[match.end():].strip(" \n:-")
        if after:
            return after
        # Le marqueur est present mais rien ne suit -- reponse tronquee par
        # manque de budget de tokens (constate en pratique : le raisonnement
        # etape par etape ecrit AVANT le marqueur peut consommer tout le
        # budget max_completion_tokens avant que la reponse elle-meme ne
        # soit ecrite). Plutot que de renvoyer une chaine vide a
        # l'utilisateur, on retombe sur le raisonnement qui PRECEDE le
        # marqueur : il contient generalement deja l'information demandee,
        # juste pas reformule proprement -- toujours mieux qu'une reponse
        # vide.
        before = raw_text[:match.start()].strip()
        return before if before else raw_text.strip()
    # Fallback si le marqueur est absent : le modele numerote souvent ses
    # etapes de raisonnement (1. ... 2. ... 5. <reponse finale>) meme sans
    # marqueur explicite -- on recupere alors le contenu du DERNIER point
    # numerote, qui contient generalement la reponse composee a la fin.
    last_steps = list(re.finditer(r'^\s*\d+[\.\)]\s*', raw_text, re.MULTILINE))
    if last_steps:
        return raw_text[last_steps[-1].end():].strip()
    return raw_text.strip()


def _to_strict_json_schema(schema: dict) -> dict:
    """
    Convertit un schema JSON "normal" (comme ROUTE_SCHEMA, GRADE_SCHEMA,
    GROUNDEDNESS_SCHEMA plus bas) vers la forme exigee par le mode strict de
    l'API Groq/OpenAI (response_format={"type": "json_schema", "strict":
    True, ...}). Ce mode EXIGE, sur CHAQUE objet du schema et recursivement
    (y compris dans les objets imbriques dans un tableau) :
      - "additionalProperties": false
      - "required" listant EXACTEMENT toutes les cles de "properties"
    Verifie empiriquement le 05/10/2026 : sans ce correctif, l'API renvoie
    une erreur 400 explicite pour les 3 schemas utilises dans ce fichier.
    """
    schema = copy.deepcopy(schema)

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(schema)
    return schema


def call_llm_json(model: str, system: str, user: str, json_schema: dict) -> Optional[dict]:
    """
    Appel Groq contraint en JSON (routage, reecriture, grading, generation,
    verification de groundedness -- tous les appels de controle du pipeline
    passent par cette fonction).
    Retourne None si le parsing echoue -- le noeud appelant DOIT gerer ce
    cas explicitement (fallback prudent), jamais planter la boucle.

    CORRECTIF (05/10/2026) : le parametre json_schema etait accepte mais
    JAMAIS transmis a l'API -- response_format restait hardcode sur
    {"type": "json_object"}, qui garantit un JSON syntaxiquement valide mais
    AUCUNE contrainte sur sa forme (champs manquants, mauvais index de base,
    etc. -- cause racine plausible des contournements deja presents ailleurs
    dans ce fichier : decalage d'index 0/1-based dans grade_evidence,
    "0/20 preuves retenues sans erreur explicite"). Le schema est desormais
    reellement applique via le mode json_schema strict de l'API Groq.
    """
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": _to_strict_json_schema(json_schema),
                    "strict": True,
                },
            },
            temperature=0.0,
            reasoning_effort="low",      # tache de jugement/classification courte -- pas
                                          # besoin de raisonnement long, ca ralentit et peut
                                          # tronquer la sortie JSON (constate en pratique :
                                          # 0/20 preuves retenues sans erreur explicite,
                                          # probablement un JSON coupe par manque de budget)
            max_completion_tokens=3000,  # marge large pour un tableau JSON de ~20 objets
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   ⚠️ Echec appel LLM controle ({model}) : {e}")
        return None


# ============================================================
# 2. DETECTION D'INTENTION (reprise telle quelle -- logique deja fiable,
#    ne necessite pas de LLM ; sert au canal Cypher structure)
# ============================================================
INTENT_PRIORITY = [
    (["medicament", "bupropion", "varenicline"], "Medicament", "médicaments de sevrage"),
    (["programme", "plan national", "pnrt", "fctc", "projet fctc"], "Programme", "programmes et plans nationaux de lutte antitabac"),
    (["loi", "legislation", "decret", "arrêté", "circulaire", "texte juridique", "cadre legal", "règlement"], "Loi", "textes législatifs et réglementaires"),
    (["etude", "enquete", "epidemio", "gyts", "medspad", "thes", "statistique", "prevalence", "donnee"], "Etude", "études et enquêtes épidémiologiques"),
    (["produit", "substitut", "patch", "gomme", "pastille", "spray", "inhalateur", "tsn"], "Produit", "produits de substitution nicotinique"),
    (["symptome", "manque", "craving", "irritabilite", "envie"], "Symptome", "symptômes de sevrage"),
    (["pathologie", "maladie", "cancer", "bpco", "avc", "maladie cardiaque"], "Pathologie", "pathologies liées au tabac"),
    # Ajoutes le 2026-10-07 : deux echecs constates (voir JOURNAL_TRAVAIL.md) avaient la
    # meme cause -- aucune entree INTENT_PRIORITY ne couvrait ces questions, qui ne
    # passaient donc QUE par la recherche vectorielle. Or les noeuds Substance et
    # Pays/Statistique existent deja dans le graphe (confirme par requete directe),
    # relies par DANS_ARTICLE au bon chunk -- il manquait juste le routage Cypher.
    # Places en dernier (priorite la plus basse) pour ne jamais voler une question
    # qui correspond deja a une categorie plus specifique ci-dessus.
    (["nicotine", "goudron", "monoxyde de carbone", "substance chimique", "cancerigene", "composition de la fumee"], "Substance", "substances chimiques du tabac (nicotine, goudron, monoxyde de carbone...)"),
    (["cout", "fardeau economique", "impact economique", "milliards de dollars", "milliards usd", "pib", "depense de sante"], "Pays", "statistiques de cout du tabagisme par pays"),
]


def _normalize(text: str) -> str:
    """
    Enleve les accents et met en minuscules, pour un matching de mots-cles
    robuste aux variations d'accentuation. Bug reel constate : le mot-cle
    "symptome" (sans accent) ne matchait jamais "symptomes" ecrit avec un
    o circonflexe, et "reglement" (e accent grave) ne matchait pas
    "reglementation" (e accent aigu) -- ce sont des caracteres Unicode
    differents, `in` ne les confond pas. Apres normalisation, les deux
    variantes deviennent la meme chaine ASCII simple.
    """
    text = text.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def detect_intent(question: str):
    """
    Retourne TOUTES les categories dont un mot-cle matche (liste, dans l'ordre
    de INTENT_PRIORITY), pas seulement la premiere -- corrige le 2026-10-07 :
    avec un "premier match gagne", une question legitimement a cheval sur deux
    categories (ex: "les MALADIES liees au tabagisme COUTENT combien aux
    Etats-Unis ?") etait monopolisee par la categorie la moins pertinente des
    deux (Pathologie, a cause de "maladies") et ne declenchait JAMAIS le canal
    Cypher Pays/Statistique pourtant pertinent ici. Une liste permet au canal
    Cypher de cumuler les candidats de chaque categorie detectee.
    """
    q = _normalize(question)
    matches = []
    for keywords, label, desc in INTENT_PRIORITY:
        if any(_normalize(k) in q for k in keywords):
            matches.append({"label": label, "description": desc})
    return matches or None


# ============================================================
# 3. ETAT PARTAGE DU GRAPHE LANGGRAPH
# ============================================================
class ChatState(TypedDict, total=False):
    question: str                  # question originale, jamais modifiee
    route: str                     # "normal" | "hors_domaine" | "urgence" | "demande_personnelle"
    route_reason: str
    search_query: str              # requete effectivement utilisee pour chercher
    retry_count: int
    rewrite_history: list          # requetes precedentes + pourquoi rejetees (debug/log)
    intent: Optional[list]          # liste de {"label","description"} (plusieurs categories possibles)
    vector_candidates: list        # [{text, source, score}]
    graph_candidates: list         # [{text, source, name}]
    graded_evidence: list          # sous-ensemble retenu APRES grading, avec raison
    groundedness_fix_count: int
    draft_answer: str
    final_answer: str
    sources: list
    llm_unavailable: bool          # route_query/grade_evidence/generate_answer sont
                                    # tombes en repli (quota Groq) -- utilise seulement
                                    # si STRICT_LLM_FAILURE_MODE est actif (app.py)


# ============================================================
# 4. NOEUD [1] -- ROUTAGE SECURITE / DOMAINE (avant tout retrieval)
# ============================================================
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["normal", "hors_domaine", "urgence", "demande_personnelle"],
        },
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
}

ROUTE_SYSTEM = """Tu classes une question destinee a un chatbot d'INFORMATION (pas de diagnostic)
sur le tabagisme, la dependance a la nicotine, le sevrage, la legislation et l'epidemiologie
en Tunisie. Le chatbot ne fait JAMAIS de diagnostic ni de prescription personnalisee.

Categories possibles (choisis-en UNE SEULE) :

- "normal" : question factuelle/generale sur le tabac, la nicotine, le sevrage, le TSN,
  la legislation, l'epidemiologie, la grossesse/allaitement et le tabac, les symptomes de
  sevrage en general. Exemple : "Quels sont les symptomes du sevrage nicotinique ?"
  Exemple IMPORTANT a bien distinguer : "Quels sont les differents dosages DISPONIBLES
  pour le patch a la nicotine ?" est "normal" -- c'est une question generale sur les
  dosages qui EXISTENT sur le marche/dans la documentation, PAS une demande de
  recommandation personnalisee. Le mot cle qui change tout : "disponibles" (= catalogue
  general) vs "je dois prendre" (= decision qui concerne la personne elle-meme).

- "urgence" : la personne decrit une situation medicale potentiellement urgente pour
  elle-meme ou un proche (douleur thoracique, malaise, difficulte respiratoire aigue,
  idees suicidaires, etc.). Exemple : "J'ai une douleur dans la poitrine depuis que j'ai
  arrete de fumer, c'est grave ?"

- "demande_personnelle" : la personne demande un diagnostic, une prescription, un dosage
  personnalise, ou "que dois-JE faire" concernant SA situation medicale individuelle --
  la question doit contenir une reference explicite a SA situation personnelle (je dois,
  je peux, mon cas, je suis enceinte/malade...), pas juste le mot "dosage" ou "patch" tout
  seul. Exemple : "Quel patch de nicotine je dois prendre, je suis enceinte de 20
  semaines ?"

- "hors_domaine" : la question n'a aucun rapport avec le tabac/nicotine/sevrage (meteo,
  culture generale, code, autre pays sans rapport, etc.)

Reponds UNIQUEMENT en JSON avec les champs "route" et "reason" (raison courte en francais)."""


def route_query(state: ChatState) -> ChatState:
    result = call_llm_json(
        CONTROL_MODEL, ROUTE_SYSTEM, state["question"], ROUTE_SCHEMA
    )
    if result is None:
        # Echec du controle -> on ne bloque PAS silencieusement une vraie question :
        # on part sur "normal" mais on log l'incident pour audit.
        print("   ⚠️ Routage LLM indisponible -- fallback route='normal' (a auditer).")
        state = {**state, "route": "normal", "route_reason": "fallback: echec parsing JSON"}
        if STRICT_LLM_FAILURE_MODE:
            # Reste "sticky" jusqu'a finalize() : meme si un appel ulterieur
            # reussit, le routage (filet de securite) a tourne en mode degrade
            # sur CETTE question -- pas de garantie a offrir a l'utilisateur.
            state["llm_unavailable"] = True
    else:
        state = {**state, "route": result["route"], "route_reason": result["reason"]}
    return _apply_safety_backstop(state)


# Filet de securite par REGLES, independant du LLM de routage. Constate en
# pratique : meme avec un exemple quasi-identique dans le prompt, le LLM de
# controle (CONTROL_MODEL, temperature 0) a laisse passer "quel patch je dois
# prendre, je suis enceinte de 20 semaines ?" en route "normal" au lieu de
# "demande_personnelle". Ce filet ne peut JAMAIS assouplir une classification
# du LLM (ex: repasser "urgence" en "normal") -- il ne fait que la durcir si
# des marqueurs textuels forts sont detectes et que le LLM a repondu "normal".
_URGENT_PATTERNS = re.compile(
    r"douleur (thoracique|dans la poitrine|a la poitrine)|mal a la poitrine|"
    r"difficult[ée] (a respirer|respiratoire)|id[ée]es suicidaires|"
    r"envie de me faire du mal|malaise (important|grave)",
    re.IGNORECASE,
)
_PERSONAL_REQUEST_PATTERNS = re.compile(
    r"\bje dois\b|\bdois-je\b|\bpuis-je prendre\b|\bcombien de mg\b|"
    r"\bquelle dose\b|\bquel dosage\b|\bquel .*je (dois|peux) prendre\b",
    re.IGNORECASE,
)


_DOMAIN_KEYWORDS = re.compile(
    r"tabac|tabagisme|cigarette|nicotine|fumer|fumeur|sevrage|\btsn\b|patch|gomme|"
    r"pastille|narguile|vapot|goudron|\boms\b|mpower|cclat|inhalateur|spray nasal",
    re.IGNORECASE,
)


def _apply_safety_backstop(state: ChatState) -> ChatState:
    # Correction d'un faux "hors_domaine" : contrairement aux escalades ci-dessous
    # (qui ne font QUE durcir un "normal" en plus strict, jamais l'inverse), ceci
    # corrige une erreur de classification pure, pas un relachement de securite --
    # "hors_domaine" n'est pas une categorie de securite, juste un routage errone
    # constate en pratique (une question sur la loi tunisienne classee hors sujet).
    if state.get("route") == "hors_domaine" and _DOMAIN_KEYWORDS.search(_normalize(state["question"])):
        print("   🛡️ Filet de securite (regle) : route corrigee hors_domaine -> normal (mots-cles du domaine detectes)")
        return {**state, "route": "normal",
                "route_reason": state.get("route_reason", "") + " [backstop regle: mots-cles domaine tabac detectes]"}

    if state.get("route") != "normal":
        return state  # le LLM a deja classe plus strict, on ne touche a rien
    q = state["question"]
    if _URGENT_PATTERNS.search(q):
        print("   🛡️ Filet de securite (regle) : route durcie normal -> urgence")
        return {**state, "route": "urgence",
                "route_reason": state.get("route_reason", "") + " [backstop regle: marqueurs d'urgence]"}
    if _PERSONAL_REQUEST_PATTERNS.search(q):
        print("   🛡️ Filet de securite (regle) : route durcie normal -> demande_personnelle")
        return {**state, "route": "demande_personnelle",
                "route_reason": state.get("route_reason", "") + " [backstop regle: demande de dosage personnel]"}
    return state


def handle_special_case(state: ChatState) -> ChatState:
    """Reponses cadrees pour les routes qui ne doivent PAS declencher de retrieval."""
    route = state["route"]
    if route == "urgence":
        answer = (
            "Je ne peux pas evaluer une situation medicale urgente. Si tu ressens une "
            "douleur thoracique, une difficulte a respirer, un malaise important, ou "
            "toute autre situation qui te semble grave, contacte immediatement les "
            "services d'urgence ou rends-toi aux urgences les plus proches. Je reste "
            "disponible ensuite pour des questions generales sur le tabac et le sevrage."
        )
    elif route == "demande_personnelle":
        answer = (
            "Je ne peux pas te donner un diagnostic ni te prescrire un traitement "
            "personnalise (dosage, produit precis pour ta situation). Je peux en "
            "revanche t'expliquer de maniere generale comment fonctionnent les "
            "traitements de substitution nicotinique, leurs formes et leurs precautions "
            "d'usage -- pour une decision qui te concerne personnellement, un medecin "
            "ou un professionnel de sante est la bonne personne a consulter."
        )
    else:  # hors_domaine
        answer = (
            "Cette question sort du champ de ce chatbot, qui se limite au tabagisme, "
            "a la nicotine, au sevrage, a la legislation et a l'epidemiologie liees au "
            "tabac (notamment en Tunisie). N'hesite pas a me reposer une question sur "
            "ces sujets."
        )
    return {**state, "final_answer": answer, "sources": []}


def route_condition(state: ChatState) -> str:
    return "retrieval" if state["route"] == "normal" else "special_case"


# ============================================================
# 5. NOEUD [2] -- REECRITURE DE REQUETE (avant l'index)
# ============================================================
REWRITE_SYSTEM = """Tu reformules une question pour maximiser la qualite d'une recherche
semantique/lexicale dans une base documentaire sur le tabagisme, la nicotine, le sevrage,
le TSN (traitement de substitution nicotinique), la legislation tunisienne et
l'epidemiologie. Le corpus utilise un vocabulaire precis : nicotine, TSN, patch, gomme,
pastille, inhalateur, spray, HSI, tabagisme passif, cigarette electronique, MPOWER,
CCLAT, THES, GYTS, MedSPAD.

Reecris la question en francais, de maniere plus explicite et en te rapprochant de ce
vocabulaire UNIQUEMENT quand un terme correspond deja a ce que la question demande.
N'AJOUTE JAMAIS un terme, un sujet ou un angle de comparaison qui n'est pas deja implique
par la question originale, meme s'il appartient a ce vocabulaire -- la reformulation doit
rester strictement equivalente en sens et en perimetre a la question de depart, juste
avec un vocabulaire plus precis. Ne reponds qu'avec la requete reformulee, une seule
phrase, rien d'autre."""


def rewrite_query(state: ChatState) -> ChatState:
    retry = state.get("retry_count", 0)
    base_question = state["question"]

    if retry == 0:
        user_prompt = base_question
    else:
        # On donne au LLM le contexte de l'echec precedent pour qu'il reformule
        # differemment, pas juste repeter la meme requete.
        history = state.get("rewrite_history", [])
        last = history[-1] if history else {}
        user_prompt = (
            f"Question originale : {base_question}\n"
            f"Requete precedente (a eviter de repeter telle quelle) : "
            f"{last.get('query', base_question)}\n"
            f"Raison du rejet : preuves jugees insuffisantes ou hors-sujet dans le corpus.\n"
            f"Propose une reformulation differente, plus specifique ou avec d'autres "
            f"synonymes du corpus."
        )

    try:
        response = groq_client.chat.completions.create(
            model=CONTROL_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            reasoning_effort="low",
            max_completion_tokens=500,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        if not rewritten:
            rewritten = base_question
    except Exception as e:
        print(f"   ⚠️ Echec reecriture de requete : {e} -- utilisation de la question brute.")
        rewritten = base_question

    print(f"   ✏️ Requete reformulee : {rewritten}")
    history = state.get("rewrite_history", [])
    history.append({"query": rewritten, "retry": retry})

    return {
        **state,
        "search_query": rewritten,
        "rewrite_history": history,
        "intent": detect_intent(base_question) or detect_intent(rewritten),
    }


# ============================================================
# 6. NOEUD [3] -- RECHERCHE HYBRIDE (vectorielle + Cypher structure)
# ============================================================
# Quand la question nomme un produit precis, on filtre le canal Cypher sur ce
# nom -- sinon le label generique "Produit" mélange patch/gomme/pastille/spray/
# inhalateur/cigarette electronique, et le LLM de generation peut fusionner
# par erreur les dosages de deux produits differents (constate en pratique :
# "patch... 2mg, 4mg, 21mg" alors que 2mg/4mg appartiennent a la gomme).
PRODUCT_NAME_FILTERS = [
    "patch", "gomme", "pastille", "spray", "inhalateur",
    "cigarette electronique", "cigarette électronique", "narguile", "narguilé",
    "tabac chauffe", "tabac chauffé", "sachet",
]

# Meme logique de filtrage par nom, generalisee le 2026-10-07 aux labels Substance
# (ex: "nicotine" parmi ~33 substances) et Pays (ex: "etats-unis" parmi ~22 pays) --
# sans filtre, le label seul remonterait les chunks de TOUTES les substances/pays
# melanges, ce qui est exactement le probleme deja identifie et corrige pour Produit.
SUBSTANCE_NAME_FILTERS = [
    "nicotine", "goudron", "monoxyde de carbone", "ammoniac", "arsenic",
    "cadmium", "plomb", "mercure", "toluene", "methanol",
]
PAYS_NAME_FILTERS = [
    "tunisie", "etats-unis", "états-unis", "japon", "allemagne", "canada",
    "chine", "inde", "bangladesh", "royaume-uni", "france",
]
NAME_FILTERS_BY_LABEL = {
    "Produit": PRODUCT_NAME_FILTERS,
    "Substance": SUBSTANCE_NAME_FILTERS,
    "Pays": PAYS_NAME_FILTERS,
}


def detect_name_filter(label: str, question: str) -> Optional[str]:
    names = NAME_FILTERS_BY_LABEL.get(label)
    if not names:
        return None
    q = _normalize(question)
    for name in names:
        if _normalize(name) in q:
            return name
    return None


def _get_graph_candidates_for_label(label: str, question: str) -> list:
    name_filter = detect_name_filter(label, question)

    if name_filter:
        # Une fois filtre par nom (produit/substance/pays), le volume est deja
        # restreint (~15-30 entites au pire) -- plafond fixe, PAS multiplie par
        # MAX_GRAPH_CANDIDATES (qui vaut 12 -- cf plus haut ; x3 donnerait un
        # volume qui recreerait le probleme de lenteur deja corrige). Un plafond
        # fixe de 30 suffit ici car le filtre par nom a deja fait le gros du tri.
        # apoc.text.clean() (accents + casse + ponctuation normalises des DEUX
        # cotes) plutot que toLower() seul : toLower() ne retire pas les accents,
        # donc un nom canonique accentue ("États-Unis", issu de la fusion de
        # doublons par accent du 2026-10-07) ne matchait plus un mot-cle ecrit
        # sans accent ("etats-unis") avec un simple CONTAINS sur toLower().
        query = f"""
        MATCH (e:{label})-[:DANS_ARTICLE]->(a:Article)-[:DANS_SOURCE]->(s:Source)
        WHERE apoc.text.clean(e.name) CONTAINS apoc.text.clean($name_filter)
        RETURN DISTINCT e.name AS name, a.text AS text, s.path AS source
        ORDER BY e.name
        LIMIT $limit
        """
        params = {"limit": 30, "name_filter": name_filter}
    else:
        query = f"""
        MATCH (e:{label})-[:DANS_ARTICLE]->(a:Article)-[:DANS_SOURCE]->(s:Source)
        RETURN DISTINCT e.name AS name, a.text AS text, s.path AS source
        ORDER BY e.name
        LIMIT $limit
        """
        params = {"limit": MAX_GRAPH_CANDIDATES}

    try:
        result = driver.execute_query(query, **params)
    except Exception as e:
        print(f"   ⚠️ Echec requete Cypher structuree ({label}) : {e}")
        return []

    return [
        {
            "text": r["text"],
            "source": short_source(r["source"]),
            "name": r["name"].strip() if r["name"] else None,
            "origin": "graphe",
        }
        for r in result.records
    ]


def get_graph_candidates(intents, question: str = "") -> list:
    """
    Canal Cypher direct : PAS de LLM ici (conformement a la consigne "si les
    etapes Cypher ne sont pas connectees a un LLM, ce n'est pas grave, mais
    il faut verifier les resultats"). On ne fait AUCUNE deduction : on ne
    remonte que les chunks reellement lies a l'entite via DANS_ARTICLE --
    jamais un filtrage par mot-cle qui pourrait choper un extrait hors-sujet.

    `intents` est une LISTE de {"label", "description"} (detect_intent peut
    desormais detecter plusieurs categories a la fois, ex: une question sur
    le COUT du tabagisme qui mentionne aussi des MALADIES -- voir le
    correctif du 2026-10-07 sur detect_intent). On cumule les candidats de
    CHAQUE label detecte -- PAS de dedup par (label, nom) : une meme entite
    (ex: "nicotine") est legitimement liee a des dizaines de chunks
    DIFFERENTS via DANS_ARTICLE, dedupliquer sur le nom seul en ecraserait
    la quasi-totalite (bug constate et corrige le 2026-10-07 : 22 candidats
    nicotine ecrases a 1 par une premiere version de ce dedup). Les labels de
    INTENT_PRIORITY etant tous distincts, deux intents ne peuvent de toute
    facon pas produire de vrai doublon exact ici.
    Accepte aussi un seul dict par compatibilite ascendante.
    """
    if not intents:
        return []
    if isinstance(intents, dict):
        intents = [intents]

    candidates = []
    for intent in intents:
        candidates.extend(_get_graph_candidates_for_label(intent["label"], question))
    return candidates


def hybrid_retrieve(state: ChatState) -> ChatState:
    query = state["search_query"]

    # --- Canal 1 : recherche vectorielle ---
    vector_candidates = []
    try:
        result = vector_retriever.search(query_text=query, top_k=TOP_K_VECTOR)
        for item in result.items:
            content = item.content if isinstance(item.content, str) else str(item.content)
            meta = getattr(item, "metadata", None) or {}
            vector_candidates.append({
                "text": content,
                "source": short_source(meta.get("source", "")),
                "score": meta.get("score"),
                "origin": "vecteur",
            })
    except Exception as e:
        print(f"   ⚠️ Echec recherche vectorielle : {e}")

    # --- Canal 2 : recherche structuree Cypher (verifiee, pas de mot-cle libre) ---
    # Desactivable via USE_GRAPH=0 (ablation) -- tout le reste du pipeline
    # (routage, reecriture, notation, generation, verification) est inchange.
    graph_candidates = get_graph_candidates(state.get("intent"), query) if USE_GRAPH else []

    print(f"   🔎 Candidats bruts : {len(vector_candidates)} (vecteur) + "
          f"{len(graph_candidates)} (graphe)")

    return {**state, "vector_candidates": vector_candidates, "graph_candidates": graph_candidates}


# ============================================================
# 7. NOEUD [4] -- GRADING / RERANKING (filtre avant generation)
# ============================================================
# NOTE POUR MONTER EN PRECISION (optionnel) :
# Cette etape remplace un reranker cross-encoder (type bge-reranker ou
# Cohere rerank) par un LLM (Groq, CONTROL_MODEL) qui note chaque passage
# pertinent/non pertinent. C'est le compromis raisonnable sans devoir heberger
# un modele de reranking dedie. Si tu veux un vrai reranking base sur des
# embeddings de cross-encoder (plus rapide et plus stable que demander a un
# LLM generatif de juger), tu peux ajouter `sentence-transformers` et un
# modele leger type "cross-encoder/ms-marco-MiniLM-L-6-v2" (tourne bien
# sur CPU) -- ca remplacerait uniquement cette fonction, le reste du
# graphe ne change pas.
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "pertinent": {"type": "boolean"},
                    "score": {"type": "integer"},
                    "raison": {"type": "string"},
                },
                "required": ["index", "pertinent", "score"],
            },
        }
    },
    "required": ["verdicts"],
}

GRADE_SYSTEM = """Tu evalues si chaque passage numerote ci-dessous contient une information
directement utile pour repondre a la question. Sois strict : un passage qui parle du meme
theme general mais ne repond pas concretement au sujet EXACT de la question doit etre juge
non pertinent, meme s'il parle de tabac/nicotine en general. Exemple : pour une question sur
la "publicite", un passage sur les mesures MPOWER en general ou sur le commerce illicite
n'est PAS pertinent, meme s'il vient du meme document -- seul un passage qui mentionne
explicitement la publicite l'est.

Donne aussi un score de 0 a 10 : 10 = repond directement et precisement a la question,
5 = meme theme general mais pas la reponse precise, 0 = hors sujet.

Reponds UNIQUEMENT en JSON avec un champ "verdicts", liste d'objets {index, pertinent, score,
raison}. Un objet par passage, dans l'ordre."""


def grade_evidence(state: ChatState) -> ChatState:
    query = state["search_query"]
    all_candidates = state.get("vector_candidates", []) + state.get("graph_candidates", [])

    if not all_candidates:
        return {**state, "graded_evidence": []}

    # On numerote les candidats et on prepare un texte court par candidat
    # (evite de saturer la fenetre de contexte du LLM de grading).
    # CORRECTIF (09/10/2026) : 700 etait trop court -- diagnostique sur Q61
    # ("effet combine tabac/radon") : le fait recherche etait bien dans le
    # chunk retrouve (SantePubliqueFrance_Consequences.txt) mais a la
    # position 1407/1776 caracteres -- voir GRADING_TRUNCATE_CHARS plus haut
    # dans le fichier pour le detail de ce bug et son historique.
    numbered = []
    for i, c in enumerate(all_candidates):
        snippet = clean_chunk(c["text"])[:GRADING_TRUNCATE_CHARS]
        numbered.append(f"[{i}] ({c['origin']}, source: {c['source']}) {snippet}")

    user_prompt = f"Question : {query}\n\nPassages :\n" + "\n\n".join(numbered)

    print(f"   ⏳ Grading de {len(all_candidates)} passage(s) en cours (peut prendre "
          f"du temps en local sur CPU)...")
    result = call_llm_json(GRADE_MODEL, GRADE_SYSTEM, user_prompt, GRADE_SCHEMA)

    if result is None:
        # Fallback prudent : si le grading echoue techniquement, on NE
        # donne PAS tout au LLM de generation (ce serait justement le
        # probleme qu'on veut eviter) -- on retient au plus les 3 premiers
        # candidats en le signalant explicitement dans les logs.
        print("   ⚠️ Grading LLM indisponible -- fallback conservateur (3 candidats max).")
        graded = [{**c, "raison": "fallback (grading indisponible)", "score": 5} for c in all_candidates[:3]]
        extra = {"llm_unavailable": True} if STRICT_LLM_FAILURE_MODE else {}
        return {**state, "graded_evidence": graded, **extra}

    verdicts_raw = result.get("verdicts", [])
    verdicts = {v["index"]: v for v in verdicts_raw if "index" in v}
    # Certains modeles (constate avec Groq/gpt-oss) numerotent leurs verdicts
    # a partir de 1 au lieu de 0, malgre l'exemple 0-indexe fourni dans le
    # prompt -- ce qui decalait TOUT et faisait 0/20 retenues. Detection :
    # si aucun verdict n'a l'index 0 mais qu'un verdict a l'index exact du
    # nombre de candidats (= dernier index en 1-based), on corrige le decalage.
    if verdicts and 0 not in verdicts and len(all_candidates) in verdicts:
        print("   ℹ️ Decalage d'index detecte dans le grading (1-based au lieu de 0-based) -- corrige.")
        verdicts = {k - 1: v for k, v in verdicts.items()}
    if verdicts_raw and not verdicts:
        print(f"   ⚠️ Grading : {len(verdicts_raw)} verdicts recus mais aucun index exploitable "
              f"(format inattendu) -- verifier la reponse brute si ca se reproduit.")

    graded = []
    for i, c in enumerate(all_candidates):
        v = verdicts.get(i)
        if v and v.get("pertinent"):
            graded.append({**c, "raison": v.get("raison", ""), "score": v.get("score", 5)})

    # Tri par score decroissant AVANT de plafonner : la preuve la plus
    # pertinente doit arriver en PREMIER dans le contexte donne au LLM de
    # generation. Constate en pratique (effet "lost in the middle" bien
    # documente sur les LLM) : une preuve correcte mais en position 5/6
    # etait ignoree par le LLM de generation au profit d'une reponse d'abstention,
    # alors que la meme preuve en tete de liste aurait probablement ete
    # utilisee.
    graded.sort(key=lambda g: g.get("score", 0), reverse=True)
    graded = graded[:MAX_EVIDENCE_TO_LLM]  # plafond dur, meme si le grading en retient plus
    print(f"   ✅ Preuves retenues apres grading (triees par score) : {len(graded)}/{len(all_candidates)}")
    for i, g in enumerate(graded, 1):
        apercu = clean_chunk(g["text"])[:90].replace("\n", " ")
        print(f"      [{i}] score={g.get('score', '?')} ({g['origin']}, {g['source']}) {apercu}...")
    return {**state, "graded_evidence": graded}


def quality_gate(state: ChatState) -> str:
    n_relevant = len(state.get("graded_evidence", []))
    retry = state.get("retry_count", 0)

    if n_relevant >= MIN_RELEVANT_EVIDENCE:
        return "generate"
    if retry < MAX_REWRITES:
        return "rewrite"
    return "insufficient"


def increment_retry(state: ChatState) -> ChatState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def insufficient_evidence_response(state: ChatState) -> ChatState:
    answer = (
        "Je n'ai pas trouve, dans les documents analyses, d'information suffisamment "
        "precise pour repondre a cette question de maniere fiable. Je prefere te le dire "
        "plutot que d'inventer une reponse. Tu peux essayer de reformuler ta question, ou "
        "consulter un professionnel de sante / une source officielle (OMS, Ministere de "
        "la Sante) pour ce point precis."
    )
    return {**state, "final_answer": answer, "sources": []}


# ============================================================
# 8. NOEUD [6] -- GENERATION (many-shot + chain-of-thought guide)
# ============================================================
# Chain-of-thought : on dit explicitement au LLM COMMENT raisonner (pas
# seulement le format de sortie voulu), et on lui demande de ne PAS
# restituer ce raisonnement -- seulement la reponse finale. C'est la
# difference entre "zero-shot" (rien de tout ca) et ce qu'on construit ici.
GENERATION_SYSTEM = """Tu es un assistant d'information sur le tabagisme, la nicotine, le
sevrage et la legislation antitabac, avec un focus Tunisie. Tu ne remplaces PAS un avis
medical et tu ne fais JAMAIS de diagnostic ni de prescription personnalisee.

COMMENT RAISONNER (fais ceci en interne, sans l'ecrire dans ta reponse) :
1. Identifie precisement ce que la question demande.
2. Relis chaque preuve fournie ci-dessous et note mentalement laquelle repond a QUOI
   exactement dans la question.
3. Verifie qu'aucune affirmation que tu vas ecrire ne depasse ce que les preuves disent
   litteralement -- pas d'ajout de connaissance generale, pas de chiffre non present dans
   les preuves.
4. Si les preuves ne couvrent qu'une partie de la question, dis-le explicitement au lieu
   de completer avec autre chose.
5. Compose la reponse finale : claire, concise, avec la source entre parenthese apres
   chaque affirmation qui en depend.

REGLES ABSOLUES :
- Tu reponds UNIQUEMENT en francais.
- Tu te bases STRICTEMENT sur les preuves fournies. Aucune loi, date, chiffre, nom ou
  dosage qui n'apparait pas dans les preuves ne doit apparaitre dans ta reponse.
- NE MELANGE JAMAIS les chiffres de deux produits/entites differents. Si les preuves
  parlent a la fois du patch et de la gomme par exemple, un dosage donne pour la gomme
  (ex: 2mg, 4mg) ne doit JAMAIS etre attribue au patch, et inversement. Verifie a quel
  produit precis chaque chiffre appartient avant de l'ecrire.
- Si les preuves distinguent des sous-categories d'un meme produit (ex: patch a
  liberation 24h vs 16h, avec des dosages differents pour chacune), precise la
  sous-categorie -- ne dis jamais juste "le patch" si le chiffre ne vaut que pour une
  des deux.
- Si les preuves contiennent un TABLEAU (plusieurs produits/lignes listes ensemble),
  repere d'abord la LIGNE exacte qui correspond au produit demande dans la question --
  ignore les autres lignes du tableau pour composer ta reponse, meme si elles sont
  juste a cote dans le texte.
- Reponds UNIQUEMENT a ce qui est explicitement demande dans la question. N'ajoute PAS
  d'informations annexes juste parce qu'elles sont presentes dans les preuves fournies --
  si la question porte sur les symptomes de sevrage, ne decris pas aussi les formes de
  traitement disponibles (patch, gomme...), meme si les preuves en parlent a cote.
- Distingue toujours un SYMPTOME DE SEVRAGE (consequence de l'ARRET du tabac/nicotine)
  d'un EFFET SECONDAIRE D'UN TRAITEMENT (consequence de la PRISE d'un produit comme la
  gomme ou le patch) -- ce sont deux choses different. Ne les fusionne jamais dans une
  meme liste sans preciser laquelle est laquelle.
- Si tu dois dire que les preuves ne couvrent pas la question (information absente),
  reformule le SUJET EXACT de la question originale dans ta phrase d'absence
  d'information -- ne substitue jamais ce sujet par un sujet voisin present dans les
  preuves. Exemple d'erreur a ne pas reproduire : question sur la "teneur maximale en
  GOUDRON", reponse d'absence qui parle de "dosage de NICOTINE" -- ce sont deux sujets
  differents, et confondre l'un pour l'autre induit en erreur meme dans un aveu
  d'ignorance.
- N'INTRODUIS JAMAIS un angle, un sujet ou une comparaison qui ne vient PAS des preuves
  fournies ET qui n'a PAS ete demande dans la question -- meme pour dire ensuite que
  cette comparaison n'est pas possible. Une fois que tu as repondu a ce qui est
  explicitement demande, ARRETE-TOI LA. N'ajoute pas de derniere phrase sur "ce qui
  manque" concernant un aspect que la question elle-meme n'a jamais souleve : cela
  revient a introduire un sujet de ta propre initiative, ce qui est aussi interdit que
  d'inventer un fait. Ta reponse doit se limiter au perimetre de la question posee, ni
  plus, ni moins.

FORMAT DE SORTIE OBLIGATOIRE :
Tu peux reflechir etape par etape si tu en as besoin, mais cette reflexion ne doit
JAMAIS apparaitre dans ce qui est montre a l'utilisateur. Termine donc TOUJOURS ta
sortie par cette ligne exacte, seule sur sa ligne :
### REPONSE FINALE ###
...suivie IMMEDIATEMENT de la reponse complete, autonome et comprehensible sans le
raisonnement qui precede (donc ne dis pas juste "7mg, 14mg, 21mg" sans contexte -- la
reponse apres le marqueur doit se suffire a elle-meme, comme si le raisonnement
n'existait pas). Tout ce qui est ecrit AVANT cette ligne est ignore et jamais montre.
- Si les preuves viennent de pays differents (Tunisie, France, Royaume-Uni, Etats-Unis,
  Australie...), precise le pays/l'organisme source quand la donnee n'est pas universelle
  (ex: un chiffre de mortalite francais n'est pas un chiffre tunisien).
- Ne restitue jamais ton raisonnement interne, seulement la reponse finale.

Voici des exemples de bonnes reponses, pour differents cas de figure.

--- EXEMPLE 1 : question factuelle simple ---
Preuves : "[patch a la nicotine] Liberation sur 24h : 7mg, 14mg, 21mg (source: CHATBOT_3)"
Question : "Quels sont les dosages disponibles pour le patch a la nicotine ?"
Reponse attendue :
"Le patch a la nicotine a liberation sur 24 heures existe en trois dosages : 7mg, 14mg et
21mg (source : CHATBOT_3_SEVRAGE_AVEC_LES_SUBSTITUTS_NICOTINIQUES)."

--- EXEMPLE 2 : preuves partielles seulement ---
Preuves : "[TSN grossesse] Le TSN peut etre introduit a n'importe quel trimestre... dose
efficace la plus faible pendant la duree la plus courte possible (source: guide Quit)"
Question : "Le TSN est-il sans danger pendant l'allaitement ?"
Reponse attendue :
"Pour l'allaitement precisement, les preuves disponibles indiquent que l'usage maternel de
TSN est considere comme une option plus sure que la poursuite du tabagisme, mais que les
informations sont limitees, et que la nicotine passe dans le lait maternel avec des effets
possibles chez le nourrisson (source : guide Quit, Australie). Je n'ai pas d'information
plus precise que celle-ci sur ce point specifique."

--- EXEMPLE 3 : incoherence entre sources (nuance a garder, pas a trancher) ---
Preuves : "[OMS] Cigarettes electroniques : nocives, non sures (source: OMS)" et
"[NHS] Les cigarettes electroniques... nettement moins nocives... aide populaire a
l'arret (source: NHS UK)"
Question : "Le vapotage est-il une bonne solution pour arreter de fumer ?"
Reponse attendue :
"Les sources disponibles ne s'accordent pas totalement : le NHS (Royaume-Uni) presente le
vapotage comme une aide efficace a l'arret, avec un accompagnement adapte, tandis que
l'OMS le classe parmi les produits nocifs necessitant une action de prevention, notamment
chez les jeunes (sources : NHS UK, OMS). Cette question depend donc aussi du contexte
reglementaire et de l'age de la personne."

--- EXEMPLE 4 : chiffre localise, ne pas generaliser ---
Preuves : "[Sante publique France] plus de 68 000 deces prematures en 2023 en France
(source: Sante publique France)"
Question : "Combien de personnes meurent du tabac chaque annee ?"
Reponse attendue :
"Le chiffre depend du pays : en France, le tabac est responsable de plus de 68 000 deces
prematures en 2023 (source : Sante publique France). Je n'ai pas, dans les preuves
fournies ici, de chiffre equivalent pour la Tunisie a cote de celui-ci -- si tu veux ce
chiffre precis, dis-le moi et je peux verifier si une autre preuve du corpus le couvre."

--- FIN DES EXEMPLES ---"""


def generate_answer(state: ChatState) -> ChatState:
    evidence = state.get("graded_evidence", [])
    evidence_text = "\n\n".join(
        f"- ({e['origin']}, source: {display_source(e['source'])})\n{clean_chunk(e['text'])[:GENERATION_TRUNCATE_CHARS]}"
        for e in evidence
    )
    user_prompt = f"Preuves disponibles :\n{evidence_text}\n\nQuestion : {state['search_query']}"

    try:
        response = groq_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            reasoning_effort="medium",   # ici le raisonnement aide (composer une reponse
                                          # correcte a partir de plusieurs preuves)
            max_completion_tokens=2500,  # releve de 1500 -- constate en pratique : le
                                          # raisonnement etape par etape ecrit avant le
                                          # marqueur peut a lui seul approcher 1500 tokens,
                                          # tronquant la reponse finale (reponse vide livree
                                          # a l'utilisateur alors que le marqueur etait bien
                                          # present).
        )
        draft = response.choices[0].message.content.strip()
        draft = extract_final_answer(draft)
        if not draft:
            print("   ⚠️ Reponse finale vide apres extraction -- generation consideree en echec.")
            draft = (
                "Je n'ai pas reussi a composer une reponse fiable a partir des preuves "
                "disponibles. Peux-tu reformuler ta question ?"
            )
    except Exception as e:
        draft = _sanitize_llm_error(e)
        if STRICT_LLM_FAILURE_MODE:
            state = {**state, "llm_unavailable": True}

    sources = sorted({e["source"] for e in evidence})
    return {**state, "draft_answer": draft, "sources": sources}


# ============================================================
# 9. NOEUD [7] -- VALIDATION DE GROUNDEDNESS (critique)
# ============================================================
GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "supportee": {"type": "boolean"},
        "affirmations_non_supportees": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["supportee"],
}

GROUNDEDNESS_SYSTEM = """Tu verifies qu'une reponse ne contient AUCUNE affirmation qui
depasse ce que les preuves fournies disent litteralement (pas de chiffre, loi, dosage ou
nom invente ou generalise au-dela des preuves). Liste les affirmations non supportees s'il
y en a. Reponds UNIQUEMENT en JSON."""


def _audit_once(evidence_text: str, draft: str) -> Optional[dict]:
    user_prompt = f"Preuves :\n{evidence_text}\n\nReponse a verifier :\n{draft}"
    # Utilise GENERATION_MODEL (openai/gpt-oss-120b, via Groq), pas CONTROL_MODEL
    # (openai/gpt-oss-20b) : constate en pratique que le modele leger rejette a tort
    # des affirmations pourtant correctes et bien appuyees par les preuves -- peu
    # fiable pour ce jugement precis. Comme la generation qui precede utilise deja
    # GENERATION_MODEL, ca ne rajoute AUCUN changement de modele supplementaire
    # (GENERATION_MODEL reste charge d'un appel a l'autre) -- juste un seul swap
    # leger->lourd par question, au lieu de plusieurs allers-retours entre modeles
    # differents.
    return call_llm_json(GENERATION_MODEL, GROUNDEDNESS_SYSTEM, user_prompt, GROUNDEDNESS_SCHEMA)


_NO_INFO_PATTERNS = re.compile(
    r"pas d'information|pas de donn[ée]es|ne mentionnent pas|ne sont pas couvert|"
    r"ne couvrent pas|n'ai pas trouv[ée]|"
    # Variante trouvee le 2026-10-11 (Q77 du jeu d'eval) : le LLM dit parfois
    # "informations fournies" au lieu de "preuves fournies" -- le regex ne
    # couvrait que ce 2e mot, ratant une vraie auto-abstention (consequence :
    # mal classee "repondre" au lieu de "abstention" cote evaluation, ET le
    # groundedness-skip de _is_self_declared_no_info ne s'appliquait pas).
    r"(preuves?|informations?|[ée]l[ée]ments?) (disponibles?|fournies?) ne",
    re.IGNORECASE,
)


def _is_self_declared_no_info(draft: str) -> bool:
    """
    Detecte si la reponse est deja elle-meme un aveu d'absence d'information.
    Bug constate : le critique de groundedness rejetait a tort des phrases
    comme "Il n'y a pas de donnees... dans les preuves fournies" comme une
    "affirmation non supportee" -- alors que ce n'est pas une affirmation
    factuelle a verifier, c'est un constat d'absence. On saute l'audit dans
    ce cas plutot que de laisser le critique se tromper dessus et gaspiller
    les tentatives de correction pour rien.
    """
    return bool(_NO_INFO_PATTERNS.search(draft))


def validate_groundedness(state: ChatState) -> ChatState:
    """
    Boucle interne BORNEE (pas un aller-retour dans le graphe LangGraph) :
    on audite, corrige si besoin, puis on RE-AUDITE la version corrigee avant
    de la livrer -- sinon une correction qui ne corrige rien passe quand meme
    (bug constate : la generalisation "les patchs" au lieu de "les patchs a
    liberation 24h" n'etait pas re-verifiee apres correction).
    Le nombre total de tentatives reste plafonne par MAX_GROUNDEDNESS_FIXES,
    donc toujours pas de boucle infinie.
    """
    evidence = state.get("graded_evidence", [])
    evidence_text = "\n\n".join(clean_chunk(e["text"])[:GROUNDEDNESS_TRUNCATE_CHARS] for e in evidence)
    draft = state["draft_answer"]
    fixes_used = 0

    while True:
        if _is_self_declared_no_info(draft):
            return {**state, "draft_answer": draft, "final_answer": draft,
                     "groundedness_fix_count": fixes_used}

        result = _audit_once(evidence_text, draft)

        if result is None:
            print("   ⚠️ Verification de groundedness indisponible -- reponse livree sans audit.")
            return {**state, "draft_answer": draft, "final_answer": draft,
                     "groundedness_fix_count": fixes_used}

        if result.get("supportee", True):
            return {**state, "draft_answer": draft, "final_answer": draft,
                     "groundedness_fix_count": fixes_used}

        unsupported = result.get("affirmations_non_supportees", [])
        print(f"   ⚠️ Affirmations non supportees detectees (tentative {fixes_used}) : {unsupported}")

        if fixes_used >= MAX_GROUNDEDNESS_FIXES:
            fallback = (
                "Je ne suis pas certain de pouvoir repondre avec assez de precision sur ce "
                "point sans risquer d'ajouter une information non verifiee. Peux-tu reformuler "
                "ou preciser ta question ?"
            )
            # On efface les sources : on rejette la reponse associee, donc on
            # ne doit pas laisser croire qu'elle est appuyee par ces sources.
            return {**state, "final_answer": fallback, "sources": [],
                     "groundedness_fix_count": fixes_used}

        correction_prompt = (
            f"Ta reponse precedente contenait des affirmations non supportees par les preuves : "
            f"{unsupported}. Reecris la reponse en retirant ou en nuançant PRECISEMENT ces "
            f"points -- par exemple, si un dosage ou une duree n'est valable que pour une "
            f"sous-categorie du produit dans les preuves, dis-le explicitement au lieu de "
            f"generaliser. Reste strictement dans les preuves fournies.\n\n"
            f"Preuves :\n{evidence_text}\n\nQuestion : {state['search_query']}\n\n"
            f"Reponse precedente a corriger :\n{draft}"
        )
        try:
            response = groq_client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM},
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.0,
                reasoning_effort="medium",
                max_completion_tokens=2500,  # meme correctif que generate_answer()
            )
            draft = response.choices[0].message.content.strip()
            draft = extract_final_answer(draft)
            if not draft:
                print("   ⚠️ Correction vide apres extraction -- on garde la reponse precedente.")
                draft = state["draft_answer"]
        except Exception as e:
            print(f"   ⚠️ Echec correction : {e}")
            return {**state, "final_answer": draft, "groundedness_fix_count": fixes_used}

        fixes_used += 1


# ============================================================
# 10. NOEUD [8] -- FINALISATION
# ============================================================
SATURATION_MESSAGE = "Service temporairement sature, reessaie dans quelques minutes."


def finalize(state: ChatState) -> ChatState:
    if STRICT_LLM_FAILURE_MODE and state.get("llm_unavailable"):
        # Le routage et/ou le grading sont tombes en repli faute de quota Groq
        # sur cette question -- on ne livre JAMAIS la reponse qui en resulte
        # (potentiellement degradee silencieusement) a un vrai utilisateur.
        # Pas de disclaimer medical ici : ce n'est pas un contenu medical,
        # juste un statut systeme.
        return {**state, "final_answer": SATURATION_MESSAGE, "sources": []}

    answer = state.get("final_answer", "")
    sources = state.get("sources", [])
    if sources:
        # state["sources"] reste les noms de fichiers bruts (identite interne,
        # utilisee par evaluation/ et tests/) -- seul l'AFFICHAGE ici passe par
        # display_source() pour une citation lisible dans le texte.
        display_names = sorted({display_source(s) for s in sources})
        answer += "\n\n📚 Sources : " + ", ".join(display_names)
    answer += "\n\n⚠️ Cette information ne remplace pas un avis medical personnalise."
    return {**state, "final_answer": answer}


# ============================================================
# 11. CONSTRUCTION DU GRAPHE LANGGRAPH
# ============================================================
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("route_query", route_query)
    graph.add_node("handle_special_case", handle_special_case)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("grade_evidence", grade_evidence)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("insufficient_evidence_response", insufficient_evidence_response)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_groundedness", validate_groundedness)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        route_condition,
        {"retrieval": "rewrite_query", "special_case": "handle_special_case"},
    )
    graph.add_edge("handle_special_case", "finalize")

    graph.add_edge("rewrite_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "grade_evidence")

    graph.add_conditional_edges(
        "grade_evidence",
        quality_gate,
        {
            "generate": "generate_answer",
            "rewrite": "increment_retry",
            "insufficient": "insufficient_evidence_response",
        },
    )
    graph.add_edge("increment_retry", "rewrite_query")
    graph.add_edge("insufficient_evidence_response", "finalize")

    graph.add_edge("generate_answer", "validate_groundedness")
    graph.add_edge("validate_groundedness", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================
# 12. MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 CHATBOT TABAGISME TUNISIE -- pipeline LangGraph controle")
    print("=" * 60)
    print("⚠️  Cet assistant ne remplace pas un avis medical.\n")
    print("Commandes : quit\n" + "=" * 60)

    app = build_graph()

    try:
        while True:
            q = input("\n❓ Ta question : ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue

            initial_state: ChatState = {
                "question": q,
                "retry_count": 0,
                "rewrite_history": [],
                "groundedness_fix_count": 0,
            }
            final_state = app.invoke(initial_state)
            print(f"\n💬 REPONSE :\n{final_state.get('final_answer', '(pas de reponse)')}")
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir.")
    finally:
        driver.close()
        print("🔌 Connexion Neo4j fermee.")