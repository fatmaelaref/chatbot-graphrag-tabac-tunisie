"""
============================================================
kg_config.py -- Configuration PARTAGEE du Knowledge Graph
============================================================
Ce module contient le splitter, le schema et le prompt utilises
par 2_build_kg_ULTIME.py. Modifie-les ICI uniquement.

Schema audite le 23/08/2026 contre les 10 documents reels du
corpus (4 PPTX + guide Quit + 5 fiches OMS/CDC/NHSx2/SPF).
Chaque type et relation ci-dessous est justifie par au moins
une occurrence verifiee dans le corpus -- rien n'est ajoute
"au cas ou", pour eviter un schema trop large qui produirait
un graphe bruite et gaspillerait des tokens en boucle.
"""

from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm.types import LLMResponse, LLMUsage
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.exceptions import LLMGenerationError


# ============================================================
# 0. CORRECTIF LLM : bug async de OllamaLLM dans neo4j_graphrag
# ============================================================
# OllamaLLM.__ainvoke_v1 (utilisee par TOUT le pipeline async, donc par
# SimpleKGPipeline.run_async) empaquette tout `model_params` dans le seul
# parametre `options=...` au lieu de le deployer avec **model_params comme
# le fait la version synchrone (__invoke_v1, correcte).
# Consequence concrete verifiee : un `format` (schema JSON de contrainte
# de sortie) place dans model_params n'atteint JAMAIS Ollama en mode async
# -- silencieusement ignore. Sans ce correctif, qwen2.5:7b continue a
# halluciner sa propre convention de cles JSON (type/source/target au lieu
# de label/start_node_id/end_node_id) malgre toutes les instructions du
# prompt et malgre la configuration de `format`.
#
# Cette sous-classe re-implemente ainvoke() pour les entrees texte (str,
# le seul cas utilise par ce pipeline) en reproduisant fidelement la
# logique de la version sync, en asynchrone.
class FixedOllamaLLM(OllamaLLM):
    async def ainvoke(
        self,
        input,
        message_history=None,
        system_instruction=None,
        response_format=None,
        **kwargs,
    ) -> LLMResponse:
        if not isinstance(input, str):
            return await super().ainvoke(
                input,
                message_history=message_history,
                system_instruction=system_instruction,
                response_format=response_format,
                **kwargs,
            )
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_messages(input, message_history, system_instruction),
                **self.model_params,  # <-- deploie options=... ET format=... correctement
            )
            content = response.message.content or ""
            usage = None
            if (
                response.prompt_eval_count is not None
                or response.eval_count is not None
            ):
                request_tokens = response.prompt_eval_count
                response_tokens = response.eval_count
                usage = LLMUsage(
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    total_tokens=(request_tokens + response_tokens)
                    if (request_tokens is not None and response_tokens is not None)
                    else None,
                )
            return LLMResponse(content=content, usage=usage)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)


# ============================================================
# 0bis. SCHEMA JSON DE CONTRAINTE OLLAMA (parametre `format`)
# ============================================================
# Contraint reellement les tokens generes par Ollama a respecter cette
# structure -- independant de la "bonne volonte" du modele a suivre des
# instructions textuelles dans le prompt.
OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["id", "label"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_node_id": {"type": "string"},
                    "end_node_id": {"type": "string"},
                    "type": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["start_node_id", "end_node_id", "type"],
            },
        },
    },
    "required": ["nodes", "relationships"],
}


# ============================================================
# 0ter. CORRECTIF MEMOIRE : forcer max_concurrency=1
# ============================================================
# SimpleKGPipeline lance par defaut 5 appels LLM EN PARALLELE
# (max_concurrency=5, code en dur, non exposee comme parametre
# configurable). Sur une machine avec RAM limitee (ex: 16 Go dont
# seulement ~2 Go libres), faire tourner 5 requetes simultanees vers
# qwen2.5:7b (chacune avec num_ctx=8192) sature la memoire et provoque
# un arret brutal du processus Python par le systeme (OOM), sans aucun
# message d'erreur exploitable dans les logs.
#
# On patche la methode interne qui construit l'extracteur pour forcer
# max_concurrency=1 : les chunks sont alors traites un par un au lieu
# de 5 a la fois -- plus lent, mais stable sur une machine aux
# ressources limitees.
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder import (
    SimpleKGPipelineConfig,
)
try:
    # Chemin recent (neo4j_graphrag >= une certaine version)
    from neo4j_graphrag.components.entity_relation_extractor import (
        LLMEntityRelationExtractor as _LLMEntityRelationExtractor,
    )
except ModuleNotFoundError:
    # Chemin ancien (versions plus anciennes de neo4j_graphrag, encore
    # valide mais marque "deprecated" dans les versions recentes)
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor as _LLMEntityRelationExtractor,
    )


def _patched_get_extractor(self):
    llm = self.get_default_llm()
    return _LLMEntityRelationExtractor(
        llm=llm,
        prompt_template=self.prompt_template,
        on_error=self.on_error,
        use_structured_output=llm.supports_structured_output,
        max_concurrency=1,  # <-- correctif : sequentiel au lieu de 5 en parallele
    )


SimpleKGPipelineConfig._get_extractor = _patched_get_extractor


# ============================================================
# 1. SPLITTER PERSONNALISE : decoupe par diapositive, page, ou section
# ============================================================
class HybridSplitter(TextSplitter):
    def __init__(self) -> None:
        self.fallback = FixedSizeSplitter(chunk_size=600, chunk_overlap=100)

    async def run(self, text: str) -> TextChunks:
        if "=== DIAPOSITIVE" in text:
            return await self._split_by_marker(text, "=== DIAPOSITIVE")
        elif "=== PAGE" in text:
            return await self._split_by_marker(text, "=== PAGE")
        elif "=== SOURCE ===" in text:
            return await self._split_by_sections(text)
        else:
            return await self.fallback.run(text)

    async def _split_by_marker(self, text: str, marker: str) -> TextChunks:
        index = 0
        chunks = []
        current_section = ""
        for line in text.split('\n'):
            if line.startswith(marker) and current_section.strip():
                chunks.append(TextChunk(text=current_section, index=index))
                current_section = ""
                index += 1
            current_section += line + "\n"
        if current_section.strip():
            chunks.append(TextChunk(text=current_section, index=index))
        return TextChunks(chunks=chunks)

    async def _split_by_sections(self, text: str) -> TextChunks:
        index = 0
        chunks = []
        current_section = ""
        for line in text.split('\n'):
            stripped = line.strip()
            is_opening_marker = (
                stripped.startswith("=== ")
                and stripped.endswith(" ===")
                and not stripped.startswith("=== FIN")
            )
            if is_opening_marker and current_section.strip():
                chunks.append(TextChunk(text=current_section, index=index))
                current_section = ""
                index += 1
            current_section += line + "\n"
        if current_section.strip():
            chunks.append(TextChunk(text=current_section, index=index))
        return TextChunks(chunks=chunks)


text_splitter = HybridSplitter()

# ============================================================
# 2. SCHEMA PERSONNALISE -- 16 types de noeuds, audites contre
#    les 10 documents reels du corpus (voir commentaire en tete
#    de fichier). Chaque description contient un exemple concret.
# ============================================================

NODE_TYPES = [
    {"label": "Loi", "description": "TOUT texte legislatif ou reglementaire tunisien encadrant le tabac, quelle que soit sa forme -- loi, decret, arrete (ministeriel ou interministeriel), circulaire -- qu'il soit identifie par un NUMERO (ex: 'Loi n° 98-17') ou par une SIMPLE DATE sans numero (ex: 'Arrete du 24 fevrier 1999', 'Circulaire du 24 Aout 2023') -- l'absence de numero ne change PAS le type, un arrete ou une circulaire restent TOUJOURS de type Loi, jamais Mesure. Vaut aussi pour un traite/convention international nomme. Exemple dans le corpus : 'Loi n° 98-17 du 23 Fevrier 1998', 'Decret n° 98-2248', 'Arrete du 24 fevrier 1999', 'Circulaire du 24 Aout 2023', 'Convention-Cadre de Lutte Anti-Tabac (CCLAT)'."},
    {"label": "Programme", "description": "Plan ou programme pluriannuel de lutte antitabac, nomme explicitement -- distinct d'une loi ponctuelle ou d'une mesure isolee. Exemple dans le corpus : 'Programme national de lutte anti-tabac (2001)', 'Plan national de reduction du tabagisme (PNRT)', 'Projet FCTC 20-30'. NE PAS INCLURE les intitules d'axes ou de sections d'une presentation (ex: 'Axe 3 : Aide au sevrage tabagique') -- ce sont des titres de diapositive/plan de presentation, pas des programmes nommes (meme regle de sobriete que pour les 'Axes strategiques' de l'exemple 10)."},
    {"label": "Etude", "description": "Etude ou enquete epidemiologique nommee avec ses annees. Exemple dans le corpus : 'THES 2016', 'GYTS 2024', 'MedSPAD 2013/2017/2021'."},
    {"label": "Substance", "description": "Substance chimique PRECISE et IDENTIFIABLE liee au tabac, nommee explicitement. Exemple dans le corpus : nicotine, goudron, monoxyde de carbone, cadmium, arsenic, ammoniac, radon. NE PAS INCLURE 'tabac', 'tabagisme' ou 'cigarette(s)' comme Substance : 'tabagisme' est un Concept (deja liste dans le type Concept), 'cigarette' est un Produit (deja liste dans le type Produit), et 'tabac' seul (sans molecule precise nommee) n'est ni l'un ni l'autre -- n'en fais pas un noeud Substance."},
    {"label": "Pathologie", "description": "Maladie ou trouble de sante nomme comme cause ou consequence du tabac. Exemple dans le corpus : cancer du poumon, BPCO, infarctus du myocarde, AVC, faible poids de naissance, DMLA, parodontite."},
    {"label": "Produit", "description": "Produit lie a la consommation ou au sevrage du tabac : formes de TSN (patch, gomme, pastille, spray, inhalateur) ET produits alternatifs au tabac classique (cigarette electronique, tabac chauffe, sachets de nicotine, narguile). Distingue les deux categories via une propriete du noeud (ex: categorie='TSN' ou categorie='produit alternatif'), pas via des labels differents. NOM CANONIQUE OBLIGATOIRE : n'inclus JAMAIS un dosage, une duree de liberation ou toute variante numerique dans le nom du produit -- le nom doit etre le meme partout ou ce produit est mentionne, dans n'importe quel document, pour que les mentions se fusionnent en UN SEUL noeud (voir EXEMPLE 3 et EXEMPLE 13 plus bas). Mets les dosages dans une propriete texte (ex: properties.dosages). Exemple dans le corpus : 'Patch a la nicotine' (PAS 'Patch a la nicotine 21mg'), 'Gomme a la nicotine', 'Cigarette electronique', 'Tabac chauffe'."},
    {"label": "Medicament", "description": "Medicament non-nicotinique pour le sevrage, ou tout medicament dont le metabolisme est affecte par le tabac, nomme explicitement. Exemple dans le corpus : bupropion, varenicline, warfarine, methadone, insuline."},
    {"label": "Institution", "description": "Organisme, ministere ou agence nomme. Exemple dans le corpus : Ministere de la Sante (Tunisie), OMS, CDC, NHS, Sante publique France, Quitline, Faculte de Medecine de Tunis, Quit, RACGP."},
    {"label": "Mesure", "description": "Mesure de politique publique nommee ou decrite precisement -- le CONTENU d'une action, jamais l'INSTRUMENT JURIDIQUE lui-meme qui l'instaure. Exemple dans le corpus : les 6 mesures MPOWER (Monitor, Protect, Offer, Warn, Enforce, Raise), interdiction de publicite, avertissement sanitaire sur les paquets, interdiction de fumer dans les lieux publics (France, 2007). NE PAS INCLURE un arrete, decret, circulaire ou loi lui-meme (ex: 'Arrete du 24 fevrier 1999' est TOUJOURS de type Loi, jamais Mesure, meme s'il n'a pas de numero -- voir description du type Loi) -- la mesure est ce que ce texte impose (ex: 'Interdiction de publicite'), pas le texte qui l'impose. NOM CANONIQUE OBLIGATOIRE : n'inclus JAMAIS de precision variable (ex: '... du tabac', '... des produits tabagiques') dans le nom -- utilise la forme la plus courte et generique possible, IDENTIQUE partout ou cette mesure est mentionnee, pour que les mentions se fusionnent en un seul noeud au lieu d'en creer un par variante de formulation. Exemple : 'Interdiction de publicite' (PAS 'Interdiction de publicite du tabac' ni 'Interdiction de publicite pour le tabac'). NE PAS INCLURE non plus un VERBE ou une expression verbale qui relie simplement deux textes legislatifs entre eux (ex: 'modifiant et completant', 'abrogeant', 'completant') -- ce n'est pas une mesure de politique publique, c'est une relation entre deux noeuds Loi : utilise la relation MODIFIE ou ABROGE directement entre les deux noeuds Loi concernes, ne cree jamais de noeud Mesure a partir d'un simple verbe de ce type."},
    {"label": "Outil", "description": "Outil, modele ou instrument clinique nomme, utilise pour evaluer ou accompagner le sevrage. Garde-le comme UN SEUL noeud avec ses details en propriete -- ne cree pas un noeud par etape ou par tranche de score. Exemple dans le corpus : 'Heaviness of Smoking Index (HSI)', monitoring du monoxyde de carbone (CO), modele 'Ask, Advise, Help'."},
    {"label": "Concept", "description": "Concept nomme lie a la dependance, au sevrage, ou au tabagisme en general. Exemple dans le corpus : dependance physique, dependance psychologique, dependance comportementale, tolerance, renforcement positif/negatif, recepteurs nicotiniques, dopamine, tabagisme, tabagisme passif."},
    {"label": "Symptome", "description": "Symptome PHYSIOLOGIQUE ou COMPORTEMENTAL de sevrage nicotinique, ou effet indesirable physiologique d'un traitement, nomme explicitement. Exemple dans le corpus : maux de tete, irritabilite, insomnie, nausees, irritation cutanee. NE PAS INCLURE, sous AUCUNE forme ni reformulation : (1) les risques materiels d'un appareil (explosion, incendie d'une cigarette electronique -- risque produit, pas symptome medical) ; (2) TOUTE citation/opinion subjective rapportee dans une diapositive de type 'cognitions' ou 'idees recues' -- cela couvre TOUTES les phrases de ce type dans le corpus (ex: 'Ca fait mal a ceux qui m'entourent', 'Risques graves pour la sante !', 'Coute une fortune !', 'Ca sent mauvais !', 'C'est cool de fumer !', 'Cela m'aide a me detendre', et toute autre phrase de la meme diapositive), pas seulement les exemples cites ici -- si une phrase est une opinion/croyance rapportee entre guillemets, exclus-la systematiquement ; (3) la mortalite ou le deces, MEME avec un contexte ajoute (ex: 'Mort', 'Deces', 'Deces chez les nourrissons', 'Deces chez les adultes non-fumeurs') -- c'est une consequence/statistique, pas un symptome ; (4) les categories d'enquete epidemiologique (ex: 'Consommation actuelle', 'Usage actuel', 'Experimentation') ; (5) les mecanismes de dependance qui sont deja des Concept (ex: 'Tolerance', 'Renforcement positif', 'Renforcement negatif') -- ne les duplique pas en Symptome ; (6) les niveaux d'un score ou index (ex: 'dependance faible/moderee/forte' d'un indice comme le HSI) -- garde-les en propriete du noeud Outil correspondant, ne cree pas un noeud Symptome par niveau ; (7) les etats emotionnels d'un diagramme simple emotion-envie-cigarette (ex: 'Stresse', 'Heureux', 'Triste', 'En colere') -- ce sont des exemples illustratifs d'une diapositive, pas des symptomes ; (8) les elements d'un diagramme de dependance comportementale/sociale (ex: 'pression des pairs', 'moments de fete', 'reflexes du quotidien', 'lieux declencheurs') -- deja couverts par le Concept 'Dependance comportementale', n'en fais pas des Symptome separes."},
    {"label": "Population", "description": "Groupe de population cible explicitement nomme dans le texte. Exemple dans le corpus : femmes enceintes, femmes allaitantes, adolescents 13-15 ans, adultes 15 ans et plus."},
    {"label": "Pays", "description": "Pays nomme dans une comparaison ou statistique. Exemple dans le corpus : Tunisie, Etats-Unis, Japon, Allemagne, Canada, Chine, Inde, Bangladesh, France, Royaume-Uni."},
    {"label": "Statistique", "description": "Chiffre, pourcentage ou montant precis tire d'un tableau ou d'une phrase (cout economique, prevalence, mortalite...), avec son unite et son annee si disponible. Toujours relie a ce qu'il decrit (un Pays, une Population, une Pathologie...) par une relation explicite (ex: COUTE, EVALUE, ASSOCIE_A) plutot que laisse comme simple texte non structure. Exemple dans le corpus : '531 milliards USD' (cout tabagisme Etats-Unis), '25,1%' (prevalence adultes Tunisie), '13200 deces/an' (mortalite Tunisie). NOM : la valeur brute avec son unite, ex '531 milliards USD', PAS de phrase complete."},
    {"label": "Personne", "description": "Personne REELLEMENT NOMMEE explicitement (prenom+nom ou titre+nom) : auteur, reviseur, ou intervenant cite dans un document. Exemple dans le corpus : Dr Salsabil Rejaibi, Yuan Loke, Sarah Malone. NE PAS INCLURE les pronoms ou roles generiques qui ne nomment personne en particulier -- ex: 'vous', 'le fumeur', 'un non-fumeur', 'le lecteur' -- ce ne sont pas des personnes identifiees, n'en fais pas des noeuds Personne."},
    {"label": "Thematique", "description": "Grand theme couvert par un document source (rattache au noeud Source, pas au contenu detaille)."},
    {"label": "Citation", "description": "Reference bibliographique ou etude EXTERNE citee dans le texte (ex: en bas de fiche OMS/CDC/NHS), qui n'est PAS l'un des 10 documents du corpus lui-meme. Exemple dans le corpus : 'WHO Report on the Global Tobacco Epidemic 2008', 'John RM, et al. Nicotine Tob Res. 2021 Jan 22;23(2):294-301.', 'Global Burden of Disease 2023'. ATTENTION : n'utilise JAMAIS le label 'Source' pour ceci -- 'Source' est reserve au document source lui-meme (gere automatiquement par le pipeline, ce n'est pas un label que tu dois produire) ; une reference citee DANS un document est TOUJOURS 'Citation', jamais 'Source'."},
]

RELATIONSHIP_TYPES = [
    # Legislation / reglementation
    "INTERDIT", "IMPOSE", "MET_A_JOUR", "ABROGE", "MODIFIE",
    "MET_EN_OEUVRE", "APPLIQUE_A",
    # Causalite et risque sante
    "CAUSE", "AUGMENTE_RISQUE", "REDUIT_RISQUE", "DECLENCHE",
    # Position clinique
    "RECOMMANDE", "DECONSEILLE", "CONTRE_INDIQUE",
    "NECESSITE_SURVEILLANCE_MEDICALE", "NECESSITE_PRUDENCE_AVEC",
    # Composition / structure
    "INCLUT", "COMPOSE_DE", "CONTIENT",
    # Traitement
    "TRAITE", "REDUIT", "FOURNIT",
    # Institutions / documents
    "PUBLIE", "AUTEUR_PAR", "REVISE_PAR", "APPROUVE_PAR", "ADHERE_A",
    # Etudes / donnees / outils
    "EVALUE", "PRODUIT", "MESURE_DANS",
    # Mecanismes biologiques
    "SE_LIE_A", "DECLENCHE_LIBERATION_DE", "IMPLIQUE_DANS",
    "AFFECTE_METABOLISME_DE", "SIMILAIRE_A",
    # Association generale / citation
    "CONCERNE", "ASSOCIE_A", "MOTIVE_PAR", "RESULTE_EN",
    "MANIFESTE_PAR", "COUTE", "TRAITE_DE", "CITE",
]

# ============================================================
# 3. PROMPT PERSONNALISE avec exemples nombreux et verifies (many-shot)
# ============================================================
# Chaque exemple ci-dessous est tire mot pour mot d'un des 10 documents
# du corpus. Les exemples 10, 11 et 12 servent aussi a montrer
# explicitement au modele de NE PAS exploser un contenu structurel
# en trop de petits noeuds -- c'est le principal garde-fou contre
# un graphe bruite/chaotique avec un schema plus large.

domain_instructions = """
Tu es un expert en extraction d'information, specialise dans le tabagisme en Tunisie.

REGLE ABSOLUE : n'extrais QUE les entites et faits explicitement ecrits dans le texte fourni.
N'utilise JAMAIS tes connaissances generales sur la Tunisie, la legislation, la medecine ou
la sante publique pour completer, corriger, deduire ou enrichir ce que tu extrais.
Si un numero de loi, une date, un pourcentage, un delai ou un nom precis n'apparait pas
litteralement dans le texte, NE L'INVENTE PAS et n'extrais pas cette entite.

REGLE DE SOBRIETE : ne cree pas un noeud pour chaque sous-detail structurel d'une liste ou
d'un tableau si ces details ne sont pas des entites du monde reel en soi (voir exemples 10 et 11
ci-dessous). Un schema trop granulaire produit un graphe bruite plutot qu'utile.

REGLE DES MENTIONS SECONDAIRES : n'ignore pas une entite nommee seulement parce qu'elle
apparait en INCISE ou ENTRE PARENTHESES dans une phrase dont le sujet principal est autre
chose. Exemple : dans "L'arrete du 21 janvier 2022 ... (avec appui du Projet FCTC 20-30)",
le sujet principal de la phrase est l'arrete (type Loi), mais "Projet FCTC 20-30" doit QUAND
MEME etre extrait comme un noeud Programme a part entiere, meme s'il n'est mentionne
qu'entre parentheses. Une entite entre parentheses reste une entite reelle a extraire.

REGLE DE NOMMAGE CANONIQUE (s'applique a TOUS les types de noeuds, pas seulement a un type en
particulier) : si un meme element du monde reel (une loi, un programme, une mesure, un produit...)
peut etre mentionne avec des formulations legerement differentes selon le passage ou le document
(ex: "Programme national de LA lutte anti-tabac" vs "Programme national de lutte anti-tabac" ;
"Interdiction de publicite" vs "Interdiction de publicite du tabac"), choisis TOUJOURS la forme la
PLUS COURTE ET LA PLUS STABLE, sans article ni precision redondante ("du tabac", "des produits
tabagiques"...), et utilise EXACTEMENT cette meme forme partout ou cet element est mentionne, quel
que soit le document source. Une variation de nom, meme minime (un article "la" en trop, un mot
ajoute), cree un doublon dans le graphe au lieu de fusionner les mentions du meme element -- c'est
une erreur aussi importante qu'une hallucination de fait.

VOCABULAIRE CANONIQUE DE REFERENCE : pour les entites suivantes, deja identifiees dans le
corpus, utilise TOUJOURS EXACTEMENT le nom ecrit ici (rien de plus, rien de moins), quel que
soit le document ou la formulation locale du passage que tu traites. Si tu rencontres une de
ces entites formulee differemment, ramene-la a cette forme canonique.

-- Loi --
"Loi n° 98-17", "Decret n° 98-2248", "Arrete du 30 janvier 1999", "Arrete du 24 fevrier 1999",
"Arrete du 20 mai 2014", "Arrete du 21 janvier 2022", "Circulaire du 24 Aout 2023",
"Convention-Cadre de Lutte Anti-Tabac (CCLAT)"
ATTENTION : inclus TOUJOURS le symbole "°" apres le "n" pour un numero de loi/decret (ex:
"n° 98-17", jamais "n 98-17" sans le symbole) -- c'est une source de doublon frequente si
le symbole est omis dans certaines extractions et present dans d'autres.

-- Programme --
"Programme national de lutte anti-tabac (2001)" (jamais "de LA lutte"),
"Plan national de reduction du tabagisme (PNRT)", "Projet FCTC 20-30"

-- Mesure --
"Interdiction de publicite" (jamais "... du tabac" ni "... pour le tabac"),
"Interdiction de fumer dans les lieux a usage collectif",
"Interdiction de fumer dans les etablissements universitaires",
"Mesures MPOWER" (UN SEUL noeud pour les 6 mesures Monitor/Protect/Offer/Warn/Enforce/Raise
-- garde le detail des 6 en propriete texte, ne cree pas 6 noeuds separes, meme regle de
sobriete que pour les Axes strategiques de l'Exemple 10)

-- Etude --
"THES 2016", "GYTS 2024", "MedSPAD" (un seul noeud regroupant les vagues 2013/2017/2021 en
propriete, pas 3 noeuds separes), "Global Burden of Disease 2023", "US Surgeon General
Report 2010", "WHO Report on the Global Tobacco Epidemic 2008"

-- Produit (categorie="TSN") --
"Gomme a la nicotine", "Patch a la nicotine", "Pastille a la nicotine",
"Spray nasal a la nicotine", "Inhalateur de nicotine"

-- Produit (categorie="produit alternatif") --
"Cigarette", "Cigarette electronique", "Narguile", "Tabac chauffe", "Cigare", "Cigarillo",
"Tabac a rouler", "Tabac a pipe", "Sachets de nicotine", "Bidis", "Kreteks",
"Produits de tabac sans fumee"
ATTENTION SYNONYME (piege de fragmentation frequent) : "Narguile" et "Pipe a eau" designent
EXACTEMENT le meme produit -- le corpus utilise les deux termes dans le MEME document
(CHATBOT 1, diapositives 3 et 9). Utilise TOUJOURS "Narguile" comme nom canonique, meme
quand le texte source ecrit "pipe a eau". De meme, "Tabac chauffe" et l'abreviation anglaise
"HTP" (Heated Tobacco Products) designent le meme produit -- canonique : "Tabac chauffe".

-- Substance --
"Nicotine", "Goudron", "Monoxyde de carbone", "Cadmium", "Plomb", "Chrome", "Mercure",
"Arsenic", "Ammoniac", "Radon", "Dopamine", "Propylene glycol", "Glycerine vegetale",
"Oxyde d'azote", "Acide cyanhydrique"
NE CREE JAMAIS de noeud Substance nomme "Tabac", "Tabagisme" ou "Cigarette(s)" (voir la
description du type Substance).

-- Pathologie (liste non exhaustive -- seulement les plus citees a travers PLUSIEURS
   documents differents, donc les plus a risque de fragmentation) --
"Cancer du poumon", "BPCO", "AVC", "Maladies cardiaques", "Diabete de type 2", "Asthme",
"Tabagisme passif" (double etiquette Pathologie+Concept, voir regle de nommage canonique)

-- Concept --
"Tabagisme", "Tabagisme passif", "Dependance physique", "Dependance psychologique",
"Dependance comportementale", "Tolerance", "Renforcement positif", "Renforcement negatif",
"Recepteurs nicotiniques", "Conditionnement operant"

-- Symptome --
"Maux de tete", "Toux", "Envies (cravings)", "Augmentation de l'appetit ou prise de poids",
"Changements d'humeur", "Agitation", "Diminution du rythme cardiaque",
"Difficulte a se concentrer", "Symptomes grippaux", "Insomnie"
(rappel : n'inclus PAS les citations "cognitions", ni la mortalite, ni les categories
d'enquete -- voir description du type Symptome)

-- Institution --
"Ministere de la Sante (Tunisie)", "OMS", "CDC", "NHS", "Sante publique France"

-- Pays --
"Tunisie", "Etats-Unis", "Japon", "Allemagne", "Canada", "Chine", "Inde", "Bangladesh",
"France", "Royaume-Uni"

-- Statistique --
Toujours accompagnee d'une relation COUTE, EVALUE ou ASSOCIE_A vers l'entite decrite (voir
Exemple 16) -- jamais un noeud Statistique isole sans relation.

-- Population --
"Adultes 15 ans et plus", "Adolescents 13-15 ans", "Adolescents 15-17 ans",
"Femmes enceintes", "Femmes allaitantes", "Nourrissons", "Enfants"

Cette liste n'est pas exhaustive -- si une entite legitime n'y figure pas, extrais-la quand
meme normalement en respectant la regle de nommage canonique ci-dessus (forme la plus courte
et stable). Cette liste sert a trancher les cas ambigus rencontres dans plusieurs documents,
pas a limiter l'extraction aux seules entites listees ici.

Voici plusieurs exemples de bonne extraction, tires des documents reels de ce projet.

--- EXEMPLE 1 : texte legislatif ---
Texte source :
"Loi n 98-17 du 23 Fevrier 1998 relative a la prevention des mefaits du tabac (18 articles) :
Interdiction de faire la propagande / publicite, Interdiction de fumer dans les lieux
affectes a l'usage collectif avec fixation de la liste de ces lieux"

Extraction attendue :
- Noeud (Loi) "Loi n 98-17 du 23 Fevrier 1998"
- Noeud (Mesure) "Interdiction de publicite du tabac"
- Noeud (Mesure) "Interdiction de fumer dans les lieux a usage collectif"
- Relation : (Loi n 98-17) -[INTERDIT]-> (Interdiction de publicite du tabac)
- Relation : (Loi n 98-17) -[INTERDIT]-> (Interdiction de fumer dans les lieux a usage collectif)
Ne PAS ajouter le decret d'application de 1998 s'il n'est pas mentionne dans ce texte precis,
meme si tu sais qu'il existe.

--- EXEMPLE 2 : donnees epidemiologiques ---
Texte source :
"Enquete Globale Sante Tunisie (THES) 2016 : 15 ans et plus. Adultes: Prevalence globale
de 25,1% plus elevee chez les hommes"

Extraction attendue :
- Noeud (Etude) "THES 2016"
- Noeud (Population) "Adultes 15 ans et plus"
- Relation : (THES 2016) -[EVALUE]-> (Adultes 15 ans et plus)
Ne PAS creer de noeud pour "25,1%" seul (donnee chiffree, pas une entite nommee -- elle peut
rester en propriete texte du chunk).

--- EXEMPLE 3 : produit de sevrage (isolement des chunks) ---
Texte source :
"Patch a la Nicotine : Libere la nicotine a travers la peau ; partie superieure du corps.
Liberation sur 24h : 7mg, 14mg, 21mg"

Extraction attendue :
- Noeud (Produit) "Patch a la nicotine", proprietes categorie="TSN", dosages="Liberation
  24h : 7mg, 14mg, 21mg"
- Noeud (Substance) "Nicotine"
- Relation : (Patch a la nicotine) -[INCLUT]-> (Nicotine)
IMPORTANT (voir aussi EXEMPLE 13) : le nom du noeud est "Patch a la nicotine", SANS le
dosage. Les dosages vont dans la propriete "dosages", jamais dans le nom -- sinon ce
meme produit mentionne ailleurs avec un autre dosage creerait un noeud different au lieu
de se fusionner avec celui-ci.
A NE PAS FAIRE : ne pas deduire que ce patch est recommande pour les femmes enceintes si
cette phrase precise ne le dit pas explicitement dans le passage en cours d'extraction --
meme si un autre passage du meme document le mentionne ailleurs. Chaque chunk s'extrait
independamment de ce qu'il contient reellement, pas de ce que tu sais du document entier.

--- EXEMPLE 4 : contre-indication medicale ---
Texte source :
"Lozenge containing aspartame is not suitable for women with PKU"

Extraction attendue :
- Noeud (Pathologie) "PKU (phenylcetonurie)"
- Noeud (Produit) "Pastille a la nicotine", propriete categorie="TSN"
- Relation : (PKU) -[CONTRE_INDIQUE]-> (Pastille a la nicotine)
Ne PAS generaliser cette contre-indication a d'autres formes de TSN (gomme, patch) si le
texte ne mentionne que la pastille dans cette phrase.

--- EXEMPLE 5 : surveillance medicale (nuance differente d'une contre-indication) ---
Texte source :
"NRT use in women with underlying cardiovascular disease is not associated with an
increased risk of adverse cardiovascular events. However, it should be used under
medical supervision."

Extraction attendue :
- Noeud (Pathologie) "Maladie cardiovasculaire"
- Noeud (Produit) "Traitement de substitution nicotinique (TSN)", propriete categorie="TSN"
- Relation : (Maladie cardiovasculaire) -[NECESSITE_SURVEILLANCE_MEDICALE]-> (TSN)
Important : ceci n'est PAS une contre-indication (le texte dit explicitement qu'il n'y a
pas de risque accru), donc n'utilise PAS la relation CONTRE_INDIQUE ici -- utilise
NECESSITE_SURVEILLANCE_MEDICALE, qui reflete la nuance reelle du texte.

--- EXEMPLE 6 : mecanisme biologique ---
Texte source :
"Renforcement positif : se lie aux recepteurs nicotiniques, provoquant la liberation
de dopamine, ce qui fait que les fumeurs se sentent bien."

Extraction attendue :
- Noeud (Substance) "Nicotine"
- Noeud (Concept) "Recepteurs nicotiniques"
- Noeud (Substance) "Dopamine"
- Relation : (Nicotine) -[SE_LIE_A]-> (Recepteurs nicotiniques)
- Relation : (Recepteurs nicotiniques) -[DECLENCHE_LIBERATION_DE]-> (Dopamine)

--- EXEMPLE 7 : interaction medicamenteuse ---
Texte source :
"Warfarin | Increased clearance, decreased plasma concentration; no measurable effect
on prothrombin time. | Monitor INR and adverse effects. Adjust dose accordingly."

Extraction attendue :
- Noeud (Medicament) "Warfarine"
- Relation : (Tabagisme actif) -[AFFECTE_METABOLISME_DE]-> (Warfarine)
Le texte precise "increased clearance" (donc le tabac accelere l'elimination du
medicament) -- ne pas inverser le sens de la relation.

--- EXEMPLE 8 : auteurs et institutions d'un document ---
Texte source :
"Lead Author Yuan Loke (Senior Medicines Information Pharmacist, The Women's).
Reviewers: Dr Sarah Malone (Consultant Obstetrician)"

Extraction attendue :
- Noeud (Personne) "Yuan Loke"
- Noeud (Personne) "Sarah Malone"
- Relation : (Document source) -[AUTEUR_PAR]-> (Yuan Loke)
- Relation : (Document source) -[REVISE_PAR]-> (Sarah Malone)
Ne PAS extraire de profession detaillee comme noeud separe (ex: pas de noeud
"Consultant Obstetrician") -- garde ce detail en propriete du noeud Personne.

--- EXEMPLE 9 : ANTI-HALLUCINATION -- ne pas ajouter de connaissances externes ---
Texte source (extrait reel du corpus) :
"Nicotine : Pas un cancerogene. Liquide dans son etat natif. Distille a partir de tabac
brule et transporte sur des gouttelettes de goudron. Demi-vie 120 minutes."

Extraction attendue :
- Noeud (Substance) "Nicotine", propriete demi_vie = "120 minutes"
ERREUR DEJA COMMISE PAR LE PASSE (a ne pas reproduire) : un modele avait ajoute la
propriete "delai pour atteindre le cerveau : 7 a 10 secondes" -- cette information est
vraie en general, mais elle N'APPARAIT PAS dans ce texte precis. C'est exactement le
type d'ajout interdit par la regle absolue en haut de ce prompt : meme un fait exact
au sens general devient une hallucination s'il n'est pas ecrit dans le passage que tu
traites en ce moment.

--- EXEMPLE 10 : programme national (et regle de sobriete sur le contenu structurel) ---
Texte source :
"Programme national de lutte anti-tabac (2001)
Axes strategiques: themes du chatbot
3-Legislation Tunisienne
4-Aide au sevrage tabagique
1-Effets des produits du tabac
2-Donnees epidemiologiques"

Extraction attendue :
- Noeud (Programme) "Programme national de lutte anti-tabac (2001)"
A NE PAS FAIRE : ne cree PAS 4 noeuds separes pour "Legislation Tunisienne", "Aide au
sevrage tabagique", etc. -- ce sont les axes/themes d'une presentation, pas des entites
du monde reel. Garde-les en propriete texte du noeud Programme si besoin, pas en noeuds.

--- EXEMPLE 11 : outil clinique (meme regle de sobriete) ---
Texte source :
"La dependance nicotinique peut etre evaluee avec le Heaviness of Smoking Index (HSI):
Question 1 - Nombre de cigarettes par jour: 10 ou moins = 0 point / 11 a 20 = 1 point /
21 a 30 = 2 points / 31 ou plus = 3 points [...]
Score total et niveau de dependance: 0 a 2 = dependance faible / 3 a 4 = dependance
moderee / 5 a 6 = dependance forte"

Extraction attendue :
- Noeud (Outil) "Heaviness of Smoking Index (HSI)", propriete bareme = resume du systeme
  de points
A NE PAS FAIRE : ne cree PAS un noeud par tranche de score (pas de noeud "dependance
faible 0-2", "dependance moderee 3-4", etc.) -- c'est le detail interne d'un seul outil,
pas 3 entites distinctes. Meme regle pour le modele "Ask, Advise, Help" : UN SEUL noeud
(Outil), jamais un noeud par etape (Ask / Advise / Help separement).

--- EXEMPLE 12 : produit alternatif au tabac (distinction avec le TSN) ---
Texte source :
"Cigarettes electroniques: chauffent un liquide pour creer des aerosols inhales; nocives
pour la sante et non sures"

Extraction attendue :
- Noeud (Produit) "Cigarette electronique", propriete categorie="produit alternatif"
Important : ne classe PAS "Cigarette electronique" comme une forme de TSN (traitement de
substitution nicotinique) -- ce sont deux categories differentes de Produit, distinguees
par la propriete "categorie", pas par des labels differents. Ici le texte dit juste
"nocif, non sur" sans nommer de pathologie precise -- donc pas de relation CAUSE vers une
pathologie ; garde cette information en propriete du noeud.

--- EXEMPLE 13 : NOM CANONIQUE -- meme produit mentionne dans deux documents differents ---
Texte source (document A, guide Quit) :
"Nicotine patches are available in 7mg, 14mg, 21mg (24-hour release) and 10mg, 15mg,
25mg (16-hour release)."
Texte source (document B, CHATBOT_3, chunk different) :
"Patch a la Nicotine : Liberation sur 24 h : 7mg, 14mg, 21mg Liberation sur 16 h: 10mg,
15mg, 25mg"

Extraction attendue POUR CHACUN de ces deux textes, separement :
- Noeud (Produit) "Patch a la nicotine", proprietes categorie="TSN", dosages="24h:
  7/14/21mg ; 16h: 10/15/25mg"
ERREUR A NE PAS REPRODUIRE : creer "Patch a la nicotine 21mg", "Patch 24 heures
(21mg,14mg,7mg)", "Patch 16 heures (25mg)" ou toute autre variante de nom differente
d'un chunk a l'autre. Le nom "Patch a la nicotine" doit etre EXACTEMENT IDENTIQUE
(memes majuscules, memes accents, pas de dosage ni de duree dans le nom) partout ou ce
produit apparait dans le corpus, quel que soit le document ou le chunk -- c'est ce qui
permet au systeme de fusion d'entites de reconnaitre qu'il s'agit du meme produit et de
regrouper toutes ses mentions en un seul noeud au lieu d'en creer un par variante.

--- EXEMPLE 14 : citations/opinions subjectives -- rien a extraire comme Symptome ---
Texte source :
"« Ca sent mauvais ! » « Coute une fortune ! » « Risques graves pour la sante ! »
Cognitions du tabagisme - inconvenients connus
« Affecte la condition physique » « Ca fait mal a ceux qui m'entourent »"

Extraction attendue :
- AUCUN noeud Symptome a partir de ce texte.
ERREUR DEJA COMMISE PAR LE PASSE (a ne pas reproduire) : un modele avait extrait "Risques
graves pour la sante" et "Ca fait mal a ceux qui m'entourent" comme des noeuds Symptome.
C'est une erreur : ce sont des citations/croyances que les fumeurs expriment (une
diapositive de type "idees recues" ou "cognitions"), pas des manifestations physiologiques
reelles. Applique ici la meme regle de sobriete que pour les exemples 10 et 11 : n'extrais
rien plutot que de forcer une categorisation qui ne correspond a aucun type reel du schema.

--- EXEMPLE 15 : mesure a sous-composantes multiples (regle de sobriete + nom canonique) ---
Texte source :
"En 2007, l'OMS a introduit l'initiative MPOWER pour renforcer la mise en oeuvre de la
Convention-cadre de l'OMS pour la lutte antitabac (CCLAT). Les six mesures:
Monitor: surveiller la consommation de tabac et les politiques de prevention.
Protect: proteger les populations de la fumee de tabac.
Offer: proposer une aide pour arreter de fumer.
Warn: avertir des dangers du tabac.
Enforce: faire respecter l'interdiction de la publicite, la promotion et le parrainage du tabac.
Raise: augmenter les taxes sur le tabac."

Extraction attendue :
- Noeud (Mesure) "Mesures MPOWER", propriete detail="Monitor, Protect, Offer, Warn, Enforce,
  Raise" (resume des 6 composantes)
- Noeud (Institution) "OMS"
- Noeud (Loi) "Convention-Cadre de Lutte Anti-Tabac (CCLAT)"
- Relation : (OMS) -[PRODUIT]-> (Mesures MPOWER)
A NE PAS FAIRE : ne cree PAS 6 noeuds Mesure separes (un "Monitor", un "Protect", etc.) --
ce sont les composantes d'UN SEUL dispositif nomme, pas 6 mesures independantes (meme regle
de sobriete que les Exemples 10 et 11). Utilise aussi le nom canonique exact "Mesures MPOWER"
(voir vocabulaire canonique de reference plus haut), pas "MPOWER" seul ni "L'initiative
MPOWER de l'OMS".
ERREUR DEJA COMMISE PAR LE PASSE SUR CE TYPE PRECIS (a ne pas reproduire) : un modele avait
classe "MPOWER" comme type Programme au lieu de Mesure, et "CCLAT" comme type Programme au
lieu de Loi. Le type d'un noeud n'est jamais un choix libre quand l'element figure dans le
vocabulaire canonique de reference : "Mesures MPOWER" est TOUJOURS de type Mesure, et
"Convention-Cadre de Lutte Anti-Tabac (CCLAT)" est TOUJOURS de type Loi (c'est un traite
juridique international, pas un programme d'action) -- jamais Programme pour ces deux-la.

--- EXEMPLE 16 : tableau de statistiques par pays (donnee chiffree structuree) ---
Texte source :
"Coûts pour la société
| Coût économique des maladies attribuables au tabagisme (USD) |  |
| Etats-unis | 531.00 milliards |
| Japon | 41.74 milliards |
| Tunisie | 2 milliards de TND |"

Extraction attendue :
- Noeud (Pays) "Etats-Unis"
- Noeud (Statistique) "531 milliards USD"
- Noeud (Pays) "Japon"
- Noeud (Statistique) "41,74 milliards USD"
- Noeud (Pays) "Tunisie"
- Noeud (Statistique) "2 milliards de TND"
- Relation : (Etats-Unis) -[COUTE]-> (531 milliards USD)
- Relation : (Japon) -[COUTE]-> (41,74 milliards USD)
- Relation : (Tunisie) -[COUTE]-> (2 milliards de TND)
ERREUR DEJA COMMISE PAR LE PASSE (a ne pas reproduire) : un modele avait extrait chaque pays
comme noeud Pays isole, SANS creer de noeud Statistique ni de relation COUTE vers son chiffre --
le montant restait uniquement dans le texte brut du chunk, invisible pour la recherche par
graphe. Chaque ligne d'un tableau chiffre DOIT produire un noeud Statistique relie a l'entite
qu'il decrit, pas juste l'entite seule.

--- EXEMPLE 17 : programme cite en appui entre parentheses dans un texte legislatif ---
Texte source :
"L'arrete du 21 janvier 2022 modifiant et completant l'arrete du 24 fevrier 1999 relatifs
aux messages et avertissements sanitaires sur les couvertures exterieures des
paquets/emballages (avec appui du Projet FCTC 20-30)"

Extraction attendue :
- Noeud (Loi) "Arrete du 21 janvier 2022"
- Noeud (Loi) "Arrete du 24 fevrier 1999"
- Noeud (Programme) "Projet FCTC 20-30"
- Relation : (Arrete du 21 janvier 2022) -[MODIFIE]-> (Arrete du 24 fevrier 1999)
- Relation : (Arrete du 21 janvier 2022) -[ASSOCIE_A]-> (Projet FCTC 20-30)
ERREUR DEJA COMMISE PAR LE PASSE SUR CE TEXTE PRECIS (a ne pas reproduire), DEUX FOIS :
(1) un modele avait extrait uniquement l'arrete (type Loi) et completement ignore "Projet
FCTC 20-30" parce qu'il n'apparait qu'entre parentheses, en fin de phrase, alors que le
sujet grammatical principal de la phrase est l'arrete -- une mention entre parentheses
reste une entite reelle du monde reel a extraire comme noeud a part entiere (voir aussi la
REGLE DES MENTIONS SECONDAIRES plus haut). (2) un modele avait aussi invente un noeud
(Mesure) "Modifiant et completant" a partir du verbe qui relie les deux arretes -- ce
n'est PAS une mesure, c'est juste la relation MODIFIE entre les deux noeuds Loi, ecrite en
toutes lettres dans la phrase. Ne cree JAMAIS de noeud a partir de ce genre de verbe de
liaison entre deux textes legislatifs. Si ce texte precis se represente (meme reformule),
n'oublie PAS le noeud Programme "Projet FCTC 20-30" et ne cree PAS de noeud Mesure a
partir du verbe "modifiant et completant".

--- FIN DES EXEMPLES ---

Une diapositive de titre, de remerciement, ou de liens web sans contenu factuel ne contient
PAS d'entites de type Loi, Etude ou Institution precises -- dans ce cas, n'extrais rien plutot
que de forcer une extraction.

Extrais uniquement les entites et relations liees au tabagisme : legislation, programmes,
donnees epidemiologiques, substances, pathologies, produits et medicaments de sevrage,
institutions, outils cliniques, symptomes de sevrage, concepts de dependance, mecanismes
biologiques, interactions medicamenteuses, populations ciblees, pays cites dans des
comparaisons, et personnes citees comme auteurs/reviseurs de documents.
"""
domain_instructions += """

FORMAT DE SORTIE JSON STRICT — À RESPECTER EXACTEMENT, SANS AUCUNE VARIATION :

Le champ "nodes" DOIT être une LISTE (jamais un dictionnaire), où chaque élément suit
exactement cette forme :
{{"id": "0", "label": "NomDuType", "properties": {{"name": "..."}}}}

Le champ "relationships" DOIT être une LISTE (jamais un dictionnaire), où chaque élément
suit exactement cette forme :
{{"start_node_id": "0", "end_node_id": "1", "type": "NOM_RELATION", "properties": {{}}}}

INTERDIT :
- N'utilise JAMAIS "type" comme clé pour désigner le label d'un nœud (utilise "label").
- N'utilise JAMAIS "source"/"target" pour désigner les extrémités d'une relation
  (utilise "start_node_id"/"end_node_id").
- N'utilise JAMAIS "label" comme clé pour désigner le type d'une relation
  (utilise "type").
- Ne transforme JAMAIS "nodes" ou "relationships" en dictionnaire indexé par id ou par
  nom -- ce sont toujours des LISTES.

Exemple correct complet :
{{
  "nodes": [
    {{"id": "0", "label": "Loi", "properties": {{"name": "Loi n° 98-17"}}}},
    {{"id": "1", "label": "Mesure", "properties": {{"name": "Interdiction de publicité"}}}}
  ],
  "relationships": [
    {{"start_node_id": "0", "end_node_id": "1", "type": "INTERDIT", "properties": {{}}}}
  ]
}}
"""

prompt_template = ERExtractionTemplate(
    template=domain_instructions + ERExtractionTemplate.DEFAULT_TEMPLATE
)