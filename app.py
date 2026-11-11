"""
============================================================
INTERFACE VISUELLE (Streamlit) v2 -- PROJET TABAC TUNISIE
============================================================
Developpement d'un ChatBot sur le tabagisme en Tunisie:
legislation, sevrage tabagique, epidemiologie, effets sur la sante.

Ce fichier NE DUPLIQUE PAS la logique du pipeline : il importe et
reutilise directement 3_chatbot_ULTIME.py (routage, recherche hybride,
grading, generation, verification de groundedness). Si tu modifies le
pipeline plus tard, cette interface suit automatiquement -- rien a
recopier ici.

Prerequis (dossier "assets" avec les 6 images a cote de ce fichier) :
    mon_projet_tabac/
        3_chatbot_ULTIME.py
        app_streamlit.py       <- ce fichier
        assets/
            banner.png
            icon_legislation.png
            icon_sevrage.png
            icon_dependance.png
            icon_epidemio.png
            icon_sante.png

Lancement :
    pip install streamlit
    streamlit run app_streamlit.py
"""

import time
import io
import contextlib
import importlib.util
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).parent
ASSETS = BASE_DIR / "assets"

# ============================================================
# 1. CHARGEMENT DU PIPELINE EXISTANT
# ============================================================
@st.cache_resource(show_spinner="Chargement du pipeline (connexion Neo4j + Groq)...")
def load_pipeline():
    module_path = BASE_DIR / "3_chatbot_ULTIME.py"
    spec = importlib.util.spec_from_file_location("chatbot_core", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


compiled_app = load_pipeline()

# ============================================================
# 2. CONFIGURATION DE LA PAGE + STYLE
# ============================================================
st.set_page_config(
    page_title="Chatbot Tabagisme Tunisie",
    page_icon="🚭",
    layout="centered",
)

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
    .theme-caption {
        text-align: center;
        font-weight: 600;
        margin-top: 4px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

banner_path = ASSETS / "banner.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

st.title("🚭 Chatbot sur le tabagisme en Tunisie")
st.caption(
    "Développement d'un ChatBot sur le tabagisme en Tunisie : législation, "
    "sevrage tabagique, épidémiologie, effets sur la santé."
)

st.info(
    "⏳ Le pipeline applique plusieurs vérifications avant de répondre "
    "(routage de sécurité, recherche hybride vecteur+graphe, notation des "
    "preuves, vérification de fiabilité). Réponse généralement en quelques "
    "secondes.",
    icon="⏳",
)

# ============================================================
# 3. THEMES, ICONES ET QUESTIONS D'EXEMPLE
# ============================================================
THEMES = {
    "legislation": {
        "label": "Législation",
        "icon": "icon_legislation.png",
        "questions": [
            "Que dit la loi tunisienne sur la publicité pour le tabac ?",
            "Quelle loi tunisienne interdit de fumer dans les lieux publics ?",
            "Que dit la circulaire de 2023 sur le tabac dans les universités tunisiennes ?",
            "Quelles sont les six mesures MPOWER de l'OMS ?",
        ],
    },
    "sevrage": {
        "label": "Sevrage (TSN)",
        "icon": "icon_sevrage.png",
        "questions": [
            "Quels sont les différents dosages disponibles pour le patch à la nicotine ?",
            "Quelles sont les différentes formes de traitement de substitution nicotinique ?",
            "Qui peut utiliser la TSN ?",
            "Quel patch de nicotine je dois prendre, je suis enceinte de 20 semaines ?",
        ],
    },
    "dependance": {
        "label": "Dépendance tabagique",
        "icon": "icon_dependance.png",
        "questions": [
            "Quels sont les symptômes du sevrage nicotinique ?",
            "Quelles sont les trois composantes de la dépendance au tabac ?",
            "Pourquoi les gens deviennent-ils dépendants à la nicotine ?",
        ],
    },
    "epidemio": {
        "label": "Épidémiologie",
        "icon": "icon_epidemio.png",
        "questions": [
            "Quelle est la prévalence du tabagisme chez les adultes en Tunisie ?",
            "Combien de personnes meurent du tabac chaque année en Tunisie ?",
            "Quelle est la prévalence du tabagisme chez les adolescents tunisiens de 13 à 15 ans ?",
            "Quel est le coût économique du tabagisme pour la Tunisie ?",
        ],
    },
    "sante": {
        "label": "Effets sur la santé",
        "icon": "icon_sante.png",
        "questions": [
            "Quelles sont les maladies causées par le tabagisme ?",
            "Combien de substances chimiques contient la fumée de tabac ?",
            "Quels sont les risques du tabagisme passif ?",
        ],
    },
}

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
if "active_theme" not in st.session_state:
    st.session_state.active_theme = None

st.subheader("1. Choisis un thème")

cols = st.columns(len(THEMES))
for col, (key, theme) in zip(cols, THEMES.items()):
    with col:
        icon_path = ASSETS / theme["icon"]
        if icon_path.exists():
            st.image(str(icon_path), use_container_width=True)
        st.markdown(f'<div class="theme-caption">{theme["label"]}</div>', unsafe_allow_html=True)
        if st.button("Voir les exemples", key=f"theme_{key}", use_container_width=True):
            st.session_state.active_theme = key

if st.session_state.active_theme:
    theme = THEMES[st.session_state.active_theme]
    st.markdown(f"**Exemples de questions — {theme['label']}**")
    with st.container(border=True):
        for q in theme["questions"]:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.selected_question = q

st.divider()

# ============================================================
# 4. ZONE DE QUESTION + LANCEMENT
# ============================================================
st.subheader("2. Pose ta question")
question = st.text_area(
    "Question",
    value=st.session_state.selected_question,
    height=80,
    placeholder="Pose ta question sur le tabagisme en Tunisie...",
    label_visibility="collapsed",
)

lancer = st.button("🔍 Poser la question", type="primary", use_container_width=True)

if lancer and question.strip():
    log_buffer = io.StringIO()
    start = time.time()

    with st.spinner("Traitement en cours (routage → recherche → vérification → génération)..."):
        with contextlib.redirect_stdout(log_buffer):
            initial_state = {
                "question": question.strip(),
                "retry_count": 0,
                "rewrite_history": [],
                "groundedness_fix_count": 0,
            }
            final_state = compiled_app.invoke(initial_state)

    duree = time.time() - start

    st.success(f"Réponse obtenue en {duree:.0f} secondes")
    st.markdown("### 💬 Réponse")
    st.write(final_state.get("final_answer", "(pas de reponse)"))

    with st.expander("🔧 Voir le détail du pipeline (routage, recherche, notation, vérification)"):
        st.code(log_buffer.getvalue() or "(aucun log -- réponse cadrée sans retrieval)", language=None)

elif lancer:
    st.warning("Écris une question avant de lancer, ou choisis un exemple ci-dessus.")

st.divider()
st.caption(
    "⚠️ Cet assistant est un outil d'information et ne remplace pas un avis médical "
    "personnalisé."
)
