"""
============================================================
INTERFACE VISUELLE (Streamlit) v2 -- PROJET TABAC TUNISIE
============================================================
Developpement d'un ChatBot sur le tabagisme en Tunisie:
legislation, sevrage tabagique, epidemiologie, effets sur la sante.

Ce fichier NE DUPLIQUE PAS la logique du pipeline : il importe et
reutilise directement chatbot_pipeline.py (routage, recherche hybride,
grading, generation, verification de groundedness). Si tu modifies le
pipeline plus tard, cette interface suit automatiquement -- rien a
recopier ici.

Prerequis (dossier "assets" avec les 6 images a cote de ce fichier) :
    mon_projet_tabac/
        chatbot_pipeline.py
        app.py                 <- ce fichier
        assets/
            banner.png
            icon_legislation.png
            icon_sevrage.png
            icon_dependance.png
            icon_epidemio.png
            icon_sante.png

Lancement :
    pip install streamlit
    streamlit run app.py
"""

import time
import io
import re
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
    module_path = BASE_DIR / "chatbot_pipeline.py"
    spec = importlib.util.spec_from_file_location("chatbot_core", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # On garde le module (pas seulement le graphe compile) pour pouvoir
    # reutiliser get_source_meta() plus bas -- une seule source de verite
    # pour le mapping fichier -> nom affiche/organisme/URL.
    return module.build_graph(), module


compiled_app, chatbot = load_pipeline()

# Citation textuelle ajoutee par chatbot_pipeline.finalize() ("\n\n📚 Sources :
# ...") -- retiree de l'affichage ici car remplacee par les cartes cliquables
# ci-dessous (sources_meta.json). Ne touche pas a l'avertissement medical qui
# suit, ni a l'etat interne (final_state["sources"] reste intact).
_SOURCES_LINE_RE = re.compile(r"\n\n📚 Sources :.*?(?=\n\n⚠️|$)", re.DOTALL)


def render_source_cards(filenames: list[str]) -> None:
    """Affiche les sources sous forme de petites cartes cliquables (initiale
    de l'organisme + nom + lien externe si disponible)."""
    if not filenames:
        return
    st.markdown("**📚 Sources**")
    cards_html = ['<div style="display:flex;flex-wrap:wrap;gap:0.6rem;">']
    for filename in sorted(set(filenames)):
        meta = chatbot.get_source_meta(filename)
        display_name = meta.get("display_name") or filename
        organization = meta.get("organization") or ""
        url = meta.get("url")
        initial = (organization or display_name).strip()[:1].upper() or "?"
        inner = f"""
          <div style="display:flex;align-items:center;gap:0.5rem;
                      border:1px solid #D7E0EC;border-radius:10px;
                      padding:0.5rem 0.75rem;background:#F7FAFD;max-width:260px;">
            <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;
                        background:#1F3864;color:white;display:flex;
                        align-items:center;justify-content:center;
                        font-weight:700;font-size:0.85rem;">{initial}</div>
            <div style="line-height:1.2;">
              <div style="font-size:0.85rem;font-weight:600;color:#1A1A1A;">{display_name}</div>
              {f'<div style="font-size:0.72rem;color:#5A6B82;">{organization}</div>' if organization else ''}
            </div>
          </div>
        """
        if url:
            cards_html.append(
                f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
                f'style="text-decoration:none;">{inner}</a>'
            )
        else:
            cards_html.append(inner)
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

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
    st.image(str(banner_path), width="stretch")

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

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "active_theme" not in st.session_state:
    st.session_state.active_theme = None

st.subheader("1. Choisis un thème")

cols = st.columns(len(THEMES))
for col, (key, theme) in zip(cols, THEMES.items()):
    with col:
        icon_path = ASSETS / theme["icon"]
        if icon_path.exists():
            st.image(str(icon_path), width="stretch")
        st.markdown(f'<div class="theme-caption">{theme["label"]}</div>', unsafe_allow_html=True)
        if st.button("Voir les exemples", key=f"theme_{key}", width="stretch"):
            st.session_state.active_theme = key

if st.session_state.active_theme:
    theme = THEMES[st.session_state.active_theme]
    st.markdown(f"**Exemples de questions — {theme['label']}**")
    with st.container(border=True):
        for q in theme["questions"]:
            if st.button(q, key=f"ex_{q}", width="stretch"):
                st.session_state.question_input = q

st.divider()

# ============================================================
# 4. ZONE DE QUESTION + LANCEMENT
# ============================================================
st.subheader("2. Pose ta question")
# st.form() + key= (plutot que value=st.session_state... sans key) : pattern
# canonique Streamlit pour lier un widget directement a une cle de
# session_state (les boutons d'exemple ecrivent dans cette MEME cle) et
# grouper texte+clic en un seul envoi atomique. Garde par principe de
# prudence, meme si l'investigation ci-dessous a fini par montrer que la
# vraie cause du bug rapporte etait ailleurs (voir plus bas, zone de
# reponse) : la premiere hypothese ("le pipeline recoit l'ancienne
# question") a ete testee rigoureusement, y compris en rejouant le code
# D'AVANT ce correctif avec une methode de verification correcte (attendre
# que la reponse affichee CHANGE reellement, pas juste qu'un texte soit
# present) -- et le pipeline a TOUJOURS recu la bonne question, meme dans le
# code d'origine. Le premier diagnostic ("course sur le text_area") etait
# donc faux ; la vraie cause est documentee au-dessus de `response_area`
# plus bas.
with st.form("question_form"):
    question = st.text_area(
        "Question",
        key="question_input",
        height=80,
        placeholder="Pose ta question sur le tabagisme en Tunisie...",
        label_visibility="collapsed",
    )
    lancer = st.form_submit_button("🔍 Poser la question", type="primary", width="stretch")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# st.empty() : zone unique reecrite pour la reponse -- sans ca, pendant les
# plusieurs secondes de calcul (routage+recherche+generation+verification),
# la DERNIERE reponse affichee reste visible a l'ecran telle quelle (Streamlit
# ne l'efface pas tant que le script n'a pas reatteint ce point). Un utilisateur
# qui regarde l'ecran pendant que le spinner tourne peut alors croire, a tort,
# qu'il voit deja la reponse a SA nouvelle question alors qu'il s'agit encore
# de l'ancienne. On remplace donc explicitement cette zone par un message de
# chargement DES LE DEBUT du calcul, avant meme d'appeler le pipeline.
response_area = st.empty()

if lancer and question.strip():
    response_area.info("⏳ Génération de la réponse en cours pour cette question -- la réponse précédente, affichée ci-dessus, ne correspond plus à ta question tant que ce message est là.")

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
    st.session_state.last_result = {
        "duree": duree,
        "answer_text": final_state.get("final_answer", "(pas de reponse)"),
        "sources": final_state.get("sources", []),
        "log": log_buffer.getvalue(),
    }
elif lancer:
    st.session_state.last_result = None
    response_area.warning("Écris une question avant de lancer, ou choisis un exemple ci-dessus.")

if st.session_state.last_result:
    r = st.session_state.last_result
    with response_area.container():
        st.success(f"Réponse obtenue en {r['duree']:.0f} secondes")
        st.markdown("### 💬 Réponse")
        st.write(_SOURCES_LINE_RE.sub("", r["answer_text"]))

        render_source_cards(r["sources"])

        with st.expander("🔧 Voir le détail du pipeline (routage, recherche, notation, vérification)"):
            st.code(r["log"] or "(aucun log -- réponse cadrée sans retrieval)", language=None)

st.divider()
st.caption(
    "⚠️ Cet assistant est un outil d'information et ne remplace pas un avis médical "
    "personnalisé."
)
