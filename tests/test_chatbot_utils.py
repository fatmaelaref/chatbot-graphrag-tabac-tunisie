"""
Tests unitaires de la logique pure de 3_chatbot_ULTIME.py (parsing,
regex, routage par mots-cles, filets de securite) -- par opposition a
smoke_test.py, qui verifie que les services externes reels repondent.
Aucun appel reseau ici (voir tests/conftest.py pour comment l'import du
module evite Neo4j/Groq).
"""
import pytest


# ---------------------------------------------------------------------------
# clean_chunk
# ---------------------------------------------------------------------------

def test_clean_chunk_removes_slide_markers(chatbot):
    text = "=== DIAPOSITIVE 4 ===\nDemi-vie 120 minutes\n=== FIN TABLEAU ==="
    assert "DIAPOSITIVE" not in chatbot.clean_chunk(text)
    assert "Demi-vie 120 minutes" in chatbot.clean_chunk(text)


def test_clean_chunk_preserves_line_breaks(chatbot):
    # Regression : aplatir les retours a la ligne detruit les tableaux
    # markdown (dosages patch/gomme melanges) -- voir docstring de
    # clean_chunk dans 3_chatbot_ULTIME.py.
    text = "Patch : 21mg\nGomme : 4mg"
    result = chatbot.clean_chunk(text)
    assert "\n" in result
    assert result.count("\n") == 1


def test_clean_chunk_collapses_blank_lines_and_spaces(chatbot):
    text = "Ligne 1\n\n\n\nLigne 2   avec   espaces"
    result = chatbot.clean_chunk(text)
    assert "\n\n\n" not in result
    assert "   " not in result


def test_clean_chunk_empty_input(chatbot):
    assert chatbot.clean_chunk("") == ""
    assert chatbot.clean_chunk(None) == ""


# ---------------------------------------------------------------------------
# short_source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("./textes/CHATBOT 1 LOI.txt", "CHATBOT 1 LOI.txt"),
    (".\\textes\\CHATBOT 1 LOI.txt", "CHATBOT 1 LOI.txt"),
    ("CHATBOT 1 LOI.txt", "CHATBOT 1 LOI.txt"),
    (None, "source inconnue"),
    ("", "source inconnue"),
])
def test_short_source(chatbot, path, expected):
    assert chatbot.short_source(path) == expected


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------

def test_extract_final_answer_with_marker_on_own_line(chatbot):
    raw = "1. Raisonnement...\n### REPONSE FINALE ###\nLe tabagisme passif est nocif."
    assert chatbot.extract_final_answer(raw) == "Le tabagisme passif est nocif."


def test_extract_final_answer_marker_same_line_as_answer(chatbot):
    # Constate en pratique (docstring) : le marqueur et la reponse sur la
    # meme ligne -- une simple recherche de "\n" apres le marqueur rate ca.
    raw = "5. ### REPONSE FINALE ### Le vapotage reste risque."
    assert chatbot.extract_final_answer(raw) == "Le vapotage reste risque."


def test_extract_final_answer_marker_present_but_empty(chatbot):
    # Le marqueur est present mais rien ne suit (budget de tokens epuise) :
    # on retombe sur le raisonnement qui PRECEDE le marqueur.
    raw = "Le fait cle est que X est vrai.\n### REPONSE FINALE ###"
    assert chatbot.extract_final_answer(raw) == "Le fait cle est que X est vrai."


def test_extract_final_answer_no_marker_falls_back_to_last_numbered_step(chatbot):
    raw = "1. Analyse de la question.\n2. Revue des preuves.\n5. La reponse est Y."
    assert chatbot.extract_final_answer(raw) == "La reponse est Y."


def test_extract_final_answer_no_marker_no_numbering(chatbot):
    raw = "Juste une reponse brute sans structure."
    assert chatbot.extract_final_answer(raw) == raw


def test_extract_final_answer_empty_input(chatbot):
    assert chatbot.extract_final_answer("") == ""
    assert chatbot.extract_final_answer(None) is None


# ---------------------------------------------------------------------------
# _to_strict_json_schema
# ---------------------------------------------------------------------------

def test_to_strict_json_schema_adds_required_and_additional_properties(chatbot):
    schema = {
        "type": "object",
        "properties": {"route": {"type": "string"}, "reason": {"type": "string"}},
    }
    strict = chatbot._to_strict_json_schema(schema)
    assert strict["additionalProperties"] is False
    assert sorted(strict["required"]) == ["reason", "route"]


def test_to_strict_json_schema_recurses_into_nested_objects_and_arrays(chatbot):
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
    }
    strict = chatbot._to_strict_json_schema(schema)
    nested = strict["properties"]["items"]["items"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["name"]


def test_to_strict_json_schema_does_not_mutate_input(chatbot):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    chatbot._to_strict_json_schema(schema)
    assert "additionalProperties" not in schema


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_strips_accents_and_lowercases(chatbot):
    assert chatbot._normalize("Règlement") == chatbot._normalize("reglement")
    assert chatbot._normalize("SYMPTÔME") == chatbot._normalize("symptome")


# ---------------------------------------------------------------------------
# detect_intent
# ---------------------------------------------------------------------------

def test_detect_intent_single_match(chatbot):
    result = chatbot.detect_intent("Quelle est la demi-vie de la nicotine ?")
    assert result is not None
    assert [m["label"] for m in result] == ["Substance"]


def test_detect_intent_no_match_returns_none(chatbot):
    assert chatbot.detect_intent("Quelle est la capitale de la Tunisie ?") is None


def test_detect_intent_accent_insensitive(chatbot):
    # Bug reel documente : "symptome" sans accent ne matchait pas
    # "symptômes" avec un o circonflexe avant la normalisation NFD.
    result = chatbot.detect_intent("Quels sont les symptômes du manque ?")
    assert result is not None
    assert {m["label"] for m in result} == {"Symptome"}


def test_detect_intent_returns_all_matching_categories(chatbot):
    # Regression du 2026-10-07 (voir JOURNAL_TRAVAIL.md, Etape A) : une
    # question a cheval sur Pathologie ET Pays doit remonter les DEUX,
    # pas seulement la premiere categorie qui matche.
    question = "Combien coutent chaque annee les maladies liees au tabagisme aux Etats-Unis ?"
    result = chatbot.detect_intent(question)
    labels = {m["label"] for m in result}
    assert "Pathologie" in labels
    assert "Pays" in labels


# ---------------------------------------------------------------------------
# detect_name_filter
# ---------------------------------------------------------------------------

def test_detect_name_filter_finds_known_country(chatbot):
    assert chatbot.detect_name_filter("Pays", "Combien coute le tabac aux Etats-Unis ?") == "etats-unis"


def test_detect_name_filter_accent_insensitive(chatbot):
    assert chatbot.detect_name_filter("Pays", "Cout du tabagisme en Tunisie") == "tunisie"


def test_detect_name_filter_no_match(chatbot):
    assert chatbot.detect_name_filter("Pays", "Cout du tabagisme en Belgique") is None


def test_detect_name_filter_unknown_label(chatbot):
    assert chatbot.detect_name_filter("Loi", "Quelle est la loi en vigueur ?") is None


# ---------------------------------------------------------------------------
# _apply_safety_backstop
# ---------------------------------------------------------------------------

def test_safety_backstop_escalates_urgent_symptoms(chatbot):
    state = {"question": "J'ai une douleur thoracique, que faire ?", "route": "normal"}
    result = chatbot._apply_safety_backstop(state)
    assert result["route"] == "urgence"


def test_safety_backstop_escalates_personal_dosage_request(chatbot):
    state = {"question": "Quelle dose de patch dois-je prendre ?", "route": "normal"}
    result = chatbot._apply_safety_backstop(state)
    assert result["route"] == "demande_personnelle"


def test_safety_backstop_never_relaxes_a_stricter_llm_classification(chatbot):
    # Le filet ne doit JAMAIS repasser "urgence" en "normal", meme si le
    # texte ne contient aucun marqueur de regle -- il ne fait que durcir.
    state = {"question": "Bonjour", "route": "urgence"}
    result = chatbot._apply_safety_backstop(state)
    assert result["route"] == "urgence"


def test_safety_backstop_corrects_false_hors_domaine(chatbot):
    state = {"question": "Quelle est la loi tunisienne sur le tabac ?", "route": "hors_domaine"}
    result = chatbot._apply_safety_backstop(state)
    assert result["route"] == "normal"


def test_safety_backstop_leaves_normal_question_untouched(chatbot):
    state = {"question": "Qu'est-ce que le tabagisme passif ?", "route": "normal"}
    result = chatbot._apply_safety_backstop(state)
    assert result["route"] == "normal"
    assert "route_reason" not in result or "backstop" not in result.get("route_reason", "")


# ---------------------------------------------------------------------------
# route_condition / quality_gate / handle_special_case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route,expected", [
    ("normal", "retrieval"),
    ("hors_domaine", "special_case"),
    ("urgence", "special_case"),
    ("demande_personnelle", "special_case"),
])
def test_route_condition(chatbot, route, expected):
    assert chatbot.route_condition({"route": route}) == expected


def test_quality_gate_generates_when_enough_evidence(chatbot):
    state = {"graded_evidence": [{"text": "x"}], "retry_count": 0}
    assert chatbot.quality_gate(state) == "generate"


def test_quality_gate_rewrites_when_retries_remain(chatbot):
    state = {"graded_evidence": [], "retry_count": 0}
    assert chatbot.quality_gate(state) == "rewrite"


def test_quality_gate_gives_up_when_retries_exhausted(chatbot):
    state = {"graded_evidence": [], "retry_count": chatbot.MAX_REWRITES}
    assert chatbot.quality_gate(state) == "insufficient"


@pytest.mark.parametrize("route", ["urgence", "demande_personnelle", "hors_domaine"])
def test_handle_special_case_never_returns_sources(chatbot, route):
    result = chatbot.handle_special_case({"route": route})
    assert result["sources"] == []
    assert result["final_answer"].strip() != ""


def test_handle_special_case_answers_differ_by_route(chatbot):
    answers = {
        route: chatbot.handle_special_case({"route": route})["final_answer"]
        for route in ["urgence", "demande_personnelle", "hors_domaine"]
    }
    assert len(set(answers.values())) == 3


# ---------------------------------------------------------------------------
# _is_self_declared_no_info (bug corrige le 2026-10-11, Q77)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [
    "Les preuves fournies ne permettent pas de repondre a cette question.",
    "Les informations fournies ne mentionnent pas ce point precis.",
    "Je n'ai pas trouve d'information sur ce sujet dans le corpus.",
    "Ces documents ne couvrent pas ce point.",
])
def test_is_self_declared_no_info_detects_abstention_phrasings(chatbot, draft):
    assert chatbot._is_self_declared_no_info(draft) is True


def test_is_self_declared_no_info_false_on_normal_answer(chatbot):
    draft = "Le tabagisme passif augmente le risque de cancer du poumon (OMS)."
    assert chatbot._is_self_declared_no_info(draft) is False
