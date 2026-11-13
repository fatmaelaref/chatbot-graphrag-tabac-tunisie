"""
Tests du mode strict (STRICT_LLM_FAILURE_MODE) : si Groq sature (429) pendant
le routage, le grading ou la generation, un vrai utilisateur ne doit JAMAIS
recevoir une reponse degradee silencieusement -- seulement un message de
statut explicite, sans source. Desactive par defaut (evaluate.py/baseline.py/
ablation_graphe.py ne doivent pas changer de comportement) ; app.py l'active
explicitement pour l'interface utilisateur reelle.

Aucun appel reseau ici : call_llm_json et groq_client.chat.completions.create
sont remplaces par des doublures (monkeypatch, auto-restaure apres chaque test
-- y compris STRICT_LLM_FAILURE_MODE lui-meme, pour ne pas polluer les autres
tests qui partagent le meme module `chatbot` charge une seule fois par
session, voir conftest.py).
"""


def test_sanitize_llm_error_detects_rate_limit(chatbot):
    err = Exception("Error code: 429 - {'error': {'message': 'rate_limit_exceeded'}}")
    assert chatbot._sanitize_llm_error(err) == "Quota Groq atteint (429)"


def test_sanitize_llm_error_never_leaks_raw_text(chatbot):
    # Un message d'erreur Groq reel peut contenir un identifiant d'organisation
    # ou de projet -- le message affiche ne doit jamais reprendre le texte brut.
    err = Exception("org_01AbCdEfGhIjKlMnOpQrSt exceeded quota, project proj_xyz123")
    sanitized = chatbot._sanitize_llm_error(err)
    assert "org_01AbCdEfGhIjKlMnOpQrSt" not in sanitized
    assert "proj_xyz123" not in sanitized


def test_route_query_sets_llm_unavailable_in_strict_mode(chatbot, monkeypatch):
    monkeypatch.setattr(chatbot, "STRICT_LLM_FAILURE_MODE", True)
    monkeypatch.setattr(chatbot, "call_llm_json", lambda *a, **kw: None)

    result = chatbot.route_query({"question": "Quels sont les dosages du patch ?"})

    assert result["route"] == "normal"
    assert result["llm_unavailable"] is True


def test_route_query_does_not_set_flag_outside_strict_mode(chatbot, monkeypatch):
    # Comportement par defaut (evaluate.py/baseline.py) : inchange.
    monkeypatch.setattr(chatbot, "STRICT_LLM_FAILURE_MODE", False)
    monkeypatch.setattr(chatbot, "call_llm_json", lambda *a, **kw: None)

    result = chatbot.route_query({"question": "Quels sont les dosages du patch ?"})

    assert result["route"] == "normal"
    assert "llm_unavailable" not in result


def test_grade_evidence_sets_llm_unavailable_in_strict_mode(chatbot, monkeypatch):
    monkeypatch.setattr(chatbot, "STRICT_LLM_FAILURE_MODE", True)
    monkeypatch.setattr(chatbot, "call_llm_json", lambda *a, **kw: None)

    state = {
        "search_query": "test",
        "vector_candidates": [{"text": "x", "source": "a.txt", "origin": "vecteur"}],
        "graph_candidates": [],
    }
    result = chatbot.grade_evidence(state)

    assert result["llm_unavailable"] is True
    assert len(result["graded_evidence"]) <= 3


def test_generate_answer_sanitizes_exception_in_strict_mode(chatbot, monkeypatch):
    monkeypatch.setattr(chatbot, "STRICT_LLM_FAILURE_MODE", True)

    def _fake_create(*a, **kw):
        raise Exception("Error code: 429 - org_SECRET123 rate_limit_exceeded")

    monkeypatch.setattr(chatbot.groq_client.chat.completions, "create", _fake_create)

    state = {
        "search_query": "Quels sont les dosages du patch ?",
        "graded_evidence": [{"origin": "vecteur", "source": "a.txt", "text": "x"}],
    }
    result = chatbot.generate_answer(state)

    assert result["llm_unavailable"] is True
    assert "org_SECRET123" not in result["draft_answer"]
    assert result["draft_answer"] == "Quota Groq atteint (429)"


def test_finalize_returns_saturation_message_when_llm_unavailable(chatbot, monkeypatch):
    monkeypatch.setattr(chatbot, "STRICT_LLM_FAILURE_MODE", True)

    state = {
        "final_answer": "une reponse potentiellement degradee",
        "sources": ["SOURCE_URL_CDC_Cigarette_Smoking.txt"],
        "llm_unavailable": True,
    }
    result = chatbot.finalize(state)

    assert result["final_answer"] == chatbot.SATURATION_MESSAGE
    assert result["sources"] == []


def test_finalize_unaffected_outside_strict_mode(chatbot):
    # STRICT_LLM_FAILURE_MODE est False par defaut (pas de monkeypatch ici) --
    # comportement d'evaluate.py/baseline.py inchange, meme avec le flag
    # llm_unavailable present dans l'etat (ne devrait jamais y etre en dehors
    # du mode strict, mais on verifie quand meme qu'il est sans effet si
    # jamais present).
    state = {
        "final_answer": "une reponse",
        "sources": ["SOURCE_URL_CDC_Cigarette_Smoking.txt"],
        "llm_unavailable": True,
    }
    result = chatbot.finalize(state)

    assert result["final_answer"] != chatbot.SATURATION_MESSAGE
    assert "une reponse" in result["final_answer"]
