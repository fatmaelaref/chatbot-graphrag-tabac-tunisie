"""
Tests unitaires de evaluation/common.py -- fonctions pures de
normalisation/scoring partagees par evaluate.py et baseline.py.
Aucune dependance externe (le fichier source le garantit deja par
conception, voir sa docstring) -- import direct, pas de fixture requise.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import common  # noqa: E402


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize_strips_accents_and_case():
    assert common.normalize("DEMI-VIE") == common.normalize("demi-vie")
    assert common.normalize("États-Unis") == common.normalize("etats-unis")


def test_normalize_typographic_dash_matches_ascii_dash():
    # Regression Q56 (JOURNAL_TRAVAIL.md, 2026-10-07) : un strip d'accents
    # seul ne neutralise pas le tiret insecable typographique utilise par
    # les modeles gpt-oss ("demi‑vie" avec U+2011).
    assert common.normalize("demi‑vie") == common.normalize("demi-vie")


def test_normalize_curly_quotes_match_straight_quotes():
    assert common.normalize("l’arret du tabac") == common.normalize("l'arret du tabac")


def test_normalize_empty_input():
    assert common.normalize("") == ""
    assert common.normalize(None) == ""


# ---------------------------------------------------------------------------
# element_recall
# ---------------------------------------------------------------------------

def test_element_recall_all_elements_found():
    answer = "La demi-vie de la nicotine est d'environ 120 minutes."
    assert common.element_recall(answer, ["demi-vie", "120 minutes"]) == 1.0


def test_element_recall_partial_match():
    answer = "La demi-vie de la nicotine est d'environ 120 minutes."
    assert common.element_recall(answer, ["demi-vie", "48 heures"]) == 0.5


def test_element_recall_normalizes_before_matching():
    # Meme bug que normalize() ci-dessus : un element attendu avec tiret
    # insecable doit quand meme matcher une reponse avec tiret ASCII.
    answer = "La demi‑vie de la nicotine."
    assert common.element_recall(answer, ["demi-vie"]) == 1.0


def test_element_recall_no_expected_elements_returns_none():
    assert common.element_recall("peu importe", []) is None
    assert common.element_recall("peu importe", None) is None


# ---------------------------------------------------------------------------
# retrieval_hit
# ---------------------------------------------------------------------------

def test_retrieval_hit_found_returns_rank():
    candidates = [{"text": "autre chose"}, {"text": "le bon chunk avec demi-vie 120 minutes"}]
    hit, rank = common.retrieval_hit(candidates, "demi-vie 120 minutes")
    assert hit is True
    assert rank == 2


def test_retrieval_hit_not_found():
    candidates = [{"text": "chunk sans le bon fait"}]
    hit, rank = common.retrieval_hit(candidates, "fait absent")
    assert hit is False
    assert rank is None


def test_retrieval_hit_no_snippet_or_no_candidates_returns_none_none():
    assert common.retrieval_hit([], "peu importe") == (None, None)
    assert common.retrieval_hit([{"text": "x"}], "") == (None, None)


# ---------------------------------------------------------------------------
# infer_behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["hors_domaine", "urgence", "demande_personnelle"])
def test_infer_behavior_special_routes_are_refus_redirection(route):
    assert common.infer_behavior(route, "peu importe la reponse", lambda t: False) == "refus_redirection"


def test_infer_behavior_abstention_detected():
    is_abstention = lambda text: "pas d'information" in text
    result = common.infer_behavior("normal", "Je n'ai pas d'information sur ce point.", is_abstention)
    assert result == "abstention"


def test_infer_behavior_repondre_when_not_abstention():
    result = common.infer_behavior("normal", "Le tabagisme passif est nocif.", lambda t: False)
    assert result == "repondre"


# ---------------------------------------------------------------------------
# avg / median / mrr
# ---------------------------------------------------------------------------

def test_avg_ignores_none_values():
    assert common.avg([1.0, None, 3.0]) == 2.0


def test_avg_empty_or_all_none_returns_none():
    assert common.avg([]) is None
    assert common.avg([None, None]) is None


def test_median_ignores_none_values():
    assert common.median([1, None, 2, 3]) == 2


def test_mrr_mix_of_found_and_not_found():
    # ranks : trouve au rang 1, pas trouve (None -> 0.0), trouve au rang 4
    result = common.mrr([1, None, 4])
    assert result == round((1.0 + 0.0 + 0.25) / 3, 3)


def test_mrr_empty_returns_none():
    assert common.mrr([]) is None


# ---------------------------------------------------------------------------
# aggregate_by
# ---------------------------------------------------------------------------

def test_aggregate_by_groups_and_averages_correctly():
    results = [
        {"category": "legislation", "route_match": True, "elements_recall": 1.0, "correctness_score": 5},
        {"category": "legislation", "route_match": False, "elements_recall": 0.0, "correctness_score": 3},
        {"category": "sante", "route_match": True, "elements_recall": 0.5, "correctness_score": 4},
    ]
    out = common.aggregate_by(results, "category")
    assert out["legislation"]["n"] == 2
    assert out["legislation"]["route_accuracy"] == 0.5
    assert out["legislation"]["elements_recall"] == 0.5
    assert out["sante"]["n"] == 1
    assert out["sante"]["route_accuracy"] == 1.0


def test_aggregate_by_handles_missing_fields_gracefully():
    results = [{"category": "securite"}]
    out = common.aggregate_by(results, "category")
    assert out["securite"]["n"] == 1
    assert out["securite"]["route_accuracy"] is None
    assert out["securite"]["elements_recall"] is None
