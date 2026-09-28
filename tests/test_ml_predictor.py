"""
tests/test_ml_predictor.py
==========================
Tests for Machine Learning prediction, partial rule matching, and NLP symptom extraction.
"""

import pytest
from ml_predictor import (
    get_ml_predictor,
    evaluate_partial_rules,
    parse_freeform_symptoms,
    MLMedicalModel,
)


class TestPartialRuleEvaluation:
    def test_partial_match_fever_and_cough(self):
        """fever and cough should partially match influenza and covid-19."""
        results = evaluate_partial_rules(["fever", "cough"])
        assert len(results) > 0
        diseases = [r["disease"] for r in results]
        assert "influenza" in diseases or "covid_19" in diseases
        top = results[0]
        assert 0.0 < top["match_score"] <= 1.0
        assert "fever" in top["matched_conditions"] or "cough" in top["matched_conditions"]

    def test_partial_match_empty_symptoms(self):
        """Empty symptoms return empty list."""
        assert evaluate_partial_rules([]) == []


class TestMLPredictor:
    def test_ml_predictor_singleton(self):
        p1 = get_ml_predictor()
        p2 = get_ml_predictor()
        assert p1 is p2

    def test_ml_predict_random_symptoms(self):
        """Predicts reasonable candidates even from single or random symptoms."""
        predictor = get_ml_predictor()
        preds = predictor.predict(["fever", "cough"], top_k=3)
        assert len(preds) > 0
        top = preds[0]
        assert "disease" in top
        assert "probability" in top
        assert 0.0 <= top["probability"] <= 1.0

    def test_ml_predict_uncommon_combination(self):
        """Arbitrary random combination yields structured predictions without error."""
        predictor = get_ml_predictor()
        preds = predictor.predict(["dizziness", "nausea", "headache"], top_k=5)
        assert len(preds) > 0
        assert any(p["disease"] in ["hypertension", "migraine", "malaria"] for p in preds)


class TestNLPSymptomParser:
    def test_exact_symptom_in_text(self):
        text = "Patient complains of chest pain and shortness of breath."
        detected = parse_freeform_symptoms(text)
        assert "chest_pain" in detected
        assert "shortness_of_breath" in detected

    def test_synonym_alias_matching(self):
        text = "I have a high temp and bad shivering since yesterday morning"
        detected = parse_freeform_symptoms(text)
        assert "fever" in detected
        assert "chills" in detected

    def test_empty_string(self):
        assert parse_freeform_symptoms("") == []
