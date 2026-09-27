"""
tests/test_forward_chain.py
============================
pytest tests for forward-chaining inference.
"""

import pytest
from working_memory import WorkingMemory
from inference_engine import forward_chain, diagnose_forward
from explanation import Tracer
from knowledge_base import RULES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flu_symptoms() -> set[str]:
    """Classic influenza symptom set matching R01."""
    return {"fever", "cough", "sore_throat", "body_ache"}


@pytest.fixture
def covid_symptoms() -> set[str]:
    """COVID-19 symptom set matching R05."""
    return {"fever", "cough", "loss_of_taste", "loss_of_smell"}


@pytest.fixture
def cold_symptoms() -> set[str]:
    """Common cold symptom set matching R03."""
    return {"runny_nose", "sneezing", "sore_throat"}


@pytest.fixture
def dengue_symptoms() -> set[str]:
    """Dengue symptom set matching R11."""
    return {"high_fever", "severe_headache", "joint_pain", "rash"}


@pytest.fixture
def uti_symptoms() -> set[str]:
    """UTI symptom set matching R20."""
    return {"burning_urination", "frequent_urination", "lower_abdominal_pain"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestForwardChainDiagnoses:

    def test_flu_derived_from_symptoms(self, flu_symptoms):
        """R01: fever + cough + sore_throat + body_ache → influenza."""
        conclusions, tracer = diagnose_forward(flu_symptoms)
        assert "influenza" in conclusions, (
            f"Expected 'influenza' in conclusions; got {list(conclusions)}"
        )

    def test_covid_derived_from_symptoms(self, covid_symptoms):
        """R05: fever + cough + loss_of_taste + loss_of_smell → covid_19."""
        conclusions, _ = diagnose_forward(covid_symptoms)
        assert "covid_19" in conclusions

    def test_common_cold_derived(self, cold_symptoms):
        """R03: runny_nose + sneezing + sore_throat → common_cold."""
        conclusions, _ = diagnose_forward(cold_symptoms)
        assert "common_cold" in conclusions

    def test_dengue_derived(self, dengue_symptoms):
        """R11: high_fever + severe_headache + joint_pain + rash → dengue."""
        conclusions, _ = diagnose_forward(dengue_symptoms)
        assert "dengue" in conclusions

    def test_uti_derived(self, uti_symptoms):
        """R20: burning_urination + frequent_urination + lower_abdominal_pain → UTI."""
        conclusions, _ = diagnose_forward(uti_symptoms)
        assert "urinary_tract_infection" in conclusions

    def test_no_diagnosis_for_empty_symptoms(self):
        """No symptoms → no diagnosis."""
        conclusions, _ = diagnose_forward(set())
        assert len(conclusions) == 0

    def test_irrelevant_symptoms_produce_no_match(self):
        """Randomly constructed symptom set that matches no rule."""
        conclusions, _ = diagnose_forward({"fever"})   # alone, no full rule matches
        assert len(conclusions) == 0, (
            f"Single symptom should not match any rule; got {conclusions}"
        )

    def test_certainty_factor_range(self, flu_symptoms):
        """Combined CF must be in [0, 1]."""
        conclusions, _ = diagnose_forward(flu_symptoms)
        for disease, cf in conclusions.items():
            assert 0.0 <= cf <= 1.0, f"CF for {disease} out of range: {cf}"

    def test_high_cf_for_strong_match(self, flu_symptoms):
        """R01 has CF=0.90; the combined CF for influenza should be ≥ 0.90."""
        conclusions, _ = diagnose_forward(flu_symptoms)
        assert conclusions.get("influenza", 0) >= 0.90

    def test_multiple_diseases_same_session(self):
        """Symptom overlap may trigger multiple diseases in one session."""
        # fever + cough + sore_throat + body_ache  → influenza  (R01)
        # runny_nose + sneezing + sore_throat       → common_cold (R03)
        combo = {"fever", "cough", "sore_throat", "body_ache", "runny_nose", "sneezing"}
        conclusions, _ = diagnose_forward(combo)
        assert "influenza" in conclusions
        assert "common_cold" in conclusions

    def test_forward_chain_via_wm_object(self, flu_symptoms):
        """Low-level API: WorkingMemory + forward_chain directly."""
        wm = WorkingMemory()
        for s in flu_symptoms:
            wm.assert_fact(s, True)
        tracer = Tracer()
        conclusions = forward_chain(wm, tracer)
        assert "influenza" in conclusions

    def test_covid_cf_is_high(self, covid_symptoms):
        """R05 CF = 0.95; should produce a very high combined CF."""
        conclusions, _ = diagnose_forward(covid_symptoms)
        assert conclusions.get("covid_19", 0) >= 0.95

    def test_rules_are_not_double_fired(self, flu_symptoms):
        """
        The same rule must not fire twice in one session
        (guard against infinite loops / double-counting CF).
        """
        _, tracer = diagnose_forward(flu_symptoms)
        rule_ids_fired = [e.rule_id for e in tracer.entries()]
        assert len(rule_ids_fired) == len(set(rule_ids_fired)), (
            "Duplicate rule firings detected!"
        )


class TestForwardChainRuleCustomisation:

    def test_forward_chain_respects_custom_rules(self):
        """forward_chain should work with an ad-hoc rule list."""
        custom_rules = [
            {
                "id": "TEST01",
                "conditions": ["purple_spots", "three_eyes"],
                "conclusion": "alien_disease",
                "cf": 0.99,
                "description": "Test rule",
            }
        ]
        wm = WorkingMemory({"purple_spots": True, "three_eyes": True})
        tracer = Tracer()
        conclusions = forward_chain(wm, tracer, rules=custom_rules)
        assert "alien_disease" in conclusions

    def test_forward_chain_missing_condition_does_not_fire(self):
        """A rule with an unmet condition must NOT fire."""
        custom_rules = [
            {
                "id": "TEST02",
                "conditions": ["symptom_a", "symptom_b"],
                "conclusion": "test_disease",
                "cf": 0.9,
                "description": "Test",
            }
        ]
        wm = WorkingMemory({"symptom_a": True})   # symptom_b missing
        tracer = Tracer()
        conclusions = forward_chain(wm, tracer, rules=custom_rules)
        assert "test_disease" not in conclusions
