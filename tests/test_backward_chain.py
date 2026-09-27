"""
tests/test_backward_chain.py
==============================
pytest tests for backward-chaining inference.
"""

import pytest
from working_memory import WorkingMemory
from inference_engine import backward_chain, diagnose_backward
from explanation import Tracer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _yes_to_all(symptom: str) -> bool:
    """Mock ask_fn that always says 'yes'."""
    return True


def _no_to_all(symptom: str) -> bool:
    """Mock ask_fn that always says 'no'."""
    return False


def _make_ask_fn(answers: dict[str, bool], default: bool = False):
    """Return an ask_fn that uses a lookup dict, falling back to *default*."""
    def ask(symptom: str) -> bool:
        return answers.get(symptom, default)
    return ask


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBackwardChainConfirm:

    def test_confirms_influenza_when_all_conditions_preloaded(self):
        """R01 conditions already in WM → influenza confirmed without asking."""
        known = {
            "fever": True,
            "cough": True,
            "sore_throat": True,
            "body_ache": True,
        }
        result, tracer = diagnose_backward("influenza", known, _no_to_all)
        assert result is True, "Influenza should be confirmed when all conditions are True"

    def test_confirms_covid19_preloaded(self):
        """R05 conditions fully preloaded → covid_19 confirmed."""
        known = {
            "fever": True,
            "cough": True,
            "loss_of_taste": True,
            "loss_of_smell": True,
        }
        result, _ = diagnose_backward("covid_19", known, _no_to_all)
        assert result is True

    def test_confirms_dengue_preloaded(self):
        """R11 conditions fully preloaded → dengue confirmed."""
        known = {
            "high_fever": True,
            "severe_headache": True,
            "joint_pain": True,
            "rash": True,
        }
        result, _ = diagnose_backward("dengue", known, _no_to_all)
        assert result is True

    def test_confirms_uti_preloaded(self):
        """R20 conditions fully preloaded → UTI confirmed."""
        known = {
            "burning_urination": True,
            "frequent_urination": True,
            "lower_abdominal_pain": True,
        }
        result, _ = diagnose_backward("urinary_tract_infection", known, _no_to_all)
        assert result is True

    def test_confirms_via_yes_to_all_ask_fn(self):
        """Engine asks about unknown symptoms; all answered 'yes' → confirmed."""
        result, _ = diagnose_backward(
            "influenza",
            known_symptoms={},        # nothing pre-loaded
            ask_fn=_yes_to_all,
        )
        assert result is True

    def test_confirms_common_cold_via_questions(self):
        """Cold confirmed when all unknown symptoms answered yes."""
        result, _ = diagnose_backward(
            "common_cold",
            known_symptoms={},
            ask_fn=_yes_to_all,
        )
        assert result is True


class TestBackwardChainReject:

    def test_rejects_influenza_when_all_no(self):
        """No symptoms at all → influenza rejected."""
        result, _ = diagnose_backward(
            "influenza",
            known_symptoms={},
            ask_fn=_no_to_all,
        )
        assert result is False, "Influenza should be rejected when all answers are No"

    def test_rejects_covid19_when_all_no(self):
        """No symptoms → covid_19 rejected."""
        result, _ = diagnose_backward("covid_19", {}, _no_to_all)
        assert result is False

    def test_rejects_when_one_critical_condition_false(self):
        """If a condition is explicitly False in WM the rule cannot fire."""
        known = {
            "fever": True,
            "cough": True,
            "sore_throat": True,
            "body_ache": False,          # explicitly absent
        }
        # R01 needs body_ache; it's False → try R02 which needs fatigue/headache
        # With ask_fn=no_to_all those too will be False → rejected
        result, _ = diagnose_backward("influenza", known, _no_to_all)
        assert result is False

    def test_rejects_with_targeted_no_answers(self):
        """Answers 'no' only to the first needed symptom kills the hypothesis."""
        # For the first rule that needs 'fever', answer no → rule fails
        answers = {"fever": False}
        result, _ = diagnose_backward(
            "influenza",
            known_symptoms={},
            ask_fn=_make_ask_fn(answers, default=False),
        )
        assert result is False


class TestBackwardChainTrace:

    def test_trace_records_rule_fired(self):
        """When influenza is confirmed, the trace must include a rule firing."""
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        _, tracer = diagnose_backward("influenza", known, _no_to_all)
        assert len(tracer.entries()) > 0, "Trace should contain at least one entry"

    def test_trace_mode_is_backward(self):
        """All entries recorded in backward mode must have mode='backward'."""
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        _, tracer = diagnose_backward("influenza", known, _no_to_all)
        for entry in tracer.entries():
            assert entry.mode == "backward", (
                f"Expected mode='backward', got '{entry.mode}'"
            )

    def test_trace_conclusion_matches_goal(self):
        """Fired rule's conclusion must equal the goal we were trying to prove."""
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        goal = "influenza"
        _, tracer = diagnose_backward(goal, known, _no_to_all)
        conclusions_in_trace = {e.conclusion for e in tracer.entries()}
        assert goal in conclusions_in_trace

    def test_trace_lists_matched_conditions(self):
        """Matched conditions in trace must be a subset of rule conditions."""
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        _, tracer = diagnose_backward("influenza", known, _no_to_all)
        for entry in tracer.entries():
            assert len(entry.matched_conditions) > 0

    def test_combined_cf_populated_after_confirm(self):
        """After confirmation, combined_cf for the goal must be > 0."""
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        _, tracer = diagnose_backward("influenza", known, _no_to_all)
        assert tracer.combined_cf("influenza") > 0.0

    def test_no_trace_entries_on_rejected(self):
        """If hypothesis is rejected with no rules firing, trace is empty."""
        result, tracer = diagnose_backward("influenza", {}, _no_to_all)
        assert result is False
        assert len(tracer.entries()) == 0, (
            "Rejected hypothesis with no matches should produce empty trace"
        )


class TestBackwardChainPartialKnowledge:

    def test_partial_preload_then_ask_confirms(self):
        """Some symptoms preloaded, rest answered yes by ask_fn → confirmed."""
        # R01: fever, cough, sore_throat, body_ache
        # Pre-load half
        known = {"fever": True, "cough": True}
        ask = _make_ask_fn({"sore_throat": True, "body_ache": True}, default=False)
        result, tracer = diagnose_backward("influenza", known, ask)
        assert result is True

    def test_questions_asked_only_for_unknowns(self):
        """ask_fn should not be called for already-known symptoms."""
        asked: list[str] = []

        def tracking_ask(symptom: str) -> bool:
            asked.append(symptom)
            return True

        # Pre-load all conditions for R01 — ask_fn should never be called
        known = {
            "fever": True, "cough": True,
            "sore_throat": True, "body_ache": True,
        }
        backward_chain(
            goal="influenza",
            wm=WorkingMemory(initial_facts=known),
            tracer=Tracer(),
            ask_fn=tracking_ask,
        )
        # None of R01's symptoms were unknown, so ask_fn should not have been
        # called for any of them
        assert not any(s in asked for s in ["fever", "cough", "sore_throat", "body_ache"]), (
            f"ask_fn was called for pre-loaded symptoms: {asked}"
        )


class TestBackwardChainCustomRules:

    def test_works_with_custom_rules(self):
        """backward_chain accepts an explicit rule list."""
        custom_rules = [
            {
                "id": "TC01",
                "conditions": ["purple_spots"],
                "conclusion": "alien_disease",
                "cf": 1.0,
                "description": "Test",
            }
        ]
        wm = WorkingMemory({"purple_spots": True})
        tracer = Tracer()
        result = backward_chain(
            goal="alien_disease",
            wm=wm,
            tracer=tracer,
            ask_fn=_no_to_all,
            rules=custom_rules,
        )
        assert result is True
