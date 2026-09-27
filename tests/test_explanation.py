"""
tests/test_explanation.py
==========================
pytest tests for the Tracer / explanation facility.
"""

import pytest
from explanation import Tracer, TraceEntry


# ── Sample rule fixtures ───────────────────────────────────────────────────────

RULE_R01 = {
    "id": "R01",
    "conditions": ["fever", "cough", "sore_throat", "body_ache"],
    "conclusion": "influenza",
    "cf": 0.90,
    "description": "Classic flu rule",
}

RULE_R02 = {
    "id": "R02",
    "conditions": ["fever", "cough", "fatigue", "headache"],
    "conclusion": "influenza",
    "cf": 0.75,
    "description": "Flu with fatigue",
}

RULE_R05 = {
    "id": "R05",
    "conditions": ["fever", "cough", "loss_of_taste", "loss_of_smell"],
    "conclusion": "covid_19",
    "cf": 0.95,
    "description": "Covid hallmark rule",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTracerRecord:

    def test_record_returns_combined_cf(self):
        """record() must return the newly combined CF (≥ rule CF for first firing)."""
        tracer = Tracer()
        cf = tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        assert cf == pytest.approx(0.90, abs=1e-6)

    def test_mycin_combination_two_rules(self):
        """Second firing of same conclusion should increase combined CF via MYCIN formula."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        # MYCIN: 0.90 + 0.75*(1-0.90) = 0.90 + 0.075 = 0.975
        cf2 = tracer.record(RULE_R02, RULE_R02["conditions"], mode="forward")
        expected = 0.90 + 0.75 * (1.0 - 0.90)
        assert cf2 == pytest.approx(expected, abs=1e-6)

    def test_entries_count(self):
        """entries() length must equal number of record() calls."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.record(RULE_R05, RULE_R05["conditions"], mode="backward")
        assert len(tracer.entries()) == 2

    def test_entries_are_copies(self):
        """entries() must return a copy — mutations should not affect internal list."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        entries_copy = tracer.entries()
        entries_copy.clear()
        assert len(tracer.entries()) == 1

    def test_step_numbers_are_sequential(self):
        """TraceEntry.step must be 1, 2, 3 …"""
        tracer = Tracer()
        for rule in [RULE_R01, RULE_R02, RULE_R05]:
            tracer.record(rule, rule["conditions"], mode="forward")
        steps = [e.step for e in tracer.entries()]
        assert steps == [1, 2, 3]

    def test_entries_for_filters_by_conclusion(self):
        """entries_for() must return only entries with the given conclusion."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.record(RULE_R02, RULE_R02["conditions"], mode="forward")
        tracer.record(RULE_R05, RULE_R05["conditions"], mode="backward")

        flu_entries = tracer.entries_for("influenza")
        assert len(flu_entries) == 2
        assert all(e.conclusion == "influenza" for e in flu_entries)

        covid_entries = tracer.entries_for("covid_19")
        assert len(covid_entries) == 1


class TestTracerCF:

    def test_combined_cf_zero_for_unknown(self):
        """combined_cf for a disease not yet derived must be 0.0."""
        tracer = Tracer()
        assert tracer.combined_cf("nonexistent") == 0.0

    def test_combined_cf_after_single_record(self):
        """After one record, combined_cf must equal the rule's CF."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        assert tracer.combined_cf("influenza") == pytest.approx(0.90, abs=1e-6)

    def test_all_conclusions(self):
        """all_conclusions() must list every derived conclusion."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.record(RULE_R05, RULE_R05["conditions"], mode="backward")
        conclusions = tracer.all_conclusions()
        assert set(conclusions.keys()) == {"influenza", "covid_19"}

    def test_cf_capped_by_mycin(self):
        """Even after many firings, combined CF must stay ≤ 1.0."""
        tracer = Tracer()
        # Fire a rule 10 times (simulating 10 overlapping rules for same disease)
        for i in range(10):
            rule = {**RULE_R01, "id": f"T{i:02d}"}
            tracer.record(rule, rule["conditions"], mode="forward")
        assert tracer.combined_cf("influenza") <= 1.0


class TestTracerReset:

    def test_reset_clears_entries(self):
        """After reset(), entries() must be empty."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.reset()
        assert len(tracer.entries()) == 0

    def test_reset_clears_cf(self):
        """After reset(), combined_cf for previously seen conclusions is 0."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.reset()
        assert tracer.combined_cf("influenza") == 0.0


class TestExplainOutput:

    def test_explain_contains_rule_id(self):
        """explain() output must mention each fired rule ID."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        output = tracer.explain()
        assert "R01" in output

    def test_explain_contains_conclusion(self):
        """explain() must mention the derived conclusion."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        output = tracer.explain()
        assert "influenza" in output

    def test_explain_empty_tracer(self):
        """explain() on an empty tracer must return a no-rules-fired message."""
        tracer = Tracer()
        output = tracer.explain()
        assert "No rules fired" in output

    def test_explain_for_specific_conclusion(self):
        """explain_for() must limit output to the specified conclusion."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        tracer.record(RULE_R05, RULE_R05["conditions"], mode="backward")
        output = tracer.explain_for("covid_19")
        assert "R05" in output
        assert "R01" not in output

    def test_explain_for_unknown_conclusion(self):
        """explain_for() for an unknown conclusion must say no rules fired."""
        tracer = Tracer()
        output = tracer.explain_for("alien_disease")
        assert "No rules fired" in output

    def test_mode_recorded_correctly_forward(self):
        """Entries recorded with mode='forward' must have mode attribute set."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward")
        assert tracer.entries()[0].mode == "forward"

    def test_mode_recorded_correctly_backward(self):
        """Entries recorded with mode='backward' must have mode attribute set."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="backward")
        assert tracer.entries()[0].mode == "backward"

    def test_note_stored_in_entry(self):
        """Custom note passed to record() must appear in the entry."""
        tracer = Tracer()
        tracer.record(RULE_R01, RULE_R01["conditions"], mode="forward", note="test note")
        assert tracer.entries()[0].note == "test note"

    def test_matched_conditions_stored(self):
        """matched_conditions in each entry must match what was passed to record()."""
        tracer = Tracer()
        conds = ["fever", "cough"]
        tracer.record(RULE_R01, conds, mode="forward")
        assert tracer.entries()[0].matched_conditions == conds


class TestTraceEntryDataclass:

    def test_to_dict_has_required_keys(self):
        """TraceEntry.to_dict() must include all required keys."""
        entry = TraceEntry(
            step=1,
            rule_id="R01",
            matched_conditions=["fever"],
            conclusion="influenza",
            cf=0.9,
            combined_cf=0.9,
            mode="forward",
        )
        d = entry.to_dict()
        for key in ("step", "rule_id", "matched_conditions", "conclusion",
                    "cf", "combined_cf", "mode", "note"):
            assert key in d, f"Key '{key}' missing from to_dict() output"
