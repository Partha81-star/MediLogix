"""
tests/test_working_memory.py
==============================
Tests for the WorkingMemory class.
"""

import pytest
from working_memory import WorkingMemory


class TestWorkingMemoryBasics:

    def test_empty_init(self):
        wm = WorkingMemory()
        assert wm.all_facts() == {}

    def test_init_with_facts(self):
        wm = WorkingMemory({"fever": True, "cough": False})
        assert wm.is_true("fever")
        assert wm.is_false("cough")

    def test_assert_fact_true(self):
        wm = WorkingMemory()
        wm.assert_fact("fever", True)
        assert wm.is_true("fever")

    def test_assert_fact_false(self):
        wm = WorkingMemory()
        wm.assert_fact("cough", False)
        assert wm.is_false("cough")

    def test_unknown_fact_is_not_true(self):
        wm = WorkingMemory()
        assert not wm.is_true("unknown_symptom")

    def test_unknown_fact_is_not_false(self):
        wm = WorkingMemory()
        assert not wm.is_false("unknown_symptom")

    def test_is_known_returns_false_for_unknown(self):
        wm = WorkingMemory()
        assert not wm.is_known("x")

    def test_is_known_returns_true_for_asserted(self):
        wm = WorkingMemory()
        wm.assert_fact("x", True)
        assert wm.is_known("x")

    def test_get_returns_none_for_unknown(self):
        wm = WorkingMemory()
        assert wm.get("z") is None

    def test_get_returns_value_for_known(self):
        wm = WorkingMemory()
        wm.assert_fact("y", False)
        assert wm.get("y") is False

    def test_retract_removes_fact(self):
        wm = WorkingMemory()
        wm.assert_fact("fever", True)
        wm.retract_fact("fever")
        assert not wm.is_known("fever")

    def test_retract_nonexistent_is_harmless(self):
        wm = WorkingMemory()
        wm.retract_fact("nonexistent")    # should not raise

    def test_positive_facts(self):
        wm = WorkingMemory({"a": True, "b": False, "c": True})
        assert wm.positive_facts() == {"a", "c"}

    def test_all_facts_returns_copy(self):
        wm = WorkingMemory({"a": True})
        copy = wm.all_facts()
        copy["extra"] = True
        assert "extra" not in wm.all_facts()

    def test_update_bulk(self):
        wm = WorkingMemory()
        wm.update({"x": True, "y": False})
        assert wm.is_true("x")
        assert wm.is_false("y")

    def test_reset_clears_all(self):
        wm = WorkingMemory({"a": True, "b": True})
        wm.reset()
        assert wm.all_facts() == {}

    def test_contains_operator(self):
        wm = WorkingMemory({"fever": True})
        assert "fever" in wm
        assert "cough" not in wm
