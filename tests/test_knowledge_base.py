"""
tests/test_knowledge_base.py
==============================
Tests for knowledge_base.py — rule integrity and helper functions.
"""

import pytest
from knowledge_base import (
    RULES, get_all_symptoms, get_all_diseases,
    get_rules_for_disease, add_rule,
)


class TestRuleIntegrity:

    def test_minimum_rule_count(self):
        """Knowledge base must have at least 15 rules."""
        assert len(RULES) >= 15, f"Expected ≥15 rules, found {len(RULES)}"

    def test_all_rules_have_required_keys(self):
        """Every rule must have id, conditions, conclusion, cf, description."""
        required = {"id", "conditions", "conclusion", "cf", "description"}
        for rule in RULES:
            missing = required - rule.keys()
            assert not missing, f"Rule {rule.get('id','?')} missing fields: {missing}"

    def test_all_rule_ids_are_unique(self):
        """Rule IDs must not repeat."""
        ids = [r["id"] for r in RULES]
        assert len(ids) == len(set(ids)), "Duplicate rule IDs found!"

    def test_certainty_factors_in_range(self):
        """All CF values must be between 0.0 and 1.0."""
        for rule in RULES:
            assert 0.0 <= rule["cf"] <= 1.0, (
                f"Rule {rule['id']}: CF={rule['cf']} out of [0,1]"
            )

    def test_conditions_are_non_empty_lists(self):
        """Every rule must have at least one condition."""
        for rule in RULES:
            assert isinstance(rule["conditions"], list)
            assert len(rule["conditions"]) >= 1, (
                f"Rule {rule['id']} has no conditions"
            )

    def test_conditions_are_strings(self):
        """All conditions must be non-empty strings."""
        for rule in RULES:
            for cond in rule["conditions"]:
                assert isinstance(cond, str) and cond.strip(), (
                    f"Rule {rule['id']} has invalid condition: {cond!r}"
                )

    def test_conclusion_is_non_empty_string(self):
        """Conclusion must be a non-empty string."""
        for rule in RULES:
            assert isinstance(rule["conclusion"], str) and rule["conclusion"].strip(), (
                f"Rule {rule['id']} has invalid conclusion"
            )


class TestHelperFunctions:

    def test_get_all_symptoms_returns_list(self):
        assert isinstance(get_all_symptoms(), list)

    def test_get_all_symptoms_non_empty(self):
        assert len(get_all_symptoms()) > 0

    def test_get_all_symptoms_sorted(self):
        symptoms = get_all_symptoms()
        assert symptoms == sorted(symptoms)

    def test_get_all_diseases_returns_list(self):
        assert isinstance(get_all_diseases(), list)

    def test_get_all_diseases_sorted(self):
        diseases = get_all_diseases()
        assert diseases == sorted(diseases)

    def test_known_diseases_present(self):
        diseases = get_all_diseases()
        expected = {"influenza", "common_cold", "covid_19", "dengue",
                    "malaria", "pneumonia", "gastroenteritis"}
        for d in expected:
            assert d in diseases, f"Expected disease '{d}' not found in KB"

    def test_get_rules_for_disease_influenza(self):
        rules = get_rules_for_disease("influenza")
        assert len(rules) >= 1
        assert all(r["conclusion"] == "influenza" for r in rules)

    def test_get_rules_for_nonexistent_disease(self):
        rules = get_rules_for_disease("unicorn_fever")
        assert rules == []


class TestAddRule:

    def test_add_valid_rule(self):
        initial_count = len(RULES)
        add_rule({
            "id": "ZTEST01",
            "conditions": ["test_symptom_x"],
            "conclusion": "test_disease_x",
            "cf": 0.77,
            "description": "Temporary test rule",
        })
        assert len(RULES) == initial_count + 1
        # Clean up
        RULES[:] = [r for r in RULES if r["id"] != "ZTEST01"]

    def test_add_rule_duplicate_id_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            add_rule({
                "id": "R01",    # duplicate
                "conditions": ["x"],
                "conclusion": "y",
                "cf": 0.5,
                "description": "duplicate",
            })

    def test_add_rule_invalid_cf_raises(self):
        with pytest.raises(ValueError, match="Certainty factor"):
            add_rule({
                "id": "ZBAD01",
                "conditions": ["x"],
                "conclusion": "y",
                "cf": 1.5,     # invalid
                "description": "bad cf",
            })

    def test_add_rule_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing required fields"):
            add_rule({
                "id": "ZBAD02",
                "conditions": ["x"],
                # 'conclusion', 'cf', 'description' missing
            })
