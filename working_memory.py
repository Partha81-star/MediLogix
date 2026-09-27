"""
working_memory.py
=================
The Working Memory (WM) holds the current set of facts the inference engine
operates on during a single reasoning session.

Facts are stored as a dict mapping fact-name → boolean value:
  {"fever": True, "cough": True, "runny_nose": False, ...}

A fact with value True  means the symptom/condition is *present*.
A fact with value False means it is *absent* (explicitly ruled out).
A missing key  means it is *unknown* (not yet queried).
"""

from __future__ import annotations


class WorkingMemory:
    """A thin wrapper around a dict that represents the agent's belief state."""

    def __init__(self, initial_facts: dict[str, bool] | None = None) -> None:
        self._facts: dict[str, bool] = dict(initial_facts) if initial_facts else {}

    # ── Core API ──────────────────────────────────────────────────────────────

    def assert_fact(self, name: str, value: bool = True) -> None:
        """Add or update a fact in working memory."""
        self._facts[name] = value

    def retract_fact(self, name: str) -> None:
        """Remove a fact from working memory (back to *unknown*)."""
        self._facts.pop(name, None)

    def is_true(self, name: str) -> bool:
        """Return True iff the fact is known AND is True."""
        return self._facts.get(name, False) is True

    def is_false(self, name: str) -> bool:
        """Return True iff the fact is known AND is False."""
        return self._facts.get(name, None) is False

    def is_known(self, name: str) -> bool:
        """Return True iff the fact has any known value (True or False)."""
        return name in self._facts

    def get(self, name: str) -> bool | None:
        """Return the fact's value, or None if unknown."""
        return self._facts.get(name, None)

    def all_facts(self) -> dict[str, bool]:
        """Return a copy of the full facts dict."""
        return dict(self._facts)

    def positive_facts(self) -> set[str]:
        """Return the set of fact names whose value is True."""
        return {k for k, v in self._facts.items() if v is True}

    def reset(self) -> None:
        """Clear all facts (start a fresh session)."""
        self._facts.clear()

    def update(self, facts: dict[str, bool]) -> None:
        """Bulk-load facts from a dict (useful for initialising from a symptom form)."""
        self._facts.update(facts)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self._facts

    def __repr__(self) -> str:  # pragma: no cover
        true_facts = [k for k, v in self._facts.items() if v]
        false_facts = [k for k, v in self._facts.items() if not v]
        return (
            f"WorkingMemory(true={true_facts}, false={false_facts})"
        )
