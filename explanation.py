"""
explanation.py
==============
Trace / Explanation Facility
-----------------------------
Every time the inference engine fires a rule it calls the Tracer.
At the end of a session the tracer can produce a human-readable
"Why did you conclude X?" explanation — the hallmark of classic
expert systems (MYCIN-style).

Trace entry schema
------------------
{
  "step":        int          — firing order (1-based)
  "rule_id":     str          — e.g. "R01"
  "matched_conditions": list  — the conditions that were True
  "conclusion":  str          — the fact that was derived
  "cf":          float        — certainty factor of the rule
  "combined_cf": float        — running combined CF for this conclusion
  "mode":        str          — "forward" | "backward"
  "note":        str          — optional free-text remark
}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TraceEntry:
    step: int
    rule_id: str
    matched_conditions: list[str]
    conclusion: str
    cf: float
    combined_cf: float
    mode: Literal["forward", "backward"]
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "rule_id": self.rule_id,
            "matched_conditions": self.matched_conditions,
            "conclusion": self.conclusion,
            "cf": self.cf,
            "combined_cf": self.combined_cf,
            "mode": self.mode,
            "note": self.note,
        }


class Tracer:
    """
    Collects rule-firing events and converts them into a readable explanation.

    Usage
    -----
    tracer = Tracer()
    tracer.record(rule, matched_conditions, combined_cf, mode="forward")
    ...
    print(tracer.explain())
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._combined_cf: dict[str, float] = {}   # conclusion → running CF

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        rule: dict,
        matched_conditions: list[str],
        mode: Literal["forward", "backward"],
        note: str = "",
    ) -> float:
        """
        Record a rule firing and compute the MYCIN-style combined certainty
        factor for that conclusion.

        MYCIN combination formula (two CFs already in range [0, 1]):
            combined = cf1 + cf2 * (1 - cf1)

        Returns the updated combined CF for the conclusion.
        """
        conclusion = rule["conclusion"]
        rule_cf = rule["cf"]

        prev_cf = self._combined_cf.get(conclusion, 0.0)
        new_cf = prev_cf + rule_cf * (1.0 - prev_cf)          # MYCIN formula
        self._combined_cf[conclusion] = new_cf

        entry = TraceEntry(
            step=len(self._entries) + 1,
            rule_id=rule["id"],
            matched_conditions=list(matched_conditions),
            conclusion=conclusion,
            cf=rule_cf,
            combined_cf=round(new_cf, 4),
            mode=mode,
            note=note,
        )
        self._entries.append(entry)
        return new_cf

    # ── Querying ──────────────────────────────────────────────────────────────

    def entries(self) -> list[TraceEntry]:
        """Return a copy of all recorded trace entries in firing order."""
        return list(self._entries)

    def entries_for(self, conclusion: str) -> list[TraceEntry]:
        """Return entries that contributed to a specific conclusion."""
        return [e for e in self._entries if e.conclusion == conclusion]

    def combined_cf(self, conclusion: str) -> float:
        """Return the final combined certainty factor for a given conclusion."""
        return self._combined_cf.get(conclusion, 0.0)

    def all_conclusions(self) -> dict[str, float]:
        """Return {conclusion: combined_cf} for every derived conclusion."""
        return dict(self._combined_cf)

    def reset(self) -> None:
        """Clear all recorded entries (call before a new session)."""
        self._entries.clear()
        self._combined_cf.clear()

    # ── Human-readable explanation ─────────────────────────────────────────────

    def explain(self) -> str:
        """
        Generate a plain-text explanation of every rule that fired,
        suitable for display in a terminal or a text widget.
        """
        if not self._entries:
            return "No rules fired — no explanation available."

        lines: list[str] = ["═" * 60, "  REASONING TRACE", "═" * 60]
        for e in self._entries:
            conds = ", ".join(e.matched_conditions)
            lines.append(
                f"\nStep {e.step}  [{e.mode.upper()}]  Rule {e.rule_id}"
            )
            lines.append(f"  IF   {conds}")
            lines.append(f"  THEN {e.conclusion}  (rule CF = {e.cf:.0%})")
            lines.append(f"  Combined CF for '{e.conclusion}' → {e.combined_cf:.0%}")
            if e.note:
                lines.append(f"  Note: {e.note}")

        lines.append("\n" + "═" * 60)
        lines.append("  CONCLUSIONS")
        lines.append("═" * 60)
        for conclusion, cf in sorted(
            self._combined_cf.items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(cf * 20)
            lines.append(f"  {conclusion:<30} {cf:>6.0%}  {bar}")

        return "\n".join(lines)

    def explain_for(self, conclusion: str) -> str:
        """
        Generate an explanation focussed on one specific conclusion —
        used by the backward-chaining mode's "Why?" button.
        """
        relevant = self.entries_for(conclusion)
        if not relevant:
            return f"No rules fired for '{conclusion}'."

        lines = [f"Explanation for: {conclusion.upper()}", "-" * 40]
        for e in relevant:
            conds = ", ".join(e.matched_conditions)
            lines.append(
                f"  Rule {e.rule_id} — IF ({conds}) THEN {conclusion} "
                f"[CF={e.cf:.0%}, combined={e.combined_cf:.0%}]"
            )
            if e.note:
                lines.append(f"    Note: {e.note}")
        return "\n".join(lines)
