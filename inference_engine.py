"""
inference_engine.py
===================
Implements two classic AI inference strategies:

  1. forward_chain  — data-driven (bottom-up)
  2. backward_chain — goal-driven  (top-down)

Both strategies share the same knowledge base (RULES) and work on a
WorkingMemory instance.  All rule firings are recorded by a Tracer so the
system can explain its reasoning afterwards.

Forward Chaining (Data-Driven)
--------------------------------
Start with the set of symptoms the user has provided.
Repeatedly scan all rules; if every condition of a rule is True in working
memory, fire it — assert its conclusion and log the trace.
Keep iterating (agenda loop) until no new facts are added in a full pass.
Classic "Rete-style" saturation, simplified.

Backward Chaining (Goal-Driven)
---------------------------------
Start with a hypothesis disease (the goal).
Look up every rule that concludes that disease.
For each such rule, check whether ALL its conditions are satisfied:
  - If a condition is True  in WM → already satisfied, continue.
  - If a condition is False in WM → rule fails, try next rule.
  - If condition is Unknown → call ask_fn(symptom) to query the user,
    record the answer, then continue.
If any rule's conditions are fully satisfied, the goal is PROVED.
If no rule can be satisfied, the goal is DISPROVED.
"""

from __future__ import annotations
from typing import Callable

from knowledge_base import RULES, get_rules_for_disease
from working_memory import WorkingMemory
from explanation import Tracer


# ═══════════════════════════════════════════════════════════════════════════════
#  Forward Chaining
# ═══════════════════════════════════════════════════════════════════════════════

def forward_chain(
    wm: WorkingMemory,
    tracer: Tracer,
    rules: list[dict] | None = None,
) -> dict[str, float]:
    """
    Data-driven inference: derive all conclusions reachable from current facts.

    Parameters
    ----------
    wm      : WorkingMemory  — must already contain the user's symptom facts
    tracer  : Tracer         — records every rule firing for explanation
    rules   : optional override list (defaults to the global RULES)

    Returns
    -------
    dict mapping  conclusion → combined certainty factor
    All derived conclusions are also asserted into *wm* so they can be used
    as conditions for chained rules (e.g. a disease can itself be a condition
    for a higher-level rule in the future).
    """
    rule_set = rules if rules is not None else RULES
    fired_rules: set[str] = set()    # rule IDs already fired this session

    changed = True
    while changed:                   # keep looping until a full pass adds nothing
        changed = False
        for rule in rule_set:
            if rule["id"] in fired_rules:
                continue             # already fired — skip (avoid infinite loops)

            # Check if ALL conditions are True in working memory
            if all(wm.is_true(cond) for cond in rule["conditions"]):
                # ── Fire the rule ──────────────────────────────────────────
                combined_cf = tracer.record(
                    rule=rule,
                    matched_conditions=rule["conditions"],
                    mode="forward",
                )
                # Assert the conclusion back into WM so downstream rules can
                # use it.  We mark it True only if combined CF ≥ 0.5.
                if combined_cf >= 0.5:
                    wm.assert_fact(rule["conclusion"], True)
                fired_rules.add(rule["id"])
                changed = True       # something new was derived — loop again

    return tracer.all_conclusions()


# ═══════════════════════════════════════════════════════════════════════════════
#  Backward Chaining
# ═══════════════════════════════════════════════════════════════════════════════

def backward_chain(
    goal: str,
    wm: WorkingMemory,
    tracer: Tracer,
    ask_fn: Callable[[str], bool],
    rules: list[dict] | None = None,
    _visited: set[str] | None = None,
) -> bool:
    """
    Goal-driven inference: try to prove *goal* by working backwards through rules.

    Parameters
    ----------
    goal    : str                  — disease or fact to prove (e.g. "influenza")
    wm      : WorkingMemory        — current belief state; updated as answers arrive
    tracer  : Tracer               — records fired rules
    ask_fn  : Callable[[str],bool] — called for unknown symptoms; returns True/False
    rules   : optional rule list override
    _visited: internal guard against infinite recursion

    Returns
    -------
    True  if the goal can be proved (at least one rule's conditions are all True)
    False if no rule can prove the goal
    """
    rule_set = rules if rules is not None else RULES
    if _visited is None:
        _visited = set()

    # ── Base case: goal is already known ─────────────────────────────────────
    if wm.is_true(goal):
        return True
    if wm.is_false(goal):
        return False

    # ── Guard: don't revisit a goal (handles circular rules) ─────────────────
    if goal in _visited:
        return False
    _visited.add(goal)

    # ── Find rules that conclude the goal ─────────────────────────────────────
    candidate_rules = get_rules_for_disease(goal) if not rules else [
        r for r in rule_set if r["conclusion"] == goal
    ]

    if not candidate_rules:
        # No rule can prove this — ask user directly as a last resort
        answer = ask_fn(goal)
        wm.assert_fact(goal, answer)
        return answer

    # ── Try each candidate rule ───────────────────────────────────────────────
    for rule in candidate_rules:
        all_satisfied = True
        satisfied_conditions: list[str] = []

        for condition in rule["conditions"]:
            if wm.is_true(condition):
                satisfied_conditions.append(condition)
                continue                          # already known True ✓

            if wm.is_false(condition):
                all_satisfied = False
                break                             # this rule fails — try next

            # Unknown: try backward-chaining on it first (sub-goal), then ask
            # Sub-goal check: maybe this condition is itself a disease conclusion
            sub_proved = backward_chain(
                goal=condition,
                wm=wm,
                tracer=tracer,
                ask_fn=ask_fn,
                rules=rule_set,
                _visited=_visited,
            )

            if not sub_proved:
                # Fall back to asking the user
                answer = ask_fn(condition)
                wm.assert_fact(condition, answer)
                if answer:
                    satisfied_conditions.append(condition)
                else:
                    all_satisfied = False
                    break
            else:
                satisfied_conditions.append(condition)

        if all_satisfied:
            # ── Fire the rule ────────────────────────────────────────────────
            tracer.record(
                rule=rule,
                matched_conditions=satisfied_conditions,
                mode="backward",
                note=f"Proving goal: {goal}",
            )
            wm.assert_fact(goal, True)
            return True

    # No rule could be satisfied
    wm.assert_fact(goal, False)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience wrapper: run forward chaining from a plain symptom set
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_forward(symptom_set: set[str]) -> tuple[dict[str, float], Tracer]:
    """
    Convenience function: create a fresh WM + Tracer, load symptoms, run
    forward chaining, and return (conclusions_with_cf, tracer).

    Parameters
    ----------
    symptom_set : set of symptom strings that are *present*

    Returns
    -------
    (dict[disease → cf], Tracer)
    """
    wm = WorkingMemory()
    for symptom in symptom_set:
        wm.assert_fact(symptom, True)

    tracer = Tracer()
    conclusions = forward_chain(wm, tracer)
    return conclusions, tracer


def diagnose_backward(
    goal: str,
    known_symptoms: dict[str, bool],
    ask_fn: Callable[[str], bool],
) -> tuple[bool, Tracer]:
    """
    Convenience function: create a fresh WM + Tracer, load known facts, run
    backward chaining on *goal*, return (result, tracer).

    Parameters
    ----------
    goal           : disease hypothesis to test
    known_symptoms : {symptom: True/False} pre-loaded facts
    ask_fn         : callback to query user for unknown symptoms

    Returns
    -------
    (bool result, Tracer)
    """
    wm = WorkingMemory(initial_facts=known_symptoms)
    tracer = Tracer()
    result = backward_chain(goal, wm, tracer, ask_fn)
    return result, tracer
