# MediLogix 🩺

> **Rule-Based Medical Diagnosis Expert System**  
> A college AI project demonstrating Forward Chaining and Backward Chaining inference with MYCIN-style Certainty Factors.

> ⚠️ **DISCLAIMER — Educational Use Only**  
> MediLogix is a *demonstration* of classical AI expert-system techniques built for a college AI course. It is **NOT** a real medical diagnostic tool and does **NOT** constitute medical advice. Always consult a qualified healthcare professional for any medical concerns.

---

## Table of Contents

1. [What Is MediLogix?](#what-is-medilogix)
2. [Forward vs. Backward Chaining — Core Concepts](#forward-vs-backward-chaining)
3. [Project Architecture](#project-architecture)
4. [Certainty Factors (MYCIN-style)](#certainty-factors-mycin-style)
5. [Rule Format & How to Extend the KB](#rule-format--how-to-extend-the-knowledge-base)
6. [Setup & Running Locally](#setup--running-locally)
7. [Running the Tests](#running-the-tests)
8. [Example Runs](#example-runs)
9. [Rule Network Diagram](#rule-network-diagram)
10. [Project Structure](#project-structure)

---

## What Is MediLogix?

MediLogix is a **rule-based expert system** — a form of AI that encodes human expert knowledge as a collection of IF-THEN rules, then uses an inference engine to apply those rules automatically.

It covers 20 rules across 10 diseases (flu, COVID-19, common cold, pneumonia, dengue, malaria, typhoid, gastroenteritis, hypertension, migraine, diabetes, asthma, UTI).

The system has **two inference modes**, each demonstrating a fundamentally different reasoning strategy:

| Mode | Direction | Starting point | What it does |
|------|-----------|----------------|--------------|
| **Forward Chaining** | Bottom-up / data-driven | Known symptoms | Derives every reachable diagnosis |
| **Backward Chaining** | Top-down / goal-driven | A hypothesis disease | Asks only the questions it needs to prove/reject |

---

## Forward vs. Backward Chaining

### 🔵 Forward Chaining (Data-Driven Inference)

```
Symptoms (facts) ──→ Rules ──→ Derived diagnoses
```

**How it works:**

1. Load all known symptoms into Working Memory as facts.
2. Scan all rules. If every condition of a rule is satisfied, **fire** it — assert the conclusion as a new fact.
3. Repeat (agenda loop) until no new facts can be derived.
4. Return all derived diagnoses with their combined certainty factors.

**Analogy:** A doctor looks at your full chart and works through every possible diagnosis systematically.

**Classic use-case:** "Given these symptoms, what could this be?"

```
Facts: {fever, cough, sore_throat, body_ache}
  → Rule R01 fires: IF fever AND cough AND sore_throat AND body_ache THEN influenza [CF=90%]
Conclusion: influenza [90%]
```

---

### 🟢 Backward Chaining (Goal-Driven Inference)

```
Hypothesis ──→ Rules for that hypothesis ──→ Ask about conditions ──→ Proven/Rejected
```

**How it works:**

1. Start with a **goal** (e.g. "Does the patient have influenza?").
2. Find all rules whose conclusion is that goal.
3. For each rule, check each condition:
   - **Known True** → satisfied, continue.
   - **Known False** → this rule fails, try next.
   - **Unknown** → recursively try to prove it as a sub-goal, or ask the user.
4. If any rule's conditions are all satisfied → goal **PROVED**.
5. If no rule can be satisfied → goal **DISPROVED**.

**Analogy:** A doctor suspects flu and asks *only* the specific questions that let them confirm or rule it out.

**Classic use-case:** "Is this hypothesis true?" — much more efficient when you already have a strong prior.

```
Goal: influenza
  → R01 needs: fever? → ask → Yes
              cough?  → ask → Yes
              sore_throat? → ask → Yes
              body_ache?   → ask → Yes
  → All conditions met → influenza CONFIRMED [CF=90%]
```

---

### Key Differences at a Glance

| Aspect | Forward Chaining | Backward Chaining |
|--------|-----------------|-------------------|
| Direction | Facts → conclusions | Goal → facts needed |
| Starting point | All known symptoms | A specific hypothesis |
| Questions asked | None (all facts pre-loaded) | Only those needed to prove/reject |
| Output | All derivable diagnoses | Confirmed / Rejected for one disease |
| Best for | Broad differential diagnosis | Hypothesis testing / confirmation |
| Analogous to | Lab-work-first approach | Clinician reasoning toward a suspicion |

---

## Project Architecture

```
medilogix/
├── knowledge_base.py       # 20 IF-THEN rules as pure data (list of dicts)
├── working_memory.py       # Belief state: facts with True/False/Unknown
├── inference_engine.py     # forward_chain() + backward_chain() algorithms
├── explanation.py          # Tracer — records rule firings, generates "Why?" output
├── app.py                  # Streamlit UI — two inference modes
├── requirements.txt
├── pyproject.toml          # pytest config
└── tests/
    ├── test_forward_chain.py   # 16 tests
    ├── test_backward_chain.py  # 22 tests
    ├── test_explanation.py     # 22 tests
    ├── test_knowledge_base.py  # 16 tests
    └── test_working_memory.py  # 17 tests
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `knowledge_base.py` | Pure data — rules as list of dicts. Zero inference logic. |
| `working_memory.py` | Session state — facts with three states: True / False / Unknown. |
| `inference_engine.py` | Algorithms only — no UI, no data. Depends on the two above. |
| `explanation.py` | Cross-cutting concern — receives events from the engine, formats them. |
| `app.py` | Presentation layer — depends on all four above, Streamlit only. |

---

## Certainty Factors (MYCIN-style)

Each rule carries a **certainty factor** (CF) between 0.0 and 1.0.

When **multiple rules** derive the same conclusion, their CFs are combined using the **MYCIN formula**:

```
combined = CF₁ + CF₂ × (1 − CF₁)
```

This is similar to combining independent probabilities but is bounded by 1.0 and stays intuitive. A result reads as **"72% likely — flu"** rather than a flat yes/no.

**Example:**

- Rule R01 derives `influenza` with CF=0.90
- Rule R02 also derives `influenza` with CF=0.75
- Combined: `0.90 + 0.75 × (1 − 0.90) = 0.90 + 0.075 = 0.975` → **97.5% likely**

---

## Rule Format & How to Extend the Knowledge Base

Rules live in [`knowledge_base.py`](knowledge_base.py) as a plain Python list. **No inference code changes are needed** to add a rule.

### Rule Schema

```python
{
    "id":          "R21",                      # unique string ID
    "conditions":  ["symptom_a", "symptom_b"], # ALL must be True (AND logic)
    "conclusion":  "disease_name",             # fact asserted when rule fires
    "cf":          0.85,                       # certainty factor 0.0–1.0
    "description": "Human-readable summary",
}
```

### Adding a Rule — three ways

**1. Edit `knowledge_base.py` directly** — append to the `RULES` list.

**2. From the UI** — use the "➕ Add a new rule" expander in the sidebar.

**3. At runtime from Python:**

```python
from knowledge_base import add_rule

add_rule({
    "id": "R21",
    "conditions": ["spotted_rash", "joint_swelling", "fever"],
    "conclusion": "chikungunya",
    "cf": 0.83,
    "description": "Spotted rash + joint swelling + fever → Chikungunya",
})
```

### Symptom naming convention

Use `snake_case` for symptom names (e.g. `loss_of_smell`, `chest_pain`). The UI automatically converts underscores to spaces and applies title case.

---

## Setup & Running Locally

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
git clone https://github.com/Partha81-star/MediLogix.git
cd MediLogix
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Running the Tests

```bash
# Run all 92 tests with verbose output
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing
```

**Test coverage summary:**

| Test file | Tests | What's covered |
|-----------|-------|----------------|
| `test_forward_chain.py` | 16 | FC derivation, CF ranges, double-fire guard, custom rules |
| `test_backward_chain.py` | 22 | Confirmation, rejection, trace, partial knowledge, ask-fn tracking |
| `test_explanation.py` | 22 | MYCIN math, entry ordering, filtering, reset, output formatting |
| `test_knowledge_base.py` | 15 | Rule integrity, helpers, add_rule validation |
| `test_working_memory.py` | 17 | All WM methods: assert, retract, get, reset, contains |
| **Total** | **92** | **All pass ✅** |

---

## Example Runs

### Forward Chaining — Flu diagnosis

```
Input symptoms: fever ✓  cough ✓  sore_throat ✓  body_ache ✓

══════════════════════════════════════════════════════════
  REASONING TRACE
══════════════════════════════════════════════════════════

Step 1  [FORWARD]  Rule R01
  IF   fever, cough, sore_throat, body_ache
  THEN influenza  (rule CF = 90%)
  Combined CF for 'influenza' → 90%

══════════════════════════════════════════════════════════
  CONCLUSIONS
══════════════════════════════════════════════════════════
  influenza                      90%  ██████████████████
```

---

### Backward Chaining — COVID-19 hypothesis

```
Hypothesis: covid_19

? Do you have Fever?           → Yes
? Do you have Cough?           → Yes
? Do you have Loss Of Taste?   → Yes
? Do you have Loss Of Smell?   → Yes

✅ CONFIRMED — Covid 19 is 95% likely

Rule R05 fired:
  IF (fever AND cough AND loss_of_taste AND loss_of_smell)
  THEN covid_19 [CF=95%]
```

---

## Rule Network Diagram

The Streamlit UI generates a live rule network diagram after Forward Chaining runs. Symptom nodes are purple, diagnosis nodes are blue, and directed edges show which rules fired.

Example for a flu + cold combined session:

```
[fever] ──────────────┐
[cough] ──────────────┤ R01 ──→ [influenza]
[sore_throat] ────────┤
[body_ache] ──────────┘

[runny_nose] ─────────┐
[sneezing] ───────────┤ R03 ──→ [common_cold]
[sore_throat] ────────┘
```

---

## Project Structure

```
.
├── app.py                  # Streamlit UI (two inference modes)
├── knowledge_base.py       # 20 medical rules as data
├── working_memory.py       # Session fact store
├── inference_engine.py     # forward_chain() + backward_chain()
├── explanation.py          # MYCIN tracer + "Why?" explanations
├── requirements.txt        # Python dependencies
├── pyproject.toml          # pytest configuration
├── README.md               # This file
└── tests/
    ├── __init__.py
    ├── test_forward_chain.py
    ├── test_backward_chain.py
    ├── test_explanation.py
    ├── test_knowledge_base.py
    └── test_working_memory.py
```

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11+ | Core language |
| Streamlit | Interactive web UI |
| NetworkX + Matplotlib | Rule-network diagram |
| Pandas | Knowledge base table display |
| pytest | Test framework |

---

*Built for the AI course project — demonstrating that even "simple" rule-based systems embody deep ideas about reasoning, knowledge representation, and explainability.*
