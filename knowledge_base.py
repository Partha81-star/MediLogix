"""
knowledge_base.py
=================
Stores all medical diagnostic rules as pure data (list of dicts).
Rules can be added, removed, or edited here WITHOUT touching any inference code.

Rule schema:
  {
    "id":          str   — unique identifier, e.g. "R01"
    "conditions":  list  — symptom strings that must ALL be present (AND logic)
    "conclusion":  str   — disease/fact that is derived when conditions hold
    "cf":          float — certainty factor  0.0 (no confidence) → 1.0 (certain)
    "description": str   — human-readable explanation of the rule
  }

Adding a new rule
-----------------
Just append a new dict to RULES following the schema above.
The inference engine will pick it up automatically.
"""

RULES = [
    # ── Influenza / Flu ─────────────────────────────────────────────────────
    {
        "id": "R01",
        "conditions": ["fever", "cough", "sore_throat", "body_ache"],
        "conclusion": "influenza",
        "cf": 0.90,
        "description": "High fever + cough + sore throat + body aches → Influenza (Flu)",
    },
    {
        "id": "R02",
        "conditions": ["fever", "cough", "fatigue", "headache"],
        "conclusion": "influenza",
        "cf": 0.75,
        "description": "Fever + cough + fatigue + headache → Influenza (moderate confidence)",
    },
    # ── Common Cold ──────────────────────────────────────────────────────────
    {
        "id": "R03",
        "conditions": ["runny_nose", "sneezing", "sore_throat"],
        "conclusion": "common_cold",
        "cf": 0.85,
        "description": "Runny nose + sneezing + sore throat → Common Cold",
    },
    {
        "id": "R04",
        "conditions": ["runny_nose", "mild_fever", "cough"],
        "conclusion": "common_cold",
        "cf": 0.70,
        "description": "Runny nose + mild fever + cough → Common Cold (lower confidence)",
    },
    # ── COVID-19 ─────────────────────────────────────────────────────────────
    {
        "id": "R05",
        "conditions": ["fever", "cough", "loss_of_taste", "loss_of_smell"],
        "conclusion": "covid_19",
        "cf": 0.95,
        "description": "Fever + cough + loss of taste/smell → COVID-19 (high confidence)",
    },
    {
        "id": "R06",
        "conditions": ["fever", "shortness_of_breath", "fatigue", "loss_of_taste"],
        "conclusion": "covid_19",
        "cf": 0.88,
        "description": "Fever + shortness of breath + fatigue + loss of taste → COVID-19",
    },
    # ── Pneumonia ─────────────────────────────────────────────────────────────
    {
        "id": "R07",
        "conditions": ["high_fever", "chest_pain", "shortness_of_breath", "productive_cough"],
        "conclusion": "pneumonia",
        "cf": 0.88,
        "description": "High fever + chest pain + shortness of breath + productive cough → Pneumonia",
    },
    {
        "id": "R08",
        "conditions": ["fever", "chills", "shortness_of_breath", "chest_pain"],
        "conclusion": "pneumonia",
        "cf": 0.75,
        "description": "Fever + chills + shortness of breath + chest pain → Pneumonia",
    },
    # ── Malaria ───────────────────────────────────────────────────────────────
    {
        "id": "R09",
        "conditions": ["cyclical_fever", "chills", "sweating", "headache"],
        "conclusion": "malaria",
        "cf": 0.87,
        "description": "Cyclical fever + chills + sweating + headache → Malaria",
    },
    {
        "id": "R10",
        "conditions": ["fever", "chills", "nausea", "muscle_pain"],
        "conclusion": "malaria",
        "cf": 0.70,
        "description": "Fever + chills + nausea + muscle pain → Malaria (moderate confidence)",
    },
    # ── Dengue ───────────────────────────────────────────────────────────────
    {
        "id": "R11",
        "conditions": ["high_fever", "severe_headache", "joint_pain", "rash"],
        "conclusion": "dengue",
        "cf": 0.90,
        "description": "High fever + severe headache + joint pain + rash → Dengue Fever",
    },
    {
        "id": "R12",
        "conditions": ["fever", "rash", "eye_pain", "muscle_pain"],
        "conclusion": "dengue",
        "cf": 0.78,
        "description": "Fever + rash + eye pain + muscle pain → Dengue Fever",
    },
    # ── Typhoid ───────────────────────────────────────────────────────────────
    {
        "id": "R13",
        "conditions": ["prolonged_fever", "abdominal_pain", "weakness", "loss_of_appetite"],
        "conclusion": "typhoid",
        "cf": 0.85,
        "description": "Prolonged fever + abdominal pain + weakness + loss of appetite → Typhoid",
    },
    # ── Gastroenteritis ───────────────────────────────────────────────────────
    {
        "id": "R14",
        "conditions": ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
        "conclusion": "gastroenteritis",
        "cf": 0.88,
        "description": "Nausea + vomiting + diarrhea + abdominal pain → Gastroenteritis",
    },
    {
        "id": "R15",
        "conditions": ["diarrhea", "abdominal_cramps", "fever"],
        "conclusion": "gastroenteritis",
        "cf": 0.72,
        "description": "Diarrhea + abdominal cramps + fever → Gastroenteritis",
    },
    # ── Hypertension ──────────────────────────────────────────────────────────
    {
        "id": "R16",
        "conditions": ["headache", "dizziness", "blurred_vision", "chest_tightness"],
        "conclusion": "hypertension",
        "cf": 0.80,
        "description": "Headache + dizziness + blurred vision + chest tightness → Hypertension",
    },
    # ── Migraine ──────────────────────────────────────────────────────────────
    {
        "id": "R17",
        "conditions": ["severe_headache", "nausea", "light_sensitivity", "sound_sensitivity"],
        "conclusion": "migraine",
        "cf": 0.87,
        "description": "Severe headache + nausea + light/sound sensitivity → Migraine",
    },
    # ── Diabetes (Type-2 indicators) ──────────────────────────────────────────
    {
        "id": "R18",
        "conditions": ["excessive_thirst", "frequent_urination", "fatigue", "blurred_vision"],
        "conclusion": "diabetes_type2",
        "cf": 0.82,
        "description": "Excessive thirst + frequent urination + fatigue + blurred vision → Diabetes T2",
    },
    # ── Asthma ────────────────────────────────────────────────────────────────
    {
        "id": "R19",
        "conditions": ["wheezing", "shortness_of_breath", "chest_tightness", "cough"],
        "conclusion": "asthma",
        "cf": 0.85,
        "description": "Wheezing + shortness of breath + chest tightness + cough → Asthma",
    },
    # ── Urinary Tract Infection ───────────────────────────────────────────────
    {
        "id": "R20",
        "conditions": ["burning_urination", "frequent_urination", "lower_abdominal_pain"],
        "conclusion": "urinary_tract_infection",
        "cf": 0.88,
        "description": "Burning urination + frequent urination + lower abdominal pain → UTI",
    },
]

# ── Convenience helpers ────────────────────────────────────────────────────────

def get_all_symptoms() -> list[str]:
    """Return a de-duplicated, sorted list of every symptom used across all rules."""
    symptoms: set[str] = set()
    for rule in RULES:
        symptoms.update(rule["conditions"])
    return sorted(symptoms)


def get_all_diseases() -> list[str]:
    """Return a de-duplicated, sorted list of every disease conclusion."""
    return sorted({rule["conclusion"] for rule in RULES})


def get_rules_for_disease(disease: str) -> list[dict]:
    """Return all rules whose conclusion matches *disease*."""
    return [r for r in RULES if r["conclusion"] == disease]


def add_rule(rule: dict) -> None:
    """
    Dynamically add a new rule at runtime.
    Validates required fields before appending.
    """
    required = {"id", "conditions", "conclusion", "cf", "description"}
    missing = required - rule.keys()
    if missing:
        raise ValueError(f"Rule is missing required fields: {missing}")
    if not (0.0 <= rule["cf"] <= 1.0):
        raise ValueError("Certainty factor 'cf' must be between 0.0 and 1.0")
    if any(r["id"] == rule["id"] for r in RULES):
        raise ValueError(f"A rule with id '{rule['id']}' already exists")
    RULES.append(rule)
