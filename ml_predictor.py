"""
ml_predictor.py
===============
Machine Learning and Probabilistic Inference for MediLogix.

Handles:
  1. Partial / Fuzzy rule-matching when exact 100% rule conditions aren't satisfied.
  2. Machine Learning Disease Classifier (Random Forest + Logistic Regression) trained
     on clinical symptom profiles, capable of predicting from ANY arbitrary or random
     symptom combination.
  3. Free-form / Random symptom NLP parser that resolves arbitrary user text, typos,
     and colloquial expressions into recognized symptoms.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import TypedDict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from knowledge_base import RULES, get_all_symptoms, get_all_diseases


# ── Natural Language Synonym / Alias Map ──────────────────────────────────────
# Allows arbitrary user descriptions or random input to resolve to known symptoms
SYMPTOM_ALIASES: dict[str, list[str]] = {
    "fever": ["high temp", "temperature", "febrile", "hot", "pyrexia", "feverish"],
    "high_fever": ["very hot", "burning up", "extreme fever", "severe fever", "103", "104"],
    "mild_fever": ["low fever", "warm", "slight fever", "subfebrile", "99", "100"],
    "cyclical_fever": ["intermittent fever", "periodic fever", "fever comes and goes"],
    "prolonged_fever": ["fever for days", "fever for weeks", "persistent fever", "continuous fever"],
    "cough": ["coughing", "dry cough", "hacking cough", "tussis"],
    "productive_cough": ["wet cough", "phlegm", "mucus cough", "sputum"],
    "sore_throat": ["throat pain", "scratchy throat", "hurts to swallow", "pharyngitis"],
    "body_ache": ["body pain", "aching all over", "generalized pain", "myalgia"],
    "muscle_pain": ["sore muscles", "muscle ache", "muscle cramps", "cramps"],
    "joint_pain": ["achy joints", "arthralgia", "knee pain", "elbow pain", "stiff joints"],
    "headache": ["head pain", "cephalea", "heavy head"],
    "severe_headache": ["terrible headache", "splitting headache", "throbbing head", "worst headache"],
    "fatigue": ["tired", "exhaustion", "lethargy", "sleepy", "drained", "worn out", "no energy"],
    "weakness": ["feeling weak", "feeble", "asthenia", "debility"],
    "chills": ["shivering", "feeling cold", "shakes", "rigors"],
    "sweating": ["night sweats", "perspiration", "diaphoresis", "sweaty"],
    "nausea": ["queasy", "feel like throwing up", "sick to stomach"],
    "vomiting": ["throwing up", "puking", "emesis"],
    "diarrhea": ["loose motions", "loose stools", "watery stools", "upset tummy"],
    "abdominal_pain": ["stomach pain", "belly ache", "tummy ache", "gut pain"],
    "abdominal_cramps": ["stomach cramps", "gut cramps", "colic"],
    "lower_abdominal_pain": ["pelvic pain", "lower tummy pain", "bladder pain"],
    "runny_nose": ["rhinorrhea", "watery nose", "nasal discharge", "running nose", "stuffy nose"],
    "sneezing": ["sneeze", "sternutation"],
    "loss_of_taste": ["ageusia", "cant taste", "no taste", "food tastes bland"],
    "loss_of_smell": ["anosmia", "cant smell", "no smell"],
    "shortness_of_breath": ["breathless", "dyspnea", "trouble breathing", "out of breath", "gasping"],
    "chest_pain": ["pain in chest", "angina", "hurts to breathe"],
    "chest_tightness": ["tight chest", "heavy chest", "constriction in chest"],
    "wheezing": ["whistling breath", "noisy breathing", "stridor"],
    "dizziness": ["dizzy", "lightheaded", "vertigo", "spinning"],
    "blurred_vision": ["blurry eyes", "cant see clear", "fuzzy vision"],
    "light_sensitivity": ["photophobia", "eyes hurt with light", "bright light hurts"],
    "sound_sensitivity": ["phonophobia", "sensitive to noise", "loud sounds hurt"],
    "excessive_thirst": ["polydipsia", "always thirsty", "dry mouth"],
    "frequent_urination": ["peeing often", "polyuria", "need to pee constantly"],
    "burning_urination": ["dysuria", "hurts when peeing", "stinging urine", "burning pee"],
    "loss_of_appetite": ["anorexia", "not hungry", "dont want to eat", "no appetite"],
    "rash": ["skin rash", "spots", "bumps", "hives", "red spots", "eruption"],
    "eye_pain": ["pain behind eyes", "retro-orbital pain", "aching eyes"],
}


def parse_freeform_symptoms(text: str) -> list[str]:
    """
    Extract recognized symptoms from arbitrary free-form user text
    using token matching, aliases, and fuzzy similarity.
    """
    if not text:
        return []

    text_lower = text.lower().strip()
    # Normalize punctuation
    normalized_text = re.sub(r"[,;.\n/]+", " ", text_lower)
    words = normalized_text.split()
    detected: set[str] = set()
    all_syms = get_all_symptoms()

    # 1. Exact alias/key phrase matching
    for sym, aliases in SYMPTOM_ALIASES.items():
        # Check canonical symptom string (e.g. "shortness of breath")
        canonical_readable = sym.replace("_", " ")
        if re.search(r"\b" + re.escape(canonical_readable) + r"\b", normalized_text):
            detected.add(sym)
            continue

        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", normalized_text):
                detected.add(sym)
                break

    # 2. Check each recognized symptom directly
    for sym in all_syms:
        sym_clean = sym.replace("_", " ")
        if sym_clean in normalized_text:
            detected.add(sym)

    # 3. Fuzzy single-token or bigram check for any remaining words
    for i in range(len(words)):
        token = words[i]
        if len(token) >= 4:
            for sym in all_syms:
                if SequenceMatcher(None, token, sym.replace("_", "")).ratio() > 0.85:
                    detected.add(sym)
        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i+1]}"
            for sym in all_syms:
                if SequenceMatcher(None, bigram, sym.replace("_", " ")).ratio() > 0.85:
                    detected.add(sym)

    return sorted(detected)


# ── Partial Rule Matcher (Fuzzy Inference) ───────────────────────────────────

class PartialMatchResult(TypedDict):
    disease: str
    match_score: float
    matched_conditions: list[str]
    missing_conditions: list[str]
    best_rule_id: str
    description: str


def evaluate_partial_rules(user_symptoms: set[str] | list[str]) -> list[PartialMatchResult]:
    """
    Evaluates rules even when ONLY A SUBSET of conditions match (e.g. fever + cough).
    Returns ranked list of candidate diseases with match statistics.
    """
    user_set = set(user_symptoms)
    if not user_set:
        return []

    disease_candidates: dict[str, dict] = {}

    for rule in RULES:
        rule_conds = set(rule["conditions"])
        matched = rule_conds.intersection(user_set)
        if not matched:
            continue

        missing = sorted(rule_conds - user_set)
        coverage = len(matched) / len(rule_conds)
        # Weight by rule's original certainty factor
        score = coverage * rule["cf"]

        disease = rule["conclusion"]
        if disease not in disease_candidates or score > disease_candidates[disease]["match_score"]:
            disease_candidates[disease] = {
                "disease": disease,
                "match_score": round(score, 3),
                "matched_conditions": sorted(matched),
                "missing_conditions": missing,
                "best_rule_id": rule["id"],
                "description": rule["description"],
            }

    results = list(disease_candidates.values())
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


# ── Machine Learning Diagnostic Model ────────────────────────────────────────

class MLMedicalModel:
    """
    Supervised Machine Learning classifier that predicts diseases from ANY
    combination of symptoms, including unseen, random, or sparse subsets.
    """
    def __init__(self):
        self.all_symptoms: list[str] = get_all_symptoms()
        self.symptom_to_idx: dict[str, int] = {s: i for i, s in enumerate(self.all_symptoms)}
        self.all_diseases: list[str] = get_all_diseases()
        self.disease_to_idx: dict[str, int] = {d: i for i, d in enumerate(self.all_diseases)}
        self.idx_to_disease: dict[int, str] = {i: d for i, d in enumerate(self.all_diseases)}

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(max_iter=500, random_state=42)
        self._is_trained = False
        self._train()

    def _generate_synthetic_cases(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds a rich training matrix from the medical knowledge base:
        - Full rule profiles
        - Random sub-combinations (simulating patients with only 1, 2, or 3 symptoms)
        - Realistic random noise / co-morbidities
        """
        np.random.seed(42)
        X_list: list[list[int]] = []
        y_list: list[int] = []

        # 1. Base cases from all rules
        for rule in RULES:
            disease_idx = self.disease_to_idx[rule["conclusion"]]
            cond_indices = [self.symptom_to_idx[c] for c in rule["conditions"] if c in self.symptom_to_idx]

            # Canonical complete presentation (repeated for high weight)
            for _ in range(12):
                vec = [0] * len(self.all_symptoms)
                for idx in cond_indices:
                    vec[idx] = 1
                X_list.append(vec)
                y_list.append(disease_idx)

            # Subsets / partial presentations (1, 2, 3 symptoms)
            for _ in range(25):
                vec = [0] * len(self.all_symptoms)
                # Pick a random subset of condition symptoms
                k = np.random.randint(1, len(cond_indices) + 1)
                chosen = np.random.choice(cond_indices, size=k, replace=False)
                for idx in chosen:
                    vec[idx] = 1
                # 10% chance of an irrelevant random background symptom
                if np.random.rand() < 0.15:
                    rand_sym = np.random.randint(0, len(self.all_symptoms))
                    vec[rand_sym] = 1
                X_list.append(vec)
                y_list.append(disease_idx)

        return np.array(X_list), np.array(y_list)

    def _train(self):
        X, y = self._generate_synthetic_cases()
        self.rf_model.fit(X, y)
        self.lr_model.fit(X, y)
        self._is_trained = True

    def vectorize_symptoms(self, symptoms: list[str] | set[str]) -> np.ndarray:
        vec = np.zeros(len(self.all_symptoms), dtype=int)
        for s in symptoms:
            if s in self.symptom_to_idx:
                vec[self.symptom_to_idx[s]] = 1
        return vec.reshape(1, -1)

    def predict(self, symptoms: list[str] | set[str], top_k: int = 5) -> list[dict]:
        """
        Predict probable diseases given any random symptom list.
        Combines Random Forest and Logistic Regression probabilities.
        """
        symptoms_set = set(symptoms)
        if not symptoms_set:
            return []

        # Find which symptoms are directly recognized
        known_symptoms = [s for s in symptoms if s in self.symptom_to_idx]
        if not known_symptoms:
            # Try parsing if freeform strings were passed
            all_parsed = []
            for s in symptoms:
                all_parsed.extend(parse_freeform_symptoms(s))
            known_symptoms = list(set(all_parsed))

        vec = self.vectorize_symptoms(known_symptoms)
        rf_probs = self.rf_model.predict_proba(vec)[0]
        lr_probs = self.lr_model.predict_proba(vec)[0]

        # Ensemble blend
        blended_probs = 0.65 * rf_probs + 0.35 * lr_probs

        # Also blend with partial rule overlap to preserve medical specificity
        partial_matches = {p["disease"]: p["match_score"] for p in evaluate_partial_rules(known_symptoms)}

        results = []
        for idx, base_prob in enumerate(blended_probs):
            disease = self.idx_to_disease[idx]
            rule_boost = partial_matches.get(disease, 0.0)

            # Combined confidence score
            if rule_boost > 0:
                final_prob = 0.5 * base_prob + 0.5 * rule_boost
            else:
                final_prob = 0.7 * base_prob

            # Identify which of the user's symptoms support this disease
            disease_rules = [r for r in RULES if r["conclusion"] == disease]
            disease_symptoms: set[str] = set()
            for r in disease_rules:
                disease_symptoms.update(r["conditions"])

            matched_for_disease = sorted(disease_symptoms.intersection(symptoms_set))
            all_typical = sorted(disease_symptoms)

            results.append({
                "disease": disease,
                "probability": float(np.clip(final_prob, 0.05, 0.98)),
                "matched_symptoms": matched_for_disease,
                "typical_symptoms": all_typical,
                "match_count": len(matched_for_disease),
            })

        # Sort by probability descending
        results.sort(key=lambda x: (x["match_count"] > 0, x["probability"]), reverse=True)
        return results[:top_k]


# Singleton instance
_ml_engine: MLMedicalModel | None = None

def get_ml_predictor() -> MLMedicalModel:
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLMedicalModel()
    return _ml_engine
