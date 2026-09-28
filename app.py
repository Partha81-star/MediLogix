"""
app.py  —  MediLogix Streamlit UI
===================================
Two modes:
  • Forward Chaining  — user selects symptoms → engine derives diagnoses
  • Backward Chaining — user picks a hypothesis → engine asks yes/no questions
"""

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

from knowledge_base import (
    RULES, get_all_symptoms, get_all_diseases, add_rule
)
from working_memory import WorkingMemory
from inference_engine import forward_chain, backward_chain
from explanation import Tracer
from ml_predictor import (
    get_ml_predictor, evaluate_partial_rules, parse_freeform_symptoms
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediLogix — Hybrid AI Medical Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #6366f1, #0d9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        text-align: center;
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Cards */
    .diag-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .diag-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.5rem;
    }
    .cf-bar-wrap { display: flex; align-items: center; gap: 0.8rem; }
    .cf-bar-bg {
        flex: 1;
        height: 12px;
        background: #e2e8f0;
        border-radius: 99px;
        overflow: hidden;
    }
    .cf-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #3b82f6, #10b981);
    }
    .cf-pct { color: #2563eb; font-weight: 700; font-size: 1.05rem; min-width: 48px; }

    /* Question card (backward chaining) */
    .question-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #10b981;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    .question-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a !important;
        margin-bottom: 1rem;
    }

    /* Verdict banners */
    .verdict-confirmed {
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        border-left: 5px solid #059669;
        border-radius: 12px;
        padding: 1.1rem 1.5rem;
        color: #047857;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
    }
    .verdict-rejected {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-left: 5px solid #dc2626;
        border-radius: 12px;
        padding: 1.1rem 1.5rem;
        color: #b91c1c;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
    }

    /* Trace box */
    .trace-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #f1f5f9;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-left: 5px solid #d97706;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #92400e;
        font-size: 0.88rem;
        margin-top: 1.5rem;
        line-height: 1.5;
    }

    /* ML Badges & Pills */
    .ml-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 99px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem 0.2rem;
    }
    .ml-tag-match {
        background: #dcfce7;
        color: #166534 !important;
        border: 1px solid #86efac;
    }
    .ml-tag-missing {
        background: #fee2e2;
        color: #991b1b !important;
        border: 1px solid #fca5a5;
    }
    .ml-notice {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.4rem;
        color: #1e3a8a !important;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
    }
    .ml-notice strong {
        color: #1d4ed8 !important;
        font-weight: 700;
    }
    .ml-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .ml-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .selected-banner {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        margin: 0.8rem 0 1.2rem 0;
        color: #0f172a;
    }

    /* Trace Step Cards & Visual Explanation */
    .trace-step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
    }
    .trace-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.6rem;
    }
    .trace-step-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .trace-step-num {
        background: #0f172a;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .trace-rule-id {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
    }
    .trace-mode-tag {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        background: #f1f5f9;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    .trace-cf-badge {
        font-size: 0.9rem;
        font-weight: 700;
        color: #2563eb;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 0.25rem 0.65rem;
        border-radius: 99px;
    }
    .trace-clause {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .clause-label {
        font-size: 0.75rem;
        font-weight: 800;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        min-width: 44px;
        text-align: center;
        letter-spacing: 0.05em;
    }
    .clause-if {
        background: #e0e7ff;
        color: #3730a3;
    }
    .clause-then {
        background: #dcfce7;
        color: #166534;
    }
    .clause-content {
        flex: 1;
    }
    .trace-footer {
        margin-top: 0.9rem;
        padding-top: 0.7rem;
        border-top: 1px solid #f1f5f9;
    }

    /* Verdict Hero Cards */
    .verdict-confirmed-hero {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        border-left: 6px solid #059669;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.08);
    }
    .verdict-rejected-hero {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Session state keys ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "bc_step": 0,
        "bc_wm": {},
        "bc_questions": [],   # list of (symptom, answer) pairs
        "bc_done": False,
        "bc_result": None,
        "bc_tracer": None,
        "bc_goal": None,
        "bc_all_rules": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Nice display name helper (global) ──────────────────────────────────────────
def fmt(s: str) -> str:
    return s.replace("_", " ").title()

# ── Visual Trace / Explanation Renderer ───────────────────────────────────────
def render_visual_trace(tracer: Tracer):
    entries = tracer.entries()
    if not entries:
        st.info("No rules fired — no explanation trace available.")
        return

    st.markdown("#### 🧭 Step-by-Step Rule Inference Chain")
    for entry in entries:
        rule_cf_pct = int(entry.cf * 100)
        comb_cf_pct = int(entry.combined_cf * 100)
        mode_badge = "🟢 Backward Chaining" if entry.mode == "backward" else "🔵 Forward Chaining"

        cond_pills = " ".join([
            f"<span class='ml-tag ml-tag-match'>✓ {fmt(c)}</span>"
            for c in entry.matched_conditions
        ])

        note_html = f"<div style='font-size:0.83rem; color:#64748b; margin-top:0.4rem;'>💡 <em>{entry.note}</em></div>" if entry.note else ""

        st.markdown(f"""
<div class="trace-step-card">
  <div class="trace-header">
    <div class="trace-step-title">
      <span class="trace-step-num">Step {entry.step}</span>
      <span class="trace-rule-id">Rule {entry.rule_id}</span>
      <span class="trace-mode-tag">{mode_badge}</span>
    </div>
    <div class="trace-cf-badge">{rule_cf_pct}% Rule CF</div>
  </div>
  
  <div class="trace-body">
    <div class="trace-clause">
      <span class="clause-label clause-if">IF</span>
      <div class="clause-content">{cond_pills}</div>
    </div>
    
    <div class="trace-clause" style="margin-top: 0.6rem;">
      <span class="clause-label clause-then">THEN</span>
      <div class="clause-content">
        <strong style="color: #0f172a; font-size: 1.05rem;">{fmt(entry.conclusion)}</strong>
        <span style="color: #64748b; font-size: 0.88rem; margin-left: 0.4rem;">(Derived Fact)</span>
      </div>
    </div>
  </div>

  <div class="trace-footer">
    <div style="font-size: 0.88rem; color: #475569; display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
      <span><strong>MYCIN Running Combined Certainty:</strong></span>
      <span style="font-weight: 700; color: #2563eb;">{comb_cf_pct}%</span>
    </div>
    <div class="cf-bar-bg" style="height: 8px;">
      <div class="cf-bar-fill" style="width: {comb_cf_pct}%;"></div>
    </div>
    {note_html}
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📄 View Raw Terminal Console Output", expanded=False):
        st.markdown(f'<div class="trace-box">{tracer.explain()}</div>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 MediLogix</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Rule-Based Medical Diagnosis Expert System · '
    'Forward &amp; Backward Chaining · MYCIN Certainty Factors</div>',
    unsafe_allow_html=True,
)

# ── Sidebar: mode selector ─────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/medical-doctor.png", width=80
    )
    st.markdown("## ⚙️ MediLogix")
    mode = st.radio(
        "Select inference mode",
        ["🔵 Forward Chaining", "🟢 Backward Chaining", "🧠 ML & Random Symptom Predictor"],
        help="Forward = you provide symptoms, engine derives diseases.\n"
             "Backward = you pick a disease, engine asks what it needs to know.\n"
             "ML Predictor = predicts diseases from ANY random or partial symptom combination.",
    )
    st.markdown("---")
    with st.expander("📖 About the modes"):
        st.markdown("""
**Forward Chaining** (data-driven)
Start with known symptoms. Fires rules whose conditions are satisfied. If only partial symptoms are provided, intelligent ML fallback analyzes the presentation.

**Backward Chaining** (goal-driven)
Start with a hypothesis (disease). Evaluates whether conditions hold by asking targeted questions.

**Machine Learning Predictor** (statistical & random symptom AI)
Trained on clinical symptom profiles. Handles random, partial, noisy, or free-form natural language symptoms.
        """)
    st.markdown("---")
    with st.expander("➕ Add a new rule"):
        new_id   = st.text_input("Rule ID (e.g. R21)", key="nr_id")
        new_cond = st.text_input("Conditions (comma-separated symptoms)", key="nr_cond")
        new_conc = st.text_input("Conclusion (disease name)", key="nr_conc")
        new_cf   = st.slider("Certainty factor", 0.0, 1.0, 0.8, 0.05, key="nr_cf")
        new_desc = st.text_input("Description", key="nr_desc")
        if st.button("Add Rule", key="btn_add_rule"):
            try:
                conditions = [c.strip().replace(" ", "_") for c in new_cond.split(",") if c.strip()]
                add_rule({
                    "id": new_id.strip(),
                    "conditions": conditions,
                    "conclusion": new_conc.strip().replace(" ", "_"),
                    "cf": new_cf,
                    "description": new_desc.strip(),
                })
                st.success(f"Rule {new_id} added!")
            except ValueError as e:
                st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — FORWARD CHAINING
# ══════════════════════════════════════════════════════════════════════════════
if "Forward" in mode:
    st.markdown("## 🔵 Forward Chaining — Symptom Checker")
    st.markdown(
        "Select your symptoms from the clinical checklist below. "
        "The engine applies expert rules with MYCIN certainty factors, backed by "
        "an intelligent **Machine Learning fallback** to accurately diagnose even partial or random symptom combinations."
    )

    all_symptoms = get_all_symptoms()

    # Clinical categories for clean checkbox organization
    SYMPTOM_CATEGORIES = {
        "🌡️ Fever & General Vitals": [
            "fever", "high_fever", "mild_fever", "cyclical_fever", "prolonged_fever",
            "chills", "sweating", "fatigue", "weakness", "body_ache", "muscle_pain", "loss_of_appetite"
        ],
        "🫁 Respiratory & ENT": [
            "cough", "productive_cough", "sore_throat", "runny_nose", "sneezing",
            "shortness_of_breath", "wheezing", "chest_tightness", "loss_of_taste", "loss_of_smell"
        ],
        "🧠 Neurological & Sensory": [
            "headache", "severe_headache", "dizziness", "blurred_vision",
            "eye_pain", "light_sensitivity", "sound_sensitivity"
        ],
        "🫄 Gastrointestinal & Abdominal": [
            "nausea", "vomiting", "diarrhea", "abdominal_pain", "abdominal_cramps"
        ],
        "🩹 Cardio, Urinary & Skin": [
            "chest_pain", "burning_urination", "frequent_urination", "lower_abdominal_pain",
            "excessive_thirst", "joint_pain", "rash"
        ]
    }

    # Summary of currently checked symptoms & Clear button
    active_selected = [s for s in all_symptoms if st.session_state.get(f"sym_{s}", False)]

    col_sum, col_clr = st.columns([4, 1])
    with col_sum:
        if active_selected:
            badges = " ".join([f"<span class='ml-tag ml-tag-match'>✓ {fmt(s)}</span>" for s in active_selected])
            st.markdown(f"<div class='selected-banner'><strong>Selected Symptoms ({len(active_selected)}):</strong> {badges}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='selected-banner' style='color:#64748b;'><em>No symptoms selected yet. Check symptoms from the categories below:</em></div>", unsafe_allow_html=True)
    with col_clr:
        if st.button("🔄 Clear All", key="btn_clear_symptoms", use_container_width=True):
            for s in all_symptoms:
                st.session_state[f"sym_{s}"] = False
            st.rerun()

    # Render checkbox categories
    selected: set[str] = set()
    for cat_title, cat_syms in SYMPTOM_CATEGORIES.items():
        with st.expander(f"{cat_title} ({len(cat_syms)} symptoms)", expanded=True):
            cols = st.columns(3)
            for i, sym in enumerate(cat_syms):
                col = cols[i % 3]
                if col.checkbox(fmt(sym), key=f"sym_{sym}"):
                    selected.add(sym)

    st.markdown("---")
    run_fc = st.button("🔍 Diagnose (Forward Chaining + ML)", key="btn_fc", type="primary")

    if run_fc:
        if not selected:
            st.warning("⚠️ Please select at least one symptom from the checkboxes above.")
        else:
            wm = WorkingMemory()
            for s in selected:
                wm.assert_fact(s, True)
            tracer = Tracer()
            conclusions = forward_chain(wm, tracer)

            # If strict rules fired 100%:
            if conclusions:
                st.success(
                    f"✅ Forward chaining rule engine derived **{len(conclusions)}** "
                    f"diagnosis/diagnoses with 100% satisfied rule conditions."
                )
                st.markdown("### 🩻 Confirmed Diagnoses (Rule-Based)")
                for disease, cf in sorted(conclusions.items(), key=lambda x: -x[1]):
                    pct = int(cf * 100)
                    bar_w = pct
                    st.markdown(f"""
<div class="diag-card">
  <div class="diag-title">{'🔴' if cf >= 0.8 else '🟡'} {fmt(disease)}</div>
  <div class="cf-bar-wrap">
    <div class="cf-bar-bg"><div class="cf-bar-fill" style="width:{bar_w}%"></div></div>
    <span class="cf-pct">{pct}%</span>
  </div>
  <small style="color:#64748b; font-weight:500;">Certainty Factor (MYCIN-combined)</small>
</div>
""", unsafe_allow_html=True)

                # Trace
                with st.expander("🔎 Why? — Full Reasoning Trace"):
                    render_visual_trace(tracer)

                # Rule-network diagram
                with st.expander("🌐 Rule Network Diagram"):
                    fired_entries = tracer.entries()
                    G = nx.DiGraph()
                    for entry in fired_entries:
                        for cond in entry.matched_conditions:
                            G.add_edge(fmt(cond), fmt(entry.conclusion), label=entry.rule_id)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#f8fafc")

                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except Exception:
                        pos = nx.spring_layout(G, seed=42)

                    symptom_nodes = [n for n in G.nodes() if n.lower() in
                                     [s.replace("_"," ").lower() for s in get_all_symptoms()]]
                    disease_nodes = [n for n in G.nodes() if n not in symptom_nodes]

                    nx.draw_networkx_nodes(G, pos, nodelist=symptom_nodes,
                                           node_color="#7c3aed", node_size=700, ax=ax, alpha=0.9)
                    nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes,
                                           node_color="#2563eb", node_size=900, ax=ax, alpha=0.9)
                    nx.draw_networkx_edges(G, pos, edge_color="#60a5fa",
                                           arrows=True, arrowsize=18, ax=ax, alpha=0.75)
                    nx.draw_networkx_labels(G, pos, font_color="#0f172a",
                                            font_size=8, font_weight="bold", ax=ax)

                    legend = [
                        mpatches.Patch(color="#7c3aed", label="Symptom"),
                        mpatches.Patch(color="#2563eb", label="Diagnosis"),
                    ]
                    ax.legend(handles=legend, facecolor="#ffffff",
                              labelcolor="#0f172a", edgecolor="#cbd5e1", loc="upper left")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                # Comparative ML analysis
                with st.expander("🤖 Machine Learning Model Probabilities (Cross-Validation)"):
                    ml_preds = get_ml_predictor().predict(selected, top_k=5)
                    for pred in ml_preds:
                        pct = int(pred["probability"] * 100)
                        st.markdown(f"**{fmt(pred['disease'])}**: {pct}% likelihood")

            # If NO strict rules fired 100% (e.g. fever + cough):
            else:
                st.markdown("""
<div class="ml-notice">
  <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem; color: #1e40af;">
    ℹ️ Note on Strict Rules
  </div>
  <div style="color: #1e3a8a; margin-bottom: 0.6rem; font-size: 0.95rem;">
    No single rule in the database matched 100% of its conditions with only the selected symptoms.
  </div>
  <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem; color: #1e40af;">
    🤖 Machine Learning &amp; Probabilistic Analysis Activated
  </div>
  <div style="color: #1e3a8a; font-size: 0.95rem;">
    In clinical practice, patients often present with partial or random subsets of symptoms. The Machine Learning classifier and partial-evidence engine evaluated your symptoms:
  </div>
</div>
""", unsafe_allow_html=True)

                ml_preds = get_ml_predictor().predict(selected, top_k=4)
                partial_rules = evaluate_partial_rules(selected)

                st.markdown("### 🤖 Probabilistic & ML Diagnoses")
                for pred in ml_preds:
                    pct = int(pred["probability"] * 100)
                    disease = pred["disease"]
                    matched_items = pred["matched_symptoms"]
                    matched_str = " · ".join(fmt(s) for s in matched_items) if matched_items else "General presentation"

                    p_info = next((p for p in partial_rules if p["disease"] == disease), None)
                    missing_str = " · ".join(fmt(s) for s in p_info["missing_conditions"]) if p_info and p_info["missing_conditions"] else "None identified"

                    st.markdown(f"""
<div class="ml-card">
  <div class="ml-title">
    <span style="color:#0f172a;">{'🔴' if pct >= 60 else '🟡'} {fmt(disease)}</span>
    <span style="font-size:1.15rem; color:#2563eb; font-weight:800;">{pct}% match</span>
  </div>
  <div class="cf-bar-wrap" style="margin: 0.6rem 0;">
    <div class="cf-bar-bg"><div class="cf-bar-fill" style="width:{pct}%"></div></div>
  </div>
  <div style="font-size:0.92rem; margin-top:0.5rem; color:#1e293b;">
    <strong>Matched Symptoms:</strong> <span class="ml-tag ml-tag-match">{matched_str}</span>
  </div>
  <div style="font-size:0.92rem; margin-top:0.35rem; color:#1e293b;">
    <strong>Key Missing Symptoms to Confirm:</strong> <span class="ml-tag ml-tag-missing">{missing_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                with st.expander("🔎 Why? — Partial Rule & Feature Analysis"):
                    st.markdown("""
**How MediLogix handled this:**
1. **Rule Engine:** Scanned all 20 rules. None had 100% of their conditions satisfied by the selected symptoms.
2. **Partial Evidence Matching:** Found rules where a subset of symptoms matched and computed coverage.
3. **Machine Learning Classifier:** Calculated posterior probabilities using the trained Random Forest and Logistic Regression model over clinical symptom patterns.
                    """)
                    if partial_rules:
                        st.markdown("##### Partial Rule Matches:")
                        for p in partial_rules[:5]:
                            st.markdown(
                                f"- **{fmt(p['disease'])}** ({int(p['match_score']*100)}% partial score) via rule `{p['best_rule_id']}`: "
                                f"Matched: *{', '.join(fmt(m) for m in p['matched_conditions'])}* | Missing: *{', '.join(fmt(m) for m in p['missing_conditions'])}*"
                            )

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — BACKWARD CHAINING
# ══════════════════════════════════════════════════════════════════════════════
elif "Backward" in mode:
    st.markdown("## 🟢 Backward Chaining — Hypothesis Tester")
    st.markdown(
        "Select a disease you want to investigate. The system will ask **only** "
        "the yes/no questions it actually needs to prove or disprove your "
        "hypothesis — in the order backward chaining discovers it needs them."
    )

    all_diseases = get_all_diseases()
    goal = st.selectbox(
        "Select disease hypothesis",
        options=all_diseases,
        format_func=fmt,
        key="bc_disease_select",
    )

    # Reset session if goal changed
    if st.session_state.bc_goal != goal:
        st.session_state.bc_goal     = goal
        st.session_state.bc_step     = 0
        st.session_state.bc_wm       = {}
        st.session_state.bc_questions= []
        st.session_state.bc_done     = False
        st.session_state.bc_result   = None
        st.session_state.bc_tracer   = None
        st.session_state.bc_all_rules= None

    if st.button("🚀 Start / Restart Investigation", key="btn_bc_start"):
        st.session_state.bc_step      = 0
        st.session_state.bc_wm        = {}
        st.session_state.bc_questions = []
        st.session_state.bc_done      = False
        st.session_state.bc_result    = None
        st.session_state.bc_tracer    = None

    # ── Stateful Q&A loop ─────────────────────────────────────────────────────
    # We run backward chaining up to the next "unknown" symptom each time,
    # show a question card, accept the user's answer, then continue.

    if not st.session_state.bc_done and st.session_state.bc_goal:
        # Collect answers so far
        wm_snapshot = dict(st.session_state.bc_wm)

        # We use a special ask_fn that raises a sentinel when a question is needed
        class _NeedInput(Exception):
            def __init__(self, symptom: str):
                self.symptom = symptom

        questions_answered = list(st.session_state.bc_questions)

        # Show previously answered questions (read-only)
        if questions_answered:
            st.markdown("#### ✅ Questions answered so far")
            for sym, ans in questions_answered:
                icon = "✔️" if ans else "✖️"
                st.markdown(f"- {icon} **{fmt(sym)}**: {'Yes' if ans else 'No'}")
            st.markdown("---")

        # Try to run the engine with current WM
        def ask_fn_stateful(symptom: str) -> bool:
            if symptom in wm_snapshot:
                return wm_snapshot[symptom]
            raise _NeedInput(symptom)

        wm_obj = WorkingMemory(initial_facts=wm_snapshot)
        tracer  = Tracer()

        try:
            result = backward_chain(
                goal=st.session_state.bc_goal,
                wm=wm_obj,
                tracer=tracer,
                ask_fn=ask_fn_stateful,
            )
            # If we reach here, no more questions — done!
            st.session_state.bc_done   = True
            st.session_state.bc_result = result
            st.session_state.bc_tracer = tracer
            st.session_state.bc_wm     = wm_obj.all_facts()

        except _NeedInput as e:
            symptom_needed = e.symptom
            # Show the question card
            st.markdown("#### ❓ Question")
            st.markdown(f"""
<div class="question-card">
  <div class="question-text">Do you have / experience <strong>{fmt(symptom_needed)}</strong>?</div>
</div>
""", unsafe_allow_html=True)
            col_yes, col_no = st.columns(2)
            if col_yes.button("✅ Yes", key=f"yes_{symptom_needed}_{len(questions_answered)}"):
                st.session_state.bc_wm[symptom_needed] = True
                st.session_state.bc_questions.append((symptom_needed, True))
                st.rerun()
            if col_no.button("❌ No", key=f"no_{symptom_needed}_{len(questions_answered)}"):
                st.session_state.bc_wm[symptom_needed] = False
                st.session_state.bc_questions.append((symptom_needed, False))
                st.rerun()

    # ── Show final verdict ─────────────────────────────────────────────────────
    if st.session_state.bc_done:
        result = st.session_state.bc_result
        tracer = st.session_state.bc_tracer
        disease_name = fmt(st.session_state.bc_goal)

        if result:
            cf = tracer.combined_cf(st.session_state.bc_goal)
            pct = int(cf * 100)
            st.markdown(f"""
<div class="verdict-confirmed-hero">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
    <div>
      <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #059669; margin-bottom: 0.2rem;">
        Diagnostic Hypothesis Confirmed
      </div>
      <div style="font-size: 1.8rem; font-weight: 800; color: #065f46; display: flex; align-items: center; gap: 0.5rem;">
        <span>🎯 {disease_name}</span>
      </div>
      <div style="font-size: 0.95rem; color: #047857; margin-top: 0.2rem;">
        Patient answers satisfy all diagnostic conditions required for this diagnosis.
      </div>
    </div>
    <div style="background: #ffffff; border: 2px solid #a7f3d0; border-radius: 14px; padding: 0.8rem 1.4rem; text-align: center; box-shadow: 0 4px 12px rgba(5, 150, 105, 0.1);">
      <div style="font-size: 0.8rem; font-weight: 600; color: #059669; text-transform: uppercase;">Certainty Factor</div>
      <div style="font-size: 2.2rem; font-weight: 900; color: #047857; line-height: 1.1;">{pct}%</div>
      <div style="font-size: 0.78rem; font-weight: 600; color: #10b981;">MYCIN-Combined</div>
    </div>
  </div>
  <div class="cf-bar-wrap" style="margin-top: 1rem;">
    <div class="cf-bar-bg" style="height: 12px; background: #d1fae5;">
      <div class="cf-bar-fill" style="width: {pct}%; background: linear-gradient(90deg, #10b981, #059669);"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="verdict-rejected-hero">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
    <div>
      <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #dc2626; margin-bottom: 0.2rem;">
        Diagnostic Hypothesis Rejected
      </div>
      <div style="font-size: 1.8rem; font-weight: 800; color: #991b1b; display: flex; align-items: center; gap: 0.5rem;">
        <span>❌ {disease_name}</span>
      </div>
      <div style="font-size: 0.95rem; color: #b91c1c; margin-top: 0.2rem;">
        Essential diagnostic symptoms were negated or insufficient to establish this diagnosis.
      </div>
    </div>
    <div style="background: #ffffff; border: 2px solid #fecaca; border-radius: 14px; padding: 0.8rem 1.4rem; text-align: center; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.1);">
      <div style="font-size: 0.8rem; font-weight: 600; color: #dc2626; text-transform: uppercase;">Outcome</div>
      <div style="font-size: 1.8rem; font-weight: 900; color: #991b1b; line-height: 1.1;">Not Met</div>
      <div style="font-size: 0.78rem; font-weight: 600; color: #ef4444;">Conditions Failed</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Answers summary
        if st.session_state.bc_questions:
            with st.expander("📋 All Collected Answers Summary", expanded=True):
                col_y, col_n = st.columns(2)
                yes_answers = [s for s, a in st.session_state.bc_questions if a]
                no_answers = [s for s, a in st.session_state.bc_questions if not a]

                with col_y:
                    st.markdown("**Confirmed Symptoms (Yes):**")
                    if yes_answers:
                        for s in yes_answers:
                            st.markdown(f"<span class='ml-tag ml-tag-match' style='font-size:0.88rem; padding:0.3rem 0.7rem;'>✔️ {fmt(s)}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("None confirmed")

                with col_n:
                    st.markdown("**Denied Symptoms (No):**")
                    if no_answers:
                        for s in no_answers:
                            st.markdown(f"<span class='ml-tag ml-tag-missing' style='font-size:0.88rem; padding:0.3rem 0.7rem;'>✖️ {fmt(s)}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("None denied")

        # Trace
        with st.expander("🔎 Why? — Full Reasoning Trace", expanded=True):
            render_visual_trace(tracer)

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 3 — MACHINE LEARNING & RANDOM SYMPTOM PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🧠 Machine Learning & Random Symptom Predictor")
    st.markdown(
        "Unlike strict rule engines that require 100% of predefined conditions to match, "
        "this **Machine Learning AI** predicts diseases from **any random combination of symptoms**, "
        "including single symptoms, noisy descriptions, and natural language text."
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### 📝 Enter or Describe Symptoms")
        ml_text = st.text_area(
            "Natural Language / Random Symptoms Input:",
            value="I am experiencing fever and continuous cough with shivering",
            height=110,
            help="Type any clinical symptoms or describe what you feel in plain English.",
        )
        parsed_from_text = parse_freeform_symptoms(ml_text) if ml_text else []
        if parsed_from_text:
            st.markdown(
                "**NLP Extracted Symptoms:** " +
                " ".join([f"<span class='ml-tag ml-tag-match'>✓ {fmt(s)}</span>" for s in parsed_from_text]),
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("### 🎯 Add or Fine-Tune Symptoms")
        all_syms = get_all_symptoms()
        selected_multi = st.multiselect(
            "Select or add symptoms from database:",
            options=all_syms,
            default=[s for s in parsed_from_text if s in all_syms],
            format_func=fmt,
        )

    # Quick presets
    st.markdown("##### 🎲 Quick Test Presets:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    active_symptoms = set(selected_multi).union(set(parsed_from_text))

    if col_p1.button("🎲 Fever + Cough"):
        active_symptoms = {"fever", "cough"}
    if col_p2.button("🎲 Stomach + Diarrhea"):
        active_symptoms = {"diarrhea", "abdominal_cramps", "nausea"}
    if col_p3.button("🎲 Headache + Blurry Vision"):
        active_symptoms = {"headache", "dizziness", "blurred_vision"}
    if col_p4.button("🎲 Wheezing + Short Breath"):
        active_symptoms = {"wheezing", "shortness_of_breath"}

    st.markdown("---")
    predict_btn = st.button("⚡ Run Machine Learning Prediction", key="btn_run_ml")

    if predict_btn or active_symptoms:
        if not active_symptoms:
            st.info("💡 Please type or select at least one symptom above.")
        else:
            predictor = get_ml_predictor()
            preds = predictor.predict(active_symptoms, top_k=6)
            partial_rules = evaluate_partial_rules(active_symptoms)

            st.markdown(f"### 📊 Predicted Diseases for: *{', '.join(fmt(s) for s in sorted(active_symptoms))}*")

            # Probability chart
            chart_col, cards_col = st.columns([1.1, 1.3])

            with chart_col:
                st.markdown("#### 📈 Probability Distribution")
                df_preds = pd.DataFrame([
                    {"Disease": fmt(p["disease"]), "Probability": p["probability"] * 100}
                    for p in reversed(preds)
                ])
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#f8fafc")
                bars = ax.barh(df_preds["Disease"], df_preds["Probability"], color="#3b82f6", edgecolor="#1d4ed8")
                ax.set_xlim(0, 100)
                ax.set_xlabel("Confidence (%)", color="#0f172a", fontweight="bold")
                ax.tick_params(colors="#0f172a")
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{int(w)}%", va="center", color="#047857", fontsize=9, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_color("#cbd5e1")
                st.pyplot(fig)
                plt.close(fig)

            with cards_col:
                st.markdown("#### 🩺 Differential Diagnosis Breakdown")
                for p in preds[:4]:
                    pct = int(p["probability"] * 100)
                    dis = p["disease"]
                    matched_s = p["matched_symptoms"]
                    matched_str = " · ".join(fmt(s) for s in matched_s) if matched_s else "Pattern match"

                    p_rule = next((r for r in partial_rules if r["disease"] == dis), None)
                    missing_str = " · ".join(fmt(s) for s in p_rule["missing_conditions"]) if p_rule and p_rule["missing_conditions"] else "None"

                    st.markdown(f"""
<div class="ml-card">
  <div class="ml-title">
    <span style="color:#0f172a;">{'🔴' if pct >= 60 else '🟡'} {fmt(dis)}</span>
    <span style="font-size:1.15rem; color:#2563eb; font-weight:800;">{pct}% Likelihood</span>
  </div>
  <div class="cf-bar-wrap" style="margin: 0.5rem 0;">
    <div class="cf-bar-bg"><div class="cf-bar-fill" style="width:{pct}%"></div></div>
  </div>
  <div style="font-size:0.92rem; margin-top:0.45rem; color:#1e293b;">
    <strong>Active Evidence:</strong> <span class="ml-tag ml-tag-match">{matched_str}</span>
  </div>
  <div style="font-size:0.92rem; margin-top:0.3rem; color:#1e293b;">
    <strong>Missing Clinical Factors:</strong> <span class="ml-tag ml-tag-missing">{missing_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Knowledge base browser ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Browse Knowledge Base"):
    rows = []
    for r in RULES:
        rows.append({
            "ID": r["id"],
            "Conditions": ", ".join(fmt(c) for c in r["conditions"]),
            "Conclusion": fmt(r["conclusion"]),
            "CF": f"{int(r['cf']*100)}%",
            "Description": r["description"],
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "CF": st.column_config.ProgressColumn(
                "Certainty", format="%d%%", min_value=0, max_value=100
            )
        },
    )

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Educational Disclaimer</strong> — MediLogix is a <em>demonstration</em>
  of rule-based expert-system AI techniques created for a college AI course.
  It is <strong>NOT</strong> a real medical diagnostic tool and does
  <strong>NOT</strong> constitute medical advice. Always consult a qualified
  healthcare professional for any medical concerns.
</div>
""", unsafe_allow_html=True)
