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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediLogix — Rule-Based Expert System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .diag-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .diag-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 0.4rem;
    }
    .cf-bar-wrap { display: flex; align-items: center; gap: 0.8rem; }
    .cf-bar-bg {
        flex: 1;
        height: 10px;
        background: rgba(255,255,255,0.12);
        border-radius: 99px;
        overflow: hidden;
    }
    .cf-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
    }
    .cf-pct { color: #60a5fa; font-weight: 600; font-size: 0.95rem; min-width: 42px; }

    /* Question card (backward chaining) */
    .question-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(96,165,250,0.4);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .question-text {
        font-size: 1.1rem;
        font-weight: 500;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Verdict banners */
    .verdict-confirmed {
        background: linear-gradient(90deg, rgba(52,211,153,0.18), rgba(52,211,153,0.06));
        border: 1px solid rgba(52,211,153,0.55);
        border-radius: 14px;
        padding: 1rem 1.4rem;
        color: #34d399;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
    }
    .verdict-rejected {
        background: linear-gradient(90deg, rgba(248,113,113,0.18), rgba(248,113,113,0.06));
        border: 1px solid rgba(248,113,113,0.55);
        border-radius: 14px;
        padding: 1rem 1.4rem;
        color: #f87171;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
    }

    /* Trace box */
    .trace-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #c9d1d9;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Sidebar */
    .css-1d391kg { background: rgba(15,12,41,0.9) !important; }

    /* Disclaimer */
    .disclaimer {
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.4);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        color: #fbbf24;
        font-size: 0.82rem;
        margin-top: 1.5rem;
    }

    div[data-testid="stSidebar"] { background: rgba(15,12,41,0.95); }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.8rem;
        font-weight: 600;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.88; }
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
        ["🔵 Forward Chaining", "🟢 Backward Chaining"],
        help="Forward = you provide symptoms, engine derives diseases.\n"
             "Backward = you pick a disease, engine asks what it needs to know.",
    )
    st.markdown("---")
    with st.expander("📖 About the modes"):
        st.markdown("""
**Forward Chaining** (data-driven)
Start with known facts (symptoms). Repeatedly fire every rule whose
conditions are all true. Keep going until nothing new can be derived.

**Backward Chaining** (goal-driven)
Start with a hypothesis (disease). Check whether the rule's conditions
hold; for any unknown symptom, ask the user a yes/no question. Prove
or disprove the hypothesis.
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
        "Tick the symptoms you are experiencing, then click **Diagnose**. "
        "The engine will fire every applicable rule and show you every "
        "disease it can derive, together with a confidence score."
    )

    all_symptoms = get_all_symptoms()
    # Group into columns for a nicer layout
    n_cols = 3
    cols = st.columns(n_cols)
    selected: set[str] = set()

    # Nice display name: replace underscores with spaces, title-case
    def fmt(s: str) -> str:
        return s.replace("_", " ").title()

    for i, sym in enumerate(all_symptoms):
        col = cols[i % n_cols]
        if col.checkbox(fmt(sym), key=f"sym_{sym}"):
            selected.add(sym)

    st.markdown("---")
    run_fc = st.button("🔍 Diagnose (Forward Chaining)", key="btn_fc")

    if run_fc:
        if not selected:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            wm = WorkingMemory()
            for s in selected:
                wm.assert_fact(s, True)
            tracer = Tracer()
            conclusions = forward_chain(wm, tracer)

            if not conclusions:
                st.error("🤷 No diagnosis could be derived from the selected symptoms.")
            else:
                st.success(
                    f"✅ Forward chaining derived **{len(conclusions)}** "
                    f"diagnosis/diagnoses from your symptoms."
                )
                st.markdown("### 🩻 Diagnoses")
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
  <small style="color:#64748b">Certainty Factor (MYCIN-combined)</small>
</div>
""", unsafe_allow_html=True)

            # ── Trace / Why? ─────────────────────────────────────────────────
            with st.expander("🔎 Why? — Full Reasoning Trace"):
                st.markdown(
                    f'<div class="trace-box">{tracer.explain()}</div>',
                    unsafe_allow_html=True,
                )

            # ── Rule-network diagram ──────────────────────────────────────────
            if conclusions:
                with st.expander("🌐 Rule Network Diagram"):
                    top_disease = max(conclusions, key=conclusions.get)
                    fired_entries = tracer.entries()
                    G = nx.DiGraph()
                    for entry in fired_entries:
                        for cond in entry.matched_conditions:
                            G.add_edge(fmt(cond), fmt(entry.conclusion),
                                       label=entry.rule_id)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor("#0d1117")
                    ax.set_facecolor("#0d1117")

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
                    nx.draw_networkx_labels(G, pos, font_color="white",
                                            font_size=7, ax=ax)

                    legend = [
                        mpatches.Patch(color="#7c3aed", label="Symptom"),
                        mpatches.Patch(color="#2563eb", label="Diagnosis"),
                    ]
                    ax.legend(handles=legend, facecolor="#1e293b",
                              labelcolor="white", loc="upper left")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — BACKWARD CHAINING
# ══════════════════════════════════════════════════════════════════════════════
else:
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
<div class="verdict-confirmed">
  ✅ CONFIRMED — {disease_name} is <strong>{pct}% likely</strong>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="verdict-rejected">
  ❌ REJECTED — {disease_name} could not be established from your answers
</div>
""", unsafe_allow_html=True)

        # Answers summary
        if st.session_state.bc_questions:
            with st.expander("📋 All answers collected"):
                rows = [
                    {"Symptom": fmt(s), "Answer": "Yes ✔️" if a else "No ✖️"}
                    for s, a in st.session_state.bc_questions
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Trace
        with st.expander("🔎 Why? — Full Reasoning Trace"):
            trace_text = tracer.explain()
            st.markdown(
                f'<div class="trace-box">{trace_text}</div>',
                unsafe_allow_html=True,
            )

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
