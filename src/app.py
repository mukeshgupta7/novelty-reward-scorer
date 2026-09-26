"""Interactive Streamlit demo for scoring a single submission."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.embeddings import get_embedder
from src.novelty import RELEVANCE_FLOOR, build_corpus, score_submission, submission_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def load_reference_data() -> tuple[str, list[dict]]:
    fixed_content = json.loads((DATA_DIR / "fixed_content.json").read_text())["text"]
    submissions = json.loads((DATA_DIR / "submissions.json").read_text())
    return fixed_content, submissions


st.set_page_config(page_title="Novelty Scoring", page_icon="N", layout="wide")
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700&display=swap');
    .stApp {
        background: radial-gradient(ellipse at 8% 0%, #e7f2ed 0, transparent 34%), #f7f9f7;
        color: #1d302c;
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
    }
    h1, h2, h3 { font-family: 'Manrope', 'Segoe UI', sans-serif; color: #173b33; }
    h1 { font-size: 2rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce8e2;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        color: #173b33 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #173b33 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #173b33 !important;
        -webkit-text-fill-color: #173b33 !important;
        caret-color: #176b59;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #687a74 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #687a74 !important;
    }
    div.stButton > button[kind="primary"] {
        background: #176b59;
        border-color: #176b59;
    }
    </style>""",
    unsafe_allow_html=True,
)

try:
    fixed_content, submissions = load_reference_data()
except (OSError, json.JSONDecodeError, KeyError) as error:
    st.error(f"Could not load the reference data: {error}")
    st.stop()

with st.sidebar:
    st.header("Scoring setup")
    backend_label = st.selectbox("Embedding backend", ("Local (offline)", "Gemini API"))
    backend = "gemini" if backend_label == "Gemini API" else "local"
    if backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        st.warning("Set GEMINI_API_KEY in the environment that starts Streamlit.")
    with st.expander("Reference story"):
        st.write(fixed_content)

st.caption("NOVELTY / RELEVANCE")
st.title("Submission scoring")

form_column, result_column = st.columns((1, 1.05), gap="large")
with form_column:
    with st.form("submission_form", clear_on_submit=False):
        headline = st.text_input("Headline", max_chars=140)
        body = st.text_area("Comment", max_chars=1200, height=170)
        stance = st.segmented_control(
            "Stance",
            options=("Support", "Oppose", "Neutral"),
            default="Neutral",
        )
        submitted = st.form_submit_button("Score submission", type="primary", use_container_width=True)

form_error = None
if submitted:
    if not headline.strip() or not body.strip():
        st.session_state.pop("last_score", None)
        form_error = "Enter both a headline and a comment."
    else:
        new_submission = {
            "id": "demo-submission",
            "headline": headline.strip(),
            "body": body.strip(),
            "stance": stance or "Neutral",
        }
        try:
            corpus = build_corpus(submissions, fixed_content) + [submission_text(new_submission)]
            embedder = get_embedder(corpus, backend=backend)
            result = score_submission(
                new_submission,
                submissions,
                fixed_content,
                embedder=embedder,
            )
            nearest = next(
                (item for item in submissions if item["id"] == result.nearest_neighbor_id),
                None,
            )
            st.session_state["last_score"] = {
                "result": result,
                "nearest": nearest,
                "backend": backend_label,
            }
        except Exception as error:
            st.session_state.pop("last_score", None)
            form_error = f"Scoring failed: {error}"

with result_column:
    st.subheader("Score")
    if form_error:
        st.error(form_error)

    last_score = st.session_state.get("last_score")
    if last_score:
        result = last_score["result"]
        relevance_column, novelty_column, reward_column = st.columns(3)
        relevance_column.metric("Relevance", f"{result.relevance:.3f}")
        novelty_column.metric("Novelty", f"{result.novelty:.3f}")
        reward_column.metric("Reward", f"{result.final_score:.3f}")
        st.caption(f"Backend: {last_score['backend']} · relevance floor: {RELEVANCE_FLOOR:.2f}")

        if result.relevance < RELEVANCE_FLOOR:
            st.warning("Reward gated: relevance is below the minimum threshold.")
        else:
            st.success("Relevance gate passed.")

        nearest = last_score["nearest"]
        if nearest:
            with st.expander(f"Closest existing submission · {nearest['id']}"):
                st.markdown(f"**{nearest['headline']}**")
                st.write(nearest["body"])
                st.caption(f"Stance: {nearest['stance']}")
    else:
        st.metric("Comparison pool", f"{len(submissions)} submissions")
        st.caption(f"Selected backend: {backend_label}")