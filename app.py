"""Streamlit UI for the ToS summarizer.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from tos_summarizer.config import (ROOT, BERT_MODEL_PATH, BART_MODEL_PATH,
                                   DEFAULT_UNFAIR_THRESHOLD, IDLE_TIMEOUT_MINUTES, get_device)
from tos_summarizer.classifier import load_bert
from tos_summarizer.summarizer import load_bart
from tos_summarizer.preprocessing import clean_tos_text, ensure_nltk_data
from tos_summarizer.pipeline import process_tos_document

SAMPLE_PATH = ROOT / "fb_tos_testing.txt"

LABEL_DISPLAY = {
    'clearly_unfair': ("🚨", "Clearly unfair"),
    'potentially_unfair': ("⚠️", "Potentially unfair"),
}

st.set_page_config(page_title="ToS Summarizer", page_icon="📜", layout="wide")

if IDLE_TIMEOUT_MINUTES > 0:
    # Leaving the page closes Streamlit's websocket, which is what lets the server
    # scale to zero. Activity timestamps live on window so reruns don't reset them
    st.html(f"""
    <script>
    if (!window.__tosIdleWatch) {{
      window.__tosIdleWatch = true;
      let last = Date.now();
      const bump = () => {{ last = Date.now(); }};
      ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart', 'wheel']
        .forEach(e => window.addEventListener(e, bump, {{ passive: true, capture: true }}));
      setInterval(() => {{
        if (Date.now() - last > {IDLE_TIMEOUT_MINUTES * 60_000:.0f}) {{
          window.location.replace('app/static/idle.html');
        }}
      }}, 30000);
    }}
    </script>
    """, unsafe_allow_javascript=True)


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    # Loaded once per server process and shared across every user/session
    ensure_nltk_data()
    device = get_device()
    bert_model, bert_tokenizer = load_bert(BERT_MODEL_PATH, device)
    bart_model, bart_tokenizer = load_bart(BART_MODEL_PATH, device)
    return bert_model, bart_model, bert_tokenizer, bart_tokenizer, device


def results_to_df(results):
    return pd.DataFrame([{
        'label': r.label,
        'unfair_probability': round(r.unfair_prob, 3),
        'summary': r.summary,
        'original_clause': r.clause,
    } for r in results])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Flagging sensitivity",
        min_value=0.10, max_value=0.90, value=DEFAULT_UNFAIR_THRESHOLD, step=0.05,
        help="A clause is flagged if BERT thinks it's unfair, or if its combined unfair "
             "probability is at least this. Lower = flag more clauses (more false alarms)."
    )
    show_potential = st.checkbox("Show 'potentially unfair' clauses", value=True)
    st.divider()
    device = load_models()[-1]
    st.caption(f"Running on: `{device}`")


# ---------------- Input ----------------
st.title("📜 Terms of Service Summarizer")
st.write(
    "Paste a Terms of Service agreement or upload a `.txt` file. Each clause is classified "
    "by a fine-tuned Legal-BERT model, and the risky ones are rewritten in plain English by a "
    "fine-tuned BART model."
)

if "tos_text" not in st.session_state:
    st.session_state.tos_text = ""

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
        st.session_state.tos_text = uploaded.read().decode("utf-8", errors="replace")
        st.session_state.uploaded_name = uploaded.name
with col_sample:
    st.write("")
    st.write("")
    if SAMPLE_PATH.exists() and st.button("Try the Facebook ToS", use_container_width=True):
        st.session_state.tos_text = SAMPLE_PATH.read_text(encoding="utf-8")

text = st.text_area("...or paste it here", key="tos_text", height=250,
                    placeholder="Paste Terms of Service text...")

analyze = st.button("Analyze", type="primary", disabled=not text.strip())


# ---------------- Run pipeline ----------------
if analyze:
    bert_model, bart_model, bert_tokenizer, bart_tokenizer, device = load_models()
    progress = st.progress(0.0, text="Classifying clauses...")

    def on_progress(done, total):
        progress.progress(done / total, text=f"Summarizing flagged clauses ({done}/{total})...")

    results = process_tos_document(
        clean_tos_text(text), bert_model, bart_model, bert_tokenizer, bart_tokenizer, device,
        unfair_threshold=threshold, progress_callback=on_progress
    )
    progress.empty()
    st.session_state.results = results


# ---------------- Results ----------------
results = st.session_state.get("results")
if results is not None:
    st.divider()
    n_clear = sum(r.label == 'clearly_unfair' for r in results)
    n_potential = sum(r.label == 'potentially_unfair' for r in results)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged clauses", len(results))
    m2.metric("🚨 Clearly unfair", n_clear)
    m3.metric("⚠️ Potentially unfair", n_potential)

    if not results:
        st.success("No risky clauses found 🎉")

    # Most worrying first: clearly unfair, then by confidence
    ordered = sorted(results, key=lambda r: (r.label != 'clearly_unfair', -r.unfair_prob))
    for r in ordered:
        if r.label == 'potentially_unfair' and not show_potential:
            continue
        emoji, name = LABEL_DISPLAY[r.label]
        with st.container(border=True):
            st.markdown(f"**{emoji} {name}** · {r.unfair_prob:.0%} unfair probability")
            st.write(r.summary)
            with st.expander("Original clause"):
                st.caption(r.clause)

    if results:
        st.download_button(
            "Download results (CSV)",
            results_to_df(results).to_csv(index=False).encode("utf-8"),
            file_name="tos_analysis.csv",
            mime="text/csv",
        )
