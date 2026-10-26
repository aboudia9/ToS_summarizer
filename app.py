"""Streamlit UI for the ToS summarizer.

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from tos_summarizer.config import (BERT_MODEL_PATH, BART_MODEL_PATH,
                                   DEFAULT_UNFAIR_THRESHOLD, IDLE_TIMEOUT_MINUTES, MAX_INPUT_CHARS,
                                   get_device)
from tos_summarizer.classifier import load_bert
from tos_summarizer.summarizer import load_bart
from tos_summarizer.preprocessing import clean_tos_text, ensure_nltk_data
from tos_summarizer.pipeline import process_tos_document
from tos_summarizer.examples import load_examples, load_cache_index, text_key, cached_results

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


@st.cache_resource
def load_example_data():
    # Example docs + their precomputed results (scripts/precompute_examples.py)
    examples = load_examples()
    cache = load_cache_index()
    by_key = {text_key(ex.read_text()): ex for ex in examples}
    return examples, cache, by_key


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

examples, example_cache, example_by_key = load_example_data()

def use_example(ex):
    # One click straight to results: the results section swaps in the cached analysis
    st.session_state.tos_text = ex.read_text()
    st.session_state.results = None
    st.session_state.results_key = text_key(st.session_state.tos_text)

if examples:
    st.markdown("**Don't have one handy? Try a real one** (results are instant):")
    for col, ex in zip(st.columns(len(examples)), examples):
        col.button(ex.name, on_click=use_example, args=(ex,), use_container_width=True)

uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
    st.session_state.tos_text = uploaded.read().decode("utf-8", errors="replace")
    st.session_state.uploaded_name = uploaded.name

text = st.text_area("...or paste it here", key="tos_text", height=250,
                    placeholder="Paste Terms of Service text...")

# Text matching a bundled example (button or pasted by hand) is served from its cache
key = text_key(text) if text.strip() else None
current_example = example_by_key.get(key)
if current_example and current_example.tip:
    st.info(f"**What to look for in {current_example.name}'s terms:** {current_example.tip}  \n"
            f"[Source]({current_example.source}), retrieved {current_example.retrieved}.", icon="🔎")

too_long = len(text) > MAX_INPUT_CHARS
if too_long:
    st.warning(f"That document is {len(text):,} characters — the limit is {MAX_INPUT_CHARS:,}. "
               "Try pasting just the sections you care about.")

analyze = st.button("Analyze", type="primary", disabled=not text.strip() or too_long)


# ---------------- Run pipeline ----------------
cached = cached_results(example_cache[key], threshold) if key in example_cache else None
if analyze and not too_long and cached is not None:
    st.session_state.results = None
    st.session_state.results_key = key
elif analyze and not too_long:
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
    st.session_state.results_key = None


# ---------------- Results ----------------
# results_key is set when the results come from an example's cache; those follow the
# sensitivity slider without rerunning, and go away once the text is changed
results_key = st.session_state.get("results_key")
if results_key is None:
    results = st.session_state.get("results")
else:
    results = cached if results_key == key else None
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
    else:
        st.caption("These are model predictions and can be wrong (summaries especially). "
                   "Open **Original clause** to check what the terms actually say.")

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
