"""
app.py

Streamlit UI for the Tax Research Assistant (IRC S280A home office
deduction). Imports the pipeline directly from pipeline.py -- see that
module for the research/verification/orchestration logic itself.

Run with:
    streamlit run app.py

Requires a GEMINI_API_KEY in the environment (e.g. via a .env file) and a
corpus/ folder next to this file containing the cleaned *_clean.txt source
files pipeline.py's `files` registry expects.
"""

import re

import streamlit as st

import pipeline

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="S280A Home Office Deduction Research Assistant", layout="wide")

LABEL_COLORS = {
    "SUPPORTED": "#1a7f37",
    "PARTIALLY SUPPORTED": "#9a6700",
    "UNSUPPORTED": "#cf222e",
    "DECLINE": "#57606a",
    "PARSE_ERROR": "#57606a",
}
LABEL_BG = {
    "SUPPORTED": "#dafbe1",
    "PARTIALLY SUPPORTED": "#fff8c5",
    "UNSUPPORTED": "#ffebe9",
    "DECLINE": "#eaeef2",
    "PARSE_ERROR": "#eaeef2",
}

st.title("Home Office Deduction Research Assistant")
st.caption("IRC S280A — eligibility and simplified vs. actual expense method questions")

st.warning(
    "**Not tax advice.** This is a research prototype for a capstone project. "
    "Outputs have not been reviewed by a licensed tax professional and should not "
    "be relied on for actual filing or advice decisions.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Cached corpus load -- runs the embedding model + chunk/embed step exactly
# once per process, not on every rerun (button click, thumbs vote, etc.)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading corpus and embedding model (first run only)...")
def get_corpus():
    return pipeline.load_corpus()


try:
    corpus_info = get_corpus()
except FileNotFoundError as e:
    st.error(
        f"Couldn't load the corpus: {e}\n\n"
        f"Expected the cleaned source .txt files in: `{pipeline.CORPUS_DIR}`"
    )
    st.stop()

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"{corpus_info['n_sources']} sources, {corpus_info['n_chunks']} chunks")
    st.caption("Cached at process start via st.cache_resource — not re-embedded per page load.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "query_id" not in st.session_state:
    st.session_state.query_id = None

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

question = st.text_area(
    "Ask a question about the home office deduction",
    placeholder="e.g. My client runs a business from a spare bedroom but also lets guests sleep there occasionally. Does it still qualify?",
    height=90,
)

run_clicked = st.button("Research", type="primary", disabled=not question.strip())

if run_clicked:
    st.session_state.question = question
    st.session_state.feedback = None

    status_box = st.status("Starting research...", expanded=True)

    def on_progress(stage, detail=""):
        messages = {
            "drafting": "Drafting an answer from the retrieved excerpts...",
            "verifying": f"Verifying claims against the corpus... {detail}".strip(),
            "retrying": f"Some claims weren't well supported — revising the answer ({detail})...",
        }
        status_box.update(label=messages.get(stage, stage))
        if stage == "verifying" and detail:
            status_box.write(detail)

    try:
        result = pipeline.orchestrate_answer(question, progress_callback=on_progress)
        st.session_state.result = result
        st.session_state.query_id = pipeline.log_query(question, result)
        status_box.update(
            label="Done." if result["status"] == "ok" else "Declined — insufficient support.",
            state="complete",
        )
    except Exception as e:
        status_box.update(label=f"Failed: {e}", state="error")
        st.session_state.result = None

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.result

if result:
    summary = result["summary"]
    used_chunks = result["used_chunks"]
    verifiable = summary["total_claims"] - summary["decline"]
    ok_count = summary["supported"]
    flagged_count = summary["unsupported"] + summary["partially_supported"]

    # --- Confidence banner --------------------------------------------------
    if result["status"] == "declined":
        st.error(
            f"**Declined to answer.** After {result['attempts']} attempt(s), too many claims "
            f"still weren't adequately supported by the corpus.",
            icon="🚫",
        )
    elif verifiable == 0:
        st.info("No verifiable claims in this answer (e.g. a full decline / out-of-scope response).", icon="ℹ️")
    elif flagged_count == 0:
        st.success(f"**{ok_count} of {verifiable} citations verified.** All claims fully supported.", icon="✅")
    else:
        st.warning(
            f"**{ok_count} of {verifiable} citations verified, {flagged_count} flagged.** "
            f"Review the verification log below before relying on this answer.",
            icon="⚠️",
        )

    st.divider()

    # --- Memo with inline citation badges -----------------------------------
    st.subheader("Answer")

    def _badge_citation(match):
        nums = match.group(1)
        return f'<span style="background:#eef1f4;border-radius:4px;padding:1px 6px;font-size:0.85em;font-weight:600;color:#1f2937;">[{nums}]</span>'

    memo_html = re.sub(r'\[([\d,\s]+)\]', _badge_citation, result["answer"])
    st.markdown(memo_html, unsafe_allow_html=True)

    if used_chunks:
        st.caption("Sources cited — click a number to see the exact excerpt it refers to:")
        cols = st.columns(min(len(used_chunks), 6) or 1)
        for i, chunk in enumerate(used_chunks):
            with cols[i % len(cols)]:
                with st.popover(f"[{i + 1}] {chunk['citation']}"):
                    st.markdown(f"**{chunk['citation']}**")
                    st.caption(f"Retrieval score: {chunk.get('score', 'n/a')}")
                    st.write(chunk["text"])

    st.divider()

    # --- Verification log table ---------------------------------------------
    st.subheader("Verification log")
    claim_results = result["claim_results"]

    if not claim_results:
        st.caption("No claims to verify.")
    else:
        rows_html = ['<table style="width:100%;border-collapse:collapse;">']
        rows_html.append(
            '<tr style="text-align:left;border-bottom:1px solid #d0d7de;">'
            '<th style="padding:6px 8px;">Status</th>'
            '<th style="padding:6px 8px;">Claim</th>'
            '<th style="padding:6px 8px;">Reasoning</th></tr>'
        )
        for r in claim_results:
            color = LABEL_COLORS.get(r["label"], "#57606a")
            bg = LABEL_BG.get(r["label"], "#eaeef2")
            rows_html.append(
                '<tr style="border-bottom:1px solid #eaeef2;">'
                f'<td style="padding:6px 8px;white-space:nowrap;">'
                f'<span style="background:{bg};color:{color};border-radius:4px;padding:2px 8px;font-weight:600;font-size:0.85em;">{r["label"]}</span></td>'
                f'<td style="padding:6px 8px;">{r["claim"]}</td>'
                f'<td style="padding:6px 8px;color:#57606a;">{r["reasoning"]}</td></tr>'
            )
        rows_html.append("</table>")
        st.markdown("".join(rows_html), unsafe_allow_html=True)

    st.divider()

    # --- Reasoning trace ------------------------------------------------------
    with st.expander("Reasoning trace (retrieval -> draft -> correction)"):
        for step in result["trace"]:
            st.markdown(f"**Attempt {step['attempt']}**")

            if step["retry_feedback_used"]:
                st.caption("Revision requested based on:")
                st.code(step["retry_feedback_used"], language=None)

            st.markdown("*Retrieved excerpts:*")
            for i, c in enumerate(step["used_chunks"]):
                st.caption(f"[{i + 1}] {c['citation']} — score {c.get('score', 'n/a')}")

            st.markdown("*Draft answer:*")
            st.write(step["draft_answer"])

            st.markdown("*Verification results:*")
            s = step["summary"]
            st.caption(
                f"{s['supported']} supported, {s['partially_supported']} partially supported, "
                f"{s['unsupported']} unsupported, {s['decline']} decline"
            )
            st.divider()

    # --- Feedback -------------------------------------------------------------
    st.subheader("Was this answer helpful?")
    fb_col1, fb_col2, _ = st.columns([1, 1, 6])
    with fb_col1:
        if st.button("👍", key="thumbs_up"):
            st.session_state.feedback = "up"
            pipeline.log_feedback(st.session_state.query_id, "up")
    with fb_col2:
        if st.button("👎", key="thumbs_down"):
            st.session_state.feedback = "down"
            pipeline.log_feedback(st.session_state.query_id, "down")

    if st.session_state.feedback:
        st.caption(f"Feedback recorded: {'👍 helpful' if st.session_state.feedback == 'up' else '👎 not helpful'}")
        st.caption(f"Saved to `logs/interactions.jsonl` on this machine.")
