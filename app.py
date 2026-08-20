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

import html
import re

import streamlit as st
import streamlit.components.v1 as components

import pipeline

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Section 280A Home Office Deduction Research Assistant", layout="wide")

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

def _memo_markdown_to_html(text):
    """Minimal markdown -> HTML for the memo: **bold** and "- " bullet
    lists, paragraph breaks on blank lines. Escapes everything else first
    so source/model text can't inject stray HTML."""
    blocks = re.split(r'\n\s*\n', text.strip())
    html_blocks = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        is_list = all(l.startswith(("- ", "* ")) and not l.startswith("**") for l in lines)
        if is_list:
            items = "".join(f"<li>{html.escape(l[2:].strip())}</li>" for l in lines)
            html_blocks.append(f"<ul>{items}</ul>")
        else:
            html_blocks.append(f"<p>{html.escape(' '.join(lines))}</p>")
    joined = "\n".join(html_blocks)
    # Restore **bold** after escaping (escape turns nothing relevant here,
    # so this is safe to run post-escape).
    joined = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', joined)
    return joined


def render_memo_with_inline_citations(answer_text, used_chunks):
    """Renders the memo as HTML where each [n] / [n, m] marker is clickable
    right where it sits in the text, expanding a panel with the cited
    source excerpt(s) directly beneath that point -- not a separate control
    elsewhere on the page. Streamlit's native widgets can't be embedded
    inside a run of text, so this is a self-contained HTML/JS component
    instead of markdown + st.popover.
    """
    body_html = _memo_markdown_to_html(answer_text)

    counter = {"n": 0}

    def _make_marker(match):
        nums = [int(x.strip()) for x in match.group(1).split(",")]
        counter["n"] += 1
        marker_id = f"cite-{counter['n']}"
        label = ",".join(str(n) for n in nums)

        panels = []
        for n in nums:
            if 1 <= n <= len(used_chunks):
                c = used_chunks[n - 1]
                panels.append(
                    f'<div class="cite-source">'
                    f'<div class="cite-source-title">[{n}] {html.escape(c["citation"])}'
                    f' <span class="cite-score">score {c.get("score", "n/a")}</span></div>'
                    f'<div class="cite-source-text">{html.escape(c["text"])}</div>'
                    f'</div>'
                )
            else:
                panels.append(f'<div class="cite-source">[{n}] (source not found)</div>')

        return (
            f'<span class="cite-marker" onclick="toggleCite(\'{marker_id}\')">[{label}]</span>'
            f'<div class="cite-panel" id="{marker_id}">{"".join(panels)}</div>'
        )

    body_html = re.sub(r'\[([\d,\s]+)\]', _make_marker, body_html)

    full_html = f"""
    <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 15px; line-height: 1.6; color: #1f2937; }}
        p {{ margin: 0 0 12px 0; }}
        ul {{ margin: 0 0 12px 0; padding-left: 22px; }}
        li {{ margin-bottom: 6px; }}
        .cite-marker {{
            display: inline-block; background: #eef1f4; border-radius: 4px;
            padding: 1px 6px; font-size: 0.85em; font-weight: 600;
            color: #1f2937; cursor: pointer; user-select: none;
        }}
        .cite-marker:hover {{ background: #dbe4ee; }}
        .cite-panel {{
            display: none; margin: 6px 0 14px 0; border-left: 3px solid #94a3b8;
            padding-left: 12px;
        }}
        .cite-panel.open {{ display: block; }}
        .cite-source {{ margin-bottom: 10px; padding: 8px 0 8px 8px; background: #f8fafc; border-radius: 4px; }}
        .cite-source + .cite-source {{ border-top: 1px solid #cbd5e1; padding-top: 10px; }}
        .cite-source-title {{ font-weight: 600; font-size: 0.9em; margin-bottom: 2px; }}
        .cite-score {{ font-weight: 400; color: #6b7280; font-size: 0.85em; }}
        .cite-source-text {{ font-size: 0.9em; color: #374151; white-space: pre-wrap; }}
    </style>
    <div id="memo-root">{body_html}</div>
    <script>
        function toggleCite(id) {{
            var el = document.getElementById(id);
            el.classList.toggle('open');
            sendHeight();
        }}
        function sendHeight() {{
            var height = document.documentElement.scrollHeight;
            window.parent.postMessage({{type: "streamlit:setFrameHeight", height: height}}, "*");
        }}
        window.addEventListener('load', sendHeight);
        setTimeout(sendHeight, 100);
    </script>
    """
    # Estimate a generous starting height from the actual content length so
    # the box doesn't visibly clip before the resize script (best-effort;
    # not guaranteed to fire in every browser/embedding context) has a
    # chance to run. scrolling=True is the real safety net -- even if the
    # estimate undershoots, nothing becomes permanently inaccessible.
    # Estimate height from the CLOSED-state content only -- citation panels
    # are display:none until clicked, so they contribute zero height at
    # render time. (Earlier version wrongly assumed every panel was already
    # open, which is why the box was rendering with far too much blank
    # space.) scrolling=True stays on as a fallback for when someone opens
    # enough panels to exceed this estimate.
    estimated_height = max(150, min(900, 80 + len(answer_text) // 4))
    components.html(full_html, height=estimated_height, scrolling=True)


def render_source_list(used_chunks):
    """Renders all cited sources as a row of side-by-side, independently
    clickable boxes beneath the answer -- a second, separate way to open
    the same source text as the inline [n] markers (which stay as-is).
    Native Streamlit widgets (columns + popover), not custom HTML, so
    sizing is handled by Streamlit itself rather than needing a manual
    height estimate.
    """
    cols = st.columns(min(len(used_chunks), 6) or 1)
    for i, chunk in enumerate(used_chunks, start=1):
        with cols[(i - 1) % len(cols)]:
            with st.popover(f"[{i}] {chunk['citation']}"):
                st.markdown(f"**{chunk['citation']}**")
                st.caption(f"Retrieval score: {chunk.get('score', 'n/a')}")
                st.write(chunk["text"])


st.title("Home Office Deduction Research Assistant")
st.caption("IRC Section 280A — eligibility and simplified vs. actual expense method questions")

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

# Re-sync pipeline.py's module-level embedder/all_chunks/chunk_embeddings from
# the cached corpus on every rerun. get_corpus() only re-runs load_corpus()
# once per process (that's the expensive, cached part) -- but the retrieval
# functions in pipeline.py depend on those three names as live module
# globals, and that global state doesn't reliably survive every Streamlit
# rerun on its own. This keeps them in sync cheaply (no re-embedding) so a
# button click never sees a stale/reset embedder.
pipeline.embedder = corpus_info["embedder"]
pipeline.all_chunks = corpus_info["all_chunks"]
pipeline.chunk_embeddings = corpus_info["chunk_embeddings"]

with st.sidebar:
    st.subheader("Sources in this corpus")

    sidebar_html = f"""
    <style>
        .src-list {{ list-style: disc; padding-left: 18px; margin: 0; line-height: 1.9; }}
        .src-top {{ font-weight: 600; font-size: 16px; color: #1a73e8; }}
        .src-top a {{ color: #1a73e8; text-decoration: underline; }}
        .src-sub {{ list-style: none; padding-left: 14px; margin: 2px 0 0; }}
        .src-sub li a {{ color: #1a73e8; text-decoration: underline; font-size: 13px; }}
        .src-caption {{ font-size: 12px; color: #6b7280; margin: 12px 0 0; }}
    </style>
    <ul class="src-list">
        <li class="src-top"><a href="https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section280A&num=0&edition=prelim" target="_blank">IRC Section 280A</a></li>
        <li class="src-top">
            Prop. Treas. Reg. Section 1.280A-2
            <ul class="src-sub">
                <li><a href="https://www.taxnotes.com/research/federal/proposed-regulations/proposed-regs-on-deductions-for-business-use-or-rental-of/1r3c8" target="_blank">Original text</a></li>
                <li><a href="https://www.taxnotes.com/research/federal/proposed-regulations/proposed-regulation-affects-deductions-of-expenses-for-business-use-or/1r38k" target="_blank">Amendment</a></li>
            </ul>
        </li>
        <li class="src-top"><a href="https://www.irs.gov/irb/2013-06_IRB#RP-2013-13" target="_blank">Rev. Proc. 2013-13</a></li>
        <li class="src-top">
            <a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000283" target="_blank">IRS Pub. 587</a>
            <ul class="src-sub">
                <li><a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000226296" target="_blank">Qualifying for the Deduction</a></li>
                <li><a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000283" target="_blank">Figuring the Deduction</a></li>
                <li><a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000390" target="_blank">Simplified Method</a></li>
                <li><a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000226361" target="_blank">Daycare Facility Test</a></li>
                <li><a href="https://www.irs.gov/publications/p587#en_US_2025_publink1000226304" target="_blank">Employee Use</a></li>
            </ul>
        </li>
        <li class="src-top"><a href="https://www.law.cornell.edu/supct/html/91-998.ZO.html" target="_blank">Comm'r v. Soliman</a></li>
    </ul>
    <p class="src-caption">{corpus_info['n_sources']} sources, {corpus_info['n_chunks']} chunks — cached at process start via st.cache_resource, not re-embedded per page load.</p>
    """
    st.markdown(sidebar_html, unsafe_allow_html=True)

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

    # --- Memo with true inline citation pop-outs -----------------------------
    # Each [n] is a clickable marker; clicking it opens the source excerpt
    # directly beneath the sentence it's in (not a separate control
    # elsewhere on the page). Needs raw HTML/JS since Streamlit's native
    # widgets can't be embedded inside a run of text.
    st.subheader("Answer")
    render_memo_with_inline_citations(result["answer"], used_chunks)

    if used_chunks:
        st.caption("Sources cited (click to expand):")
        render_source_list(used_chunks)

    st.divider()

    # --- Verification log table ---------------------------------------------
    st.subheader("Verification log")
    claim_results = result["claim_results"]

    if not claim_results:
        st.caption("No claims to verify.")
    else:
        rows_html = ['<table style="width:100%;border-collapse:collapse;font-size:0.82em;">']
        rows_html.append(
            '<tr style="text-align:left;border-bottom:1px solid #d0d7de;">'
            '<th style="padding:4px 6px;">Status</th>'
            '<th style="padding:4px 6px;">Claim</th>'
            '<th style="padding:4px 6px;">Reasoning</th></tr>'
        )
        for r in claim_results:
            color = LABEL_COLORS.get(r["label"], "#57606a")
            bg = LABEL_BG.get(r["label"], "#eaeef2")
            rows_html.append(
                '<tr style="border-bottom:1px solid #eaeef2;">'
                f'<td style="padding:4px 6px;white-space:nowrap;">'
                f'<span style="background:{bg};color:{color};border-radius:4px;padding:1px 6px;font-weight:600;font-size:0.9em;">{r["label"]}</span></td>'
                f'<td style="padding:4px 6px;">{r["claim"]}</td>'
                f'<td style="padding:4px 6px;color:#57606a;">{r["reasoning"]}</td></tr>'
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
                st.markdown(step["retry_feedback_used"])

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
        st.caption("Logged to Google Sheets.")
