"""
pipeline.py

Importable version of the Tax Research Assistant pipeline (IRC S280A home
office deduction), extracted from Tax_Research_Assistant_Clean.ipynb.

This module is intentionally NOT a copy-paste of the notebook cells as-is.
One change was made on purpose:

  - `_call_gemini` is a new shared retry/backoff wrapper. The notebook's
    retry logic only lived inside `ask_research_agent`; `verify_claim` made
    a bare `generate_content` call with no protection. Since `verify_answer`
    calls `verify_claim` once per claim in a loop, a single transient error
    (e.g. a 503) partway through verification would crash a live UI session
    with no retry. Both call sites now go through the same wrapper.

Everything else (prompts, chunking, scoring logic, orchestration thresholds)
is carried over unchanged from the notebook.

Usage from app.py:

    import pipeline

    corpus = pipeline.load_corpus()   # wrap this call in st.cache_resource
    result = pipeline.orchestrate_answer("Can I deduct my home office if...?")
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.1-flash-lite"

# Directory holding the cleaned corpus .txt files. Point this at wherever
# your *_clean.txt files actually live -- defaults to a "corpus" folder
# next to this module.
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Shared Gemini call wrapper with retry/backoff (covers 503s and other
# transient errors for BOTH the research agent and the verification agent)
# ---------------------------------------------------------------------------

def _call_gemini(contents, system_instruction, temperature, max_output_tokens,
                  label, max_retries=5, base_wait=15):
    """Calls generate_content with retry/backoff. Raises after max_retries.

    Backoff schedule matches the notebook's original ask_research_agent
    pattern: wait = base_wait * (attempt + 1), i.e. 15s, 30s, 45s, 60s, 75s.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            _check_finish_reason(response, label)
            return response
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"    (retrying {label} after error: {e}; waiting {wait}s)")
            time.sleep(wait)
    raise last_err  # pragma: no cover


def _check_finish_reason(response, label):
    """Warns if a response was cut off by the max_output_tokens cap rather
    than finishing naturally -- a truncated answer/verification can silently
    drop claims or citations."""
    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError):
        return
    if "MAX_TOKENS" in str(finish_reason):
        print(f"  WARNING: {label} response was truncated (finish_reason={finish_reason})")


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

def clean_legal_text(text):
    text = re.sub(r'\n?\d{0,2}\s*Publication 587 \(2025\)(?!,)\s*\n?', '\n', text)
    text = re.sub(r'\n?Publication 587 \(2025\)(?!,)\s*\d{1,2}\s*\n?', '\n', text)
    text = re.sub(
        r'\n\s*\d{3}\s*\n\s*\d{3}\s*\n\s*\d{3}\s*\n\s*\d{3}\s*\n\s*Schedule C \(Form 1040\)\s*\n\s*\d{4}\s*\n\s*\d{4}\s*\n',
        '\n', text,
    )

    def _wrap_join(m):
        left, frag = m.group(1), m.group(2)
        merged = left + frag
        if zipf_frequency(merged.lower(), "en") >= 4.0:
            return merged
        return m.group(0)

    text = re.sub(r'(\w+)\n(\w{1,15})\b', _wrap_join, text)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Corpus registry
# ---------------------------------------------------------------------------

files = {
    "280A_statute": "280a_statute_clean.txt",
    "prop_reg_280A2": "1_280a2_proposed_reg_business_use_clean.txt",
    "pub587_qualifying": "pub587_qualifying_only_clean.txt",
    "pub587_method_choice": "pub587_method_choice_categorization_clean.txt",
    "pub587_simplified_method": "pub587_simplified_method_clean.txt",
    "rev_proc_2013-13": "rev_proc_2013-13_simplified_method_clean.txt",
    "soliman": "soliman_506us168_1993_majority_opinion_clean.txt",
    "pub587_daycare": "pub587_daycare_facility_clean.txt",
    "pub587_employee_use": "pub587_employee_use_clean.txt",
}

# Retrieval fix carried over from the notebook (recall@5 82% -> 91%, zero
# regressions on the rest of the gold set): only 280A_statute and
# rev_proc_2013-13 got more specific embedding prefixes; the rest are
# unchanged.
source_titles = {
    "280A_statute": (
        "IRC S280A - Disallowance of Certain Expenses in Connection with "
        "Business Use of Home; the regular use, exclusive use, and principal "
        "place of business tests that determine eligibility"
    ),
    "prop_reg_280A2": "Proposed Treasury Regulation S1.280A-2 - Business Use of a Dwelling Unit",
    "pub587_qualifying": "IRS Publication 587 - Qualifying for the Home Office Deduction",
    "pub587_method_choice": "IRS Publication 587 - Figuring the Home Office Deduction",
    "pub587_simplified_method": "IRS Publication 587 - Simplified Method",
    "rev_proc_2013-13": (
        "Revenue Procedure 2013-13 - requirements and mechanics for electing "
        "the simplified method, including how the election is made on a tax "
        "return and its irrevocability for the taxable year"
    ),
    "soliman": "Commissioner v. Soliman, 506 U.S. 168 (1993)",
    "pub587_daycare": "IRS Publication 587 - Daycare Facility Qualifying Test",
    "pub587_employee_use": "IRS Publication 587 - Employee Use (Figure A)",
}

citation_labels = {
    "280A_statute": "IRC Section 280A",
    "prop_reg_280A2": "Prop. Treas. Reg. Section 1.280A-2",
    "pub587_qualifying": "IRS Pub. 587 (Qualifying for the Deduction)",
    "pub587_method_choice": "IRS Pub. 587 (Figuring the Deduction)",
    "pub587_simplified_method": "IRS Pub. 587 (Simplified Method)",
    "rev_proc_2013-13": "Rev. Proc. 2013-13",
    "soliman": "Commissioner v. Soliman, 506 U.S. 168 (1993)",
    "pub587_daycare": "IRS Pub. 587 (Daycare Facility Test)",
    "pub587_employee_use": "IRS Pub. 587 (Employee Use)",
}

# Which underlying legal/guidance document each registry entry belongs to.
# `files` has 9 entries because Pub 587 is split into 5 pieces for
# retrieval/citation granularity -- but that's still only 5 underlying
# documents (statute, proposed reg, Pub 587, Rev. Proc. 2013-13, Soliman),
# per the proposal. Used to report the accurate document count in the UI.
DOCUMENT_GROUPS = {
    "280A_statute": "IRC S280A",
    "prop_reg_280A2": "Prop. Treas. Reg. S1.280A-2",
    "pub587_qualifying": "IRS Publication 587",
    "pub587_method_choice": "IRS Publication 587",
    "pub587_simplified_method": "IRS Publication 587",
    "pub587_daycare": "IRS Publication 587",
    "pub587_employee_use": "IRS Publication 587",
    "rev_proc_2013-13": "Rev. Proc. 2013-13",
    "soliman": "Commissioner v. Soliman, 506 U.S. 168 (1993)",
}

AUTHORITATIVE_SOURCES = set(files.keys())

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunk_text(text, source_label, chunk_size=300, overlap=50):
    tokenizer = embedder.tokenizer
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    chunks = []
    i = 0
    chunk_num = 1
    while i < len(offsets):
        window = offsets[i:i + chunk_size]
        start_char, end_char = window[0][0], window[-1][1]
        chunk_str = text[start_char:end_char]
        chunks.append({
            "id": f"{source_label}_{chunk_num}",
            "source": source_label,
            "text": chunk_str,
        })
        i += chunk_size - overlap
        chunk_num += 1
    return chunks


# ---------------------------------------------------------------------------
# Corpus loading -- call this once, cache the result (st.cache_resource)
# ---------------------------------------------------------------------------

embedder = None
all_chunks = None
chunk_embeddings = None


def load_corpus(corpus_dir=None):
    """Loads the embedding model, cleans/chunks/embeds every file in the
    registry, and sets the module-level globals every retrieval function
    depends on. Expensive (loads a sentence-transformer model + embeds the
    whole corpus) -- call exactly once per process and cache the result.
    """
    global embedder, all_chunks, chunk_embeddings

    corpus_dir = corpus_dir or CORPUS_DIR
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    all_chunks = []
    for label, filename in files.items():
        path = os.path.join(corpus_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, label, chunk_size=300, overlap=50))

    chunk_texts = [f"{source_titles[c['source']]}. {c['text']}" for c in all_chunks]
    chunk_embeddings = embedder.encode(chunk_texts)

    return {
        "embedder": embedder,
        "all_chunks": all_chunks,
        "chunk_embeddings": chunk_embeddings,
        "n_chunks": len(all_chunks),
        "n_sources": len(set(DOCUMENT_GROUPS.values())),
    }


# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------

def retrieve_full(query, top_k=5):
    """Full chunk text + citation label, for feeding to the research agent."""
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = embedder.encode([prefixed_query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "id": all_chunks[idx]["id"],
            "source": all_chunks[idx]["source"],
            "citation": citation_labels[all_chunks[idx]["source"]],
            "score": round(float(similarities[idx]), 3),
            "text": all_chunks[idx]["text"],
        })
    return results


def retrieve_scoped(query, top_k=5, allowed_sources=None):
    """Retrieval restricted to a source subset -- used by the verifier."""
    if allowed_sources is None:
        allowed_sources = AUTHORITATIVE_SOURCES

    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = embedder.encode([prefixed_query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    allowed_indices = [i for i, c in enumerate(all_chunks) if c["source"] in allowed_sources]
    allowed_similarities = [(i, similarities[i]) for i in allowed_indices]
    allowed_similarities.sort(key=lambda x: x[1], reverse=True)
    top = allowed_similarities[:top_k]

    results = []
    for idx, score in top:
        results.append({
            "id": all_chunks[idx]["id"],
            "source": all_chunks[idx]["source"],
            "citation": citation_labels[all_chunks[idx]["source"]],
            "score": round(float(score), 3),
            "text": all_chunks[idx]["text"],
        })
    return results


def build_context(chunks):
    parts = [f"[{i + 1}] ({c['citation']})\n{c['text']}" for i, c in enumerate(chunks)]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# 6. Research Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a tax research assistant answering questions about the IRC \
S280A home office deduction — specifically qualification requirements and the choice \
between the regular and simplified methods. Depreciation mechanics, MACRS tables, and \
calculation-only content are out of scope; if a question is only about those, say so \
and decline rather than answering from general knowledge.

This scope restriction applies even within an otherwise in-scope answer, and covers the \
whole category of depreciation-classification concepts, not just MACRS specifically — \
this includes terms like "section 1250 property," depreciation recapture, useful life, \
placed-in-service dates, or any other detail whose meaning depends on depreciation rules. \
If — and ONLY if — one of the retrieved excerpts you were actually given for THIS question \
explicitly contains such a detail, do not restate, define, or explain it; instead note that \
an additional depreciation-related requirement exists in that excerpt and that its specifics \
are out of scope, citing that excerpt's number. Do NOT add this kind of disclaimer, or any \
mention of depreciation/MACRS/out-of-scope content, if none of the retrieved excerpts for \
this question actually contain such a detail — never add it as a generic caveat, hedge, or \
boilerplate closing remark.

You will be given retrieved source excerpts, each labeled with a citation. Follow these \
rules exactly:

1. Begin your answer with a one-sentence paraphrase of the user's question, in your own \
words — do not copy the question verbatim, and do not skip this step even if the answer \
is short or begins with a direct conclusion.
2. Answer using ONLY the information in the provided excerpts. Do not use outside \
knowledge of tax law, even if you believe it's correct.
3. Every substantive claim must end with a citation using the bracketed number of \
the excerpt it came from, e.g. [1] or [2]. Do not invent numbers not present in the \
excerpts, and do not add source labels or other text inside the brackets — just the \
number.
4. If the excerpts do not fully answer the question — even if they're topically \
related — say so explicitly. Do not infer, extrapolate, or fill gaps with general tax \
knowledge or assumptions about what the law "probably" says.
5. If excerpts conflict or address only part of the question, note that rather than \
silently picking one reading.
6. When an answer includes multiple distinct sub-points, requirements, or \
exceptions — such as a list of tests, conditions, or exceptions — present them as a \
bulleted or numbered list rather than a single dense paragraph. Use prose for answers \
that are a single continuous point.

=== EXAMPLE ===

RETRIEVED EXCERPTS:
[1] Pub. 587 (Simplified Method): The simplified method lets you multiply the allowable \
square footage of your home office by a prescribed rate, instead of tracking actual \
expenses. The maximum allowable square footage is 300 square feet.
[2] Rev. Proc. 2013-13: A taxpayer electing the simplified method may not deduct \
depreciation for the portion of the home used in the business for that taxable year.

QUESTION: Can I use the simplified method if my home office is larger than 300 square \
feet?

ANSWER: You are asking whether the simplified method is available when a home office \
exceeds 300 square feet.

- The simplified method caps the deductible area at 300 square feet, even if the actual \
office space is larger [1].
- Electing the simplified method also means you cannot separately deduct depreciation \
for the business-use portion of the home for that year [2].

The excerpts do not state whether a taxpayer with an office larger than 300 square feet \
may still elect the simplified method and simply cap the deduction at 300 square feet, or \
whether the method is unavailable entirely in that case — this distinction is not \
addressed in the provided sources.

=== END EXAMPLE ===
"""


def ask_research_agent(question, top_k=5, retry_feedback=None):
    chunks = retrieve_full(question, top_k=top_k)
    context = build_context(chunks)
    contents = f"=== RETRIEVED EXCERPTS ===\n\n{context}\n\n=== QUESTION ===\n{question}"

    if retry_feedback:
        contents += (
            f"\n\n=== REVISION NEEDED ===\n"
            f"Your previous answer to this question included claims that a verification "
            f"check found were not adequately supported by the retrieved excerpts:\n\n"
            f"{retry_feedback}\n\n"
            f"Revise your answer: remove or soften each flagged claim so it no longer "
            f"overstates what the excerpts say, while keeping everything else that was "
            f"correct. Do not introduce new claims."
        )

    response = _call_gemini(
        contents=contents,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=500,
        label="ask_research_agent",
    )
    return response.text, chunks


# ---------------------------------------------------------------------------
# 7. Claim splitting
# ---------------------------------------------------------------------------

RESTATEMENT_PATTERN = re.compile(
    r'^(you are (asking|inquiring)|what (is|are|does)|the user is asking)',
    re.IGNORECASE,
)

ABBREVIATIONS = r'(?<!\bv\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bNo\.)(?<!\bSec\.)'


def _is_header_only(text):
    stripped = text.strip('*').strip()
    return stripped.endswith(":") and not re.search(r'\[[\d,\s]+\]', text)


def split_claims(answer):
    lines = answer.strip().split("\n")
    claims = []
    checked_for_restatement = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not checked_for_restatement:
            checked_for_restatement = True
            if not line.startswith("*") and RESTATEMENT_PATTERN.match(line):
                continue

        if line.startswith("*") and not line.startswith("**"):
            claim = re.sub(r"^\*\s*", "", line)
            if not _is_header_only(claim):
                claims.append(claim)
        else:
            sentences = re.split(ABBREVIATIONS + r'(?<=[.?!])\s+(?=[A-Z])', line)
            for s in sentences:
                s = s.strip()
                if s and not _is_header_only(s):
                    claims.append(s)

    cleaned = []
    for c in claims:
        c = re.sub(r'\s*\[[\d,\s]+\]([.?!]?)\s*$', r'\1', c).strip()
        if c:
            cleaned.append(c)

    return cleaned


# ---------------------------------------------------------------------------
# 8. Verification Agent
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are a verification agent checking whether factual claims \
about IRC S280A (home office deduction) are actually grounded in source material, or whether they \
go beyond what any retrieved excerpt supports. You will be given the OTHER CLAIMS from the same \
answer as this claim (for context only), the SPECIFIC CLAIM you are judging, and excerpts retrieved \
independently from the full source corpus (statute, proposed regulation, IRS Publication 587, \
Revenue Procedure 2013-13, and the Soliman opinion).

Judge only the SPECIFIC CLAIM, not the answer as a whole. Use the other claims only to check \
whether a general statement is later qualified or completed by another claim in the same answer \
— for example, if this claim states a general rule and another claim from the same answer states \
its exceptions, do not penalize the general claim for omitting exceptions it doesn't need to \
restate. Do not use the other claims as a source of evidence themselves — only the retrieved \
excerpts count as evidence for whether a claim is true.

Judge the claim using exactly one of these three labels:

- SUPPORTED: the excerpts directly state or clearly entail the claim, OR the claim is a general \
statement whose exceptions/caveats are addressed in another claim from the same answer and are \
themselves supported by the excerpts.
- PARTIALLY SUPPORTED: the excerpts touch on the same topic but don't fully confirm the claim as \
stated, and the gap is NOT resolved by another claim from the same answer — e.g. they support part \
of it, use different scope/conditions, or the claim adds detail not present in the excerpts or the \
other claims.
- UNSUPPORTED: the excerpts don't address this claim at all, or contradict it.

Do not guess or use outside legal knowledge. If the excerpts are simply irrelevant to the claim, \
say UNSUPPORTED and note that the excerpts don't address it — do not stretch a weak match into \
PARTIALLY SUPPORTED just because something was retrieved.

If a claim states an unconditional rule (e.g., "no deduction, even if other tests are met") and \
the retrieved excerpts state that same rule without qualification, do not introduce an exception, \
condition, or limitation from outside legal knowledge — even if you believe one exists elsewhere \
in tax law. Judge the claim against what the excerpts actually say, not against the full state of \
the law. If you believe an excerpt is outdated or incomplete relative to current law, note that as \
a limitation of the retrieved excerpts, not as a reason to downgrade or contradict the claim.

When a claim's overall framing states a general rule (e.g., "the deduction is calculated based \
on X") but then names specific items the rule applies to (e.g., "...not the full amount of \
mortgage interest and utilities"), the named items define the claim's actual scope — do not treat \
the general framing language as extending the claim to categories of items it never names. A claim \
about "mortgage interest and utilities" should not be judged against rules for a different expense \
category (such as expenses that exist only because of business use) unless the claim itself \
mentions that category.

Before treating a claim as contradicted, confirm the contradicting excerpt actually addresses the \
SAME specific rule or requirement the claim is about — not merely the same statute section or a \
nearby, unrelated provision. If you identify a provision that is genuinely unrelated to the \
specific rule the claim describes, do not cite it as a reason to downgrade the claim's label.

If your reasoning asserts a specific count or quantity (e.g. "more than two," "at least three"), \
you must list each item counted by name in your reasoning. Do not assert a count without \
enumerating what makes it up.

Respond in exactly this format:
LABEL: <SUPPORTED|PARTIALLY SUPPORTED|UNSUPPORTED>
REASONING: <one or two sentences>
"""

DECLINE_PATTERN = re.compile(
    r'(the (provided )?excerpts? (do|does) not|'
    r'do not (address|cover|define|provide|contain)|'
    r'not (addressed|covered) (by|in) the (provided )?excerpts?|'
    r'out of scope|'
    r'do not offer guidance)',
    re.IGNORECASE,
)


def is_decline_statement(text):
    return bool(DECLINE_PATTERN.search(text))


def verify_claim(claim, other_claims, top_k=5):
    evidence = retrieve_scoped(claim, top_k=top_k)
    context = build_context(evidence)

    if other_claims:
        other_claims_block = "\n".join(f"- {c}" for c in other_claims)
    else:
        other_claims_block = "(none -- this is the only claim in the answer)"

    contents = (
        f"=== OTHER CLAIMS FROM THE SAME ANSWER (for context only) ===\n{other_claims_block}\n\n"
        f"=== SPECIFIC CLAIM TO JUDGE ===\n{claim}\n\n"
        f"=== EXCERPTS ===\n\n{context}"
    )

    response = _call_gemini(
        contents=contents,
        system_instruction=VERIFICATION_SYSTEM_PROMPT,
        temperature=0,
        max_output_tokens=400,
        label="verify_claim",
    )
    return response.text, evidence


def verify_answer(answer, top_k=5, delay_seconds=5, progress_callback=None):
    """Splits `answer` into claims and verifies each independently.

    progress_callback(done, total, claim_text), if given, is called after
    each claim is scored -- lets the UI show live progress instead of a
    blank spinner for what can be 20+ seconds of sequential, rate-limited
    calls.
    """
    claims = split_claims(answer)
    results = []
    verifiable_claims = [c for c in claims if not is_decline_statement(c)]
    decline_claims = [c for c in claims if is_decline_statement(c)]

    for c in decline_claims:
        results.append({
            "claim": c,
            "label": "DECLINE",
            "reasoning": "Meta-statement about absence of information in the excerpts; not sent to verifier.",
            "evidence_ids": [],
        })

    total = len(verifiable_claims)
    for i, claim in enumerate(verifiable_claims):
        other_claims = [c for j, c in enumerate(verifiable_claims) if j != i]
        result_text, evidence = verify_claim(claim, other_claims, top_k=top_k)
        label_match = re.search(r'LABEL:\s*(SUPPORTED|PARTIALLY SUPPORTED|UNSUPPORTED)', result_text)
        reasoning_match = re.search(r'REASONING:\s*(.+)', result_text, re.DOTALL)

        label = label_match.group(1) if label_match else "PARSE_ERROR"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text

        results.append({
            "claim": claim,
            "label": label,
            "reasoning": reasoning,
            "evidence_ids": [e["id"] for e in evidence],
        })

        if progress_callback:
            progress_callback(i + 1, total, claim)

        if i < total - 1:
            time.sleep(delay_seconds)

    summary = {
        "total_claims": len(results),
        "supported": sum(1 for r in results if r["label"] == "SUPPORTED"),
        "partially_supported": sum(1 for r in results if r["label"] == "PARTIALLY SUPPORTED"),
        "unsupported": sum(1 for r in results if r["label"] == "UNSUPPORTED"),
        "decline": sum(1 for r in results if r["label"] == "DECLINE"),
        "parse_errors": sum(1 for r in results if r["label"] == "PARSE_ERROR"),
    }

    return results, summary


# ---------------------------------------------------------------------------
# 9. Orchestrator
# ---------------------------------------------------------------------------

def build_retry_feedback(claim_results):
    problems = [r for r in claim_results if r["label"] in ("UNSUPPORTED", "PARTIALLY SUPPORTED")]
    lines = []
    for p in problems:
        lines.append(f'- Claim: "{p["claim"]}"\n  Problem ({p["label"]}): {p["reasoning"]}')
    return "\n".join(lines)


def orchestrate_answer(question, max_retries=1, unsupported_threshold=0.25, top_k=5,
                        progress_callback=None):
    """Runs the research agent, verifies the answer, and retries with
    targeted feedback if too many claims fail verification.

    Returns a dict with: status ("ok" | "declined"), answer, attempts,
    claim_results, summary, used_chunks (the excerpt-number -> source
    mapping behind the answer's [n] citations, for the UI to resolve
    citations back to source text).

    progress_callback(stage, detail), if given, is called at each pipeline
    stage transition (e.g. "drafting", "verifying", "retrying") so the UI
    can show a meaningful status instead of a single opaque spinner.
    """

    def _notify(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    trace = []  # one entry per attempt: retrieval -> draft -> verification, for the UI's reasoning-trace panel

    _notify("drafting")
    answer, used_chunks = ask_research_agent(question, top_k=top_k)
    retry_feedback = None

    for attempt in range(max_retries + 1):
        _notify("verifying")

        def _claim_progress(done, total, claim_text):
            _notify("verifying", f"claim {done} of {total}")

        claim_results, summary = verify_answer(
            answer, top_k=top_k, progress_callback=_claim_progress
        )
        verifiable = summary["total_claims"] - summary["decline"]

        trace.append({
            "attempt": attempt + 1,
            "retry_feedback_used": retry_feedback,
            "used_chunks": used_chunks,
            "draft_answer": answer,
            "claim_results": claim_results,
            "summary": summary,
        })

        if verifiable == 0:
            return {"status": "ok", "answer": answer, "attempts": attempt + 1,
                    "claim_results": claim_results, "summary": summary,
                    "used_chunks": used_chunks, "trace": trace}

        fail_count = summary["unsupported"] + summary["partially_supported"]
        fail_rate = fail_count / verifiable

        if fail_rate <= unsupported_threshold:
            return {"status": "ok", "answer": answer, "attempts": attempt + 1,
                    "claim_results": claim_results, "summary": summary,
                    "used_chunks": used_chunks, "trace": trace}

        if attempt < max_retries:
            _notify("retrying", f"{fail_count}/{verifiable} claims flagged")
            retry_feedback = build_retry_feedback(claim_results)
            answer, used_chunks = ask_research_agent(question, top_k=top_k, retry_feedback=retry_feedback)

    decline_message = (
        "The available sources do not provide sufficiently reliable support to fully "
        "and confidently answer this question. Rather than risk stating claims that "
        "aren't adequately grounded in the retrieved excerpts, this assistant is "
        "declining to provide a final answer. You may want to consult a primary source "
        "directly or rephrase the question."
    )
    return {"status": "declined", "answer": decline_message, "attempts": max_retries + 1,
            "claim_results": claim_results, "summary": summary,
            "used_chunks": used_chunks, "trace": trace}


# ---------------------------------------------------------------------------
# Google Sheets logging
#
# Replaces the earlier local-file logging entirely -- this is the one
# logging path for the deployed app, so it survives Streamlit Cloud
# restarts instead of living in an ephemeral local file.
#
# Two worksheets in the same Google Sheet, joined by a shared query_id:
#
#   "queries"  -- one row per completed orchestrate_answer() call, with the
#                 full per-attempt draft/citation-outcome detail per the
#                 proposal's logging requirement ("Every pipeline run will
#                 log the original draft memo, the Verification Agent's
#                 outcome per citation, and the corrected memo, if a retry
#                 was triggered.")
#   "feedback" -- one row per thumbs up/down vote, referencing query_id.
#
# Requires two secrets (set the same way as GEMINI_API_KEY, in Streamlit's
# Secrets manager or a local .streamlit/secrets.toml):
#   - GOOGLE_SHEET_ID: the sheet's ID from its URL
#   - [gcp_service_account]: the full contents of the service-account JSON
#     key, as a TOML table
# ---------------------------------------------------------------------------

QUERIES_HEADER = [
    "timestamp", "query_id", "question", "status", "attempts", "answer",
    "supported", "partially_supported", "unsupported", "decline",
    "attempt_history_json",
]
FEEDBACK_HEADER = ["timestamp", "query_id", "feedback"]


def _get_sheet():
    """Returns an authorized gspread Spreadsheet handle, or None if the
    secrets aren't configured (so logging can fail quietly rather than
    crash the app for a demo where logging is secondary to the pipeline
    itself)."""
    try:
        import streamlit as st
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        gc = gspread.authorize(creds)
        return gc.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
    except Exception as e:
        print(f"  WARNING: could not connect to Google Sheets logging: {e}")
        return None


def _get_or_create_worksheet(sheet, title, header):
    try:
        ws = sheet.worksheet(title)
    except Exception:
        ws = sheet.add_worksheet(title=title, rows=1000, cols=len(header))
        ws.append_row(header)
    return ws


def log_query(question, result):
    """Logs a completed orchestrate_answer() call to the "queries"
    worksheet. Returns a query_id to pass to log_feedback() later if the
    user votes on this answer. Returns None (and prints a warning) if
    Sheets logging isn't configured or the write fails -- this should
    never block the app from showing the answer it already has."""
    query_id = str(uuid.uuid4())
    sheet = _get_sheet()
    if sheet is None:
        return query_id  # still usable for in-session feedback linking

    try:
        ws = _get_or_create_worksheet(sheet, "queries", QUERIES_HEADER)
        summary = result["summary"]
        ws.append_row([
            datetime.now(timezone.utc).isoformat(),
            query_id,
            question,
            result["status"],
            result["attempts"],
            result["answer"],
            summary["supported"],
            summary["partially_supported"],
            summary["unsupported"],
            summary["decline"],
            json.dumps([
                {
                    "attempt": a["attempt"],
                    "retry_feedback_used": a["retry_feedback_used"],
                    "draft_answer": a["draft_answer"],
                    "used_chunks": [c["citation"] for c in a["used_chunks"]],
                    "citation_outcomes": [
                        {"claim": c["claim"], "label": c["label"], "reasoning": c["reasoning"]}
                        for c in a["claim_results"]
                    ],
                }
                for a in result["trace"]
            ], ensure_ascii=False),
        ], value_input_option="RAW")
    except Exception as e:
        print(f"  WARNING: failed to log query to Google Sheets: {e}")

    return query_id


def log_feedback(query_id, feedback):
    """feedback: 'up' or 'down'."""
    sheet = _get_sheet()
    if sheet is None:
        return

    try:
        ws = _get_or_create_worksheet(sheet, "feedback", FEEDBACK_HEADER)
        ws.append_row([
            datetime.now(timezone.utc).isoformat(),
            query_id,
            feedback,
        ], value_input_option="RAW")
    except Exception as e:
        print(f"  WARNING: failed to log feedback to Google Sheets: {e}")
