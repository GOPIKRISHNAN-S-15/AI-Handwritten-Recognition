"""
Synthetic Intelligence Research Interface (SIRI) — Language Layer
Semantic post-processing, contextual error correction, and structured entity extraction.
"""

import streamlit as st
import json
import re

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header, render_entity_cards,
)
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service

st.set_page_config(page_title="LANGUAGE LAYER — HWR LAB", page_icon="🔬", layout="wide")
load_css()

# ── System Runtime Checks ──
mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")
emnist_letters_model = load_trained_model("emnist_letters")
cnn_loaded = any(m is not None for m in (mnist_model, emnist_model, emnist_letters_model))
genai_service = get_genai_service()
genai_available = genai_service.check_connection() if hasattr(genai_service, 'check_connection') else genai_service.is_available
genai_status = getattr(genai_service, 'status', 'uninitialized')

# ── TopAppBar HUD ──
render_top_app_bar(
    cnn_online=cnn_loaded,
    genai_online=genai_available,
)

# ── Sidebar Drawer ──
render_sidebar_drawer(cnn_loaded, genai_available)

# ── Header ──
render_section_hud_header(
    "LANGUAGE LAYER",
    "Semantic post-processing: contextual OCR repair, document synthesis, and categorized entity extraction."
)

# Status diagnostics derived from a REAL runtime probe (see genai/ai_service.py).
# The banner reflects the actual failure mode instead of a static STANDBY label.
if not genai_available:
    if genai_status == "quota_limited":
        status_title = "LANGUAGE LAYER · QUOTA LIMIT REACHED"
        status_detail = (
            "The Gemini API rejected the connection probe with a 429 "
            "RESOURCE_EXHAUSTED error: the free-tier quota attached to this "
            "GEMINI_API_KEY is fully consumed. Quota typically resets "
            "periodically; retry the check below, or attach a key with a "
            "higher quota in the .env file."
        )
        status_color = "#f59e0b"
        quota_limit = getattr(genai_service, "quota_limit", None)
        retry_after = getattr(genai_service, "retry_after_seconds", None)
        if quota_limit is not None:
            status_detail += (
                f" The probe observed a daily limit of {quota_limit} requests "
                f"attached to this key."
            )
        if retry_after is not None:
            status_detail += (
                f" The API reports it will accept requests again in about "
                f"{retry_after}s; the retry below is timed to that window."
            )
    elif genai_status == "key_missing":
        status_title = "LANGUAGE LAYER · API KEY NOT CONFIGURED"
        status_detail = (
            "No usable GEMINI_API_KEY was found in the environment or .env "
            "file. Configure it to activate semantic post-processing."
        )
        status_color = "#ef4444"
    elif genai_status == "model_unavailable":
        status_title = "LANGUAGE LAYER · MODEL UNAVAILABLE"
        status_detail = (
            f"The configured model ({getattr(genai_service, 'model_name', '?')}) "
            "is no longer served by the Gemini API (404). Update GEMINI_MODEL "
            "in your .env file to a currently supported model."
        )
        status_color = "#ef4444"
    else:
        status_title = "LANGUAGE LAYER · STANDBY"
        status_detail = (
            "The Gemini connection probe failed. Configure GEMINI_API_KEY and "
            "check your network connection."
        )
        status_color = "#fbbf24"

    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid {status_color}; padding: 1.5rem; border-radius: 4px;">
        <div style="font-family: var(--font-primary); font-size: 1.1rem; font-weight: 600; color: {status_color}; margin-bottom: 0.5rem;">
            {status_title}
        </div>
        <div style="color: var(--text-secondary); font-size: 0.85rem; max-width: 640px; margin: 0 auto 0.8rem; line-height: 1.6;">
            {status_detail}
        </div>
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">
            Active model identifier: {getattr(genai_service, 'model_name', '—')} · probe status: {genai_status} · CNN recognition and document segmentation remain fully active.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("RE-RUN CONNECTION PROBE", key="btn_genai_recheck"):
        retry_after = getattr(genai_service, "retry_after_seconds", None)
        genai_service.clear_probe_cache()
        if retry_after is not None:
            import time
            time.sleep(min(int(retry_after) + 5, 300))
        genai_available = genai_service.check_connection()
        genai_status = getattr(genai_service, 'status', 'uninitialized')
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════
# TEXT INTAKE & PAYLOAD BUFFER
# ══════════════════════════════════════════════
default_text = st.session_state.get("workspace_text", "Sample ID: 4092. Patient John Doe attended on 14-Aug-2026 for 3 neural evaluations. Total fee: $450.")

st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;'>RAW RECOGNITION PAYLOAD</div>", unsafe_allow_html=True)
input_text = st.text_area(
    "Raw text for semantic processing",
    value=default_text,
    height=130,
    placeholder="Paste OCR text stream or dispatch from Document Pipeline...",
    key="genai_input_text",
    label_visibility="collapsed"
)

if not input_text.strip():
    st.info("Enter or paste text in the payload buffer above to run semantic operations.")
    st.stop()

st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SEMANTIC OPERATIONS
# ══════════════════════════════════════════════
tab_correct, tab_summary, tab_entities, tab_insights = st.tabs([
    "Contextual Correction",
    "Document Synthesis",
    "Entity Extraction",
    "Deep Insights",
])

# ── Tab 1: Text Correction ──
with tab_correct:
    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin: 1rem 0;'>RESOLVES OPTICAL AMBIGUITIES WHILE PRESERVING RAW CNN INTENT.</div>", unsafe_allow_html=True)

    if st.button("RUN CORRECTION", key="btn_correct_exec"):
        with st.spinner("Processing..."):
            res = genai_service.correct_text(input_text)
            if res.success:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;'>RAW OCR STREAM</div>", unsafe_allow_html=True)
                    st.code(input_text, language="text")
                with col_c2:
                    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;'>ENHANCED TEXT</div>", unsafe_allow_html=True)
                    st.code(res.content, language="text")
            else:
                st.error(res.error)

# ── Tab 2: Document Summary ──
with tab_summary:
    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin: 1rem 0;'>GENERATES HIGH-DENSITY SYNTHESIS.</div>", unsafe_allow_html=True)

    if st.button("RUN SYNTHESIS", key="btn_summary_exec"):
        with st.spinner("Synthesizing..."):
            res = genai_service.summarize_text(input_text)
            if res.success:
                st.markdown(f"""
                <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
                    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.8rem;">EXECUTIVE SUMMARY</div>
                    <div style="font-family: var(--font-primary); font-size: 0.95rem; color: var(--text-primary); line-height: 1.6;">
                        {res.content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(res.error)

# ── Tab 3: Structured Entity Extraction ──
with tab_entities:
    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin: 1rem 0;'>EXTRACTS PERSON, TEMPORAL, QUANTITATIVE, AND IDENTIFIERS.</div>", unsafe_allow_html=True)

    if st.button("RUN EXTRACTION", key="btn_entities_exec"):
        with st.spinner("Extracting..."):
            prompt = f"""Extract and categorize all structured entities from the following text into 4 JSON lists:
1. "Person": Names of individuals
2. "Temporal": Dates, times, durations, years
3. "Quantitative": Numbers, quantities, currencies, percentages
4. "Identifiers": Codes, IDs, task names, reference labels

TEXT:
{input_text}

Respond ONLY with valid JSON in this format:
{{
  "Person": ["..."],
  "Temporal": ["..."],
  "Quantitative": ["..."],
  "Identifiers": ["..."]
}}"""
            res = genai_service._generate(prompt, "extract_entities_json")
            if res.success:
                try:
                    # Clean json markdown if present
                    json_str = res.content
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    entities_dict = json.loads(json_str.strip())
                    render_entity_cards(entities_dict)
                except Exception:
                    # Fallback to standard extraction
                    st.markdown(res.content)
            else:
                st.error(res.error)

# ── Tab 4: Document Insights ──
with tab_insights:
    st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin: 1rem 0;'>ANALYZES DOCUMENT TYPE, TONE, AND STRUCTURAL OBSERVATIONS.</div>", unsafe_allow_html=True)

    if st.button("RUN INSIGHTS", key="btn_insights_exec"):
        with st.spinner("Generating..."):
            res = genai_service.get_insights(input_text)
            if res.success:
                st.markdown(f"""
                <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
                    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.8rem;">CONTEXTUAL INSIGHTS</div>
                    <div style="font-family: var(--font-primary); font-size: 0.95rem; color: var(--text-primary); line-height: 1.6;">
                        {res.content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(res.error)
