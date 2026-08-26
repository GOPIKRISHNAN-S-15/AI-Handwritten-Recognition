"""
Synthetic Intelligence Research Interface (SIRI) — Model Lab
Overview of the four-model architecture and experimental findings.
"""

import streamlit as st

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header,
)
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service

# ── Page Config ──
st.set_page_config(page_title="MODEL LAB — HWR LAB", page_icon="🔬", layout="wide")
load_css()

# ── System Runtime Checks ──
emnist_model = load_trained_model("emnist")
cnn_loaded = emnist_model is not None
genai_service = get_genai_service()
genai_available = genai_service.check_connection() if hasattr(genai_service, 'check_connection') else genai_service.is_available

# ── TopAppBar HUD ──
render_top_app_bar(
    cnn_online=cnn_loaded,
    genai_online=genai_available,
)

# ── Sidebar Drawer ──
render_sidebar_drawer(cnn_loaded, genai_available)

# ── Header ──
render_section_hud_header(
    "MODEL LAB ARCHITECTURE",
    "Technical inspection of the four-model recognition architecture and empirical observations."
)

st.html("<div style='height: 10px;'></div>")

col1, col2 = st.columns(2)

with col1:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-orange); border-radius: 4px; padding: 1.5rem; height: 100%;">
    <div style="font-family: var(--font-primary); font-size: 1.2rem; color: var(--accent-orange); font-weight: 600; margin-bottom: 1rem;">Custom CNN</div>
    <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Architecture</td><td style="padding: 8px 0; text-align: right;">CNN</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Dataset</td><td style="padding: 8px 0; text-align: right;">EMNIST Balanced</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Classes</td><td style="padding: 8px 0; text-align: right;">47</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Task</td><td style="padding: 8px 0; text-align: right;">Character classification</td></tr>
        <tr><td style="padding: 8px 0; color: var(--text-secondary);">Test Accuracy</td><td style="padding: 8px 0; text-align: right; color: var(--accent-orange); font-weight: bold;">89.5%</td></tr>
    </table>
</div>
""")

with col2:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-cyan); border-radius: 4px; padding: 1.5rem; height: 100%;">
    <div style="font-family: var(--font-primary); font-size: 1.2rem; color: var(--accent-cyan); font-weight: 600; margin-bottom: 1rem;">TrOCR</div>
    <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Architecture</td><td style="padding: 8px 0; text-align: right;">Transformer</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Task</td><td style="padding: 8px 0; text-align: right;">Handwritten text recognition</td></tr>
        <tr><td style="padding: 8px 0; color: var(--text-secondary);">Input</td><td style="padding: 8px 0; text-align: right;">Text-line image</td></tr>
    </table>
</div>
""")

st.html("<div style='height: 15px;'></div>")

col3, col4 = st.columns(2)

with col3:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-purple); border-radius: 4px; padding: 1.5rem; height: 100%;">
    <div style="font-family: var(--font-primary); font-size: 1.2rem; color: var(--accent-purple); font-weight: 600; margin-bottom: 1rem;">CNN-BiLSTM-CTC</div>
    <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Architecture</td><td style="padding: 8px 0; text-align: right;">CNN + BiLSTM + CTC</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Dataset</td><td style="padding: 8px 0; text-align: right;">IAM</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Task</td><td style="padding: 8px 0; text-align: right;">Handwritten text recognition</td></tr>
        <tr><td style="padding: 8px 0; color: var(--text-secondary);">Strength</td><td style="padding: 8px 0; text-align: right;">Connected/cursive handwriting</td></tr>
    </table>
</div>
""")

with col4:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-pink); border-radius: 4px; padding: 1.5rem; height: 100%;">
    <div style="font-family: var(--font-primary); font-size: 1.2rem; color: var(--accent-pink); font-weight: 600; margin-bottom: 1rem;">Gemini</div>
    <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Architecture</td><td style="padding: 8px 0; text-align: right;">Multimodal generative AI</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Model</td><td style="padding: 8px 0; text-align: right;">Gemini 3.5 Flash</td></tr>
        <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Task</td><td style="padding: 8px 0; text-align: right;">Document transcription</td></tr>
        <tr><td style="padding: 8px 0; color: var(--text-secondary);">Input</td><td style="padding: 8px 0; text-align: right;">Image / PDF</td></tr>
    </table>
</div>
""")

st.html("<hr style='border-color: var(--border-glass); margin: 2rem 0;'>")

# ══════════════════════════════════════════════
# EXPERIMENTAL FINDINGS
# ══════════════════════════════════════════════
render_section_hud_header("CURRENT EXPERIMENTAL FINDINGS", "Observations derived from current document testing set.")

st.html("""
<div style="display: flex; flex-direction: column; gap: 1rem;">
    <div style="background: var(--bg-secondary); border-left: 4px solid var(--accent-orange); border-radius: 4px; padding: 1.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase;">Canvas Input</div>
        <div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--text-primary); font-weight: 600; margin-bottom: 4px;">Custom CNN</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Best suited to isolated character recognition.</div>
    </div>

    <div style="background: var(--bg-secondary); border-left: 4px solid var(--accent-cyan); border-radius: 4px; padding: 1.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase;">Block Handwriting</div>
        <div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--text-primary); font-weight: 600; margin-bottom: 4px;">TrOCR</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Currently strongest result in tested block-letter document.</div>
    </div>

    <div style="background: var(--bg-secondary); border-left: 4px solid var(--accent-purple); border-radius: 4px; padding: 1.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase;">Cursive Handwriting</div>
        <div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--text-primary); font-weight: 600; margin-bottom: 4px;">CNN-BiLSTM-CTC</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Currently strongest result in tested cursive document.</div>
    </div>

    <div style="background: var(--bg-secondary); border-left: 4px solid var(--accent-pink); border-radius: 4px; padding: 1.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase;">Document Intelligence & Entities</div>
        <div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--text-primary); font-weight: 600; margin-bottom: 4px;">Gemini 3.5 Flash</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Best suited for zero-shot manuscript transcription and contextual error correction.</div>
    </div>
</div>

<div style="margin-top: 1.5rem; padding: 1rem; border: 1px dashed var(--border-glass); border-radius: 4px; color: var(--text-muted); font-size: 0.8rem; text-align: center;">
    <strong>Note:</strong> These are experimental observations from the current test set, not permanent universal claims. 
    Performance may vary based on document quality, handwriting style, and preprocessing conditions.
</div>
""")
