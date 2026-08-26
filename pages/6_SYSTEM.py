"""
Synthetic Intelligence Research Interface (SIRI) — System Architecture & About
Comprehensive technical documentation, model topology, and system verification.
"""

import streamlit as st
import sys
import platform

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header, render_neural_gauges,
)
from utils.constants import APP_TITLE, APP_VERSION, GEMINI_MODEL
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service

# ── Page Config ──
st.set_page_config(page_title="SYSTEM — HWR LAB", page_icon="🔬", layout="wide")
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
    "SYSTEM ARCHITECTURE CORE",
    "Deep neural network topology, adaptive image preprocessing pipeline, and generative OCR layer."
)

# ── System Telemetry Gauges ──
render_section_hud_header("HARDWARE & RUNTIME TELEMETRY")

render_neural_gauges([
    {
        "label": "Python Runtime",
        "value": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "meta": f"{platform.system()} {platform.machine()}",
    },
    {
        "label": "Deep Learning Engine",
        "value": "TensorFlow 2.x",
        "meta": "Keras Sequential Conv2D",
    },
    {
        "label": "Transformer Framework",
        "value": "Hugging Face",
        "meta": "Transformers Library",
    },
    {
        "label": "Interface Framework",
        "value": "Streamlit Native",
        "meta": "Digital Laboratory Theme",
    },
])

st.html("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>")

# ══════════════════════════════════════════════
# FOUR-MODEL RECOGNITION PIPELINE
# ══════════════════════════════════════════════
render_section_hud_header("FOUR-MODEL RECOGNITION ARCHITECTURE")

st.html("""
<div style="display: flex; flex-direction: column; gap: 1rem;">
<div style="background: var(--bg-card); border-left: 4px solid var(--accent-orange); border-radius: 4px; padding: 1.2rem;">
<div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-orange); font-weight: 600; margin-bottom: 4px;">1. Custom CNN (Primary Baseline)</div>
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
<strong>Topology:</strong> Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(256) → Dense(47)<br>
<strong>Role:</strong> Character-level recognition pipeline with explicit line/word/character segmentation and projection profile analysis.
</div>
</div>

<div style="background: var(--bg-card); border-left: 4px solid var(--accent-cyan); border-radius: 4px; padding: 1.2rem;">
<div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-cyan); font-weight: 600; margin-bottom: 4px;">2. TrOCR (Transformer Vision-Encoder-Decoder)</div>
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
<strong>Topology:</strong> ViT Encoder + RoBERTa Decoder (microsoft/trocr-small-handwritten)<br>
<strong>Role:</strong> Line-level printed and handwritten text recognition using attention mechanisms.
</div>
</div>

<div style="background: var(--bg-card); border-left: 4px solid var(--accent-purple); border-radius: 4px; padding: 1.2rem;">
<div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-purple); font-weight: 600; margin-bottom: 4px;">3. CNN-BiLSTM-CTC (Sequence Modeling)</div>
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
<strong>Topology:</strong> ConvNet Feature Extractor + Bidirectional LSTM + CTC Loss<br>
<strong>Role:</strong> Alignment-free continuous cursive handwriting recognition trained on the IAM dataset.
</div>
</div>

<div style="background: var(--bg-card); border-left: 4px solid var(--accent-pink); border-radius: 4px; padding: 1.2rem;">
<div style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-pink); font-weight: 600; margin-bottom: 4px;">4. Gemini 3.5 Flash (Generative Document OCR)</div>
<div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
<strong>Topology:</strong> Multimodal Large Language Model (Google GenAI)<br>
<strong>Role:</strong> Zero-shot document transcription, providing contextual and semantic correction over full document payloads.
</div>
</div>
</div>
""")

st.html("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>")

# ══════════════════════════════════════════════
# ADAPTIVE PREPROCESSING PIPELINE
# ══════════════════════════════════════════════
render_section_hud_header("ADAPTIVE PREPROCESSING PIPELINE")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.8rem;">1. SIGNAL ANALYSIS & RESTORATION</div>
    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin: 0; padding-left: 1.2rem;">
        <li><strong>Content Analysis:</strong> Document-vs-character detection (uniform-border test) to route operations</li>
        <li><strong>Noise Filtering:</strong> Non-local means denoising, applied only to full-document scans</li>
        <li><strong>Contrast:</strong> CLAHE applied only to low-contrast document scans</li>
        <li><strong>Binarization:</strong> Otsu thresholding with polarity inversion; adaptive threshold for uneven backgrounds</li>
        <li><strong>Deskewing:</strong> Moment-based angle correction (applies to rotated document scans only)</li>
    </ul>
</div>
""")

with col_a2:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.8rem;">2. TENSOR NORMALIZATION</div>
    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin: 0; padding-left: 1.2rem;">
        <li><strong>Bounding Box Isolation:</strong> Tight crop around active ink pixels</li>
        <li><strong>Aspect-Ratio Preserved Scaling:</strong> Scales to max 20×20 bounding box</li>
        <li><strong>Moments Centering:</strong> Center-of-mass aligned to canvas centroid (14, 14)</li>
        <li><strong>Tensor Reshape:</strong> <code>(1, 28, 28, 1)</code> float32 normalized in <code>[0.0, 1.0]</code></li>
    </ul>
</div>
""")
