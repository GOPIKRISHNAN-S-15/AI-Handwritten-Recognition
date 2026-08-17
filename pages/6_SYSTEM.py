"""
Synthetic Intelligence Research Interface (SIRI) — System Architecture & About
Comprehensive technical documentation, CNN topology, and system verification.
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
mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")
cnn_loaded = mnist_model is not None or emnist_model is not None
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
    "Deep neural network topology, adaptive image preprocessing pipeline, and generative language layer."
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
        "label": "Language Layer Model",
        "value": GEMINI_MODEL,
        "meta": "Google GenAI SDK",
    },
    {
        "label": "Interface Framework",
        "value": "Streamlit Native",
        "meta": "Digital Laboratory Theme",
    },
])

st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CNN ARCHITECTURE SPECIFICATION
# ══════════════════════════════════════════════
render_section_hud_header("CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE")

st.markdown("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
    <div style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary); line-height: 2;">
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Input Layer]</span> 28 × 28 × 1 Grayscale Tensor (Normalized 0.0 - 1.0)<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Block 1 Conv]</span> Conv2D(32 filters, 3×3 kernel, ReLU) + BatchNormalization<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Block 1 Pool]</span> Conv2D(32 filters, 3×3) + MaxPooling2D(2×2) + Dropout(0.25)<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Block 2 Conv]</span> Conv2D(64 filters, 3×3, ReLU) + BatchNormalization<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Block 2 Pool]</span> Conv2D(64 filters, 3×3) + MaxPooling2D(2×2) + Dropout(0.25)<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Dense Head]</span> Flatten → Dense(256 units, ReLU) + BatchNormalization + Dropout(0.5)<br>
        <span style="color: var(--text-secondary); width: 150px; display: inline-block;">[Classifier]</span> Dense(10/47 units, Softmax Activation) → Softmax Probability Vector
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ADAPTIVE PREPROCESSING PIPELINE
# ══════════════════════════════════════════════
render_section_hud_header("ADAPTIVE PREPROCESSING PIPELINE")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown("""
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
    """, unsafe_allow_html=True)

with col_a2:
    st.markdown("""
    <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.8rem;">2. TENSOR NORMALIZATION</div>
        <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin: 0; padding-left: 1.2rem;">
            <li><strong>Bounding Box Isolation:</strong> Tight crop around active ink pixels</li>
            <li><strong>Aspect-Ratio Preserved Scaling:</strong> Scales to max 20×20 bounding box</li>
            <li><strong>Moments Centering:</strong> Center-of-mass aligned to canvas centroid (14, 14)</li>
            <li><strong>Tensor Reshape:</strong> <code>(1, 28, 28, 1)</code> float32 normalized in <code>[0.0, 1.0]</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
