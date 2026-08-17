"""
Synthetic Intelligence Research Interface (SIRI) — Neural Core Dashboard
Main Application Entry Point (NEURAL_CORE v2.0)
"""

import streamlit as st

# ── Page Configuration ──
st.set_page_config(
    page_title="HWR LAB — System Core",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──
from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_hero_hud, render_neural_gauges, render_feature_tiles,
    render_reconstruction_pipeline, render_section_hud_header,
)
import os
import time

from utils.constants import APP_TITLE, APP_DESCRIPTION, MNIST_INFO
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service

# ── Live dashboard metrics (real data, never hardcoded) ──

@st.cache_data(show_spinner=False)
def _live_dashboard_metrics(mnist_model):
    """Compute real telemetry: measured accuracy + inference latency."""
    try:
        from analytics.model_analysis import load_evaluation
        from tensorflow import keras
        import numpy as np
        out = {}
        eval_m = load_evaluation("mnist")
        out["mnist_accuracy"] = eval_m["test_accuracy"] if eval_m else None
        out["emnist_accuracy"] = load_evaluation("emnist")["test_accuracy"] if load_evaluation("emnist") else None
        out["emnist_model_present"] = os.path.exists("models/emnist_model.keras")
        # Measured latency on MNIST
        if mnist_model is not None:
            (_, _), (xt, _) = keras.datasets.mnist.load_data()
            xb = xt[:100].astype(np.float32) / 255.0
            xb = xb[:, :, :, None]
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                mnist_model.predict(xb, verbose=0)
                times.append((time.perf_counter() - t0) * 1000.0 / xb.shape[0])
            out["latency_ms"] = float(np.median(times))
        else:
            out["latency_ms"] = None
        return out
    except Exception:
        return {}

# ── Load CSS ──
load_css()

# ── System Runtime Checks ──
mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")
cnn_loaded = mnist_model is not None or emnist_model is not None

dash = _live_dashboard_metrics(mnist_model)

genai_service = get_genai_service()
genai_available = genai_service.check_connection() if hasattr(genai_service, 'check_connection') else genai_service.is_available

# ── TopAppBar HUD ──
render_top_app_bar(
    title="NEURAL_CORE",
    version="v2.0",
    cnn_online=cnn_loaded,
    genai_online=genai_available,
)

# ── Sidebar Navigation Drawer ──
render_sidebar_drawer(cnn_loaded, genai_available)

# ══════════════════════════════════════════════
# NEURAL CORE DASHBOARD
# ══════════════════════════════════════════════

# ── Hero Section ──
render_hero_hud(
    title="DOCUMENT DIGITIZATION",
    subtitle="LABORATORY",
    description=(
        "Professional neural interface for handwriting classification, manuscript segmentation, and semantic analysis."
    ),
)

# ── Live Neural Gauges ──
render_section_hud_header("Neural Core Telemetry", "Real-time hardware, model memory allocation, and pipeline load", icon="📡")

gauges = [
    {
        "icon": "🧠",
        "label": "Neural Classifier",
        "value": f"MNIST {dash.get('mnist_accuracy', 0) * 100:.2f}%" if dash.get("mnist_accuracy") is not None else "No eval data",
        "meta": "2-Stage Conv2D + BN + Dense (measured)",
        "color": "var(--accent-cyan)",
    },
    {
        "icon": "⚡",
        "label": "Inference Latency",
        "value": f"{dash.get('latency_ms', 0):.1f} ms" if dash.get("latency_ms") is not None else "—",
        "meta": "CPU Vectorized Kernel (measured)",
        "color": "var(--accent-green)",
    },
    {
        "icon": "📊",
        "label": "Dataset Core",
        "value": f"{MNIST_INFO['train_samples']:,}",
        "meta": f"{MNIST_INFO['num_classes']} Verified Digit Classes",
        "color": "var(--accent-purple)",
    },
    {
        "icon": "✨",
        "label": "Gemini AI Link",
        "value": (f"{genai_service.model}" if hasattr(genai_service, 'model') else "Connected"),
        "meta": "Contextual Correction & Entities",
        "color": "var(--accent-pink)",
    },
]
if dash.get("emnist_model_present"):
    gauges.append({
        "icon": "🔤",
        "label": "EMNIST Engine",
        "value": (f"Balanced {dash.get('emnist_accuracy', 0) * 100:.2f}%" if dash.get("emnist_accuracy") is not None else "Trained"),
        "meta": "47 Classes · Digits + Letters",
        "color": "var(--accent-orange)",
    })

render_neural_gauges(gauges)

# ── Quick Action Triggers ──
render_section_hud_header("Global Action Triggers", "Direct launch workflows across neural research modules", icon="🚀")

col_act1, col_act2, col_act3, col_act4 = st.columns(4)

with col_act1:
    if st.button("CAPTURE MODULE", key="nav_recog", width='stretch'):
        st.switch_page("pages/1_CAPTURE.py")

with col_act2:
    if st.button("DOCUMENTS", key="nav_doc", width='stretch'):
        st.switch_page("pages/2_DOCUMENTS.py")

with col_act3:
    if st.button("MODEL LAB", key="nav_model", width='stretch'):
        st.switch_page("pages/4_MODEL_LAB.py")

with col_act4:
    if st.button("LANGUAGE LAYER", key="nav_genai", width='stretch'):
        st.switch_page("pages/5_LANGUAGE.py")

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# ── Pipeline Visualization ──
render_reconstruction_pipeline(active_step="CNN_CLASSIFY")

# ── Core Architecture Modules ──
render_section_hud_header("Research Modules & Capabilities", "Comprehensive toolkit for digit intelligence and extraction", icon="🔬")

render_feature_tiles([
    {
        "icon": "✍️",
        "title": "AI Recognition Engine",
        "desc": "Single-digit and character classification via canvas drawing or high-res image upload with live adaptive preprocessing and candidate scoring.",
    },
    {
        "icon": "📑",
        "title": "Document Pipeline",
        "desc": "Multi-stage manuscript reconstruction: horizontal projection line detection, bounding box character segmentation, and text assembly.",
    },
    {
        "icon": "✨",
        "title": "Gemini AI Workspace",
        "desc": "High-level semantic post-processing: automated OCR error correction, concise summaries, structured entity extraction, and contextual insights.",
    },
    {
        "icon": "📊",
        "title": "Neural Analytics Hub",
        "desc": "Dynamic evaluation analytics, interactive Plotly loss and accuracy training curves, class distribution bar charts, and test metrics.",
    },
    {
        "icon": "🧠",
        "title": "Model Intelligence",
        "desc": "Deep confusion matrix analysis, commonly confused character pairs (0↔O, 1↔I, 5↔S), and adaptive preprocessing recommendations.",
    },
    {
        "icon": "⚙️",
        "title": "System Architecture Core",
        "desc": "Complete technical specifications, Conv2D layer topological graphs, hardware telemetry, and environment verification.",
    },
])

