"""
Synthetic Intelligence Research Interface (SIRI) — Model Lab
Confusion matrix heatmaps, per-class error distribution, and technical metrics.
"""

import streamlit as st
import numpy as np
import pandas as pd

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header,
)
from utils.constants import MNIST_INFO, EMNIST_BALANCED_INFO
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service
from analytics.model_analysis import (
    load_evaluation, plot_confusion_matrix_plotly,
    plot_per_class_metrics, get_confused_pairs_table,
    get_metrics_summary,
)

# ── Page Config ──
st.set_page_config(page_title="MODEL LAB — HWR LAB", page_icon="🔬", layout="wide")
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
    "MODEL LAB",
    "Technical inspection of model architecture, empirical evaluation metrics, and error analytics."
)

# ── Model Selector ──
# Gating: only models that are actually loaded AND evaluated can be
# inspected. This prevents a "selected EMNIST, shown MNIST metrics"
# mismatch.
ready_options = []
if mnist_model is not None and load_evaluation("mnist") is not None:
    ready_options.append("DIGITS -> MNIST (0-9)")
if emnist_model is not None and load_evaluation("emnist") is not None:
    ready_options.append("CHARACTERS -> EMNIST (Balanced)")

if not ready_options:
    st.error("⚠️ No inspected model is available (model file + evaluation artifacts missing).")
    st.stop()

model_tab = st.selectbox(
    "ACTIVE MODEL INSPECTION",
    ready_options,
    key="intelligence_model",
)

is_mnist = "MNIST" in model_tab
info = MNIST_INFO if is_mnist else EMNIST_BALANCED_INFO
model_type = "mnist" if is_mnist else "emnist"
model = mnist_model if is_mnist else emnist_model

evaluation = load_evaluation(model_type)

if not evaluation:
    st.error("Evaluation telemetry not found for this model. Run model evaluation first.")
    st.stop()

metrics = get_metrics_summary(evaluation)

# ── Measured inference latency (real runtime benchmark) ──
@st.cache_data(show_spinner=False)
def _measure_latency(_model, model_type):
    """Benchmark the actual model on 200 real images, return median ms."""
    import time
    import os
    try:
        from tensorflow import keras
        if model_type == "mnist":
            (_, _), (xt, yt) = keras.datasets.mnist.load_data()
        else:
            npz = os.path.expanduser("~/.emnist_balanced/emnist_balanced.npz")
            if os.path.exists(npz):
                d = np.load(npz)
                xt, yt = d["x_test"], d["y_test"]
            else:
                return None, None
        xt = xt[:200].astype(np.float32) / 255.0
        sample = np.random.RandomState(42).choice(xt.shape[0], 100, replace=False)
        x_batch = xt[sample][:, :, :, None]
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _model.predict(x_batch, verbose=0)
            times.append((time.perf_counter() - t0) * 1000.0 / x_batch.shape[0])
        med = float(np.median(times))
        mn = float(np.min(times))
        return med, mn
    except Exception:
        return None, None

inference_ms, inference_min_ms = _measure_latency(model, model_type)

# ══════════════════════════════════════════════
# TECHNICAL INFORMATION
# ══════════════════════════════════════════════
render_section_hud_header("ARCHITECTURE & METRICS")

col_arch, col_metrics = st.columns(2)

with col_arch:
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase;">ARCHITECTURE SPECIFICATION</div>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Active Model</td><td style="padding: 8px 0; text-align: right;">{info['name']}</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Dataset</td><td style="padding: 8px 0; text-align: right;">{info.get('train_samples', 0) + info.get('test_samples', 0):,} samples</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Classes</td><td style="padding: 8px 0; text-align: right;">{info['num_classes']}</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Input Size</td><td style="padding: 8px 0; text-align: right;">28x28x1</td></tr>
            <tr><td style="padding: 8px 0; color: var(--text-secondary);">Parameters</td><td style="padding: 8px 0; text-align: right;">{model.count_params():,} (measured)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase;">EVALUATION METRICS</div>
        <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Accuracy</td><td style="padding: 8px 0; text-align: right;">{metrics['accuracy']*100:.2f}%</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Precision</td><td style="padding: 8px 0; text-align: right;">{metrics['precision']*100:.2f}%</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">Recall</td><td style="padding: 8px 0; text-align: right;">{metrics['recall']*100:.2f}%</td></tr>
            <tr style="border-bottom: 1px solid var(--border-glass);"><td style="padding: 8px 0; color: var(--text-secondary);">F1 Score</td><td style="padding: 8px 0; text-align: right;">{metrics['f1_score']*100:.2f}%</td></tr>
            <tr><td style="padding: 8px 0; color: var(--text-secondary);">Inference Time</td><td style="padding: 8px 0; text-align: right;">{"%.1f ms/sample (median, measured)" % inference_ms if inference_ms is not None else "—"}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CONFUSION ANALYSIS
# ══════════════════════════════════════════════
st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
render_section_hud_header("CONFUSION ANALYSIS")

fig_cm = plot_confusion_matrix_plotly(evaluation, info["name"])
st.plotly_chart(fig_cm, width='stretch')

st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MOST CONFUSED CLASSES
# ══════════════════════════════════════════════
render_section_hud_header("MOST CONFUSED CLASSES")

pairs_df = get_confused_pairs_table(evaluation)

if not pairs_df.empty:
    st.dataframe(
        pairs_df.head(10),
        width='stretch',
        hide_index=True,
    )
else:
    st.info("No significant character misclassification pairs found in this test partition.")
