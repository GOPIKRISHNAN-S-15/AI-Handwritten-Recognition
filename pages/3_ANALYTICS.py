"""
Synthetic Intelligence Research Interface (SIRI) — Analytics
Real-time training telemetry, loss convergence graphs, and test performance.
"""

import streamlit as st
import numpy as np

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header,
)
from utils.constants import MNIST_INFO, EMNIST_BALANCED_INFO
from models.cnn_model import load_trained_model
from genai.ai_service import get_genai_service
from analytics.model_analysis import (
    load_training_history, load_evaluation,
    plot_training_curves, plot_class_distribution,
    get_metrics_summary,
)

# ── Page Config ──
st.set_page_config(page_title="ANALYTICS — HWR LAB", page_icon="🔬", layout="wide")
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
    "ANALYTICS WORKSPACE",
    "Empirical training telemetry, loss convergence graphs, and test set performance."
)

# ── Dataset / Model Selector ──
# Gating: EMNIST appears only when the real model AND its evaluation
# artifacts exist on disk. This prevents a dead UI entry backed by
# MNIST data.
mnist_eval_ready = load_evaluation("mnist") is not None and load_training_history("mnist") is not None
emnist_ready = (load_trained_model("emnist") is not None
                and load_evaluation("emnist") is not None
                and load_training_history("emnist") is not None)

model_options = ["MNIST Digit Dataset (0-9)"]
if emnist_ready:
    model_options.append("EMNIST Balanced (Alphanumeric)")

model_tab = st.selectbox(
    "DATASET SELECTION",
    model_options,
    key="analytics_model",
)

is_mnist = "MNIST" in model_tab
info = MNIST_INFO if is_mnist else EMNIST_BALANCED_INFO
model_type = "mnist" if is_mnist else "emnist"

# ══════════════════════════════════════════════
# REAL DATASET METRICS
# ══════════════════════════════════════════════
evaluation = load_evaluation(model_type)
history = load_training_history(model_type)

train_count = info["train_samples"]
test_count = info["test_samples"]

if evaluation and "total_test_samples" in evaluation:
    test_count = evaluation["total_test_samples"]
if evaluation and "num_classes" in evaluation:
    info = dict(info)  # keep original metadata untouched
    info["num_classes"] = evaluation["num_classes"]

st.markdown(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem; margin-bottom: 2rem; display: flex; justify-content: space-between; font-family: var(--font-mono);">
    <div style="text-align: center;">
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Train Samples</div>
        <div style="font-size: 1.5rem; color: var(--text-primary); font-weight: 600;">{train_count:,}</div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Test Samples</div>
        <div style="font-size: 1.5rem; color: var(--text-primary); font-weight: 600;">{test_count:,}</div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Classes</div>
        <div style="font-size: 1.5rem; color: var(--text-primary); font-weight: 600;">{info['num_classes']}</div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Input Shape</div>
        <div style="font-size: 1.5rem; color: var(--text-primary); font-weight: 600;">{info['image_size'][0]}x{info['image_size'][1]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# REAL TRAINING CONVERGENCE CURVES
# ══════════════════════════════════════════════
render_section_hud_header("TRAINING CONVERGENCE")

if history:
    fig_curves = plot_training_curves(history, info["name"])
    # Modify plotly template to remove neon glow
    fig_curves.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    # Reset line colors
    fig_curves.data[0].line.color = '#3b82f6' # Train acc
    fig_curves.data[1].line.color = '#9ca3af' # Val acc
    fig_curves.data[2].line.color = '#3b82f6' # Train loss
    fig_curves.data[3].line.color = '#9ca3af' # Val loss

    st.plotly_chart(fig_curves, width='stretch')

    final_train_acc = history.get("accuracy", [0])[-1]
    final_val_acc = history.get("val_accuracy", [0])[-1]
    final_train_loss = history.get("loss", [0])[-1]
    final_val_loss = history.get("val_loss", [0])[-1]

    st.markdown(f"""
    <div style="background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1rem; margin-top: 1rem; display: flex; justify-content: space-around; font-family: var(--font-mono); font-size: 0.85rem;">
        <div><span style="color: var(--text-secondary);">Train Acc (Final):</span> <span style="color: var(--text-primary);">{final_train_acc * 100:.2f}%</span></div>
        <div><span style="color: var(--text-secondary);">Val Acc (Final):</span> <span style="color: var(--text-primary);">{final_val_acc * 100:.2f}%</span></div>
        <div><span style="color: var(--text-secondary);">Train Loss:</span> <span style="color: var(--text-primary);">{final_train_loss:.4f}</span></div>
        <div><span style="color: var(--text-secondary);">Val Loss:</span> <span style="color: var(--text-primary);">{final_val_loss:.4f}</span></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Training history JSON is not yet available for this model.")

st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TEST SET PERFORMANCE & CLASS DISTRIBUTION
# ══════════════════════════════════════════════
render_section_hud_header("CLASS DISTRIBUTION")

if evaluation:
    fig_dist = plot_class_distribution(evaluation, info["name"])
    # Adjust bar colors
    fig_dist.data[0].marker.colorscale = [[0, '#3b82f6'], [1, '#3b82f6']]
    fig_dist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dist, width='stretch')
else:
    st.info("Evaluation metrics JSON not found.")
