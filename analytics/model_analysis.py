"""
Model analysis utilities for analytics and intelligence pages.

Loads saved training histories and evaluation results to generate
charts, metrics, and error analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


def load_training_history(model_type: str = "mnist") -> Optional[Dict]:
    """Load saved training history."""
    if model_type == "mnist":
        path = _resolve("training/training_history_mnist.json")
    else:
        path = _resolve("training/training_history_emnist.json")

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_evaluation(model_type: str = "mnist") -> Optional[Dict]:
    """Load saved evaluation results."""
    if model_type == "mnist":
        path = _resolve("training/evaluation_mnist.json")
    else:
        path = _resolve("training/evaluation_emnist.json")

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ──────────────────────────────────────────────
# Plotly Charts (Interactive)
# ──────────────────────────────────────────────

def plot_training_curves(history: Dict, model_name: str = "Model") -> go.Figure:
    """Create interactive training accuracy and loss curves."""
    epochs = list(range(1, len(history.get("accuracy", [])) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training & Validation Accuracy", "Training & Validation Loss"),
        horizontal_spacing=0.12,
    )

    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history.get("accuracy", []),
            mode='lines+markers', name='Train Accuracy',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=4),
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history.get("val_accuracy", []),
            mode='lines+markers', name='Val Accuracy',
            line=dict(color='#ec4899', width=2, dash='dash'),
            marker=dict(size=4),
        ), row=1, col=1
    )

    # Loss
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history.get("loss", []),
            mode='lines+markers', name='Train Loss',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=4),
        ), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history.get("val_loss", []),
            mode='lines+markers', name='Val Loss',
            line=dict(color='#ec4899', width=2, dash='dash'),
            marker=dict(size=4),
        ), row=1, col=2
    )

    fig.update_layout(
        title=f"{model_name} — Training Performance",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        font=dict(color="#e0e0ff"),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )

    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    return fig


def plot_class_distribution(evaluation: Dict, model_name: str = "Model") -> go.Figure:
    """Create interactive class distribution bar chart."""
    mapping = evaluation.get("class_mapping", {})
    support = evaluation.get("per_class_support", [])

    if not mapping or not support:
        return go.Figure()

    labels = [mapping.get(str(i), str(i)) for i in range(len(support))]

    fig = go.Figure(data=[
        go.Bar(
            x=labels, y=support,
            marker=dict(
                color=support,
                colorscale=[[0, '#7c3aed'], [0.5, '#00d4ff'], [1, '#10b981']],
                line=dict(width=0),
            ),
            hovertemplate='Class: %{x}<br>Samples: %{y}<extra></extra>',
        )
    ])

    fig.update_layout(
        title=f"{model_name} — Test Set Class Distribution",
        xaxis_title="Class",
        yaxis_title="Number of Samples",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        font=dict(color="#e0e0ff"),
        height=400,
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    return fig


def plot_per_class_metrics(evaluation: Dict, model_name: str = "Model") -> go.Figure:
    """Create per-class precision/recall/F1 chart."""
    mapping = evaluation.get("class_mapping", {})
    precision = evaluation.get("per_class_precision", [])
    recall = evaluation.get("per_class_recall", [])
    f1 = evaluation.get("per_class_f1", [])

    if not mapping or not precision:
        return go.Figure()

    labels = [mapping.get(str(i), str(i)) for i in range(len(precision))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Precision', x=labels, y=precision,
        marker_color='#00d4ff', opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        name='Recall', x=labels, y=recall,
        marker_color='#7c3aed', opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        name='F1 Score', x=labels, y=f1,
        marker_color='#10b981', opacity=0.8,
    ))

    fig.update_layout(
        title=f"{model_name} — Per-Class Metrics",
        xaxis_title="Class",
        yaxis_title="Score",
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        font=dict(color="#e0e0ff"),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", range=[0, 1.05])

    return fig


# ──────────────────────────────────────────────
# Matplotlib / Seaborn Charts
# ──────────────────────────────────────────────

def plot_confusion_matrix_sns(
    evaluation: Dict, model_name: str = "Model", max_display: int = 20
) -> plt.Figure:
    """Create a seaborn confusion matrix heatmap."""
    cm = np.array(evaluation.get("confusion_matrix", []))
    mapping = evaluation.get("class_mapping", {})

    if cm.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig

    num_classes = cm.shape[0]
    labels = [mapping.get(str(i), str(i)) for i in range(num_classes)]

    # For large matrices, show only subset or normalize
    if num_classes > max_display:
        # Show normalized version for readability
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        data = cm_normalized
        fmt = '.2f'
        title_suffix = " (Normalized)"
    else:
        data = cm
        fmt = 'd'
        title_suffix = ""

    fig_size = max(8, num_classes * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        data, annot=(num_classes <= 20), fmt=fmt,
        xticklabels=labels, yticklabels=labels,
        cmap='YlOrRd', linewidths=0.5,
        ax=ax, square=True,
        cbar_kws={'label': 'Count' if fmt == 'd' else 'Rate'},
    )

    ax.set_title(f"{model_name} — Confusion Matrix{title_suffix}", fontsize=14, pad=20)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.tick_params(axis='both', labelsize=8)

    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#111128')
    ax.title.set_color('#e0e0ff')
    ax.xaxis.label.set_color('#e0e0ff')
    ax.yaxis.label.set_color('#e0e0ff')
    ax.tick_params(colors='#9ca3af')

    plt.tight_layout()
    return fig


def plot_confusion_matrix_plotly(evaluation: Dict, model_name: str = "MNIST Digit Model") -> go.Figure:
    """Create interactive Plotly confusion matrix heatmap."""
    cm = np.array(evaluation.get("confusion_matrix", []))
    mapping = evaluation.get("class_mapping", {})

    if cm.size == 0:
        return go.Figure()

    num_classes = cm.shape[0]
    labels = [mapping.get(str(i), str(i)) for i in range(num_classes)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale=[
            [0.0, "#111318"],
            [0.05, "#1e1b4b"],
            [0.3, "#7c3aed"],
            [0.7, "#00f2ff"],
            [1.0, "#ffffff"]
        ],
        hoverongaps=False,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title=f"{model_name} — Interactive Confusion Matrix Heatmap",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(24,26,36,0.8)",
        font=dict(color="#f1f5f9", family="Sora, sans-serif"),
        height=500,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", autorange="reversed"),
    )
    return fig


def compute_live_metrics_sklearn(model, x_test: np.ndarray, y_test: np.ndarray, class_mapping: Dict[int, str]) -> Dict:
    """
    Compute real evaluation metrics directly on test data using scikit-learn.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Predict
    if x_test.ndim == 3:
        x_norm = x_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
    else:
        x_norm = x_test

    probs = model.predict(x_norm, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Per class metrics
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    _, support = np.unique(y_test, return_counts=True)

    # Confused pairs
    confused_pairs = []
    cm_arr = np.array(cm)
    for i in range(len(cm_arr)):
        for j in range(i + 1, len(cm_arr)):
            err_ij = int(cm_arr[i, j])
            err_ji = int(cm_arr[j, i])
            total_err = err_ij + err_ji
            if total_err > 0:
                confused_pairs.append({
                    "class_a": class_mapping.get(i, str(i)),
                    "class_b": class_mapping.get(j, str(j)),
                    "total_errors": total_err,
                    "errors_a_as_b": err_ij,
                    "errors_b_as_a": err_ji,
                })
    confused_pairs.sort(key=lambda x: x["total_errors"], reverse=True)

    return {
        "test_accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "per_class_precision": prec_pc,
        "per_class_recall": rec_pc,
        "per_class_f1": f1_pc,
        "per_class_support": support.tolist(),
        "confusion_matrix": cm,
        "confused_pairs": confused_pairs,
        "class_mapping": {str(k): v for k, v in class_mapping.items()},
        "total_test_samples": int(len(y_test)),
        "total_correct": int(np.sum(y_true == y_pred) if 'y_true' in locals() else int(np.sum(y_test == y_pred))),
        "total_misclassified": int(np.sum(y_test != y_pred)),
        "num_classes": len(class_mapping),
    }


def get_confused_pairs_table(evaluation: Dict) -> pd.DataFrame:
    """Get commonly confused pairs as a DataFrame."""
    pairs = evaluation.get("confused_pairs", [])
    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs)
    df = df[["class_a", "class_b", "total_errors", "errors_a_as_b", "errors_b_as_a"]]
    df.columns = ["Character A", "Character B", "Total Errors",
                   "A→B Errors", "B→A Errors"]
    return df


def get_metrics_summary(evaluation: Dict) -> Dict:
    """Extract key metrics for display."""
    return {
        "accuracy": evaluation.get("test_accuracy", 0.0),
        "precision": evaluation.get("precision_macro", 0.0),
        "recall": evaluation.get("recall_macro", 0.0),
        "f1_score": evaluation.get("f1_macro", 0.0),
        "total_test": evaluation.get("total_test_samples", 0),
        "total_correct": evaluation.get("total_correct", 0),
        "total_misclassified": evaluation.get("total_misclassified", 0),
        "num_classes": evaluation.get("num_classes", 10),
    }
