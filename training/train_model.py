"""
Standalone training script for MNIST and EMNIST CNN models.

Run this script ONCE to train and save models + metrics.
Do NOT run this on every Streamlit application start.

Usage:
    python training/train_model.py
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models.cnn_model import build_mnist_cnn, build_emnist_cnn
from utils.constants import (
    MNIST_HYPERPARAMS, EMNIST_HYPERPARAMS,
    MNIST_MAPPING, EMNIST_BALANCED_MAPPING,
    MNIST_MODEL_PATH, EMNIST_MODEL_PATH,
    MNIST_HISTORY_PATH, EMNIST_HISTORY_PATH,
    MNIST_EVAL_PATH, EMNIST_EVAL_PATH,
)


def ensure_dirs():
    """Create output directories if they don't exist."""
    for path in [MNIST_MODEL_PATH, EMNIST_MODEL_PATH,
                 MNIST_HISTORY_PATH, EMNIST_HISTORY_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def load_mnist():
    """Load and preprocess MNIST dataset."""
    print("\n" + "=" * 60)
    print("LOADING MNIST DATASET")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Test samples:  {x_test.shape[0]}")
    print(f"  Image shape:   {x_train.shape[1:]}")
    print(f"  Classes:        {len(np.unique(y_train))}")

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def load_emnist():
    """Load and preprocess EMNIST Balanced dataset."""
    print("\n" + "=" * 60)
    print("LOADING EMNIST BALANCED DATASET")
    print("=" * 60)

    try:
        from emnist import extract_training_samples, extract_test_samples
        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test = extract_test_samples('balanced')
    except ImportError:
        print("  ERROR: 'emnist' package not installed.")
        print("  Install with: pip install emnist")
        return None, None
    except Exception as e:
        print(f"  ERROR: Could not download or extract EMNIST dataset. {e}")
        return None, None

    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Test samples:  {x_test.shape[0]}")
    print(f"  Image shape:   {x_train.shape[1:]}")
    print(f"  Classes:        {len(np.unique(y_train))}")

    # EMNIST images need to be transposed and flipped for correct orientation
    x_train = np.array([np.flipud(np.transpose(img)) for img in x_train])
    x_test = np.array([np.flipud(np.transpose(img)) for img in x_test])

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def create_data_augmenter():
    """Create an ImageDataGenerator for training augmentation."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=5,
    )


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hyperparams: dict,
    model_name: str,
):
    """Train a model with data augmentation and callbacks."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING {model_name}")
    print(f"{'=' * 60}")

    # Data augmentation
    datagen = create_data_augmenter()
    datagen.fit(x_train)

    # Split training data for validation
    val_split = hyperparams["validation_split"]
    val_size = int(len(x_train) * val_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train_split = x_train[val_size:]
    y_train_split = y_train[val_size:]

    print(f"  Training samples:   {len(x_train_split)}")
    print(f"  Validation samples: {len(x_val)}")
    print(f"  Test samples:       {len(x_test)}")
    print(f"  Batch size:         {hyperparams['batch_size']}")
    print(f"  Epochs:             {hyperparams['epochs']}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        datagen.flow(x_train_split, y_train_split, batch_size=hyperparams["batch_size"]),
        epochs=hyperparams["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_mapping: dict,
    model_name: str,
):
    """Evaluate model and generate comprehensive metrics."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATING {model_name}")
    print(f"{'=' * 60}")

    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    num_classes = len(class_mapping)
    target_names = [class_mapping[i] for i in range(num_classes)]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0,
    )

    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0,
    )

    # Find commonly confused pairs
    confused_pairs = find_confused_pairs(cm, class_mapping, top_n=10)

    # Find misclassified examples (indices)
    misclassified_mask = y_pred != y_test
    misclassified_indices = np.where(misclassified_mask)[0][:50]  # Save first 50

    evaluation = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class_precision": [float(p) for p in precision],
        "per_class_recall": [float(r) for r in recall],
        "per_class_f1": [float(f) for f in f1],
        "per_class_support": [int(s) for s in support],
        "confusion_matrix": cm.tolist(),
        "confused_pairs": confused_pairs,
        "misclassified_indices": [int(i) for i in misclassified_indices],
        "misclassified_true": [int(y_test[i]) for i in misclassified_indices],
        "misclassified_pred": [int(y_pred[i]) for i in misclassified_indices],
        "class_mapping": {str(k): v for k, v in class_mapping.items()},
        "num_classes": num_classes,
        "total_test_samples": int(len(y_test)),
        "total_correct": int(np.sum(y_pred == y_test)),
        "total_misclassified": int(np.sum(misclassified_mask)),
    }

    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")
    print(f"  F1 Score (macro):  {f1_macro:.4f}")
    print(f"  Misclassified:     {np.sum(misclassified_mask)} / {len(y_test)}")

    print("\n  Top Confused Pairs:")
    for pair in confused_pairs[:5]:
        print(f"    {pair['class_a']} <-> {pair['class_b']}: {pair['total_errors']} errors")

    return evaluation


def find_confused_pairs(
    cm: np.ndarray,
    class_mapping: dict,
    top_n: int = 10,
):
    """Find the most commonly confused class pairs from confusion matrix."""
    num_classes = cm.shape[0]
    pairs = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            # Errors in both directions
            errors_ij = int(cm[i][j])  # True=i, Pred=j
            errors_ji = int(cm[j][i])  # True=j, Pred=i
            total = errors_ij + errors_ji

            if total > 0:
                pairs.append({
                    "class_a": class_mapping.get(i, str(i)),
                    "class_b": class_mapping.get(j, str(j)),
                    "label_a": i,
                    "label_b": j,
                    "errors_a_as_b": errors_ij,
                    "errors_b_as_a": errors_ji,
                    "total_errors": total,
                })

    # Sort by total errors descending
    pairs.sort(key=lambda x: x["total_errors"], reverse=True)
    return pairs[:top_n]


def save_history(history, path: str):
    """Save training history to JSON."""
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    with open(path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"  History saved to: {path}")


def save_evaluation(evaluation: dict, path: str):
    """Save evaluation results to JSON."""
    with open(path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to: {path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  AI HANDWRITTEN OCR — MODEL TRAINING PIPELINE")
    print("=" * 60)

    ensure_dirs()

    # ── Train MNIST ──
    mnist_data = load_mnist()
    if mnist_data[0] is not None:
        (x_train, y_train), (x_test, y_test) = mnist_data

        model = build_mnist_cnn()
        model.summary()

        history = train_model(
            model, x_train, y_train, x_test, y_test,
            MNIST_HYPERPARAMS, "MNIST CNN"
        )

        # Save model
        model.save(MNIST_MODEL_PATH)
        print(f"\n  Model saved to: {MNIST_MODEL_PATH}")

        # Save history
        save_history(history, MNIST_HISTORY_PATH)

        # Evaluate
        evaluation = evaluate_model(model, x_test, y_test, MNIST_MAPPING, "MNIST")
        save_evaluation(evaluation, MNIST_EVAL_PATH)

    # ── Train EMNIST ──
    emnist_data = load_emnist()
    if emnist_data[0] is not None:
        (x_train, y_train), (x_test, y_test) = emnist_data

        model = build_emnist_cnn()
        model.summary()

        history = train_model(
            model, x_train, y_train, x_test, y_test,
            EMNIST_HYPERPARAMS, "EMNIST Balanced CNN"
        )

        # Save model
        model.save(EMNIST_MODEL_PATH)
        print(f"\n  Model saved to: {EMNIST_MODEL_PATH}")

        # Save history
        save_history(history, EMNIST_HISTORY_PATH)

        # Evaluate
        evaluation = evaluate_model(model, x_test, y_test, EMNIST_BALANCED_MAPPING, "EMNIST Balanced")
        save_evaluation(evaluation, EMNIST_EVAL_PATH)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
