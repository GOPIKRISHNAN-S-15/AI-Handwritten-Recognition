"""
Quick MNIST-only training script.

Run this to train and save the MNIST digit recognition model.
EMNIST can be trained later using the full train_model.py script.

Usage:
    python training/train_mnist_only.py
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from models.cnn_model import build_mnist_cnn
from utils.constants import (
    MNIST_HYPERPARAMS,
    MNIST_MAPPING,
    MNIST_MODEL_PATH,
    MNIST_HISTORY_PATH,
    MNIST_EVAL_PATH,
)


def ensure_dirs():
    for path in [MNIST_MODEL_PATH, MNIST_HISTORY_PATH, MNIST_EVAL_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def find_confused_pairs(cm, class_mapping, top_n=10):
    num_classes = cm.shape[0]
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            errors_ij = int(cm[i][j])
            errors_ji = int(cm[j][i])
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
    pairs.sort(key=lambda x: x["total_errors"], reverse=True)
    return pairs[:top_n]


def main():
    print("=" * 60)
    print("  AI HANDWRITTEN OCR -- MNIST TRAINING")
    print("=" * 60)

    ensure_dirs()

    # Load MNIST
    print("\n" + "=" * 60)
    print("LOADING MNIST DATASET")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"  Train samples: {x_train.shape[0]:,}")
    print(f"  Test samples:  {x_test.shape[0]:,}")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = build_mnist_cnn()
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=5,
    )
    datagen.fit(x_train)

    # Validation split
    val_split = MNIST_HYPERPARAMS["validation_split"]
    val_size = int(len(x_train) * val_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train_split = x_train[val_size:]
    y_train_split = y_train[val_size:]

    print(f"\n  Training samples:   {len(x_train_split):,}")
    print(f"  Validation samples: {len(x_val):,}")
    print(f"  Batch size:         {MNIST_HYPERPARAMS['batch_size']}")
    print(f"  Max epochs:         {MNIST_HYPERPARAMS['epochs']}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print("TRAINING MNIST CNN")
    print("=" * 60)

    history = model.fit(
        datagen.flow(x_train_split, y_train_split, batch_size=MNIST_HYPERPARAMS["batch_size"]),
        epochs=MNIST_HYPERPARAMS["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    model.save(MNIST_MODEL_PATH)
    print(f"\n  Model saved to: {MNIST_MODEL_PATH}")

    # Save history
    history_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(MNIST_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  History saved to: {MNIST_HISTORY_PATH}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING MNIST CNN")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    num_classes = len(MNIST_MAPPING)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    confused_pairs = find_confused_pairs(cm, MNIST_MAPPING, top_n=10)

    misclassified_mask = y_pred != y_test
    misclassified_indices = np.where(misclassified_mask)[0][:50]

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
        "class_mapping": {str(k): v for k, v in MNIST_MAPPING.items()},
        "num_classes": num_classes,
        "total_test_samples": int(len(y_test)),
        "total_correct": int(np.sum(y_pred == y_test)),
        "total_misclassified": int(np.sum(misclassified_mask)),
    }

    with open(MNIST_EVAL_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to: {MNIST_EVAL_PATH}")

    print(f"\n  Precision (macro): {precision_macro * 100:.2f}%")
    print(f"  Recall (macro):    {recall_macro * 100:.2f}%")
    print(f"  F1 Score (macro):  {f1_macro * 100:.2f}%")
    print(f"  Misclassified:     {int(np.sum(misclassified_mask))} / {len(y_test)}")

    print("\n  Top Confused Pairs:")
    for pair in confused_pairs[:5]:
        print(f"    {pair['class_a']} <-> {pair['class_b']}: {pair['total_errors']} errors")

    print("\n" + "=" * 60)
    print("  MNIST TRAINING COMPLETE")
    print("  Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
