"""
Standalone training script for a REAL EMNIST Balanced CNN model.

Downloads EMNIST Balanced (112,800 train / 18,800 test, 47 classes) via
TensorFlow Datasets, trains the project's EMNIST CNN architecture with
augmentation + callbacks, and saves:
  - models/emnist_model.keras
  - training/emnist_history.json
  - training/emnist_evaluation.json

Run ONCE (training takes time):
    python training/train_emnist.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models.cnn_model import build_emnist_cnn
from utils.constants import (
    EMNIST_HYPERPARAMS,
    EMNIST_BALANCED_MAPPING,
    EMNIST_MODEL_PATH,
    EMNIST_HISTORY_PATH,
    EMNIST_EVAL_PATH,
)

NPZ_PATH = os.path.expanduser("~/.emnist_balanced/emnist_balanced.npz")


def load_emnist():
    """Load EMNIST Balanced from the prepared npz (TFDS download)."""
    print("\n" + "=" * 60)
    print("LOADING EMNIST BALANCED DATASET")
    print("=" * 60)

    if not os.path.exists(NPZ_PATH):
        print(f"  ERROR: {NPZ_PATH} not found.")
        print("  Run: python load_emnist.py (requires tensorflow-datasets)")
        return None, None, None, None

    d = np.load(NPZ_PATH)
    x_train, y_train = d["x_train"].astype("float32") / 255.0, d["y_train"].astype("int64")
    x_test, y_test = d["x_test"].astype("float32") / 255.0, d["y_test"].astype("int64")
    x_train = np.transpose(x_train, (0, 2, 1, 3)).reshape(-1, 28, 28, 1)
    x_test = np.transpose(x_test, (0, 2, 1, 3)).reshape(-1, 28, 28, 1)

    num_classes = len(np.unique(y_train))
    expected_classes = len(EMNIST_BALANCED_MAPPING)
    split_name = "Balanced" if expected_classes == 47 else ("ByClass" if expected_classes == 62 else "Letters")
    print(f"  EMNIST Split:  {split_name}")
    print(f"  Expected classes: {expected_classes}")
    print(f"  Actual classes found: {num_classes}")
    print(f"  Train shape:   {x_train.shape}")
    print(f"  Test shape:    {x_test.shape}")
    return x_train, y_train, x_test, y_test


def create_data_augmenter():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=5,
    )


def find_confused_pairs(cm, class_mapping, top_n=10):
    num_classes = cm.shape[0]
    pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            total = int(cm[i][j]) + int(cm[j][i])
            if total > 0:
                pairs.append({
                    "class_a": class_mapping.get(i, str(i)),
                    "class_b": class_mapping.get(j, str(j)),
                    "label_a": i,
                    "label_b": j,
                    "errors_a_as_b": int(cm[i][j]),
                    "errors_b_as_a": int(cm[j][i]),
                    "total_errors": total,
                })
    pairs.sort(key=lambda p: p["total_errors"], reverse=True)
    return pairs[:top_n]


def main():
    print("=" * 60)
    print("  REAL EMNIST BALANCED TRAINING PIPELINE")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_emnist()
    if x_train is None:
        sys.exit(1)

    for path in (EMNIST_MODEL_PATH, EMNIST_HISTORY_PATH, EMNIST_EVAL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    model = build_emnist_cnn()
    model.summary()

    # Validation split
    val_split = EMNIST_HYPERPARAMS["validation_split"]
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_tr, y_tr = x_train[val_size:], y_train[val_size:]
    print(f"  Training: {len(x_tr)} | Validation: {len(x_val)} | Test: {len(x_test)}")
    print(f"  Batch size: {EMNIST_HYPERPARAMS['batch_size']} | Epochs: {EMNIST_HYPERPARAMS['epochs']}")

    datagen = create_data_augmenter()
    datagen.fit(x_tr)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        datagen.flow(x_tr, y_tr, batch_size=EMNIST_HYPERPARAMS["batch_size"]),
        epochs=EMNIST_HYPERPARAMS["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(EMNIST_MODEL_PATH)
    print(f"\n  Model saved to: {EMNIST_MODEL_PATH}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(EMNIST_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  History saved to: {EMNIST_HISTORY_PATH}")

    # Evaluate on the full 18,800-sample test set
    print("\n" + "=" * 60)
    print("EVALUATING ON REAL EMNIST TEST SET (18,800 samples)")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    num_classes = len(EMNIST_BALANCED_MAPPING)
    target_names = [EMNIST_BALANCED_MAPPING[i] for i in range(num_classes)]
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0,
    )
    mask = y_pred != y_test
    mis_idx = np.where(mask)[0][:50]

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
        "confused_pairs": find_confused_pairs(cm, EMNIST_BALANCED_MAPPING),
        "misclassified_indices": [int(i) for i in mis_idx],
        "misclassified_true": [int(y_test[i]) for i in mis_idx],
        "misclassified_pred": [int(y_pred[i]) for i in mis_idx],
        "class_mapping": {str(k): v for k, v in EMNIST_BALANCED_MAPPING.items()},
        "num_classes": num_classes,
        "total_test_samples": int(len(y_test)),
        "total_correct": int(np.sum(y_pred == y_test)),
        "total_misclassified": int(np.sum(mask)),
        "per_class_report": {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                             for k, v in report.items() if isinstance(v, dict)},
    }

    with open(EMNIST_EVAL_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"  Evaluation saved to: {EMNIST_EVAL_PATH}")
    print(f"  Misclassified: {np.sum(mask)} / {len(y_test)}")
    print("\n  TRAINING COMPLETE")


if __name__ == "__main__":
    main()
