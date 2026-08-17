"""
Evaluation-only run for the already-trained EMNIST Balanced model.
Reuses the trained model in models/emnist_model.keras and produces the full
evaluation JSON (test accuracy, per-class metrics, confusion matrix, confused
pairs) without retraining.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
)
from utils.constants import EMNIST_BALANCED_MAPPING


def find_confused_pairs(cm, mapping):
    pairs = []
    cm_arr = np.array(cm)
    for i in range(len(cm_arr)):
        for j in range(i + 1, len(cm_arr)):
            err_ij = int(cm_arr[i, j])
            err_ji = int(cm_arr[j, i])
            total_err = err_ij + err_ji
            if total_err > 0:
                pairs.append({
                    "class_a": mapping.get(i, str(i)),
                    "class_b": mapping.get(j, str(j)),
                    "total_errors": total_err,
                    "errors_a_as_b": err_ij,
                    "errors_b_as_a": err_ji,
                })
    pairs.sort(key=lambda x: x["total_errors"], reverse=True)
    return pairs


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "emnist_model.keras")
    eval_path = os.path.join(project_root, "training", "evaluation_emnist.json")
    npz_path = os.path.expanduser("~/.emnist_balanced/emnist_balanced.npz")

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"Loading EMNIST test set from: {npz_path}")
    d = np.load(npz_path)
    x_test = d["x_test"].astype(np.float32) / 255.0
    y_test = d["y_test"]
    x_test = x_test[:, :, :, None]

    print(f"Evaluating on {len(y_test):,} test samples...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
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

    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation saved to: {eval_path}")
    print(f"Misclassified: {np.sum(mask)} / {len(y_test)}")
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
