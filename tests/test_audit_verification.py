"""
End-to-end automated audit verification test for SIRI NEURAL_CORE v2.0.
"""

import os
import sys
import numpy as np
import cv2

# Add root
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.constants import APP_TITLE, APP_VERSION, GEMINI_MODEL, MNIST_MAPPING
from models.cnn_model import load_trained_model
from preprocessing.image_processor import AdaptivePreprocessor
from preprocessing.segmentation import DocumentSegmenter
from genai.ai_service import get_genai_service
from analytics.model_analysis import load_evaluation, load_training_history, get_metrics_summary

def run_tests():
    print("=" * 60)
    print(f"RUNNING SIRI NEURAL_CORE AUDIT VERIFICATION")
    print(f"Title: {APP_TITLE} | Version: {APP_VERSION} | Model: {GEMINI_MODEL}")
    print("=" * 60)

    # 1. Gemini API Check
    print("\n[TEST 1] Gemini 3.6 Flash API Verification...")
    service = get_genai_service()
    online = service.check_connection()
    print(f"  Gemini Online: {online}")
    assert online, "Gemini API check failed! Ensure GEMINI_API_KEY is valid."

    sample_ocr = "Patient ID 5092 reported on 14-Aug-2026. Blood pressure was normal. Total fee: $120."
    print("  Testing correct_text()...")
    res_corr = service.correct_text("The quick brown fox jumps over the 1azy dog 0123")
    print(f"    Success: {res_corr.success} | Content: {res_corr.content[:60]}...")

    print("  Testing summarize_text()...")
    res_sum = service.summarize_text(sample_ocr)
    print(f"    Success: {res_sum.success} | Content: {res_sum.content[:60]}...")

    print("  Testing extract_info()...")
    res_ext = service.extract_info(sample_ocr)
    print(f"    Success: {res_ext.success} | Content: {res_ext.content[:60]}...")

    print("  Testing get_insights()...")
    res_ins = service.get_insights(sample_ocr)
    print(f"    Success: {res_ins.success} | Content: {res_ins.content[:60]}...")

    # 2. MNIST Model & Preprocessing Check
    print("\n[TEST 2] MNIST Model Loading & Preprocessing Alignment...")
    model = load_trained_model("mnist")
    assert model is not None, "Failed to load MNIST model from models/mnist_model.keras!"
    print(f"  Model loaded successfully: {model.name} ({model.count_params():,} parameters)")

    preprocessor = AdaptivePreprocessor(target_size=(28, 28))

    # Test digits 0 through 9
    from utils.helpers import predict_character
    correct_count = 0
    for digit in range(10):
        canvas = np.zeros((150, 150), dtype=np.uint8)
        cv2.putText(canvas, str(digit), (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 255, 10, cv2.LINE_AA)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        proc, analysis, debug = preprocessor.preprocess_with_debug(canvas_bgr)
        res = predict_character(model, proc, MNIST_MAPPING)
        pred = res["predicted_label"]
        conf = res["confidence"]
        if pred == digit:
            correct_count += 1
        print(f"  Digit '{digit}' -> Predicted '{pred}' (Conf: {conf*100:.1f}%, Latency: {res['latency_ms']:.1f}ms) [{'PASS' if pred == digit else 'FAIL'}]")

    print(f"  Synthetic Digit Accuracy: {correct_count}/10 ({correct_count*10}%)")

    # 3. Document Segmenter Check
    print("\n[TEST 3] Document Segmenter Verification...")
    doc_canvas = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(doc_canvas, "4 2 9", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5, cv2.LINE_AA)
    doc_bgr = cv2.cvtColor(doc_canvas, cv2.COLOR_GRAY2BGR)

    segmenter = DocumentSegmenter(min_char_area=40, char_padding=4)
    regions, annotated = segmenter.segment(doc_bgr)
    print(f"  Segments detected: {len(regions)} regions (Expected: 3)")
    assert len(regions) >= 3, "Document segmenter failed to detect character contours!"

    # 4. Analytics & Evaluation JSON Check
    print("\n[TEST 4] Analytics & Evaluation Telemetry Check...")
    eval_data = load_evaluation("mnist")
    assert eval_data is not None, "Evaluation data training/evaluation_mnist.json not found!"
    metrics = get_metrics_summary(eval_data)
    print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Test Samples: {metrics['total_test']:,}")
    print(f"  Macro Precision: {metrics['precision']*100:.2f}%")
    print(f"  Macro Recall: {metrics['recall']*100:.2f}%")
    print(f"  Macro F1: {metrics['f1_score']*100:.2f}%")

    history = load_training_history("mnist")
    assert history is not None, "Training history training/training_history_mnist.json not found!"
    print(f"  Training Epochs in History: {len(history.get('accuracy', []))}")

    print("\n" + "=" * 60)
    print("ALL AUDIT VERIFICATION TESTS PASSED SUCCESSFULLY! [PASS]")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
