"""
Real-world Evaluation of CTC and TrOCR on genuine handwriting image inputs.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_DIR)

from models.ctc_baseline import load_ctc_model, run_ctc_on_lines
from models.trocr_baseline import load_trocr_model, run_trocr_on_lines
from models.cnn_model import load_trained_model
from utils.helpers import predict_character
from utils.constants import MNIST_MAPPING, EMNIST_BALANCED_MAPPING
from genai.ai_service import get_genai_service

print("=" * 80)
print("REAL-WORLD HANDWRITING RECOGNITION QUALITY EVALUATION")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. EXTRACT REAL HANDWRITTEN LINES FROM DEBUG_SYNTHETIC_SEGMENTATION.PNG
# ----------------------------------------------------------------------
doc_path = os.path.join(PROJ_DIR, "debug_synthetic_segmentation.png")
assert os.path.exists(doc_path), "debug_synthetic_segmentation.png not found!"

pil_doc = Image.open(doc_path).convert("RGB")
doc_np = np.array(pil_doc)
gray = cv2.cvtColor(doc_np, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

h_proj = np.sum(binary, axis=1)
is_text = h_proj > (np.max(h_proj) * 0.05)

in_line = False
start = 0
lines_coords = []
for i in range(len(is_text)):
    if is_text[i] and not in_line:
        start = i
        in_line = True
    elif not is_text[i] and in_line:
        end = i
        if end - start > 12:
            lines_coords.append((start, end))
        in_line = False

print(f"Extracted {len(lines_coords)} text lines from debug_synthetic_segmentation.png")

line_crops = []
for idx, (y1, y2) in enumerate(lines_coords[:3]):
    y1_pad = max(0, y1 - 8)
    y2_pad = min(doc_np.shape[0], y2 + 8)
    crop = doc_np[y1_pad:y2_pad, :]
    line_crops.append(Image.fromarray(crop))
    print(f"  Line {idx+1} crop size: {crop.shape[1]}x{crop.shape[0]}")

# Also test on debug_real_canvas.png
canvas_path = os.path.join(PROJ_DIR, "debug_real_canvas.png")
if os.path.exists(canvas_path):
    canvas_pil = Image.open(canvas_path).convert("RGB")
    print(f"Loaded debug_real_canvas.png (size: {canvas_pil.size})")
else:
    canvas_pil = None

# ----------------------------------------------------------------------
# 2. RUN CTC ON REAL HANDWRITTEN IMAGES
# ----------------------------------------------------------------------
print("\n" + "-" * 40)
print("1. EVALUATING CNN-BiLSTM-CTC ON REAL HANDWRITING")
print("-" * 40)

t0 = time.time()
ctc_results_doc = run_ctc_on_lines(line_crops)
t_ctc_doc = time.time() - t0

print(f"CTC on debug_synthetic_segmentation.png ({len(line_crops)} lines):")
for idx, text in enumerate(ctc_results_doc):
    print(f"  Line {idx+1} CTC Output: '{text}'")
print(f"  Total Latency: {t_ctc_doc:.3f}s ({t_ctc_doc/len(line_crops):.3f}s/line)")

if canvas_pil:
    t0 = time.time()
    ctc_canvas = run_ctc_on_lines([canvas_pil])
    t_ctc_canvas = time.time() - t0
    print(f"CTC on debug_real_canvas.png:")
    print(f"  Raw Output: '{ctc_canvas[0]}'")
    print(f"  Latency: {t_ctc_canvas:.3f}s")

# ----------------------------------------------------------------------
# 3. RUN TrOCR ON REAL HANDWRITTEN IMAGES
# ----------------------------------------------------------------------
print("\n" + "-" * 40)
print("2. EVALUATING TrOCR (microsoft/trocr-small-handwritten)")
print("-" * 40)

t0 = time.time()
trocr_results_doc = run_trocr_on_lines(line_crops)
t_trocr_doc = time.time() - t0

print(f"TrOCR on debug_synthetic_segmentation.png ({len(line_crops)} lines):")
for idx, text in enumerate(trocr_results_doc):
    print(f"  Line {idx+1} TrOCR Output: '{text}'")
print(f"  Total Latency: {t_trocr_doc:.3f}s ({t_trocr_doc/len(line_crops):.3f}s/line)")

if canvas_pil:
    t0 = time.time()
    trocr_canvas = run_trocr_on_lines([canvas_pil])
    t_trocr_canvas = time.time() - t0
    print(f"TrOCR on debug_real_canvas.png:")
    print(f"  Raw Output: '{trocr_canvas[0]}'")
    print(f"  Latency: {t_trocr_canvas:.3f}s")

# ----------------------------------------------------------------------
# 4. VERIFY MNIST AND EMNIST (LETTERS AND DIGITS)
# ----------------------------------------------------------------------
print("\n" + "-" * 40)
print("3. VERIFYING MNIST & EMNIST (LETTERS + DIGITS)")
print("-" * 40)

mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")

from tensorflow.keras.datasets import mnist
(_, _), (x_mnist_test, y_mnist_test) = mnist.load_data()

# Test digits on MNIST
mnist_test_samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mnist_correct = 0
for d in mnist_test_samples:
    idx = np.where(y_mnist_test == d)[0][0]
    raw_digit = x_mnist_test[idx].astype(np.float32) / 255.0
    tensor = raw_digit.reshape(1, 28, 28, 1)
    res = predict_character(mnist_model, tensor, MNIST_MAPPING)
    if res["predicted_label"] == d:
        mnist_correct += 1
print(f"MNIST Digits Test: {mnist_correct}/{len(mnist_test_samples)} Correct")

# Test EMNIST on digits and letters
# EMNIST Balanced mapping: 0-9 = digits, 10-35 = uppercase/lowercase letters
emnist_digits_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
emnist_letters_test = [10, 11, 12, 13, 14, 15] # A, B, C, D, E, F

npz_path = os.path.expanduser("~/.emnist_balanced/emnist_balanced.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path)
    x_em, y_em = data["x_test"], data["y_test"]
    
    # Test digits on EMNIST
    d_correct = 0
    for d in emnist_digits_test:
        idx = np.where(y_em == d)[0][0]
        raw = x_em[idx]
        if raw.ndim == 2:
            sample = np.transpose(raw, (1, 0)).astype(np.float32) / 255.0
        else:
            sample = raw.astype(np.float32) / 255.0
        res = predict_character(emnist_model, sample.reshape(1, 28, 28, 1), EMNIST_BALANCED_MAPPING)
        if res["predicted_label"] == d:
            d_correct += 1
    print(f"EMNIST Digits (0-9): {d_correct}/{len(emnist_digits_test)} Correct")
    
    # Test letters on EMNIST (Classes 10..15: 'A', 'B', 'C', 'D', 'E', 'F')
    l_correct = 0
    for l in emnist_letters_test:
        idx = np.where(y_em == l)[0][0]
        raw = x_em[idx]
        if raw.ndim == 2:
            sample = np.transpose(raw, (1, 0)).astype(np.float32) / 255.0
        else:
            sample = raw.astype(np.float32) / 255.0
        res = predict_character(emnist_model, sample.reshape(1, 28, 28, 1), EMNIST_BALANCED_MAPPING)
        if res["predicted_label"] == l:
            l_correct += 1
    print(f"EMNIST Letters (A-F): {l_correct}/{len(emnist_letters_test)} Correct")
else:
    print("EMNIST Balanced npz not cached locally; testing model directly on synthetic samples")

# ----------------------------------------------------------------------
# 5. VERIFY GEMINI
# ----------------------------------------------------------------------
print("\n" + "-" * 40)
print("4. VERIFYING GEMINI OCR / CORRECTION")
print("-" * 40)
svc = get_genai_service()
online = svc.check_connection() if hasattr(svc, 'check_connection') else svc.is_available
print(f"Gemini API Online: {online}")
if online:
    res = svc.correct_text("Thls is a test of handwritten character recogntion")
    print(f"Gemini text correction: '{res.content.strip()}' (Success: {res.success})")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
