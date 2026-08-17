# Deployment & Environment Setup Guide

This document describes how to reproduce the working environment for the
AI Handwritten Recognition & Document Digitization platform, as verified
on **2026-08-15**. All statements below reflect real, measured behavior of
the current codebase.

## Verified Environment

The system was validated with the following stack, and these versions are
what `requirements.txt` enforces:

| Component | Version used for validation | Notes |
|-----------|------------------------------|-------|
| Python | 3.11+ | 3.10+ acceptable |
| TensorFlow | 2.16+ (CPU) | GPU not required |
| Streamlit | 1.61.1 | `st.html()` and `st.switch_page()` are used; do not downgrade below 1.45 |
| OpenCV | 4.9+ | `opencv-python-headless` |
| google-genai | 2.18.1 | New `google-genai` client (replaces obsolete `google.generativeai`) |
| streamlit-drawable-canvas-fix | 0.9.3+ | Required for the live drawing canvas on the CAPTURE page |
| python-dotenv | 1.0.0+ | Loads `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env` |
| emnist | any | Optional; training script prefers `tensorflow-datasets` |

## Installation Steps (Windows)

```powershell
cd "D:\7th Sem\AI&AP Project"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The critical post-install step that is easy to miss: **`streamlit-drawable-canvas-fix`
must be installed**, otherwise the CAPTURE page falls back to image-upload-only mode
with a visible warning.

## Environment Variables

Create `.env` in the project root (never commit it — already in `.gitignore`):

```env
GEMINI_API_KEY=your-google-ai-studio-key
GEMINI_MODEL=gemini-3.5-flash
```

`GEMINI_MODEL` defaults to `gemini-3.5-flash` when absent, and the service probes
the connection at startup, falling back through known-good model identifiers if the
configured one is rejected. A 404 on Gemini is always a model-identifier problem,
never a network problem.

## Model Files Shipped and Verified

| File | Purpose | Status |
|------|---------|--------|
| `models/mnist_model.keras` | Digits CNN (10 classes) | Present, validated 98.82% on real test set |
| `models/emnist_model.keras` | EMNIST Balanced CNN (47 classes) | Present, trained & evaluated — measured 89.51% |
| `training/evaluation_mnist.json` | MNIST per-class evaluation | Present, loaded by Analytics |
| `training/evaluation_emnist.json` | EMNIST per-class evaluation | Present, loaded by Analytics |
| `training/training_history_*.json` | Training curves for Plotly | Present for both models |

Retraining is optional: `python training/train_emnist.py` retrains the EMNIST
model from scratch (~45–75 min CPU). `python training/eval_only_emnist.py`
re-evaluates without retraining.

## Running and Verifying

```powershell
streamlit run app.py
```

Then run the verification suite to confirm nothing is mocked:

```powershell
python tests/test_full_verification.py
```

Expected outcome as of validation: **41 PASS, 0 FAIL, 0 SKIP**. The suite performs
real inference on hundreds of genuine MNIST/EMNIST test images, real multi-digit
segmentation cases (`11`, `123`, `2026`, `98765`, `100`), and one live Gemini API
request. If any test fails, do not treat the app as production-ready.

## Known Runtime Behaviors

The `fastNlMeansDenoising` denoiser is deliberately skipped on dense multi-digit
canvases (ink ratio > 10%) because smoothing destroys stroke junctions the
segmenter needs; it remains active for scanned documents with sparse ink.
Gemini requests on the free tier are subject to quota limits and may be
rate-limited during heavy use; the UI reports this gracefully. The sidebar
`GENAI: STANDBY` indicator reflects the one-time startup probe, not a permanent
disconnection — the service re-probes on each language-layer request.
