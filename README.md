# 🧠 AI Handwritten OCR — Document Digitization

An AI-powered handwriting recognition and document digitization platform combining CNN-based visual recognition with Generative AI intelligence.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Project Overview

This application accepts handwritten digits, characters, words, or document images and converts them into machine-readable digital text through a multi-stage AI pipeline:

1. **Image Quality Analysis** — Detects brightness, contrast, noise, and skew
2. **Adaptive Preprocessing** — Applies appropriate transformations using OpenCV
3. **Character Segmentation** — Detects and extracts individual characters
4. **CNN Recognition** — Convolutional Neural Network classifies each character
5. **Confidence Analysis** — Provides prediction certainty and alternatives
6. **GenAI Enhancement** — Contextual correction, summarization, and insights

The CNN performs the actual recognition. The Generative AI layer provides enhancement and does **not** replace the trained ML model.

---

## ✨ Features

### Core Recognition
- ✍️ **Single Character Recognition** — Upload or draw a character/digit
- 📄 **Document Digitization** — Full document processing pipeline
- 🎨 **Drawing Canvas** — Draw characters for instant recognition
- 🔍 **Adaptive Preprocessing** — Smart image analysis and transformation

### AI Intelligence
- 🧠 **CNN-Based Classification** — MNIST (digits) + EMNIST (alphanumeric)
- 📊 **Confidence Visualization** — Animated bars and gauges
- ⚠️ **Low-Confidence Warnings** — Actionable recommendations
- 🔮 **Alternative Predictions** — Top-N predictions with probabilities

### Generative AI
- ✨ **Contextual Text Correction** — Fix OCR errors using Gemini
- 📋 **Document Summarization** — AI-generated summaries
- 🔍 **Key Information Extraction** — Names, dates, numbers, tasks
- 💡 **Contextual Insights** — Document type, structure analysis

### Analytics & Visualization
- 📊 **Interactive Charts** — Plotly training curves and distributions
- 🔢 **Confusion Matrix** — Seaborn heatmaps with zoom
- ❌ **Error Analysis** — Commonly confused character pairs
- 📈 **Per-Class Metrics** — Precision, Recall, F1 per class

---

## 🏗️ Architecture

```
handwritten-ai/
├── app.py                          # Main Streamlit entry point
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── pages/
│   ├── 1_✍️_Recognition.py         # Character recognition
│   ├── 2_📄_Document_Digitization.py
│   ├── 3_📊_Analytics.py           # Data analytics dashboard
│   ├── 4_🧠_Model_Intelligence.py  # Evaluation & error analysis
│   ├── 5_✨_GenAI_Insights.py      # GenAI panel
│   └── 6_ℹ️_About.py
├── models/
│   ├── cnn_model.py                # CNN architectures
│   ├── mnist_model.keras           # Trained MNIST model
│   └── emnist_model.keras          # Trained EMNIST model
├── preprocessing/
│   ├── image_processor.py          # Adaptive preprocessing
│   └── segmentation.py             # Document segmentation
├── genai/
│   └── ai_service.py               # Gemini API integration
├── analytics/
│   └── model_analysis.py           # Charts and metrics
├── utils/
│   ├── helpers.py                  # Prediction utilities
│   ├── constants.py                # Configuration
│   └── ui_components.py            # Custom UI components
├── styles/
│   └── main.css                    # Dark glassmorphism theme
├── training/
│   ├── train_model.py              # Standalone training script
│   ├── training_history_*.json     # Saved training metrics
│   └── evaluation_*.json           # Saved evaluation results
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Datasets

| Dataset | Classes | Train | Test | Image Size |
|---------|---------|-------|------|------------|
| **MNIST** | 10 (digits 0-9) | 60,000 | 10,000 | 28×28 |
| **EMNIST Balanced** | 47 (digits + letters) | 112,800 | 18,800 | 28×28 |

---

## 🧠 CNN Model

### MNIST Architecture (~99.3% accuracy)
```
Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
Flatten → Dense(256) → BN → Dropout(0.5) → Dense(10, softmax)
```

### EMNIST Architecture (~86-89% accuracy)
```
Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.25)
Flatten → Dense(512) → BN → Dropout(0.5) → Dense(47, softmax)
```

Both models use data augmentation (rotation, shift, zoom, shear) during training.

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ai-handwritten-ocr.git
cd ai-handwritten-ocr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Train Models (One-Time)

**Quick start (MNIST digits only, ~10-15 min):**
```bash
python training/train_mnist_only.py
```

**Full training (MNIST + EMNIST alphanumeric, ~45-75 min):**
```bash
python training/train_model.py
```

This will:
- Download MNIST and/or EMNIST datasets automatically
- Train the CNN model(s) with data augmentation
- Save trained models to `models/`
- Save training history and evaluation metrics to `training/`

### Configure GenAI API (Optional)

1. Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your-api-key-here
```

> **Security:** Never commit `.env` to version control. It is already listed in `.gitignore`.

> **Note:** The app works fully for CNN recognition without a GenAI API key.

---

## 💻 Running Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (**ensure `.env` is NOT committed**)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `app.py`
5. In the app settings → **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
6. Deploy!

> The app reads `GEMINI_API_KEY` from Streamlit Secrets on Cloud, or from `.env` locally.

---

## 📸 Screenshots

*Screenshots will be added after first deployment.*

---

## ⚠️ Limitations

- Character segmentation works best on well-spaced handwriting
- EMNIST accuracy (~87%) is lower than MNIST (~99%) due to 47-class complexity
- Document digitization assumes simple left-to-right, top-to-bottom layout
- Very cursive or overlapping handwriting may not segment correctly
- GenAI features require an active internet connection and API key

---

## 🔮 Future Improvements

- Support for more languages and scripts
- Transformer-based sequence recognition for whole words
- Custom fine-tuning on user-provided handwriting samples
- PDF document input support
- Real-time video/camera recognition
- On-device model optimization (TFLite)

---

## 📋 Academic Evaluation Alignment

| Component | Coverage |
|-----------|----------|
| **SSA 1 — Core AI** | CNN, MNIST/EMNIST, training, evaluation, NumPy, Pandas |
| **SSA 2 — Visualization** | Matplotlib, Seaborn, Plotly, confusion matrix, error analysis |
| **AL 1 — Generative AI** | Gemini API, correction, summarization, extraction, insights |
| **AL 2 — Deployment** | Streamlit app, model integration, monitoring, error handling |

---

## 📄 License

This project is developed for academic purposes.
