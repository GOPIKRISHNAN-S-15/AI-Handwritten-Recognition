"""
Application-wide constants, configuration, and dataset metadata.
"""

# ──────────────────────────────────────────────
# Application Info & Branding
# ──────────────────────────────────────────────
APP_TITLE = "NEURAL_CORE v2.0"
APP_SUBTITLE = "Synthetic Intelligence Research Interface"
APP_ICON = "⚡"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = (
    "Synthetic Intelligence Research Interface (SIRI) — "
    "A high-fidelity platform combining CNN neural recognition with Gemini AI semantic document analysis."
)

# ──────────────────────────────────────────────
# Model Paths
# ──────────────────────────────────────────────
MNIST_MODEL_PATH = "models/mnist_model.keras"
EMNIST_MODEL_PATH = "models/emnist_model.keras"
MNIST_HISTORY_PATH = "training/training_history_mnist.json"
EMNIST_HISTORY_PATH = "training/training_history_emnist.json"
MNIST_EVAL_PATH = "training/evaluation_mnist.json"
EMNIST_EVAL_PATH = "training/evaluation_emnist.json"

# ──────────────────────────────────────────────
# Dataset Metadata
# ──────────────────────────────────────────────
MNIST_INFO = {
    "name": "MNIST",
    "full_name": "Modified National Institute of Standards and Technology",
    "num_classes": 10,
    "train_samples": 60000,
    "test_samples": 10000,
    "image_size": (28, 28),
    "channels": 1,
    "description": "Handwritten digit dataset (0-9)",
}

EMNIST_BALANCED_INFO = {
    "name": "EMNIST Balanced",
    "full_name": "Extended MNIST — Balanced Split",
    "num_classes": 47,
    "train_samples": 112800,
    "test_samples": 18800,
    "image_size": (28, 28),
    "channels": 1,
    "description": "Handwritten digits and letters (47 balanced classes)",
}

# ──────────────────────────────────────────────
# EMNIST Balanced Label Mapping (label → character)
# 0-9: digits '0'-'9'
# 10-35: uppercase 'A'-'Z'
# 36-46: lowercase letters that are visually distinct
#         from uppercase: a, b, d, e, f, g, h, n, q, r, t
# ──────────────────────────────────────────────
EMNIST_BALANCED_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't',
}

MNIST_MAPPING = {i: str(i) for i in range(10)}

# ──────────────────────────────────────────────
# CNN Hyperparameters
# ──────────────────────────────────────────────
MNIST_HYPERPARAMS = {
    "batch_size": 128,
    "epochs": 2,
    "learning_rate": 0.001,
    "validation_split": 0.1,
}

EMNIST_HYPERPARAMS = {
    "batch_size": 128,
    "epochs": 12,
    "learning_rate": 0.001,
    "validation_split": 0.1,
}

# ──────────────────────────────────────────────
# Image Processing
# ──────────────────────────────────────────────
TARGET_IMG_SIZE = (28, 28)
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.50

# ──────────────────────────────────────────────
# UI Theme Colors (Cyberpunk Glassmorphism)
# ──────────────────────────────────────────────
COLORS = {
    "bg_primary": "#111318",
    "bg_secondary": "#181a24",
    "bg_card": "rgba(24, 26, 36, 0.75)",
    "accent_cyan": "#00f2ff",
    "accent_purple": "#a855f7",
    "accent_pink": "#ec4899",
    "accent_green": "#10b981",
    "accent_orange": "#f59e0b",
    "accent_red": "#ef4444",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "glass_bg": "rgba(255, 255, 255, 0.04)",
    "glass_border": "rgba(255, 255, 255, 0.09)",
}

# ──────────────────────────────────────────────
# GenAI Settings
# ──────────────────────────────────────────────
# Currently supported Gemini model; the service reads GEMINI_MODEL from the
# environment (.env) and falls back through known-good models if this fails.
GEMINI_MODEL = "gemini-3.5-flash"
GENAI_TIMEOUT = 30  # seconds
GENAI_TEMPERATURE = 0.2  # low temperature for precision

# ──────────────────────────────────────────────
# Supported Upload Formats
# ──────────────────────────────────────────────
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "bmp", "webp"]

