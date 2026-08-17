"""
CNN model architectures for MNIST and EMNIST handwriting recognition.
"""

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10) -> keras.Model:
    """
    Build a CNN for MNIST digit recognition (10 classes).

    Architecture:
        Conv2D(32) → BN → Conv2D(32) → Pool → Drop(0.25)
        Conv2D(64) → BN → Conv2D(64) → Pool → Drop(0.25)
        Flatten → Dense(256) → BN → Drop(0.5) → Dense(10, softmax)

    Expected accuracy: ~99.3% on MNIST test set.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def build_emnist_cnn(input_shape=(28, 28, 1), num_classes=47) -> keras.Model:
    """
    Build a CNN for EMNIST Balanced recognition (47 classes).

    Architecture:
        Conv2D(32) → BN → Conv2D(32) → Pool → Drop(0.25)
        Conv2D(64) → BN → Conv2D(64) → Pool → Drop(0.25)
        Conv2D(128) → BN → Pool → Drop(0.25)
        Flatten → Dense(512) → BN → Drop(0.5) → Dense(47, softmax)

    Expected accuracy: ~86-89% on EMNIST Balanced test set.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


@st.cache_resource
def load_trained_model(model_type: str = "mnist") -> keras.Model:
    """
    Load a pre-trained model from disk.

    Uses Streamlit's cache_resource to avoid reloading on every interaction.

    Args:
        model_type: "mnist" or "emnist"

    Returns:
        Loaded Keras model, or None if file not found.
    """
    if model_type == "mnist":
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mnist_model.keras')
    elif model_type == "emnist":
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'emnist_model.keras')
    else:
        return None

    path = os.path.normpath(path)

    if not os.path.exists(path):
        return None

    try:
        model = keras.models.load_model(path)
        expected_classes = 10 if model_type == "mnist" else 47
        if model.output_shape[-1] != expected_classes:
            print(f"Error: model {model_type} loaded with {model.output_shape[-1]} classes, expected {expected_classes}.")
            return None
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None


def get_model_summary_text(model: keras.Model) -> str:
    """Get model summary as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
