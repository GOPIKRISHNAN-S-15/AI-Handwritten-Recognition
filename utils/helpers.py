"""
Shared helper utilities for prediction, confidence analysis,
and recommendation generation.
"""

import numpy as np
import io
import json
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image


def predict_character(
    model,
    image: np.ndarray,
    class_mapping: Dict[int, str],
    top_n: int = 5,
) -> Dict:
    """
    Run CNN inference on a preprocessed image.

    Args:
        model: Loaded Keras model.
        image: Preprocessed image (28x28, normalized 0-1).
        class_mapping: Label → character mapping.
        top_n: Number of top predictions to return.

    Returns:
        Dict with prediction, confidence, alternatives, entropy, and latency.
    """
    import time
    # Ensure correct shape: (1, 28, 28, 1)
    if image.ndim == 2:
        image = image.reshape(1, 28, 28, 1)
    elif image.ndim == 3:
        image = image.reshape(1, *image.shape)

    image = image.astype(np.float32)

    # Predict with timing
    t0 = time.perf_counter()
    probs = model.predict(image, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate prediction entropy: -sum(p * log(p))
    epsilon = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + epsilon)))

    # Top-N predictions
    top_indices = np.argsort(probs)[::-1][:top_n]

    predicted_label = top_indices[0]
    predicted_char = class_mapping.get(int(predicted_label), "?")
    confidence = float(probs[predicted_label])

    alternatives = []
    for idx in top_indices:
        alternatives.append({
            "label": int(idx),
            "character": class_mapping.get(int(idx), "?"),
            "confidence": float(probs[idx]),
            "percentage": float(probs[idx] * 100),
        })

    return {
        "predicted_label": int(predicted_label),
        "predicted_character": predicted_char,
        "confidence": confidence,
        "confidence_percentage": float(confidence * 100),
        "confidence_level": get_confidence_level(confidence),
        "alternatives": alternatives,
        "entropy": entropy,
        "latency_ms": latency_ms,
        "all_probabilities": probs.tolist(),
    }


def get_confidence_level(confidence: float) -> Dict:
    """
    Categorize confidence into levels with appropriate indicators.

    Args:
        confidence: Float between 0 and 1.

    Returns:
        Dict with level name, emoji, color, and description.
    """
    if confidence >= 0.85:
        return {
            "level": "high",
            "emoji": "✅",
            "color": "#10b981",
            "label": "High Confidence",
            "description": "The model is confident about this prediction.",
        }
    elif confidence >= 0.50:
        return {
            "level": "medium",
            "emoji": "⚠️",
            "color": "#f59e0b",
            "label": "Medium Confidence",
            "description": "The model has moderate certainty. Review alternatives.",
        }
    else:
        return {
            "level": "low",
            "emoji": "🔴",
            "color": "#ef4444",
            "label": "Low Confidence",
            "description": "The model is uncertain. Consider the alternatives below.",
        }


def generate_recommendations(
    confidence: float,
    analysis=None,
) -> List[str]:
    """
    Generate actionable recommendations based on confidence and image analysis.

    Args:
        confidence: Prediction confidence (0-1).
        analysis: Optional ImageAnalysis from preprocessing.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if confidence < 0.85:
        recommendations.append("Upload a clearer image with better handwriting")

    if analysis is not None:
        if analysis.is_low_contrast:
            recommendations.append("Increase the contrast — use darker ink on lighter paper")
        if analysis.is_noisy:
            recommendations.append("Reduce background noise — use a clean, uniform background")
        if analysis.is_dark:
            recommendations.append("Improve lighting — the image appears too dark")
        if analysis.is_bright:
            recommendations.append("Reduce glare — the image appears overexposed")
        if analysis.is_skewed:
            recommendations.append("Try to write more straight — the text appears rotated")
        if analysis.has_uneven_background:
            recommendations.append("Use a uniform background — shadows were detected")

    if confidence < 0.50:
        recommendations.append("Crop the image to show only the character")
        recommendations.append("Draw the character larger and more centered")

    if not recommendations:
        recommendations.append("The image quality looks good!")

    return recommendations


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """Convert a numpy array to image bytes."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image)

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()


def generate_report(
    prediction_result: Dict,
    analysis=None,
    enhanced_text: Optional[str] = None,
    model_type: str = "unknown",
) -> str:
    """
    Generate a downloadable recognition report.

    Args:
        prediction_result: Output from predict_character().
        analysis: ImageAnalysis from preprocessing.
        enhanced_text: Optional GenAI-enhanced text.
        model_type: "mnist" or "emnist".

    Returns:
        Report as formatted string.
    """
    lines = [
        "=" * 50,
        "  AI HANDWRITTEN OCR — RECOGNITION REPORT",
        "=" * 50,
        "",
        f"Model: {model_type.upper()}",
        f"Predicted Character: {prediction_result['predicted_character']}",
        f"Confidence: {prediction_result['confidence_percentage']:.2f}%",
        f"Confidence Level: {prediction_result['confidence_level']['label']}",
        "",
        "Top Predictions:",
        "-" * 30,
    ]

    for alt in prediction_result["alternatives"]:
        bar = "█" * int(alt["percentage"] / 5) + "░" * (20 - int(alt["percentage"] / 5))
        lines.append(f"  {alt['character']:>3s}  {bar}  {alt['percentage']:.2f}%")

    if analysis is not None:
        lines.extend([
            "",
            "Preprocessing Applied:",
            "-" * 30,
        ])
        for op in analysis.applied_operations:
            lines.append(f"  • {op}")

        lines.extend([
            "",
            "Image Analysis:",
            "-" * 30,
            f"  Brightness:  {analysis.brightness:.1f}",
            f"  Contrast:    {analysis.contrast:.1f}",
            f"  Noise Level: {analysis.noise_level:.1f}",
        ])

    if enhanced_text:
        lines.extend([
            "",
            "AI-Enhanced Text:",
            "-" * 30,
            f"  {enhanced_text}",
        ])

    recommendations = generate_recommendations(
        prediction_result["confidence"], analysis
    )
    lines.extend([
        "",
        "Recommendations:",
        "-" * 30,
    ])
    for rec in recommendations:
        lines.append(f"  • {rec}")

    lines.extend(["", "=" * 50])
    return "\n".join(lines)


def generate_report_json(
    prediction_result: Dict,
    analysis=None,
    enhanced_text: Optional[str] = None,
    model_type: str = "unknown",
) -> str:
    """Generate a JSON-formatted recognition report."""
    report = {
        "model_type": model_type,
        "prediction": prediction_result["predicted_character"],
        "confidence": prediction_result["confidence_percentage"],
        "confidence_level": prediction_result["confidence_level"]["label"],
        "alternatives": prediction_result["alternatives"],
    }

    if analysis is not None:
        report["preprocessing"] = {
            "operations_applied": analysis.applied_operations,
            "brightness": analysis.brightness,
            "contrast": analysis.contrast,
            "noise_level": analysis.noise_level,
            "is_skewed": analysis.is_skewed,
        }

    if enhanced_text:
        report["ai_enhanced_text"] = enhanced_text

    report["recommendations"] = generate_recommendations(
        prediction_result["confidence"], analysis
    )

    return json.dumps(report, indent=2)


def load_json_safe(path: str) -> Optional[Dict]:
    """Safely load a JSON file, returning None on failure."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
