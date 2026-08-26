"""
Synthetic Intelligence Research Interface (SIRI) — Capture Module
Multi-character classification with drawing canvas and high-res upload.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header, render_prediction_hud,
    render_candidate_bars, render_reconstruction_pipeline
)
from utils.constants import (
    MNIST_MAPPING, EMNIST_BALANCED_MAPPING, SUPPORTED_IMAGE_FORMATS,
)
from utils.helpers import (
    predict_character, generate_recommendations, generate_report,
    generate_report_json, image_to_bytes,
)
from models.cnn_model import load_trained_model
from preprocessing.image_processor import AdaptivePreprocessor
from genai.ai_service import get_genai_service

# ── Page Config ──
st.set_page_config(page_title="CAPTURE MODULE — HWR LAB", page_icon="🔬", layout="wide")
load_css()

# ── System Runtime Checks ──
mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")
cnn_loaded = mnist_model is not None or emnist_model is not None
genai_service = get_genai_service()
genai_available = genai_service.check_connection() if hasattr(genai_service, 'check_connection') else genai_service.is_available

# ── TopAppBar HUD ──
render_top_app_bar(
    cnn_online=cnn_loaded,
    genai_online=genai_available,
)

# ── Sidebar Drawer ──
render_sidebar_drawer(cnn_loaded, genai_available)

# ── Header ──
render_section_hud_header(
    "CAPTURE MODULE",
    "Multi-character neural input (tactile drawing canvas or image upload) with real-time adaptive preprocessing and segmentation."
)

# ── Model Selection & Mode ──
col_mode, col_info = st.columns(2)

with col_mode:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1rem;">
    <div style="font-family: var(--font-primary); font-size: 1rem; color: var(--text-primary); font-weight: 600;">Interactive Drawing Canvas</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-green); margin-top: 4px;">● Active Input Mode</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Direct tactile character input</div>
</div>
""")
    input_mode = "Interactive Drawing Canvas"

with col_info:
    st.html("""
<div style="background: var(--bg-card); border: 1px solid var(--accent-orange); border-radius: 4px; padding: 1rem;">
    <div style="font-family: var(--font-primary); font-size: 1rem; color: var(--accent-orange); font-weight: 600;">Custom CNN</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">EMNIST Balanced · 47 Classes</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Test Accuracy: 89.5%</div>
</div>
""")

model = emnist_model
class_mapping = EMNIST_BALANCED_MAPPING
model_type = "emnist"

st.html("<div style='height: 10px;'></div>")

# ══════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════
input_image = None
active_step = "IMAGE_RECEIVED"

if "Canvas" in input_mode:
    st.html("""
<div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem; font-family: var(--font-mono);">
    DRAW MULTIPLE CHARACTERS IN THE WORKSPACE BELOW.
</div>
""")

    try:
        from streamlit_drawable_canvas import st_canvas

        col_canvas, col_info = st.columns([2, 1])
        with col_canvas:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=12,
                stroke_color="#FFFFFF",
                background_color="#1a1a1a",
                height=280,
                width=600,
                drawing_mode="freedraw",
                key="drawing_canvas",
                display_toolbar=True,
            )

        with col_info:
            st.html("""
<div class="glass-panel" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem;">
    <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-primary); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem; font-family: var(--font-mono);">
        WORKSPACE GUIDELINES
    </div>
    <ul style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.8; margin: 0; padding-left: 1.2rem; font-family: var(--font-mono);">
        <li>Draw left-to-right</li>
        <li>Ensure clear separation between characters</li>
        <li>Auto-segmentation is active</li>
        <li>Use trash bin to clear</li>
    </ul>
</div>
""")

        if canvas_result.image_data is not None:
            canvas_img = canvas_result.image_data
            if np.any(canvas_img[:, :, :3] > 15):
                gray = cv2.cvtColor(canvas_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                input_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite('debug_real_canvas.png', input_image)
                active_step = "PREPROCESS"

    except ImportError:
        st.warning("⚠️ `streamlit-drawable-canvas-fix` is required for live canvas drawing.")


# ══════════════════════════════════════════════
# RECOGNITION & PREPROCESSING PIPELINE
# ══════════════════════════════════════════════
if input_image is not None:
    st.html("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>")
    
    preprocessor = AdaptivePreprocessor(target_size=(28, 28))
    
    # Segment Characters
    active_step = "SEGMENTATION"
    segments = preprocessor.segment_characters(input_image)
    
    # Draw bounding boxes on original image for display
    display_img = input_image.copy()
    for (crop, (x, y, w, h)) in segments:
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (52, 211, 153), 2)
    
    col_img, col_timeline = st.columns([1, 1])
    with col_img:
        st.html("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;'>SEGMENTATION VIEW</div>")
        st.image(display_img, channels="BGR", width='stretch')
        
    with col_timeline:
        render_reconstruction_pipeline(active_step="CNN_CLASSIFY")

    # ── CNN Neural Inference ──
    active_step = "CNN_CLASSIFY"
    render_section_hud_header("RECOGNITION RESULTS")
    
    input_hash = hash(input_image.tobytes())
    model_key = f"{model_type}_{input_hash}"
    
    if st.session_state.get("last_model_key") != model_key:
        st.session_state["last_model_key"] = model_key
        print(f"--- INFERENCE: '{model_type}' ({model.output_shape[-1]} classes) ---")
        
        results = []
        for i, (crop, (x, y, w, h)) in enumerate(segments):
            processed_model, analysis = preprocessor.preprocess(crop, for_model=True)
            result = predict_character(model, processed_model, class_mapping)
            results.append(result)
            
        st.session_state["last_results"] = results
    else:
        results = st.session_state.get("last_results", [])
    
    # Create columns for each character
    cols = st.columns(max(len(segments), 1))
    
    for i, result in enumerate(results):
        
        with cols[i]:
            char = result["predicted_character"]
            conf = result["confidence"] * 100
            st.html(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1rem; text-align: center;">
    <div style="font-family: var(--font-primary); font-size: 2.5rem; font-weight: 600; color: var(--text-primary);">{char}</div>
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;">{conf:.1f}%</div>
</div>
""")
            
    reconstructed_text = "".join([r["predicted_character"] for r in results])
    
    st.html(f"""
<div style="background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem; text-align: center; margin-top: 1.5rem;">
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">RECONSTRUCTED TEXT</div>
    <div style="font-family: var(--font-primary); font-size: 3rem; font-weight: 600; color: var(--text-primary); letter-spacing: 4px;">{reconstructed_text}</div>
</div>
""")

    st.html("<div style='height: 15px;'></div>")
    
    # ── Export Buttons ──
    if reconstructed_text:
        from utils.export_helper import generate_txt, generate_pdf, generate_docx
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📄 Download TXT", data=generate_txt(reconstructed_text), file_name="capture_output.txt", mime="text/plain", key="btn_capture_txt", use_container_width=True)
        with c2:
            st.download_button("📄 Download PDF", data=generate_pdf(reconstructed_text), file_name="capture_output.pdf", mime="application/pdf", key="btn_capture_pdf", use_container_width=True)
        with c3:
            st.download_button("📄 Download DOCX", data=generate_docx(reconstructed_text), file_name="capture_output.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="btn_capture_docx", use_container_width=True)
