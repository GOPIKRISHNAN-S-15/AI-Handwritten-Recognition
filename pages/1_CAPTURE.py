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
col_mode, col_model = st.columns(2)

with col_mode:
    input_mode = st.selectbox(
        "INPUT MODE",
        ["Interactive Drawing Canvas", "Upload Manuscript Image"],
        key="input_mode",
    )

with col_model:
    model_options = []
    if mnist_model is not None:
        model_options.append("DIGITS -> MNIST")
    if emnist_model is not None:
        model_options.append("CHARACTERS / MIXED -> EMNIST")

    if not model_options:
        st.error("⚠️ No trained models found in `models/`. Please train models first.")
        st.stop()

    selected_model = st.selectbox("ACTIVE MODEL", model_options, key="model_select")

# Resolve model and mapping
if selected_model == "DIGITS -> MNIST":
    model = mnist_model
    class_mapping = MNIST_MAPPING
    model_type = "mnist"
else:
    model = emnist_model
    class_mapping = EMNIST_BALANCED_MAPPING
    model_type = "emnist"

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════
input_image = None
active_step = "IMAGE_RECEIVED"

if "Canvas" in input_mode:
    st.markdown("""
    <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem; font-family: var(--font-mono);">
        DRAW MULTIPLE CHARACTERS IN THE WORKSPACE BELOW.
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("""
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
            """, unsafe_allow_html=True)

        if canvas_result.image_data is not None:
            canvas_img = canvas_result.image_data
            if np.any(canvas_img[:, :, :3] > 15):
                gray = cv2.cvtColor(canvas_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                input_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite('debug_real_canvas.png', input_image)
                active_step = "PREPROCESS"

    except ImportError:
        st.warning("⚠️ `streamlit-drawable-canvas-fix` is required for live canvas drawing. Please use image upload mode.")

else:
    st.markdown("""
    <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem; font-family: var(--font-mono);">
        UPLOAD A HANDWRITTEN IMAGE.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload handwritten characters",
        type=SUPPORTED_IMAGE_FORMATS,
        label_visibility="collapsed",
        key="char_upload",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            input_image = np.array(pil_image)
            active_step = "PREPROCESS"
        except Exception as e:
            st.error(f"⚠️ Could not decode image file: {type(e).__name__}")


# ══════════════════════════════════════════════
# RECOGNITION & PREPROCESSING PIPELINE
# ══════════════════════════════════════════════
if input_image is not None:
    st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
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
        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;'>SEGMENTATION VIEW</div>", unsafe_allow_html=True)
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
        print(f"--- INFERENCE: '{selected_model}' -> '{model_type}' ({model.output_shape[-1]} classes) ---")
        
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
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1rem; text-align: center;">
                <div style="font-family: var(--font-primary); font-size: 2.5rem; font-weight: 600; color: var(--text-primary);">{char}</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;">{conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
    reconstructed_text = "".join([r["predicted_character"] for r in results])
    
    st.markdown(f"""
    <div style="background: var(--bg-secondary); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem; text-align: center; margin-top: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">RECONSTRUCTED TEXT</div>
        <div style="font-family: var(--font-primary); font-size: 3rem; font-weight: 600; color: var(--text-primary); letter-spacing: 4px;">{reconstructed_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gemini AI Quick Refinement Link ──
    active_step = "GEMINI_REFINE"
    st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    render_section_hud_header("LANGUAGE LAYER ENHANCEMENT")

    if genai_available:
        if st.button("RUN LANGUAGE ENHANCEMENT", key="btn_gemini_char"):
            with st.spinner("Processing through Language Layer..."):
                prompt = f"The CNN recognized the handwritten text as '{reconstructed_text}'. If this looks like a misspelling of a common word, or a slight error in a sequence, provide the corrected text. Output only the corrected text."
                gem_res = genai_service._generate(prompt, "char_verify") if hasattr(genai_service, '_generate') else genai_service.correct_text(reconstructed_text)
                if gem_res.success:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--accent-green); border-radius: 4px; padding: 1rem; margin-top: 1rem;">
                        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent-green); margin-bottom: 0.5rem;">ENHANCED TEXT</div>
                        <div style="font-family: var(--font-primary); font-size: 1.2rem; color: var(--text-primary);">{gem_res.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(gem_res.error)
    else:
        st.info("Language Layer is currently in standby.")
