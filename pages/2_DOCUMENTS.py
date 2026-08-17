"""
Synthetic Intelligence Research Interface (SIRI) — Document Digitization Pipeline
Bulk manuscript processing, contour/projection segmentation, and semantic extraction.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import time

from utils.ui_components import (
    load_css, render_top_app_bar, render_sidebar_drawer,
    render_section_hud_header, render_reconstruction_pipeline,
)
from utils.constants import MNIST_MAPPING, EMNIST_BALANCED_MAPPING, SUPPORTED_IMAGE_FORMATS
from models.cnn_model import load_trained_model
from preprocessing.image_processor import AdaptivePreprocessor
from preprocessing.segmentation import DocumentSegmenter, detect_spaces
from genai.ai_service import get_genai_service
from utils.helpers import predict_character, image_to_bytes

st.set_page_config(page_title="Document Digitization Pipeline — SIRI", page_icon="📄", layout="wide")
load_css()

# ── System Runtime Checks ──
mnist_model = load_trained_model("mnist")
emnist_model = load_trained_model("emnist")
cnn_loaded = mnist_model is not None or emnist_model is not None
genai_service = get_genai_service()
genai_available = genai_service.check_connection() if hasattr(genai_service, 'check_connection') else genai_service.is_available

# ── TopAppBar HUD ──
render_top_app_bar(
    title="DOCUMENT PIPELINE",
    version="v2.0",
    cnn_online=cnn_loaded,
    genai_online=genai_available,
)

# ── Sidebar Drawer ──
render_sidebar_drawer(cnn_loaded, genai_available)

# ── Header ──
render_section_hud_header(
    "Document Digitization Pipeline",
    "Bulk manuscript processing: line projection analysis, character bounding box segmentation, and neural text assembly",
    icon="📄",
)

# ── Model Selector ──
model_options = []
if mnist_model: model_options.append("MNIST (Digits 0-9) • High-Fidelity")
if emnist_model: model_options.append("EMNIST (Alphanumeric)")

if not model_options:
    st.error("⚠️ No trained models found. Please train MNIST model first.")
    st.stop()

selected_model = st.selectbox("Recognition Engine Core", model_options, key="doc_model")
if selected_model.startswith("MNIST"):
    model = mnist_model
    class_mapping = MNIST_MAPPING
    model_type = "mnist"
else:
    model = emnist_model
    class_mapping = EMNIST_BALANCED_MAPPING
    model_type = "emnist"


render_reconstruction_pipeline(active_step="SEG_LINE/WORD")

# ══════════════════════════════════════════════
# MANUSCRIPT UPLOAD
# ══════════════════════════════════════════════
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem;">
    Upload a handwritten document image. The pipeline will detect text lines, isolate character regions, and perform CNN classification.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload document image", type=SUPPORTED_IMAGE_FORMATS, label_visibility="collapsed", key="doc_upload")

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        input_image = np.array(pil_image)

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            st.markdown("##### 📷 Uploaded Manuscript")
            st.image(input_image, width='stretch')

        with col_in2:
            st.markdown("##### ⚙️ Pipeline Control")
            st.markdown("""
            <div class="glass-panel" style="padding: 1.2rem;">
                <div style="font-size: 0.75rem; font-weight: 700; color: var(--accent-cyan); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.6rem;">
                    RECONSTRUCTION PARAMETERS
                </div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.8;">
                    • <strong>Segmentation:</strong> Adaptive Otsu Gaussian + Projection Profile<br>
                    • <strong>Character Normalization:</strong> 20×20 Centered Bounding Box<br>
                    • <strong>Classification Core:</strong> TensorFlow Conv2D Layer Stack<br>
                </div>
            </div>
            """, unsafe_allow_html=True)

            run_digitize = st.button("🚀 Run Neural Digitization Pipeline", key="digitize_btn", type="primary", width='stretch')

        if run_digitize:
            with st.spinner("Processing document reconstruction pipeline..."):
                # Use DocumentSegmenter
                segmenter = DocumentSegmenter(min_char_area=40, char_padding=0)
                character_regions, annotated_img = segmenter.segment(input_image)

                preprocessor = AdaptivePreprocessor(target_size=(28, 28))
                recognized_chars = []
                confidences = []

                print(f"--- INFERENCE (DOCUMENTS): '{selected_model}' -> '{model_type}' ({model.output_shape[-1]} classes) ---")

                # Classify each segmented character
                saved_crops = 0
                for region in character_regions:
                    if region.image is not None and region.image.size > 0:
                        char_img_rgb = cv2.cvtColor(region.image, cv2.COLOR_GRAY2BGR) if len(region.image.shape) == 2 else region.image
                        processed, _ = preprocessor.preprocess(char_img_rgb, for_model=True)
                        
                        if saved_crops < 4:
                            # Save the visual crop that goes into the preprocessor
                            cv2.imwrite(f"debug_doc_crop_{saved_crops}.png", region.image)
                            saved_crops += 1
                            
                        res = predict_character(model, processed, class_mapping, top_n=1)
                        predicted_char = res["predicted_character"]
                        conf = res["confidence_percentage"]
                        region.confidence = res["confidence"]
                        recognized_chars.append(predicted_char)
                        confidences.append(conf)

                # STAGE 4 - Sequence Reconstruction
                # Reassemble predictions back into the original structure:
                # characters -> words -> lines -> full document
                
                line_words = {}
                for region, char in zip(character_regions, recognized_chars):
                    if region.image is not None and region.image.size > 0:
                        line_idx = region.line_index
                        word_idx = region.word_index
                        
                        if line_idx not in line_words:
                            line_words[line_idx] = {}
                        if word_idx not in line_words[line_idx]:
                            line_words[line_idx][word_idx] = []
                            
                        line_words[line_idx][word_idx].append(char)
                        
                recognized_lines = []
                for line_idx in sorted(line_words.keys()):
                    words_in_line = []
                    for word_idx in sorted(line_words[line_idx].keys()):
                        word = "".join(line_words[line_idx][word_idx])
                        words_in_line.append(word)
                    
                    recognized_lines.append(" ".join(words_in_line))
                    
                recognized_text = "\n".join(recognized_lines) if recognized_lines else ""
                st.session_state["workspace_text"] = recognized_text

            st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            render_section_hud_header("Reconstruction Output & Segmentation", icon="📑")

            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown("##### 🔍 Segmented Character Bounding Boxes")
                st.image(annotated_img, width='stretch')

            with col_res2:
                st.markdown("##### 📝 Digitized Text Output")
                if recognized_text:
                    st.text_area("Raw CNN Recognized Stream", value=recognized_text, height=140, key="rec_text_area")
                    avg_conf = np.mean(confidences) if confidences else 0.0
                    min_conf = np.min(confidences) if confidences else 0.0
                    st.markdown(f"""
                    <div class="glass-panel" style="padding: 0.8rem 1rem;">
                        <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-secondary);">
                            • Characters Segmented: <strong style="color: var(--accent-cyan);">{len(recognized_chars)}</strong> | 
                            • Avg Certainty: <strong style="color: #10b981;">{avg_conf:.1f}%</strong> | 
                            • Min Certainty: <strong style="color: #f59e0b;">{min_conf:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No valid character regions could be segmented from this image. Try an image with higher ink contrast.")

            # ── Gemini Workspace Direct Hand-Off ──
            st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            render_section_hud_header("Gemini AI Workspace Hand-off", "Directly route digitized payload to Gemini for correction & entity extraction", icon="✨")

            col_gem1, col_gem2 = st.columns([1.5, 1])
            with col_gem1:
                st.markdown(f"""
                <div class="glass-panel">
                    <div style="font-size: 0.85rem; font-weight: 700; color: var(--accent-purple); margin-bottom: 0.4rem;">
                        ✨ READY FOR SEMANTIC ENHANCEMENT
                    </div>
                    <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.6;">
                        Digitized text payload ({len(recognized_text)} chars) is stored in session buffer. 
                        Launch the Gemini AI Workspace to perform contextual OCR error correction, document summarization, and named entity extraction.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_gem2:
                if st.button("🚀 Open in Gemini AI Workspace", key="btn_goto_workspace", type="primary", width='stretch'):
                    st.switch_page("pages/5_LANGUAGE.py")

            # ── Downloads ──
            st.markdown("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("##### 📥 Export Digitized Artifacts")

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.download_button(
                    "📄 Export Text Stream (.txt)",
                    recognized_text,
                    file_name="digitized_document.txt",
                    mime="text/plain",
                    width='stretch',
                )
            with col_d2:
                st.download_button(
                    "🖼️ Export Segmented Visual (.png)",
                    image_to_bytes(annotated_img),
                    file_name="segmented_manuscript.png",
                    mime="image/png",
                    width='stretch',
                )
            with col_d3:
                report_dict = {
                    "model": model_type,
                    "character_count": len(recognized_chars),
                    "raw_text": recognized_text,
                    "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                }
                st.download_button(
                    "📋 Export Full Pipeline JSON (.json)",
                    json.dumps(report_dict, indent=2),
                    file_name="digitization_pipeline_report.json",
                    mime="application/json",
                    width='stretch',
                )

    except Exception as e:
        st.error(f"⚠️ Error executing document pipeline: {str(e)}")

