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
import hashlib

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

DOC_STATE_KEYS = [
    "doc_id",
    "has_run",
    "doc_cnn_text",
    "doc_trocr_text",
    "doc_ctc_text",
    "doc_gemini_text",
    "doc_cnn_status",
    "doc_trocr_status",
    "doc_ctc_status",
    "doc_gemini_status",
    "doc_metrics",
]

def clear_document_state():
    """Wipe all document-specific OCR and model output states."""
    for k in DOC_STATE_KEYS:
        if k in st.session_state:
            del st.session_state[k]

def render_export_buttons(text_content, prefix, doc_key=""):
    if text_content and not text_content.startswith("Not evaluated") and "⚠️" not in text_content:
        from utils.export_helper import generate_txt, generate_pdf, generate_docx
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📄 TXT", data=generate_txt(text_content), file_name=f"{prefix}_output.txt", mime="text/plain", key=f"btn_txt_{prefix}_{doc_key}", use_container_width=True)
        with c2:
            st.download_button("📄 PDF", data=generate_pdf(text_content), file_name=f"{prefix}_output.pdf", mime="application/pdf", key=f"btn_pdf_{prefix}_{doc_key}", use_container_width=True)
        with c3:
            st.download_button("📄 DOCX", data=generate_docx(text_content), file_name=f"{prefix}_output.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"btn_docx_{prefix}_{doc_key}", use_container_width=True)
    else:
        st.caption("Run this model first to export output.")

st.set_page_config(page_title="Document Digitization Pipeline — SIRI", page_icon="🧬", layout="wide")
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

model = emnist_model
class_mapping = EMNIST_BALANCED_MAPPING
model_type = "emnist"

render_reconstruction_pipeline(active_step="SEG_LINE/WORD")

# ══════════════════════════════════════════════
# MANUSCRIPT UPLOAD
# ══════════════════════════════════════════════
st.html("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem;">
    Upload a handwritten document image. The pipeline will detect text lines, isolate character regions, and perform CNN classification.
</div>
""")

uploaded_file = st.file_uploader("Upload document image or PDF", type=SUPPORTED_IMAGE_FORMATS + ["pdf"], label_visibility="collapsed", key="doc_upload")

if uploaded_file is None:
    # If a file was removed or is absent, ensure all stale document state is cleared
    if st.session_state.get("doc_id") is not None:
        clear_document_state()
else:
    # Compute deterministic fingerprint of uploaded content
    raw_bytes = uploaded_file.getvalue()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    base_doc_id = f"{uploaded_file.name}_{content_hash}"
    
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            from preprocessing.pdf_handler import render_pdf_pages
            
            with st.spinner("Rendering PDF pages..."):
                images, error = render_pdf_pages(raw_bytes)
                
            if error:
                st.error(error)
                st.stop()
                
            if not images:
                st.error("⚠️ No pages found in PDF.")
                st.stop()
                
            # Page selector
            if len(images) > 1:
                page_num = st.selectbox("Select Page to Process", range(1, len(images) + 1), format_func=lambda x: f"Page {x} of {len(images)}")
                pil_image = images[page_num - 1]
                current_doc_id = f"{base_doc_id}_p{page_num}"
            else:
                pil_image = images[0]
                current_doc_id = f"{base_doc_id}_p1"
                st.info("📄 Processing single-page PDF.")
                
        else:
            pil_image = Image.open(uploaded_file).convert("RGB")
            current_doc_id = base_doc_id
            
        # Invalidate state if a different document or page was uploaded
        if st.session_state.get("doc_id") != current_doc_id:
            clear_document_state()
            st.session_state["doc_id"] = current_doc_id
            st.session_state["has_run"] = False
            
        input_image = np.array(pil_image)

        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            st.markdown("##### 📷 Uploaded Manuscript")
            st.image(input_image, width='stretch')

        with col_in2:
            st.markdown("##### ⚙️ Pipeline Control")
            st.html("""
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
""")

            is_pdf_file = uploaded_file.name.lower().endswith(".pdf")
            if is_pdf_file and 'images' in locals() and len(images) > 1:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    run_digitize = st.button("Process Current Page", key="digitize_btn", type="primary", width='stretch')
                with col_btn2:
                    run_batch = st.button("Process All Pages", key="digitize_batch_btn", type="secondary", width='stretch')
            else:
                run_digitize = st.button("Process Document", key="digitize_btn", type="primary", width='stretch')
                run_batch = False

        if run_batch or run_digitize:
            images_to_process = images if run_batch else [pil_image]
            total_pages = len(images_to_process)
            
            with st.spinner("Processing document reconstruction pipeline..."):
                from models.trocr_baseline import run_trocr_on_lines
                from models.ctc_baseline import run_ctc_on_lines
                
                cnn_full_text, trocr_full_text, ctc_full_text, gemini_full_text = [], [], [], []
                
                segmenter = DocumentSegmenter(min_char_area=40, char_padding=0)
                preprocessor = AdaptivePreprocessor(target_size=(28, 28))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                cnn_status = "COMPLETE"
                trocr_status = "COMPLETE"
                ctc_status = "COMPLETE"
                gemini_status = "COMPLETE"
                
                # Keep track of metrics for the CNN output
                total_chars = 0
                total_words = 0
                total_lines = 0
                all_confidences = []
                
                for idx, img in enumerate(images_to_process):
                    if run_batch:
                        status_text.markdown(f"**Processing page {idx + 1} / {total_pages}...**")
                    else:
                        status_text.markdown("**Processing current page...**")
                    
                    page_input_image = np.array(img)
                    
                    # 1. Run CNN
                    character_regions, annotated_img = segmenter.segment(page_input_image)
                    recognized_chars = []
                    
                    for region in character_regions:
                        if region.image is not None and region.image.size > 0:
                            char_img_rgb = cv2.cvtColor(region.image, cv2.COLOR_GRAY2BGR) if len(region.image.shape) == 2 else region.image
                            processed, _ = preprocessor.preprocess(char_img_rgb, for_model=True)
                            res = predict_character(model, processed, class_mapping, top_n=1)
                            recognized_chars.append(res["predicted_character"])
                            all_confidences.append(res["confidence_percentage"])
                            
                    total_chars += len(recognized_chars)
                            
                    line_words = {}
                    for region, char in zip(character_regions, recognized_chars):
                        if region.image is not None and region.image.size > 0:
                            line_idx = region.line_index
                            word_idx = region.word_index
                            if line_idx not in line_words: line_words[line_idx] = {}
                            if word_idx not in line_words[line_idx]: line_words[line_idx][word_idx] = []
                            line_words[line_idx][word_idx].append(char)
                            
                    recognized_lines = []
                    for line_idx in sorted(line_words.keys()):
                        words_in_line = []
                        for word_idx in sorted(line_words[line_idx].keys()):
                            word = "".join(line_words[line_idx][word_idx])
                            words_in_line.append(word)
                            total_words += 1
                        recognized_lines.append(" ".join(words_in_line))
                        total_lines += 1
                        
                    cnn_text = "\n".join(recognized_lines) if recognized_lines else ""
                    
                    # 2. Prepare line images for TrOCR/CTC
                    lines_dict = {}
                    for region in character_regions:
                        if region.line_index not in lines_dict:
                            lines_dict[region.line_index] = []
                        lines_dict[region.line_index].append(region)
                        
                    line_pil_images = []
                    for line_idx in sorted(lines_dict.keys()):
                        chars = lines_dict[line_idx]
                        min_x = max(0, min(c.bbox[0] for c in chars) - 5)
                        min_y = max(0, min(c.bbox[1] for c in chars) - 5)
                        max_x = min(page_input_image.shape[1], max(c.bbox[0] + c.bbox[2] for c in chars) + 5)
                        max_y = min(page_input_image.shape[0], max(c.bbox[1] + c.bbox[3] for c in chars) + 5)
                        
                        line_crop = page_input_image[min_y:max_y, min_x:max_x]
                        line_pil_images.append(Image.fromarray(line_crop))
                        
                    # 3. Run TrOCR
                    trocr_text = ""
                    try:
                        if line_pil_images:
                            trocr_results = run_trocr_on_lines(line_pil_images)
                            trocr_text = "\n".join(trocr_results)
                    except Exception as e:
                        trocr_status = "ERROR"
                        trocr_text = f"Error: {e}"
                        
                    # 4. Run CTC
                    ctc_text = ""
                    try:
                        if line_pil_images:
                            ctc_results = run_ctc_on_lines(line_pil_images)
                            ctc_text = "\n".join(ctc_results)
                    except Exception as e:
                        ctc_status = "ERROR"
                        ctc_text = f"CTC model unavailable.\nError: {e}"
                        
                    # 5. Run Gemini
                    gemini_text = ""
                    try:
                        if genai_available:
                            from models.gemini_ocr import process_image
                            gemini_text = process_image(img)
                            if "quota" in gemini_text.lower() or "429" in gemini_text:
                                gemini_status = "UNAVAILABLE"
                                gemini_text = "GEMINI UNAVAILABLE\n\nDaily API quota exhausted.\n\nCNN, TrOCR and CNN-BiLSTM-CTC remain available."
                        else:
                            gemini_status = "UNAVAILABLE"
                            gemini_text = "Gemini API not configured or unavailable."
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            gemini_status = "UNAVAILABLE"
                            gemini_text = "GEMINI UNAVAILABLE\n\nDaily API quota exhausted.\n\nCNN, TrOCR and CNN-BiLSTM-CTC remain available."
                        else:
                            gemini_status = "ERROR"
                            gemini_text = f"Error: {e}"
                        
                    # Aggregate
                    marker = f"--- Page {idx + 1} ---" if total_pages > 1 else ""
                    if marker:
                        cnn_full_text.append(marker + "\n" + cnn_text)
                        trocr_full_text.append(marker + "\n" + trocr_text)
                        ctc_full_text.append(marker + "\n" + ctc_text)
                        gemini_full_text.append(marker + "\n" + gemini_text)
                    else:
                        cnn_full_text.append(cnn_text)
                        trocr_full_text.append(trocr_text)
                        ctc_full_text.append(ctc_text)
                        gemini_full_text.append(gemini_text)
                    
                    progress_bar.progress((idx + 1) / total_pages)
                    
                if run_batch:
                    status_text.markdown(f"**{total_pages} / {total_pages} pages processed**")
                else:
                    status_text.empty()
                    
                st.session_state["doc_id"] = current_doc_id
                st.session_state["doc_cnn_text"] = "\n\n".join(cnn_full_text)
                st.session_state["doc_trocr_text"] = "\n\n".join(trocr_full_text)
                st.session_state["doc_ctc_text"] = "\n\n".join(ctc_full_text)
                st.session_state["doc_gemini_text"] = "\n\n".join(gemini_full_text)
                
                st.session_state["doc_cnn_status"] = cnn_status
                st.session_state["doc_trocr_status"] = trocr_status
                st.session_state["doc_ctc_status"] = ctc_status
                st.session_state["doc_gemini_status"] = gemini_status
                
                st.session_state["doc_metrics"] = {
                    "chars": total_chars,
                    "words": total_words,
                    "lines": total_lines,
                    "avg_conf": float(np.mean(all_confidences)) if all_confidences else 0.0
                }
                
                st.session_state["has_run"] = True

        if st.session_state.get("has_run", False) and st.session_state.get("doc_id") == current_doc_id:
            st.html("<hr style='border-color: var(--border-glass); margin: 1.5rem 0;'>")
            render_section_hud_header("FOUR-MODEL RECOGNITION COMPARISON", "Parallel execution of architectures on input payload")
            
            cnn_status = st.session_state.get("doc_cnn_status", "READY")
            trocr_status = st.session_state.get("doc_trocr_status", "READY")
            ctc_status = st.session_state.get("doc_ctc_status", "READY")
            gemini_status = st.session_state.get("doc_gemini_status", "READY")
            
            cnn_text = st.session_state.get("doc_cnn_text", "")
            trocr_text = st.session_state.get("doc_trocr_text", "")
            ctc_text = st.session_state.get("doc_ctc_text", "")
            gemini_text = st.session_state.get("doc_gemini_text", "")
            
            metrics = st.session_state.get("doc_metrics", {"chars": 0, "words": 0, "lines": 0, "avg_conf": 0.0})
            doc_key = current_doc_id[:16]
            
            # Row 1
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.html(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-orange); border-radius: 4px; padding: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-orange); font-weight: 600;">CUSTOM CNN</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">● {cnn_status}</span>
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">EMNIST Balanced</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">47 classes</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">Character-level recognition</div>
""")
                st.text_area("CNN Output", value=cnn_text, height=200, key=f"txt_cnn_{doc_key}", label_visibility="collapsed", disabled=True)
                st.html(f"""
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
        Metrics: {metrics['lines']} Lines | {metrics['words']} Words | {metrics['chars']} Chars | {metrics['avg_conf']:.1f}% Confidence
    </div>
</div>
""")
                render_export_buttons(cnn_text, "cnn", doc_key)

            with col_m2:
                st.html(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-cyan); border-radius: 4px; padding: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-cyan); font-weight: 600;">TrOCR</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">● {trocr_status}</span>
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">TrOCR Small Handwritten</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Transformer-based HTR</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">Line-level recognition</div>
""")
                st.text_area("TrOCR Output", value=trocr_text, height=200, key=f"txt_trocr_{doc_key}", label_visibility="collapsed", disabled=True)
                st.html(f"""
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
        Metrics: Auto-calculated
    </div>
</div>
""")
                render_export_buttons(trocr_text, "trocr", doc_key)

            st.html("<div style='height: 15px;'></div>")
            
            # Row 2
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                st.html(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-purple); border-radius: 4px; padding: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-purple); font-weight: 600;">CNN-BiLSTM-CTC</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">● {ctc_status}</span>
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">IAM Handwriting</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Sequence-based HTR</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">Designed for connected/cursive handwriting</div>
""")
                st.text_area("CTC Output", value=ctc_text, height=200, key=f"txt_ctc_{doc_key}", label_visibility="collapsed", disabled=True)
                st.html(f"""
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
        Metrics: Auto-calculated
    </div>
</div>
""")
                render_export_buttons(ctc_text, "ctc", doc_key)

            with col_m4:
                st.html(f"""
<div style="background: var(--bg-card); border: 1px solid var(--border-glass); border-left: 4px solid var(--accent-pink); border-radius: 4px; padding: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-family: var(--font-primary); font-size: 1.1rem; color: var(--accent-pink); font-weight: 600;">GEMINI</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">● {gemini_status}</span>
    </div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Gemini 3.5 Flash</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">Multimodal Document OCR</div>
    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">AI-assisted transcription</div>
""")
                st.text_area("Gemini Output", value=gemini_text, height=200, key=f"txt_gemini_{doc_key}", label_visibility="collapsed", disabled=True)
                st.html(f"""
    <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
        Status: API Response Logged
    </div>
</div>
""")
                render_export_buttons(gemini_text, "gemini", doc_key)

    except Exception as e:
        st.error(f"⚠️ Error executing document pipeline: {str(e)}")
