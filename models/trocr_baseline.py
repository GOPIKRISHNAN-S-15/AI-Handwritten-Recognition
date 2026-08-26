import streamlit as st
from PIL import Image
import numpy as np

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

@st.cache_resource(show_spinner="Loading TrOCR Baseline Model...")
def load_trocr_model():
    if not TROCR_AVAILABLE:
        raise RuntimeError("PyTorch or transformers is not installed in the environment.")
    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
    
    device = torch.device("cpu")
    model.to(device)
    
    return processor, model, device

def run_trocr_on_lines(line_images):
    """
    Given a list of cropped PIL Images (one for each line),
    run TrOCR and return a list of recognized text strings.
    """
    processor, model, device = load_trocr_model()
    
    recognized_lines = []
    for img in line_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_lines.append(generated_text)
        
    return recognized_lines
