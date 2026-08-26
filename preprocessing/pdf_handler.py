import pymupdf  # PyMuPDF
from PIL import Image
import streamlit as st

@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_bytes, dpi=300):
    """
    Renders pages of a PDF from bytes to PIL Images.
    Returns: (images_list, error_message)
    """
    try:
        # Open PDF from bytes
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.needs_pass:
            return None, "⚠️ PDF is password-protected."
            
        if len(doc) == 0:
            return None, "⚠️ PDF is empty."
            
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            
            # Convert PyMuPDF pixmap to PIL Image
            if pix.alpha:
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                # Convert to RGB (white background)
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
            images.append(img)
            
        return images, None
        
    except Exception as e:
        return None, f"⚠️ Error rendering PDF: {str(e)}"
