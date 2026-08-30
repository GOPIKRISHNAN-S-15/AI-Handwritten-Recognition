import os
import streamlit as st
from google import genai
from google.genai import types

from genai.ai_service import GeminiService

@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client from Streamlit secrets or environment."""
    api_key = GeminiService._get_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in Streamlit secrets or .env.")
        
    return genai.Client(api_key=api_key)

# The prompt exactly matches the strict rules specified
TRANSCRIPTION_PROMPT = (
    "Transcribe the text in this document exactly as it appears. "
    "Rules:\n"
    "- No summarization\n"
    "- No correction of spelling or grammar\n"
    "- No invented text\n"
    "- Mark unclear content as [UNCLEAR]\n"
    "- Preserve line breaks, punctuation, and capitalization"
)

def process_image(image):
    """
    Process a PIL image and return the raw transcription.
    Returns a string (either the transcription or a friendly error message).
    """
    try:
        client = get_gemini_client()
        # image should be a PIL Image or bytes
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[image, TRANSCRIPTION_PROMPT],
        )
        if response.text:
            return response.text
        return "⚠️ Gemini returned an empty response."
    except ValueError as ve:
        return f"⚠️ Configuration Error: {str(ve)}"
    except Exception as e:
        return f"⚠️ Network/API Error during image processing: {str(e)}"

def process_pdf(pdf_file_path):
    """
    Process a PDF file and return the raw transcription.
    pdf_file_path should be a local file path.
    """
    try:
        client = get_gemini_client()
        # Upload the file using the Files API
        uploaded_file = client.files.upload(file=pdf_file_path)
        
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[uploaded_file, TRANSCRIPTION_PROMPT]
        )
        
        # Cleanup the file from Gemini storage
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass # Ignore cleanup errors to return the transcript safely
            
        if response.text:
            return response.text
        return "⚠️ Gemini returned an empty response."
    except ValueError as ve:
        return f"⚠️ Configuration Error: {str(ve)}"
    except Exception as e:
        return f"⚠️ Network/API Error during PDF processing: {str(e)}"
