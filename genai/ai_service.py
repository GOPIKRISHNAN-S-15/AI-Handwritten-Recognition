"""
Google Gemini API integration for GenAI enhancement layer.

The CNN performs recognition. This module provides a secondary intelligence
layer for contextual correction, summarization, and information extraction.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GenAIResult:
    """Result from a GenAI operation."""
    success: bool
    content: str
    error: Optional[str] = None
    operation: str = ""


class GeminiService:
    """
    Gemini API service for text enhancement.

    Handles initialization, connection checking, and all GenAI operations
    with graceful error handling.
    """

    def __init__(self):
        self.status = "uninitialized"
        self.is_available = False
        self.quota_limit = None
        self.retry_after_seconds = None
        # Human-readable runtime status of the last probe, used by the UI.
        # Possible values: "uninitialized", "ready", "quota_limited",
        # "model_unavailable", "key_missing", "network_error".
        self.status = "uninitialized"
        self.client = None
        
        # Model identifier from Streamlit secrets or environment variable
        self.model_name = self._get_model_name()
        self._initialize()

    @staticmethod
    def _get_model_name() -> str:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
                m = st.secrets["GEMINI_MODEL"]
                if m and str(m).strip():
                    return str(m).strip()
        except Exception:
            pass
        return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip()

    def _initialize(self):
        """Initialize the Gemini client with API key from environment or Streamlit secrets."""
        api_key = self._get_api_key()
        if not api_key:
            self.is_available = False
            self.client = None
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.is_available = True
        except ImportError:
            self.is_available = False
            self.client = None
        except Exception:
            self.is_available = False
            self.client = None

    @staticmethod
    def _get_api_key() -> Optional[str]:
        """Get API key from Streamlit secrets (Streamlit Cloud) or .env file (local)."""
        invalid_placeholders = {
            "", "your-gemini-api-key-here", "your_gemini_api_key_here",
            "your-google-ai-studio-key", "your_api_key_here",
        }

        # 1. Try Streamlit secrets (for Streamlit Community Cloud)
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"]:
                    if key_name in st.secrets:
                        val = st.secrets[key_name]
                        if val and str(val).strip() and str(val).strip().lower() not in invalid_placeholders:
                            return str(val).strip()
                # Check nested dicts e.g. [gemini] api_key = "..."
                for sec in ["gemini", "general", "google"]:
                    if sec in st.secrets and hasattr(st.secrets[sec], "get"):
                        for key_name in ["api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                            val = st.secrets[sec].get(key_name)
                            if val and str(val).strip() and str(val).strip().lower() not in invalid_placeholders:
                                return str(val).strip()
        except Exception:
            pass

        # 2. Fall back to environment / .env file (local development)
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        for env_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            key = os.environ.get(env_name, "")
            if key and key.strip() and key.strip().lower() not in invalid_placeholders:
                return key.strip()

        return None

    # Probe results are cached per session so the Language page (and the
    # top HUD on every page) does not burn API quota on every Streamlit
    # rerun. Cache lifetime is 60 seconds.
    _PROBE_CACHE_TTL = 60.0

    def check_connection(self) -> bool:
        """Check if the Gemini API is reachable and valid.

        Performs a REAL request. If the configured model fails with 404
        (obsolete identifier), it probes the known-good fallback models
        exactly once and switches to the first one that responds.

        Results are cached per instance for _PROBE_CACHE_TTL seconds to
        avoid exhausting the free-tier quota through repeated probes.
        """
        import time
        cache = getattr(self, "_probe_cache", None)
        if cache is not None:
            (cached_ok, ts) = cache
            if time.perf_counter() - ts < self._PROBE_CACHE_TTL:
                return cached_ok

        if not self.client:
            self._initialize()
        if not self.client:
            self.is_available = False
            self.status = "key_missing"
            self._set_probe(False)
            return False
        if self._try_model(self.model_name):
            self.status = "ready"
            self._set_probe(True)
            return True
        # Record the primary model's failure before probing fallbacks, so the
        # most relevant diagnostic (e.g. quota exhaustion) is not lost if a
        # fallback fails with a less actionable error later.
        primary_status = self.status
        # Probe supported fallbacks if the configured model is dead (404)
        for candidate in ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-2.5-flash"]:
            if candidate == self.model_name:
                continue
            if self._try_model(candidate):
                self.model_name = candidate
                self.status = "ready"
                self._set_probe(True)
                return True
        self.is_available = False
        # Report the most actionable failure: quota limit beats everything,
        # then the primary model's own status, then whatever the fallbacks said.
        if primary_status == "quota_limited" or self.status == "quota_limited":
            self.status = "quota_limited"
        elif primary_status not in ("uninitialized",):
            self.status = primary_status
        self._set_probe(False)
        return False

    def _set_probe(self, ok: bool):
        """Store the probe result together with a timestamp."""
        import time
        self._probe_cache = (ok, time.perf_counter())

    def _try_model(self, model_name: str) -> bool:
        """Issue a real request against one model; return True on success."""
        try:
            chat = self.client.chats.create(model=model_name)
            response = chat.send_message("Respond with only: OK")
            connected = bool(response and response.text)
            if connected:
                self.is_available = True
                self.status = "ready"
                return True
            return False
        except Exception as e:
            self._record_probe_error(str(e))
            return False

    def _record_probe_error(self, error_text: str):
        """Classify the last API error into a human-readable status.

        Accepts either a raw exception string (REST-style JSON error body
        embedded in the message) or a google-genai ClientError object. In
        both cases it attempts to extract quota-limit and retry-delay
        details for user-facing diagnostics.
        """
        import json
        import re
        status_code = None
        err = {}
        if not isinstance(error_text, str):
            # google-genai ClientError: inspect its response payload
            try:
                resp = getattr(error_text, "response", None)
                status_code = getattr(error_text, "status", None) or (
                    getattr(resp, "status_code", None) if resp is not None else None
                )
                text = getattr(resp, "text", "") or ""
                try:
                    payload = json.loads(text)
                    err = payload.get("error", payload) if isinstance(payload, dict) else {}
                except Exception:
                    error_text = text
            except Exception:
                error_text = str(error_text)
        else:
            m = re.search(r"(\d{3})", error_text)
            if m:
                status_code = int(m.group(1))
        # Try to parse quota details (limit + retry delay) from the raw error
        try:
            if isinstance(error_text, str) and error_text.strip().startswith("{"):
                payload = json.loads(error_text)
                err = payload.get("error", payload) if isinstance(payload, dict) else {}
            details = err.get("details", [])
            for detail in details:
                if detail.get("@type", "").endswith("QuotaFailure"):
                    for v in detail.get("violations", []):
                        metric = v.get("quotaMetric", "")
                        if metric and "requests" in metric:
                            self.quota_limit = v.get("quotaValue", "?")
                if detail.get("@type", "").endswith("RetryInfo"):
                    rd = detail.get("retryDelay", "")
                    # Duration formats: "927.592s", "60s", or protoref JSON
                    # ("seconds":"927"). Capture the leading whole seconds.
                    mm = re.search(r"(?:^|:|\s)(\d+)s$|^\s*(\d+)", rd)
                    if mm:
                        self.retry_after_seconds = int(mm.group(1) or mm.group(2))
                    elif rd.strip().isdigit():
                        self.retry_after_seconds = int(rd.strip())
        except Exception:
            pass
        # From here on treat error_text as a plain string for classification.
        if not isinstance(error_text, str):
            error_text = str(error_text)
        if status_code == 429 or "429" in error_text:
            self.status = "quota_limited"
        elif "404" in error_text:
            self.status = "model_unavailable"
        elif "401" in error_text or "403" in error_text:
            self.status = "key_missing"
        elif "network" in error_text.lower() or "connect" in error_text.lower():
            self.status = "network_error"
        else:
            self.status = "quota_limited" if "RESOURCE_EXHAUSTED" in error_text else "network_error"

    def clear_probe_cache(self):
        """Force a fresh connection probe on the next check_connection()."""
        self._probe_cache = None

    def _generate(self, prompt: str, operation: str) -> GenAIResult:
        """Internal method to call the Gemini API with error handling."""
        if not self.is_available or not self.client:
            return GenAIResult(
                success=False,
                content="",
                error="GenAI service is not available. Please configure your GEMINI_API_KEY.",
                operation=operation,
            )

        try:
            chat = self.client.chats.create(model=self.model_name)
            response = chat.send_message(prompt)
            text = response.text or ""
            return GenAIResult(
                success=True,
                content=text.strip(),
                operation=operation,
            )

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                error_msg = (
                    "Gemini free-tier quota exhausted. Please wait (quota typically "
                    "resets periodically) or use a key with a higher quota, then retry."
                )
            elif "401" in error_msg or "403" in error_msg:
                error_msg = "Invalid API key. Please check your GEMINI_API_KEY configuration."
            elif "404" in error_msg:
                error_msg = (
                    "The configured Gemini model identifier was not found. "
                    "Update the GEMINI_MODEL value in your .env file."
                )
            elif "timeout" in error_msg.lower():
                error_msg = "Request timed out. Please try again."
            elif "connect" in error_msg.lower():
                error_msg = "Network error. Please check your internet connection."
            else:
                error_msg = f"An error occurred: {error_msg}"

            return GenAIResult(
                success=False,
                content="",
                error=error_msg,
                operation=operation,
            )

    def correct_text(self, raw_text: str) -> GenAIResult:
        """
        Contextual correction of OCR output.

        Uses the LLM to fix common OCR errors like character confusion
        (0↔O, 1↔I, 5↔S, etc.) while preserving the intended meaning.
        """
        if not raw_text or not raw_text.strip():
            return GenAIResult(
                success=False, content="", operation="correct_text",
                error="No text provided for correction."
            )

        prompt = f"""You are an OCR post-processing expert. The following text was recognized from a handwritten document using a CNN-based OCR system. It may contain character recognition errors.

Please correct the text to produce the most likely intended text. Common OCR errors include:
- '0' confused with 'O'
- '1' confused with 'I' or 'l'
- '5' confused with 'S'
- '8' confused with 'B'
- '2' confused with 'Z'
- Missing or extra spaces

RAW OCR OUTPUT:
{raw_text}

IMPORTANT:
- Return ONLY the corrected text, nothing else.
- If the text appears correct already, return it as-is.
- Do not add explanations or commentary.
- Preserve the original formatting (line breaks, capitalization style).

CORRECTED TEXT:"""

        return self._generate(prompt, "correct_text")

    def summarize_text(self, text: str) -> GenAIResult:
        """
        Generate a concise summary of the recognized document text.
        """
        if not text or not text.strip():
            return GenAIResult(
                success=False, content="", operation="summarize",
                error="No text provided for summarization."
            )

        prompt = f"""Analyze and summarize the following text that was extracted from a handwritten document.

Provide a clear, concise summary in 2-4 sentences covering the main content and purpose of the document.

DOCUMENT TEXT:
{text}

SUMMARY:"""

        return self._generate(prompt, "summarize")

    def extract_info(self, text: str) -> GenAIResult:
        """
        Extract key information from the recognized text.

        Extracts names, dates, times, tasks, numbers, and important points.
        Only reports categories that are actually found.
        """
        if not text or not text.strip():
            return GenAIResult(
                success=False, content="", operation="extract_info",
                error="No text provided for information extraction."
            )

        prompt = f"""Extract key information from the following text that was recognized from a handwritten document.

Identify and list ONLY the categories that are actually present:
- **Names**: Any person names mentioned
- **Dates**: Any dates or time references
- **Numbers**: Any numerical values
- **Tasks/Actions**: Any action items or tasks
- **Key Points**: Important facts or decisions

DOCUMENT TEXT:
{text}

IMPORTANT:
- Only include categories where you actually find relevant information.
- Do NOT invent or fabricate information that isn't in the text.
- Format each category with a header and bullet points.
- If only one or two categories apply, that's fine.

EXTRACTED INFORMATION:"""

        return self._generate(prompt, "extract_info")

    def get_insights(self, text: str) -> GenAIResult:
        """
        Generate contextual insights about the document.
        """
        if not text or not text.strip():
            return GenAIResult(
                success=False, content="", operation="insights",
                error="No text provided for analysis."
            )

        prompt = f"""Provide brief, factual observations about the following text that was recognized from a handwritten document.

DOCUMENT TEXT:
{text}

Provide 3-5 concise observations about the content. For example:
- Document type (note, letter, list, etc.)
- Number of action items if any
- Number of dates/numbers mentioned
- Tone (formal, informal, etc.)
- Completeness (appears complete, seems truncated, etc.)

IMPORTANT:
- Only state what you can actually observe from the text.
- Do NOT fabricate information not present in the text.
- Keep each observation to one line.
- Use bullet points.

DOCUMENT INSIGHTS:"""

        return self._generate(prompt, "insights")


def get_genai_service() -> GeminiService:
    """
    Get or create the GeminiService singleton.
    Uses Streamlit's caching to avoid re-initializing.
    """
    try:
        import streamlit as st
        if "genai_service" not in st.session_state:
            st.session_state.genai_service = GeminiService()
        return st.session_state.genai_service
    except ImportError:
        return GeminiService()
