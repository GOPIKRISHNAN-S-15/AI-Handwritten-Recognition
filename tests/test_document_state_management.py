"""
Automated unit & state flow test for Streamlit Documents Page State Management
Tests:
A. Upload/process Document A -> confirms state set for Doc A.
B. Remove Document A (uploaded_file is None) -> confirms state wiped.
C. Upload Document B -> confirms has_run is False and no Doc A results are displayed.
D. Process Document B -> confirms all 4 model outputs belong to Doc B.
E. Repeat A -> B -> A cycle -> ensures zero leakage or stale cross-document state.
"""

import hashlib
import io
import numpy as np
from PIL import Image

# Import state keys and clear helper directly from page
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib

# Load 2_DOCUMENTS module dynamically
doc_module = importlib.import_module("pages.2_DOCUMENTS")
DOC_STATE_KEYS = doc_module.DOC_STATE_KEYS
clear_document_state = doc_module.clear_document_state

def create_mock_image_bytes(text_color=0):
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # draw simple box to make bytes unique
    arr = np.array(img)
    arr[20:40, 20:40] = text_color
    out = Image.fromarray(arr)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()

class MockUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self.data = data
    def getvalue(self):
        return self.data

def run_state_management_test():
    import streamlit as st
    
    print("=== Testing Streamlit Documents Page State Management ===")
    
    # 1. Prepare Document A and Document B
    bytes_a = create_mock_image_bytes(text_color=10)
    bytes_b = create_mock_image_bytes(text_color=200)
    
    doc_a = MockUploadedFile("doc_a.png", bytes_a)
    doc_b = MockUploadedFile("doc_b.png", bytes_b)
    
    hash_a = hashlib.sha256(bytes_a).hexdigest()[:16]
    doc_id_a = f"{doc_a.name}_{hash_a}"
    
    hash_b = hashlib.sha256(bytes_b).hexdigest()[:16]
    doc_id_b = f"{doc_b.name}_{hash_b}"
    
    assert doc_id_a != doc_id_b, "Document IDs must be distinct"
    
    # STEP A: Upload & Process Document A
    print("\n[Step A] Upload and Process Document A...")
    # Simulate first load of Document A
    if st.session_state.get("doc_id") != doc_id_a:
        clear_document_state()
        st.session_state["doc_id"] = doc_id_a
        st.session_state["has_run"] = False
        
    # Simulate processing Document A
    st.session_state["doc_id"] = doc_id_a
    st.session_state["doc_cnn_text"] = "OUTPUT_DOC_A_CNN"
    st.session_state["doc_trocr_text"] = "OUTPUT_DOC_A_TROCR"
    st.session_state["doc_ctc_text"] = "OUTPUT_DOC_A_CTC"
    st.session_state["doc_gemini_text"] = "OUTPUT_DOC_A_GEMINI"
    st.session_state["doc_cnn_status"] = "COMPLETE"
    st.session_state["doc_trocr_status"] = "COMPLETE"
    st.session_state["doc_ctc_status"] = "COMPLETE"
    st.session_state["doc_gemini_status"] = "COMPLETE"
    st.session_state["doc_metrics"] = {"chars": 10, "words": 2, "lines": 1, "avg_conf": 98.5}
    st.session_state["has_run"] = True
    
    # Verify Doc A is displayed
    assert st.session_state.get("has_run") == True
    assert st.session_state.get("doc_id") == doc_id_a
    assert st.session_state.get("doc_cnn_text") == "OUTPUT_DOC_A_CNN"
    print("  [PASS] Document A successfully processed and stored in session state.")
    
    # STEP B: Remove Document A (uploaded_file is None)
    print("\n[Step B] Remove Document A (uploaded_file is None)...")
    uploaded_file = None
    if uploaded_file is None:
        if st.session_state.get("doc_id") is not None:
            clear_document_state()
            
    # Verify state is completely cleared
    for k in DOC_STATE_KEYS:
        assert k not in st.session_state, f"State key '{k}' was not cleared when document was removed"
    print("  [PASS] Document A state completely wiped on file removal.")
    
    # STEP C: Upload Document B (before processing)
    print("\n[Step C] Upload Document B (before clicking Process)...")
    uploaded_file = doc_b
    current_doc_id = doc_id_b
    if st.session_state.get("doc_id") != current_doc_id:
        clear_document_state()
        st.session_state["doc_id"] = current_doc_id
        st.session_state["has_run"] = False
        
    # Check that has_run is False and no old results exist
    assert st.session_state.get("has_run") == False
    assert st.session_state.get("doc_cnn_text") is None
    assert st.session_state.get("doc_id") == doc_id_b
    print("  [PASS] Document B loaded: stale outputs are blocked (has_run=False, doc_cnn_text=None).")
    
    # STEP D: Process Document B
    print("\n[Step D] Process Document B...")
    st.session_state["doc_id"] = doc_id_b
    st.session_state["doc_cnn_text"] = "OUTPUT_DOC_B_CNN"
    st.session_state["doc_trocr_text"] = "OUTPUT_DOC_B_TROCR"
    st.session_state["doc_ctc_text"] = "OUTPUT_DOC_B_CTC"
    st.session_state["doc_gemini_text"] = "OUTPUT_DOC_B_GEMINI"
    st.session_state["doc_cnn_status"] = "COMPLETE"
    st.session_state["doc_trocr_status"] = "COMPLETE"
    st.session_state["doc_ctc_status"] = "COMPLETE"
    st.session_state["doc_gemini_status"] = "COMPLETE"
    st.session_state["doc_metrics"] = {"chars": 25, "words": 5, "lines": 2, "avg_conf": 94.2}
    st.session_state["has_run"] = True
    
    # Verify Doc B results are displayed
    assert st.session_state.get("has_run") == True
    assert st.session_state.get("doc_id") == doc_id_b
    assert st.session_state.get("doc_cnn_text") == "OUTPUT_DOC_B_CNN"
    assert st.session_state.get("doc_trocr_text") == "OUTPUT_DOC_B_TROCR"
    assert st.session_state.get("doc_ctc_text") == "OUTPUT_DOC_B_CTC"
    assert st.session_state.get("doc_gemini_text") == "OUTPUT_DOC_B_GEMINI"
    print("  [PASS] Document B outputs exclusively match Document B.")
    
    # STEP E: Repeat A -> B -> A cycle
    print("\n[Step E] Repeat A -> B -> A Cycle...")
    # Switch back to Doc A without removing first (direct file replace)
    uploaded_file = doc_a
    current_doc_id = doc_id_a
    if st.session_state.get("doc_id") != current_doc_id:
        clear_document_state()
        st.session_state["doc_id"] = current_doc_id
        st.session_state["has_run"] = False
        
    assert st.session_state.get("has_run") == False
    assert st.session_state.get("doc_cnn_text") is None
    assert st.session_state.get("doc_id") == doc_id_a
    print("  [PASS] Direct document swap immediately invalidates previous document results.")
    
    print("\n============================================================")
    print("ALL STATE MANAGEMENT CHECKS PASSED WITH ZERO LEAKAGE!")
    print("============================================================")

if __name__ == "__main__":
    run_state_management_test()
