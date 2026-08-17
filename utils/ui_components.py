"""
Reusable styled UI components for the Synthetic Intelligence Research Interface (SIRI).

Uses st.html for native HTML/CSS rendering to prevent raw HTML leakage.
"""

import os
import streamlit as st
from typing import List, Dict, Optional, Tuple, Any


def load_css():
    """Load the main CSS stylesheet into Streamlit."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "styles", "main.css")
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        if hasattr(st, 'html'):
            st.html(f"<style>{css}</style>")
        else:
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        print(f"Error loading CSS: {e}")


def _render_html(html_content: str):
    """Helper to render HTML cleanly without Markdown parsing interference."""
    clean_html = html_content.strip()
    if hasattr(st, 'html'):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


def render_top_app_bar(
    title: str = "HWR / LAB",
    version: str = "Handwriting Reconstruction Engine",
    cnn_online: bool = True,
    genai_online: bool = False,
):
    """Render the fixed/floating TopAppBar."""
    cnn_color = "#34d399" if cnn_online else "#f87171"
    cnn_status = "MODEL: READY" if cnn_online else "MODEL: OFFLINE"

    genai_color = "#34d399" if genai_online else "#fbbf24"
    genai_status = "GENAI: READY" if genai_online else "GENAI: STANDBY"
    
    sys_status = "SYSTEM: ONLINE"

    html = f"""
    <div class="top-app-bar" style="background: var(--bg-card); border-radius: 4px; padding: 0.8rem 1.2rem; border-bottom: 1px solid var(--border-glass); display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
        <div class="top-bar-brand" style="display: flex; align-items: center; gap: 12px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            <span class="top-bar-title" style="font-family: var(--font-primary); font-weight: 600; font-size: 1.1rem; color: var(--text-primary); letter-spacing: 0.5px;">{title}</span>
            <span class="top-bar-version" style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); border-left: 1px solid var(--border-glass); padding-left: 12px; margin-left: 4px;">{version}</span>
        </div>
        <div class="top-bar-status" style="display: flex; align-items: center; gap: 16px; font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);">
            <span style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 6px; height: 6px; border-radius: 50%; background-color: {cnn_color};"></div>
                {cnn_status}
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 6px; height: 6px; border-radius: 50%; background-color: {genai_color};"></div>
                {genai_status}
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 6px; height: 6px; border-radius: 50%; background-color: #34d399;"></div>
                {sys_status}
            </span>
        </div>
    </div>
    """
    _render_html(html)


def render_sidebar_drawer(cnn_loaded: bool, genai_available: bool):
    """Render the sidebar navigation drawer branding and system status."""
    html = f"""
    <div class="sidebar-brand-siri" style="padding: 1rem 0; border-bottom: 1px solid var(--border-glass); margin-bottom: 1rem;">
        <div class="sidebar-logo-text" style="font-family: var(--font-primary); font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">HWR LAB</div>
        <div class="sidebar-sub-text" style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">Workstation v2.0</div>
    </div>
    """
    _render_html(html)


def render_hero_hud(
    title: str = "DOCUMENT DIGITIZATION",
    subtitle: str = "LABORATORY",
    description: str = "Professional neural interface for handwriting classification, manuscript segmentation, and semantic analysis.",
):
    """Render the Hero HUD section."""
    html = f"""
    <div class="hero-hud" style="padding: 2rem; background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; margin-bottom: 2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px;">SYSTEM / {title}</div>
        <h1 class="hero-hud-title" style="font-family: var(--font-primary); font-size: 2rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            {title}
            <br>
            <span style="color: var(--text-secondary);">{subtitle}</span>
        </h1>
        <p class="hero-hud-desc" style="color: var(--text-secondary); font-size: 0.9rem; max-width: 600px; line-height: 1.5;">{description}</p>
    </div>
    """
    _render_html(html)


def render_neural_gauges(gauges: List[Dict[str, Any]]):
    """Render circular/metric neural telemetry gauges."""
    cards_html = ""
    for g in gauges:
        value = g.get("value", "--")
        label = g.get("label", "Metric")
        meta = g.get("meta", "")
        
        meta_html = f'<div class="gauge-meta" style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">{meta}</div>' if meta else ""

        cards_html += f"""
        <div class="gauge-card" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem; display: flex; flex-direction: column;">
            <div class="gauge-label" style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px;">{label}</div>
            <div class="gauge-value" style="font-family: var(--font-primary); font-size: 1.4rem; font-weight: 600; color: var(--text-primary);">{value}</div>
            {meta_html}
        </div>
        """

    html = f'<div class="gauges-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">{cards_html}</div>'
    _render_html(html)


def render_feature_tiles(features: List[Dict[str, str]]):
    """Render clean feature tiles."""
    tiles_html = ""
    for f in features:
        tiles_html += f"""
        <div class="feature-tile" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.2rem;">
            <div class="tile-title" style="font-family: var(--font-primary); font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">{f.get('title', '')}</div>
            <div class="tile-desc" style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">{f.get('desc', '')}</div>
        </div>
        """
    html = f'<div class="feature-grid-siri" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 2rem;">{tiles_html}</div>'
    _render_html(html)


def render_section_hud_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled section header."""
    sub_html = f'<div class="section-hud-sub" style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="section-hud-header" style="margin: 2rem 0 1.5rem 0; border-bottom: 1px solid var(--border-glass); padding-bottom: 0.5rem;">
        <div class="section-hud-title" style="font-family: var(--font-primary); font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">
            {title}
        </div>
        {sub_html}
    </div>
    """
    _render_html(html)


def render_prediction_hud(
    character: str,
    confidence: float,
    confidence_level: Dict,
    latency_ms: float = 8.5,
    entropy: float = 0.02,
):
    """Render the technical prediction display HUD."""
    pct = confidence * 100
    color = "var(--text-primary)"

    html = f"""
    <div class="prediction-hud" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 2rem; display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div class="prediction-hud-tag" style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase;">CLASSIFICATION RESULT</div>
        <div class="prediction-hud-char" style="font-family: var(--font-primary); font-size: 4rem; font-weight: 600; color: var(--text-primary); line-height: 1; margin-bottom: 1rem;">{character}</div>
        <div class="prediction-hud-conf" style="font-family: var(--font-mono); font-size: 1rem; color: {color}; margin-bottom: 1.5rem;">
            CONFIDENCE: {pct:.2f}%
        </div>
        <div class="prediction-meta-row" style="display: flex; gap: 12px; font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary);">
            <span style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-glass);">LAT: {latency_ms:.1f}ms</span>
            <span style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-glass);">ENT: {entropy:.3f}</span>
            <span style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-glass);">DIM: 28x28</span>
        </div>
    </div>
    """
    _render_html(html)


def render_candidate_bars(alternatives: List[Dict]):
    """Render technical candidate probability bars."""
    bars_html = ""
    for i, alt in enumerate(alternatives):
        char = alt.get("character", "?")
        pct = alt.get("confidence", 0.0) * 100
        
        bars_html += f"""
        <div class="candidate-bar" style="margin-bottom: 12px;">
            <div class="candidate-bar-label" style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.8rem; margin-bottom: 4px; color: var(--text-primary);">
                <span>RANK {i+1} [{char}]</span>
                <span>{pct:.2f}%</span>
            </div>
            <div class="candidate-bar-track" style="height: 6px; background: var(--bg-secondary); border-radius: 2px; overflow: hidden; border: 1px solid var(--border-glass);">
                <div class="candidate-bar-fill" style="height: 100%; width: {pct:.1f}%; background: var(--text-secondary);"></div>
            </div>
        </div>
        """

    html = f"""
    <div class="glass-panel" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1.2rem; text-transform: uppercase;">PROBABILITY DISTRIBUTION</div>
        {bars_html}
    </div>
    """
    _render_html(html)


def render_reconstruction_pipeline(active_step: str = "CNN_CLASSIFY"):
    """Render the visual reconstruction pipeline graph."""
    nodes = [
        {"name": "IMAGE RECEIVED", "id": "IMAGE_RECEIVED"},
        {"name": "PREPROCESSING", "id": "PREPROCESS"},
        {"name": "SEGMENTATION", "id": "SEGMENTATION"},
        {"name": "CLASSIFICATION", "id": "CNN_CLASSIFY"},
        {"name": "RECONSTRUCTION", "id": "RECONSTRUCTION"},
        {"name": "LANGUAGE ENHANCEMENT", "id": "GEMINI_REFINE"},
    ]

    nodes_html = ""
    for i, n in enumerate(nodes):
        is_active = n["id"] == active_step
        color = "var(--text-primary)" if is_active else "var(--text-muted)"
        bg = "var(--bg-secondary)" if is_active else "transparent"
        border = "var(--border-glass-hover)" if is_active else "var(--border-glass)"

        nodes_html += f"""
        <div class="pipe-node" style="padding: 8px 12px; border: 1px solid {border}; background: {bg}; color: {color}; border-radius: 4px; font-family: var(--font-mono); font-size: 0.7rem; text-align: center;">
            {n['name']}
        </div>
        """
        if i < len(nodes) - 1:
            nodes_html += '<div class="pipe-arrow" style="color: var(--text-muted); font-size: 0.8rem; margin: 0 4px;">→</div>'

    html = f"""
    <div class="glass-panel" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1rem; margin-bottom: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase; text-align: left;">PROCESSING TIMELINE</div>
        <div class="pipeline-node-grid" style="display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 4px;">
            {nodes_html}
        </div>
    </div>
    """
    _render_html(html)


def render_entity_cards(entities: Dict[str, List[str]]):
    """Render categorized entity extraction chips."""
    groups_html = ""
    category_styles = {
        "Person": ("PERSON", "PERSON"),
        "Temporal": ("TEMPORAL", "TEMPORAL / DATES"),
        "Quantitative": ("QUANTITATIVE", "QUANTITATIVE / NUMERICAL"),
        "Identifiers": ("IDENTIFIER", "KEY IDENTIFIERS"),
    }

    for cat, (chip_class, title) in category_styles.items():
        items = entities.get(cat, [])
        if items:
            chips_html = "".join([f'<span class="entity-chip" style="font-family: var(--font-mono); font-size: 0.75rem; padding: 4px 8px; border: 1px solid var(--border-glass); border-radius: 4px; background: var(--bg-secondary); color: var(--text-primary); margin-right: 6px; margin-bottom: 6px; display: inline-block;">{item}</span>' for item in items])
            groups_html += f"""
            <div class="entity-group" style="margin-bottom: 1rem;">
                <div class="entity-group-header" style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px;">
                    {title} ({len(items)})
                </div>
                <div class="entity-chips">{chips_html}</div>
            </div>
            """

    if not groups_html:
        groups_html = """
        <div class="entity-group" style="text-align: left; color: var(--text-muted); font-family: var(--font-mono); font-size: 0.8rem;">
            No structured entities detected.
        </div>
        """

    html = f"""
    <div class="entity-container" style="background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 4px; padding: 1.5rem;">
        {groups_html}
    </div>
    """
    _render_html(html)
