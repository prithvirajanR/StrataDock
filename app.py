"""
StrataDock — Parallelized Molecular Docking Pipeline
Multi-page progressive UI — no sidebar expanders.
"""

import os, subprocess, tempfile, re, shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, SDMolSupplier
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import meeko
from Bio import PDB

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="StrataDock", page_icon="⚗️",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Molecular background watermark (top-right) ──────────────────────────
import base64 as _b64
_bg_path = os.path.join(os.path.dirname(__file__), "assets", "molecular_bg.png")
if os.path.exists(_bg_path):
    _bg_b64 = _b64.b64encode(open(_bg_path, "rb").read()).decode()
    st.markdown(f"""<img src="data:image/png;base64,{_bg_b64}" style="
        position:fixed; top:-20px; right:-20px; width:650px; height:650px;
        opacity:0.22; pointer-events:none; z-index:1; object-fit:cover;
        mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
        -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
    "/>""", unsafe_allow_html=True)

# =============================================================================
# CSS — minimal, no expander overrides
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Reset ─────────────────────────────────────────── */
*, html, body { 
    font-family: 'Inter', sans-serif !important; 
    -webkit-font-smoothing: antialiased; 
    -moz-osx-font-smoothing: grayscale;
    line-height: 1.55;
}

/* ── App Background ───────────────────────────────────────── */
.stApp { 
    background: #242424;
    color: #D2D2D2;
}

/* Force text visibility on ALL elements */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown a,
.stMarkdown div, .stMarkdown strong, .stMarkdown em, .stMarkdown code,
.stCaption, .stCaption p, label, span, p,
.stSelectbox label, .stNumberInput label, .stTextInput label,
.stSlider label, .stRadio label, .stCheckbox label, .stToggle label,
.stRadio div[role="radiogroup"] label, .stRadio div[role="radiogroup"] label div,
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] a, [data-testid="stMarkdownContainer"] strong,
[data-testid="stText"], [data-testid="stCaptionContainer"],
.element-container, .stAlert p, .stAlert div {
    color: #D2D2D2 !important;
}
h1, h2, h3, h4, h5, h6 { 
    color: #EAEAEA !important; -webkit-text-fill-color: #EAEAEA !important; 
    display: inline-block;
}

/* Links */
a, a:visited, a:hover { color: #D2D2D2 !important; }

/* Custom HTML classes — force bright text on dark cards */
.help-card, .help-card *, .help-desc, .eng-card, .eng-card *,
.card, .card *, .param-desc, .param-name, .faq-body,
.box-cell, .box-cell *, .setting-group, .setting-group * {
    color: #D2D2D2 !important;
}
.help-title, .eng-title, .card-title, .box-val { color: #EAEAEA !important; }
.help-title, .eng-title, .card-title { 
    display: inline-block; 
}
.help-desc, .param-desc, .eng-sub, .faq-body, .box-lbl { color: #BBBBBB !important; }

/* Button text — light buttons get dark text */
.stButton > button, .stDownloadButton > button {
    color: #242424 !important; -webkit-text-fill-color: #242424 !important;
    background: #D2D2D2 !important;
}

/* File uploader browse button — keep it readable on dark bg */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
    background: #D2D2D2 !important; color: #242424 !important; 
    -webkit-text-fill-color: #242424 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}

/* File uploader text ("Drag and drop", "Limit XMB", etc) */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: #BBBBBB !important;
}

/* Streamlit native info/warning/error/success text */
.stAlert [data-testid="stMarkdownContainer"] p { color: #D2D2D2 !important; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #4a4a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #666666; }

section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom Footer ──────────────────────────────────────── */
.custom-footer {
    position: fixed; bottom: 0; left: 0; width: 100%;
    background: #1a1a1a;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding: 0.55rem 2rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; color: #6b6b6b;
    letter-spacing: 0.06em; z-index: 9999;
    font-family: 'Inter', sans-serif; text-transform: uppercase;
}
.block-container { 
    padding: 2.5rem 2rem 4rem; max-width: 95% !important; 
    animation: fadeInUp 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Logo bar ───────────────────────────────────────────── */
.logo-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 0 0; margin-bottom: 2.5rem;
}
.logo-left { display:flex; align-items:center; gap:20px; }
.logo-icon {
    width:72px; height:72px;
    background: #1a1a1a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px; display:flex; align-items:center;
    justify-content:center; flex-shrink:0; font-size:2.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    transform: skewX(-8deg);
}
.logo-icon:hover { transform: scale(1.05); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
.logo-icon::after {
    content:''; position:absolute; top:0; left:-100%; width:40%; height:100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.06), transparent);
    animation: shineIcon 5s infinite ease-in;
}
@keyframes shineIcon { 
    0% { left: -100% } 
    20% { left: 100% } 
    100% { left: 100% } 
}
.logo-name { 
    font-size:2.2rem; font-weight:800; 
    color: #EAEAEA;
    -webkit-text-fill-color: #EAEAEA;
    letter-spacing:-0.04em; margin-bottom: 2px;
    text-transform: uppercase;
    transform: skewX(-8deg);
    transform-origin: left bottom;
}
@keyframes shineText {
    0%   { background-position: -50% center; }
    12%  { background-position: -50% center; }
    25%  { background-position: 200% center; }
    100% { background-position: 200% center; }
}
.logo-sub { font-size:0.82rem; color:#888888; letter-spacing:0.08em; font-weight:500; text-transform: uppercase; }
.logo-gpu {
    display:flex;align-items:center;gap:8px;
    padding:0.4rem 1.2rem;
    background: #1a1a1a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: all 0.25s ease;
}
.logo-gpu:hover { background: #333333; border-color: rgba(255,255,255,0.15); transform: translateY(-1px); }
.logo-gpu-dot { width:7px;height:7px;border-radius:50%; display:inline-block; box-shadow: 0 0 6px currentColor; }
.logo-gpu-text { font-size:0.78rem; font-weight: 600; color:#999999; text-transform:uppercase; letter-spacing:0.04em; }

.eng-badge.cpu { background:#1a1a1a; color:#999999; border:1px solid #3a3a3a; }
.eng-badge.gpu { background:#1a1a1a; color:#7acc7a; border:1px solid #3a5a3a; }

/* ── Typography headers ──────────────────────────────────── */
.page-title {
    font-size: 2.4rem; font-weight: 800;
    color: #EAEAEA; -webkit-text-fill-color: #EAEAEA;
    margin-bottom: 0.25rem; line-height: 1.15; letter-spacing: -0.03em;
    text-transform: uppercase;
    display: inline-block;
}
.page-sub {
    font-size: 0.9rem; color: #888888; margin-bottom: 2rem; font-weight: 400; letter-spacing: 0.02em;
}

/* ── Cards ────────────────────────────────────────────────── */
.card {
    background: #2e2e2e;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px; padding: 2rem; margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.card:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5); 
    border-color: rgba(255,255,255,0.12);
}
.card-title { 
    font-size: 1.05rem; font-weight: 700; color: #EAEAEA; margin-bottom: 0.9rem; letter-spacing: 0.01em; text-transform: uppercase; 
    display: inline-block;
}

/* ── Input Elements ──────────────────────────────────────── */
.stButton > button {
    background: #D2D2D2 !important;
    color: #242424 !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; font-size: 0.85rem !important; padding: 0.6rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { 
    transform: translateY(-2px) !important; 
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important; 
    background: #ffffff !important;
    color: #1a1a1a !important; 
}
.stButton > button:active { transform: translateY(1px) !important; }


/* Download buttons */
.stDownloadButton > button { }
.stDownloadButton > button:hover { transform: translateY(-2px) !important; }

.stButton > button[data-testid="stBaseButton-primary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
}
.stButton > button[data-testid="stBaseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
}
.stButton > button[data-testid="stBaseButton-primary"]:active,
.stButton > button[data-testid="stBaseButton-secondary"]:active {
    transform: translateY(1px) !important;
}

/* ── Custom Loader ───────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: #D2D2D2 !important;
    border-right-color: rgba(210, 210, 210, 0.2) !important;
    border-bottom-color: rgba(210, 210, 210, 0.05) !important;
    border-left-color: rgba(210, 210, 210, 0.6) !important;
    border-width: 2px !important;
}

/* Base input hover states */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
    transition: all 0.2s ease;
}
.stTextInput>div>div>input:hover, .stSelectbox>div>div>div:hover, .stNumberInput>div>div>input:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.25) !important;
}

/* Metrics — compact */
[data-testid="stMetric"] {
    background: #2e2e2e;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid #FF4B4B;
    border-radius: 10px; padding: 0.7rem 1rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 4px 16px rgba(0,0,0,0.3); 
}
[data-testid="stMetricLabel"] { color: #888888 !important; font-size: 0.7rem !important; font-weight:600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { 
    font-size: 1.3rem !important; font-weight: 700 !important; 
    color: #EAEAEA !important; -webkit-text-fill-color: #EAEAEA !important;
}

/* Upload zones */
[data-testid="stFileUploaderDropzone"] {
    background: #2e2e2e !important;
    border: 2px dashed rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important; transition: all 0.3s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover { 
    border-color: #D2D2D2 !important; 
    background: #353535 !important; 
    transform: scale(1.01) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #FF4B4B, #FF7B7B) !important;
    border-radius: 4px !important;
}

/* Alerts */
.stAlert { 
    border-radius: 12px !important; 
    background: #2e2e2e !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-left: 3px solid #D2D2D2 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
}

/* ── Terminal ─────────────────────────────────────────────── */
.terminal-block {
    background: #1a1a1a;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 0.85rem;
    color: #8fbc8f;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    height: 320px;
    overflow-y: auto;
    margin-bottom: 1.2rem;
    display: flex;
    flex-direction: column-reverse;
    transition: all 0.3s ease;
}

/* Active Terminal */
.term-active {
    border-color: rgba(210, 210, 210, 0.25) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5), 0 0 1px rgba(210,210,210,0.2) !important;
    animation: termPulse 2.5s infinite ease-in-out;
}
@keyframes termPulse {
    0%   { border-color: rgba(210,210,210,0.1); }
    50%  { border-color: rgba(210,210,210,0.3); }
    100% { border-color: rgba(210,210,210,0.1); }
}
.terminal-block .err { color: #e06c60; }
.terminal-block .warn { color: #d4a844; }
.terminal-block .info { color: #8fbc8f; }
.terminal-block::-webkit-scrollbar { width: 5px; }
.terminal-block::-webkit-scrollbar-track { background: transparent; }
.terminal-block::-webkit-scrollbar-thumb { background: #4a4a4a; border-radius: 3px; }

/* ── Engine choice cards ──────────────────────────────── */
.eng-card {
    border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1.4rem;
    transition: all .25s ease; 
    background: #2e2e2e;
    margin-bottom: 0.8rem; box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    color: #D2D2D2;
}
.eng-card.selected { 
    border-color: rgba(210,210,210,0.3);
    background: #383838;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35); 
    transform: translateY(-2px);
}
.eng-card:hover:not(.selected) { 
    border-color: rgba(255,255,255,0.12); 
    transform: translateY(-2px); 
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.eng-title { font-size:1.05rem; font-weight:700; color:#EAEAEA; margin-bottom:0.3rem; text-transform: uppercase; }
.eng-sub   { font-size:0.8rem; color:#888888; }
.eng-badge {
    display:inline-block; font-size:0.68rem; font-weight:700; padding:3px 8px;
    border-radius:8px; margin-left:8px; vertical-align:middle; text-transform:uppercase; letter-spacing:0.04em;
}

/* Box display — compact */
.box-grid { display: grid; grid-template-columns: repeat(6,1fr); gap: 6px; margin-top:0.5rem; }
.box-cell {
    background: #2e2e2e; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 8px 4px; text-align: center;
    transition: all 0.2s ease; color: #D2D2D2;
}
.box-cell:hover { border-color: rgba(255,255,255,0.15); background: #353535; }
.box-val { font-size: 0.95rem !important; }
.box-lbl { font-size: 0.65rem !important; }
.box-val { font-size: 1.05rem; font-weight: 800; color: #EAEAEA; }
.box-lbl { font-size: 0.65rem; color: #888888; margin-top: 3px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── FAQ accordion ───────────────────────────────────────── */
details.faq-item {
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 0 1.2rem;
    margin-bottom: 0.6rem; background: #2e2e2e;
    transition: all 0.2s ease;
}
details.faq-item:hover { border-color: rgba(255,255,255,0.12); background: #353535; }
details.faq-item summary {
    padding: 1rem 0; cursor: pointer; font-size: 0.9rem;
    font-weight: 600; color: #EAEAEA; list-style: none;
    display: flex; align-items: center; gap: 10px;
}
details.faq-item summary::-webkit-details-marker { display:none; }
details.faq-item summary::before { content:'▶'; font-size:0.6rem; color:#666666; transition:transform .2s; flex-shrink:0; }
details.faq-item[open] summary::before { transform:rotate(90deg); color: #D2D2D2; }
details.faq-item .faq-body { padding: 0 0 1rem; font-size: 0.85rem; color: #999999; line-height: 1.7; }

.sect {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #666666; margin: 2rem 0 0.8rem;
}

/* Settings row */
.setting-row { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1.5rem; }
.setting-group { background: #2e2e2e; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1.5rem; }
.setting-group-title { font-size: 0.78rem; font-weight: 700; color: #888888; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 1.2rem; }

/* Help */
.help-card { background: #2e2e2e; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1.8rem; margin-bottom: 1.2rem; color: #D2D2D2; }
.help-title { font-size: 1.1rem; font-weight: 800; color: #EAEAEA; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.02em; }
.help-desc { font-size: 0.9rem; color: #999999; line-height: 1.6; }
.param-row { display: flex; gap: 1rem; padding: 0.65rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.param-name { font-size: 0.85rem; font-weight: 700; color: #EAEAEA; min-width: 170px; }
.param-desc { font-size: 0.85rem; color: #999999; }

/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #888888 !important;
    border-radius: 8px 8px 0 0 !important; font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important; text-transform: uppercase !important;
    font-size: 0.82rem !important; letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #D2D2D2 !important; background: rgba(255,255,255,0.03) !important; }
.stTabs [aria-selected="true"] {
    background: #2e2e2e !important; color: #EAEAEA !important;
    font-weight: 700 !important;
}

/* ── Dataframe overrides ──────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }
""", unsafe_allow_html=True)


# =============================================================================
# GPU DETECTION
# =============================================================================
@st.cache_resource
def detect_gpu() -> dict:
    mode, detail = "cpu", "CPU only"
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            mode, detail = "nvidia", r.stdout.strip().splitlines()[0]
    except Exception: pass
    return {"mode": mode, "detail": detail}


# =============================================================================
# SESSION DEFAULTS
# =============================================================================
def init_state():
    defaults = {
        "page": "Upload",
        "results": [], "boxes": {}, "rec_path": None,
        "work_dir": None, "rec_bytes": None, "lig_files_data": [],
        "multi_pocket": False, "cancel_run": False,
        # RDKit
        "embed_method": "ETKDGv3", "max_attempts": 250,
        "force_field": "MMFF94", "ff_steps": 500,
        "add_hs": True, "keep_chirality": True,
        # Meeko
        "merge_nphs": True, "hydrate": False,
        # Engine
        "engine": "AutoDock Vina",
        # GNINA
        "exhaustiveness": 16, "num_poses": 9, "cnn_model": "default",
        "cnn_weight": 1.0, "min_rmsd": 1.0, "gnina_seed": 0, "gpu_device": "auto",
        "flexdist": False, "cnn_mode": "none", "max_flex_res": 4,
        # Vina
        "v_exhaustiveness": 8, "v_num_poses": 9,
        "energy_range": 3, "scoring_fn": "vina", "vina_seed": 0,
        # Box
        "box_padding": 8,
        "box_padding": 8,
        # Default target pH
        "target_ph": 7.4,
        # Protein Preparation
        "prep_remove_water": True, "prep_remove_het": True, "prep_keep_metals": True,
        "prep_add_h": True, "prep_fix_atoms": True, "prep_fix_loops": True,
        "prep_propka": False, "prep_minimize": False, "prep_min_steps": 100,
        # Parallelization
        "n_jobs": max(1, (os.cpu_count() or 4) - 1), "backend": "loky",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
gpu = detect_gpu()


# =============================================================================
# STEPPER NAV
# =============================================================================
STEPS = [
    ("Upload",   "1", "⏏ Upload"),
    ("Settings", "2", "⚙ Settings"),
    ("Run",      "3", "▶ Run"),
    ("Results",  "4", "📊 Results"),
    ("Help",     "5", "ℹ Help"),
]

def render_stepper():
    """Logo bar + pill-button nav row."""

    # Logo bar
    gpu_color = {"nvidia":"#3fb950","cpu":"#FF4B4B"}.get(gpu["mode"], "#FF4B4B")
    svg_icon = """
    <svg width="34" height="34" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- bonds -->
      <line x1="13" y1="13" x2="6" y2="6"  stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.55"/>
      <line x1="13" y1="13" x2="20" y2="6"  stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.55"/>
      <line x1="13" y1="13" x2="6" y2="20" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.55"/>
      <line x1="13" y1="13" x2="20" y2="20" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.55"/>
      <!-- peripheral atoms -->
      <circle cx="6"  cy="6"  r="2.8" fill="white" opacity="0.7"/>
      <circle cx="20" cy="6"  r="2.8" fill="white" opacity="0.7"/>
      <circle cx="6"  cy="20" r="2.8" fill="white" opacity="0.7"/>
      <circle cx="20" cy="20" r="2.8" fill="white" opacity="0.7"/>
      <!-- central atom -->
      <circle cx="13" cy="13" r="4" fill="white"/>
    </svg>"""

    st.markdown(f"""
    <div class="logo-bar">
        <div class="logo-left">
            <div class="logo-icon">{svg_icon}</div>
            <div>
                <div class="logo-name">&gt;_ STRATA DOCK</div>
                <div class="logo-sub" style="text-align:right;">Alpha V0.7.02</div>
            </div>
        </div>
        <div class="logo-gpu">
            <span class="logo-gpu-dot" style="background:{gpu_color}"></span>
            <span class="logo-gpu-text">{gpu["detail"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav — slanted parallelogram tabs
    order   = [s[0] for s in STEPS]
    cur_idx = order.index(st.session_state.page) if st.session_state.page in order else 0
    done    = {"Upload":   st.session_state.rec_bytes is not None and bool(st.session_state.lig_files_data),
               "Settings": bool(st.session_state.boxes),
               "Run":      bool(st.session_state.results),
               "Results":  bool(st.session_state.results),
               "Help":     False}

    cols = st.columns(len(STEPS))
    for col, (i, (key, num, label)) in zip(cols, enumerate(STEPS)):
        with col:
            is_active = (key == st.session_state.page)
            is_done   = (i < cur_idx) or done.get(key, False)
            prefix = "✓ " if is_done and not is_active else f"{num}. "
            btn_type = "primary" if is_active else "secondary"
            if st.button(prefix + label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.page = key
                st.rerun()

    pct = ((cur_idx + 1) / len(STEPS)) * 100
    st.markdown(f"""
    <div style="width: 100%; height: 3px; background: rgba(255,255,255,0.05); margin: 0.4rem 0 1.5rem; border-radius: 2px;">
        <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #FF4B4B, #FF7B7B); 
             box-shadow: 0 0 12px rgba(255, 75, 75, 0.4); border-radius: 2px; 
             transition: width 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275);"></div>
    </div>
    """, unsafe_allow_html=True)




# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================
def compute_autobox(pdb_path, ref_path, padding):
    coords = []
    if ref_path:
        for mol in SDMolSupplier(ref_path, removeHs=True):
            if mol:
                conf = mol.GetConformer()
                coords.extend([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    if not coords:
        parser = PDB.PDBParser(QUIET=True)
        for atom in parser.get_structure("r", pdb_path).get_atoms():
            if atom.name == "CA": coords.append(atom.coord)
    if not coords: raise ValueError("No atoms found for autoboxing.")
    coords = np.array([[c[0],c[1],c[2]] if hasattr(c,"__len__") else [c.x,c.y,c.z] for c in coords])
    mins, maxs = coords.min(0), coords.max(0)
    return {k: round(float(v), 3) for k, v in zip(
        ["cx","cy","cz","sx","sy","sz"], list((mins+maxs)/2) + list(maxs-mins+padding))}


def compute_admet(mol):
    from rdkit.Chem import Descriptors, Lipinski, QED
    try:
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        qed = QED.qed(mol)
        
        # Lipinski Rule of 5 Violations
        fails = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        return {"MW": mw, "LogP": logp, "TPSA": tpsa, "HBD": hbd, "HBA": hba, "QED": qed, "Lipinski Fails": fails}
    except Exception:
        return {"MW": None, "LogP": None, "TPSA": None, "HBD": None, "HBA": None, "QED": None, "Lipinski Fails": None}


def analyze_interactions(rec_pdb, pose_sdf, distance_cutoff=4.0):
    """Find receptor residues interacting with the docked ligand pose."""
    try:
        from Bio.PDB import PDBParser, NeighborSearch, Selection
        # Parse receptor
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rec", rec_pdb)
        rec_atoms = Selection.unfold_entities(structure, "A")

        # Parse ligand from SDF
        supp = SDMolSupplier(pose_sdf, removeHs=False)
        lig_mol = next((m for m in supp if m is not None), None)
        if lig_mol is None:
            return pd.DataFrame()

        conf = lig_mol.GetConformer()
        lig_coords = [conf.GetAtomPosition(i) for i in range(lig_mol.GetNumAtoms())]
        lig_elements = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(lig_mol.GetNumAtoms())]

        # Build NeighborSearch from receptor atoms
        ns = NeighborSearch(rec_atoms)

        POLAR = {"N", "O", "S", "F"}
        HYDROPHOBIC = {"C"}
        CHARGED_POS = {"ARG", "LYS", "HIS"}
        CHARGED_NEG = {"ASP", "GLU"}

        contacts = []
        seen = set()
        for i, (lc, le) in enumerate(zip(lig_coords, lig_elements)):
            nearby = ns.search(np.array([lc.x, lc.y, lc.z]), distance_cutoff, "A")
            for atom in nearby:
                res = atom.get_parent()
                res_key = (res.get_resname(), res.get_id()[1], res.get_full_id()[2])
                if res_key in seen:
                    continue
                seen.add(res_key)
                
                dist = np.linalg.norm(np.array([lc.x, lc.y, lc.z]) - atom.get_vector().get_array())
                rec_elem = atom.element.strip() if atom.element else atom.get_name()[0]
                resname = res.get_resname()
                
                # Classify bond type
                if rec_elem in POLAR and le in POLAR and dist < 3.5:
                    bond_type = "H-bond"
                elif rec_elem in HYDROPHOBIC and le in HYDROPHOBIC and dist < 4.0:
                    bond_type = "Hydrophobic"
                elif ((resname in CHARGED_POS and le in POLAR) or (resname in CHARGED_NEG and le in POLAR)) and dist < 4.0:
                    bond_type = "Salt bridge"
                else:
                    bond_type = "van der Waals"

                contacts.append({
                    "Residue": resname,
                    "Chain": res.get_full_id()[2],
                    "ResID": res.get_id()[1],
                    "Receptor Atom": atom.get_name(),
                    "Distance (Å)": round(dist, 2),
                    "Bond Type": bond_type
                })

        df = pd.DataFrame(contacts)
        if not df.empty:
            df = df.sort_values("Distance (Å)").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def build_complex_pdb(rec_pdb, pose_sdf, out_path=None):
    """Merge receptor PDB and docked ligand SDF into a single PDB complex file."""
    try:
        if not Path(rec_pdb).exists() or not Path(pose_sdf).exists():
            return None
        if out_path is None:
            out_path = pose_sdf.replace("_out.sdf", "_complex.pdb")

        rec_lines = Path(rec_pdb).read_text().splitlines()
        # Remove END/ENDMDL lines from receptor
        rec_clean = [l for l in rec_lines if not l.startswith("END")]

        # Convert ligand SDF to PDB HETATM records via RDKit
        supp = SDMolSupplier(pose_sdf, removeHs=False)
        lig_mol = next((m for m in supp if m is not None), None)
        if lig_mol is None:
            return None

        lig_pdb = Chem.MolToPDBBlock(lig_mol)
        # Convert ATOM to HETATM for the ligand
        lig_lines = []
        for l in lig_pdb.splitlines():
            if l.startswith("ATOM"):
                lig_lines.append("HETATM" + l[6:])
            elif not l.startswith("END"):
                lig_lines.append(l)

        combined = "\n".join(rec_clean) + "\nTER\n" + "\n".join(lig_lines) + "\nEND\n"
        Path(out_path).write_text(combined)
        return out_path
    except Exception:
        return None




def prepare_ligand(mol, name, cfg, out_dir):
    try:
        # If Target pH is requested and molecule came from SMILES, run OpenBabel pH correction
        target_ph = cfg.get("target_ph", 7.4)
        if target_ph is not None and not mol.GetNumConformers():
            try:
                import subprocess, shutil, sys
                paths = [shutil.which("obabel"), os.path.join(os.path.dirname(sys.executable), "obabel"), "obabel"]
                ob_exe = next((p for p in paths if p and Path(p).exists()), None)
                if ob_exe:
                    smi = Chem.MolToSmiles(mol)
                    # Run obabel: read smiles, calculate protonation at pH, add hydrogens, output smiles
                    res = subprocess.run([ob_exe, f"-:{smi}", "-ismi", "-osmi", f"-p{target_ph}", "-h"], 
                                         capture_output=True, text=True, timeout=15)
                    if res.returncode == 0 and res.stdout.strip():
                        # Re-ingest the pH-corrected SMILES back into RDKit
                        corr_smi = res.stdout.split()[0].strip()
                        new_mol = Chem.MolFromSmiles(corr_smi)
                        if new_mol: mol = new_mol
            except Exception as e:
                pass # Fallback to original unprotonated RDKit molecule if OpenBabel fails
                
        mol.SetProp("_Name", name)
        if cfg.get("add_hs", True): mol = Chem.AddHs(mol, addCoords=True)
        pm = {"ETKDGv3":AllChem.ETKDGv3(),"ETKDGv2":AllChem.ETKDGv2(),"ETKDG":AllChem.ETKDG()}
        params = pm.get(cfg.get("embed_method","ETKDGv3"), AllChem.ETKDGv3())
        params.randomSeed = 42; params.maxIterations = cfg.get("max_attempts",250)
        if AllChem.EmbedMolecule(mol, params) == -1:
            return None, f"RDKit EmbedMolecule failed to generate a 3D conformer within {cfg.get('max_attempts')} attempts."
        ff, steps = cfg.get("force_field","MMFF94"), cfg.get("ff_steps",500)
        if ff in ("MMFF94","MMFF94s"):
            p = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=ff)
            if p: AllChem.MMFFOptimizeMolecule(mol, mmffVariant=ff, maxIters=steps)
        elif ff == "UFF": AllChem.UFFOptimizeMolecule(mol, maxIters=steps)
        try:
            # Meeko >= 0.5.0 API
            from meeko.setup import builtin_o3
            prep = meeko.MoleculePreparation(
                setup=builtin_o3,
                merge_hydrogens=cfg.get("merge_nphs",True),
                hydrate=cfg.get("hydrate",False))
        except (TypeError, ImportError):
            # Fallback for older Meeko < 0.5.0
            prep = meeko.MoleculePreparation(
                merge_these_atom_types=["H"] if cfg.get("merge_nphs",True) else [],
                hydrate=cfg.get("hydrate",False))
            
        prep.prepare(mol)
        out = os.path.join(out_dir, f"{name}.pdbqt")
        Path(out).write_text(prep.write_pdbqt_string())
        return out, None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()


def prepare_protein(pdb_path, cfg):
    """Full protein preparation pipeline (steps 1-7). Returns path to prepared PDB."""
    prep_log = []
    out_path = pdb_path.replace(".pdb", "_prepared.pdb")
    
    # ── Steps 1-5: PDBFixer-based preparation ──
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        
        # Load directly from original PDB (avoid intermediate file issues)
        fixer = PDBFixer(filename=pdb_path)
        
        # Step 1+2: Remove water and heteroatoms together via PDBFixer
        if cfg.get("prep_remove_het", True) or cfg.get("prep_remove_water", True):
            keep_water = not cfg.get("prep_remove_water", True)
            fixer.removeHeterogens(keepWater=keep_water)
            prep_log.append("Removed heteroatoms" + (" + water" if not keep_water else ""))
        elif cfg.get("prep_remove_water", True):
            fixer.removeHeterogens(keepWater=False)
            prep_log.append("Removed water molecules")
        
        # Step 5: Find and model missing residues/loops
        if cfg.get("prep_fix_loops", True):
            fixer.findMissingResidues()
            prep_log.append("Identified missing loops/residues")
        
        # Step 4: Fix missing atoms
        if cfg.get("prep_fix_atoms", True):
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            prep_log.append("Repaired missing heavy atoms")
        
        # Step 3: Add hydrogens at pH
        if cfg.get("prep_add_h", True):
            ph = cfg.get("target_ph", 7.4)
            fixer.addMissingHydrogens(ph)
            prep_log.append(f"Added hydrogens at pH {ph}")
        
        # Write PDBFixer output
        with open(out_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        prep_log.append("PDBFixer preparation complete")
        
    except ImportError:
        prep_log.append("[WARN] PDBFixer/OpenMM not installed — skipping steps 2-5")
        # Manual fallback: at least strip water via line filtering
        if cfg.get("prep_remove_water", True):
            pdb_text = Path(pdb_path).read_text()
            lines = [l for l in pdb_text.splitlines() if not (l.startswith(("HETATM", "ATOM")) and "HOH" in l)]
            Path(out_path).write_text("\n".join(lines))
            prep_log.append("Removed water molecules (manual fallback)")
    except Exception as e:
        prep_log.append(f"[WARN] PDBFixer error: {str(e)[:100]} — using raw PDB")
    
    # ── Step 6: Protonation states via propka ──
    if cfg.get("prep_propka", False):
        try:
            import propka.run as propka_run
            # propka needs a file path
            mol_container = propka_run.single(out_path)
            pka_info = []
            for group in mol_container.conformations["AVR"].groups:
                if hasattr(group, "pka_value") and group.pka_value is not None:
                    pka_info.append(f"  {group.residue_type} {group.atom.chain_id}{group.atom.res_num}: pKa={group.pka_value:.2f}")
            if pka_info:
                prep_log.append(f"propka3: {len(pka_info)} titratable residues analyzed")
            else:
                prep_log.append("propka3: No titratable residues found")
        except ImportError:
            prep_log.append("[WARN] propka not installed — skipping protonation analysis")
        except Exception as e:
            prep_log.append(f"[WARN] propka error: {str(e)[:100]}")
    
    # ── Step 7: Energy minimization via OpenMM ──
    if cfg.get("prep_minimize", False):
        try:
            from openmm.app import PDBFile, ForceField, Modeller, Simulation
            from openmm import LangevinMiddleIntegrator, LocalEnergyMinimizer
            import openmm.unit as unit
            
            pdb = PDBFile(out_path)
            forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            modeller = Modeller(pdb.topology, pdb.positions)
            
            system = forcefield.createSystem(modeller.topology, 
                                              nonbondedMethod=0,  # NoCutoff
                                              constraints=None)
            integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            simulation = Simulation(modeller.topology, system, integrator)
            simulation.context.setPositions(modeller.positions)
            
            max_steps = cfg.get("prep_min_steps", 100)
            LocalEnergyMinimizer.minimize(simulation.context, maxIterations=max_steps)
            
            positions = simulation.context.getState(getPositions=True).getPositions()
            with open(out_path, "w") as f:
                PDBFile.writeFile(simulation.topology, positions, f)
            prep_log.append(f"Energy minimization complete ({max_steps} steps, Amber14)")
        except ImportError:
            prep_log.append("[WARN] OpenMM not installed — skipping minimization")
        except Exception as e:
            prep_log.append(f"[WARN] Minimization error: {str(e)[:100]} — using unminimized structure")
    
    return out_path, prep_log


def prepare_receptor(pdb_path, out_dir):
    out = os.path.join(out_dir, "receptor.pdbqt")
    # Convert receptor to PDBQT using OpenBabel (reliable)
    
    # Fallback: use OpenBabel to prepare receptor
    r_ob = None
    import shutil
    import sys
    try:
        paths = [
            shutil.which("obabel"),
            os.path.join(os.path.dirname(sys.executable), "obabel"),
            os.path.expanduser("~/miniconda3/envs/stratadock/bin/obabel"),
            "/home/rajan/miniconda3/envs/stratadock/bin/obabel",
            "obabel"
        ]
        ob_exe = next((p for p in paths if p and Path(p).exists()), None)
        if ob_exe:
            r_ob = subprocess.run([ob_exe, pdb_path, "-O", out, "-xr"],
                               capture_output=True, text=True, timeout=60)
            if Path(out).exists(): return out
    except FileNotFoundError: pass

    # Last resort fallback: meeko Python API
    try:
        import meeko
        rec = meeko.PDBQTReceptor(pdb_path)
        Path(out).write_text(rec.get_pdbqt_string())
        return out
    except Exception: pass
        
    err = r_ob.stderr if (r_ob and hasattr(r_ob, "stderr")) else (
        f"obabel missing. Searched paths: {[p for p in paths if p]}." if 'paths' in locals() else "obabel missing"
    )
    raise RuntimeError(f"Failed to convert receptor to PDBQT. Obabel stderr: {err}")


def parse_gnina(sdf, log):
    s = {"cnn_score":None,"cnn_affinity":None,"vina_score":None}
    if Path(sdf).exists():
        for mol in SDMolSupplier(sdf, removeHs=False):
            if mol is None: continue
            for prop, key in [("CNNscore","cnn_score"),("CNNaffinity","cnn_affinity"),("minimizedAffinity","vina_score")]:
                try: s[key] = float(mol.GetProp(prop))
                except: pass
            break
    if s["vina_score"] is None:
        m = re.search(r"^\s+1\s+([-\d.]+)", log, re.MULTILINE)
        if m: s["vina_score"] = float(m.group(1))
    return s


def run_single(lig_pdbqt, receptor, box, cfg, out_dir, name, term_el=None, log_lines=None):
    out_sdf = os.path.join(out_dir, f"{name}_out.sdf")
    res = {"name":name,"cnn_score":None,"cnn_affinity":None,"vina_score":None,"status":"failed","pose_sdf":None}
    if log_lines is None: log_lines = []
    
    def update_term(new_line="", temp_line=""):
        if not term_el: return
        if new_line:
            # Color coding the output
            l = new_line.strip()
            if "error" in l.lower() or "exception" in l.lower(): l = f"<span class='err'>{l}</span>"
            elif "warning" in l.lower(): l = f"<span class='warn'>{l}</span>"
            elif "score" in l.lower() or "affinity" in l.lower() or "RMSD" in l: l = f"<span class='info'>{l}</span>"
            log_lines.append(l)
        
        # Keep only last 100 lines to prevent lag
        display_lines = log_lines[-100:]
        if temp_line:
            display_lines = display_lines + [f"<span class='info'>{temp_line}</span>"]
            
        # Place text in a single child div inside a column-reverse parent.
        # Natively anchors scrollbar to the bottom (newest text) without JS.
        active_cls = " term-active" if temp_line else ""
        html = f"""
        <div class='terminal-block{active_cls}'>
            <div style="width: 100%;">
                {'<br>'.join(display_lines)}
            </div>
        </div>
        """
        term_el.markdown(html, unsafe_allow_html=True)

    try:
        engine = cfg.get("engine","AutoDock Vina")
        if engine == "GNINA":
            update_term(f"> Initiating GNINA protocol for {name}...")
            # 1. Bulletproof the path lookup (same logic used for OpenBabel)
            gnina_paths = [
                shutil.which("gnina"),
                os.path.expanduser("~/.local/bin/gnina"),
                os.path.expanduser("~/moderndock/gnina_binary"),
                "/usr/local/bin/gnina",
                "gnina"
            ]
            gnina_exe = next((p for p in gnina_paths if p and Path(p).exists()), None) or "gnina"
            
            gpu_flag = ["--no_gpu"] if cfg.get("gpu_device")=="cpu" else \
                       ["--device",cfg["gpu_device"]] if cfg.get("gpu_device") in ("0","1") else []
            cmd = [gnina_exe,"--receptor",receptor,"--ligand",lig_pdbqt,"--out",out_sdf,
                   "--center_x",f"{box['cx']:.3f}","--center_y",f"{box['cy']:.3f}","--center_z",f"{box['cz']:.3f}",
                   "--size_x",f"{box['sx']:.3f}","--size_y",f"{box['sy']:.3f}","--size_z",f"{box['sz']:.3f}",
                   "--exhaustiveness",str(cfg.get("exhaustiveness",16)),
                   "--num_modes",str(cfg.get("num_poses",9)),
                   "--min_rmsd_filter",str(cfg.get("min_rmsd",1.0)),
                   "--cnn_scoring",cfg.get("cnn_mode", "rescore")] + gpu_flag
            
            cnn = cfg.get("cnn_model","default")
            if cnn != "default": cmd += ["--cnn",cnn]
            if cfg.get("gnina_seed",0): cmd += ["--seed",str(cfg["gnina_seed"])]
            if cfg.get("flexdist", False): 
                cmd += ["--flexdist", "3.5", "--flexdist_ligand", lig_pdbqt]
            
            update_term(f"> Executing: {' '.join(cmd)}")
            
            try:
                import time
                import threading
                import queue
                
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                full_log = []
                q = queue.Queue()
                
                def enqueue_output(out, queue):
                    for line in iter(out.readline, ''):
                        if not line: break
                        queue.put(line)
                    out.close()
                    
                t = threading.Thread(target=enqueue_output, args=(proc.stdout, q))
                t.daemon = True
                t.start()
                
                start_time = time.time()
                while True:
                    if st.session_state.get("cancel_run", False):
                        proc.terminate()
                        break
                        
                    try:
                        line = q.get(timeout=0.1)
                        full_log.append(line)
                        update_term(new_line=line)
                    except queue.Empty:
                        if proc.poll() is not None:
                            break
                        elapsed = int(time.time() - start_time)
                        update_term(temp_line=f"⚙️ GNINA is computing flexible docking ({elapsed}s elapsed)...")
                        
                log = "".join(full_log)
            except FileNotFoundError:
                err = "error: GNINA executable not found on system path — use AutoDock Vina"
                update_term(err); res["status"] = err; return res
                
            # 3. Smart CPU Fallback. If a user forces GPU but their drivers are broken or VRAM runs out, 
            # don't just crash. Catch the exception and automatically re-run GNINA purely on the CPU.
            if ("libcudnn" in log or "libcuda" in log or "out of memory" in log.lower()) and "--no_gpu" not in cmd:
                update_term("> [WARNING] GPU execution failed. Triggering automatic CPU fallback protocol...")
                cmd = [c for c in cmd if c not in ("--device", "0", "1")] + ["--no_gpu"]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                full_log = []
                q2 = queue.Queue()
                t2 = threading.Thread(target=enqueue_output, args=(proc.stdout, q2))
                t2.daemon = True
                t2.start()
                
                start_time = time.time()
                while True:
                    if st.session_state.get("cancel_run", False):
                        proc.terminate()
                        break
                        
                    try:
                        line = q2.get(timeout=0.1)
                        full_log.append(line)
                        update_term(new_line=line)
                    except queue.Empty:
                        if proc.poll() is not None:
                            break
                        elapsed = int(time.time() - start_time)
                        update_term(temp_line=f"⚙️ GNINA CPU Fallback is computing flexible docking ({elapsed}s elapsed)...")
                        
                log = "".join(full_log)

            # GNINA is known to throw non-zero exit codes (like OpenBabel) if it encounters 
            # minor non-fatal warnings with ligand topology. If the file exists, it worked.
            if not Path(out_sdf).exists():
                err_msg = log[:150].replace('\n', ' ')
                update_term(f"> [FATAL] {err_msg}")
                res["status"] = f"error: {err_msg}"; return res
                
            res.update(parse_gnina(out_sdf, log))
            res["pose_sdf"] = out_sdf
            update_term(f"> [SUCCESS] {name} docking complete.")
        else:
            update_term(f"> Initiating AutoDock Vina protocol for {name}...")
            from vina import Vina
            v = Vina(sf_name=cfg.get("scoring_fn","vina"), seed=cfg.get("vina_seed",0), verbosity=0)
            v.set_receptor(receptor)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=[box["cx"],box["cy"],box["cz"]],
                                box_size=[box["sx"],box["sy"],box["sz"]])
            update_term(f"> Grid maps calculated. Running Monte Carlo search...")
            v.dock(exhaustiveness=cfg.get("v_exhaustiveness",8), n_poses=cfg.get("v_num_poses",9))
            out_pdbqt = os.path.join(out_dir, f"{name}_out.pdbqt")
            v.write_poses(out_pdbqt, n_poses=cfg.get("v_num_poses",9), overwrite=True)
            try:
                pm = meeko.PDBQTMolecule.from_file(out_pdbqt, skip_typing=True)
                rm = meeko.RDKitMolCreate.from_pdbqt_mol(pm)
                if rm and rm[0]:
                    w = SDWriter(out_sdf); w.write(rm[0]); w.close()
            except Exception: out_sdf = out_pdbqt
            en = v.energies(n_poses=1)
            if en is not None and len(en): 
                res["vina_score"] = float(en[0][0])
                update_term(f"> Optimization reached. Best sequence energy: {res['vina_score']:.2f} kcal/mol")
            res["pose_sdf"] = out_sdf if Path(out_sdf).exists() else out_pdbqt
            update_term(f"> [SUCCESS] {name} docking complete.")
            
        res["status"] = "success"
        res["pose_sdf"] = res.get("pose_sdf") or out_sdf
    except Exception as e:
        err = f"error: {str(e)[:120]}"
        if term_el: update_term(f"> [FATAL] {err}")
        res["status"] = err
    return res


def batch_dock(rec_items, lig_items, boxes, cfg, out_dir, prog, status_el):
    n, backend = cfg.get("n_jobs",1), cfg.get("backend","loky")
    tasks = []
    for rname, rpdbqt, rpdb in rec_items:
        rec_boxes = boxes.get(rname, list(boxes.values())[0] if boxes else [])
        for lname, lpdbqt in lig_items:
            for box in rec_boxes:
                tasks.append((rname, rpdbqt, rpdb, lname, lpdbqt, box))
            
    results = []
    st.session_state.cancel_run = False
    
    # Create the terminal element in the UI
    term_el = st.empty()
    shared_log = []
    
    if n == 1 or len(tasks) == 1:
        for i, (rn, rp, rpdb, ln, lp, bx) in enumerate(tasks):
            if st.session_state.get("cancel_run", False): 
                status_el.warning("Run cancelled by user.")
                break
                
            status_el.caption(f"Docking **{ln}** into **{rn}** [{bx['name']}] ({i+1}/{len(tasks)})…")
            res = run_single(lp, rp, bx, cfg, out_dir, f"r_{rn}_l_{ln}_{bx['name'].replace(' ', '')}", term_el, shared_log)
            res["Ligand"] = ln
            res["Receptor"] = rn
            res["Pocket"] = bx['name']
            res["rec_pdb"] = rpdb
            # Generate complex PDB and binding site analysis for successful docks
            if res["status"] == "success" and res.get("pose_sdf") and Path(res["pose_sdf"]).exists():
                res["complex_pdb"] = build_complex_pdb(rpdb, res["pose_sdf"])
                res["interactions"] = analyze_interactions(rpdb, res["pose_sdf"])
            results.append(res)
            prog.progress((i+1)/len(tasks))
    else:
        status_el.caption(f"Running {len(tasks)} docking pairs sequentially because threaded execution blocked Streamlit...")
        for i, (rn, rp, rpdb, ln, lp, bx) in enumerate(tasks):
            if st.session_state.get("cancel_run", False): 
                status_el.warning("Run cancelled by user.")
                break
                
            status_el.caption(f"Docking **{ln}** into **{rn}** [{bx['name']}] ({i+1}/{len(tasks)})…")
            res = run_single(lp, rp, bx, cfg, out_dir, f"r_{rn}_l_{ln}_{bx['name'].replace(' ', '')}", term_el, shared_log)
            res["Ligand"] = ln
            res["Receptor"] = rn
            res["Pocket"] = bx['name']
            res["rec_pdb"] = rpdb
            if res["status"] == "success" and res.get("pose_sdf") and Path(res["pose_sdf"]).exists():
                res["complex_pdb"] = build_complex_pdb(rpdb, res["pose_sdf"])
                res["interactions"] = analyze_interactions(rpdb, res["pose_sdf"])
            results.append(res)
            prog.progress((i+1)/len(tasks))
            
    # Save terminal log to session state
    st.session_state.terminal_log = "\n".join([re.sub(r'<[^>]+>', '', l) for l in shared_log])
    return results


def load_ligands(files):
    mols = []
    for f in files:
        suffix = Path(f.name).suffix.lower()
        content = f.read()
        if suffix == ".zip":
            import zipfile, io
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for zname in zf.namelist():
                        zsuffix = Path(zname).suffix.lower()
                        if zsuffix in [".sdf", ".smi", ".smiles", ".txt"] and not zname.startswith("__MACOSX"):
                            class ZFile:
                                def __init__(self,n,c): self.name=n; self.c=c
                                def read(self): return self.c
                            mols.extend(load_ligands([ZFile(zname, zf.read(zname))]))
            except Exception: pass
        elif suffix == ".sdf":
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
                tmp.write(content); tp = tmp.name
            for i, mol in enumerate(SDMolSupplier(tp, removeHs=False, sanitize=True)):
                if mol:
                    n = mol.GetProp("_Name").strip().replace(" ","_") if mol.HasProp("_Name") else f"mol_{i}"
                    mols.append((n or f"mol_{i}", mol))
            os.unlink(tp)
        elif suffix in (".smi",".smiles",".txt"):
            for i, line in enumerate(content.decode("utf-8",errors="ignore").strip().splitlines()):
                p = line.strip().split()
                if not p: continue
                mol = Chem.MolFromSmiles(p[0])
                if mol: mols.append((p[1] if len(p)>1 else f"smi_{i}", mol))
    return mols


def render_3d(rec_pdb, pose_sdf, view_style="Cartoon"):
    try:
        import py3Dmol
        from stmol import showmol
        
        # Hard override for Streamlit iframe containers to perfectly center them
        st.markdown(
            """<style>
            iframe[title="stmol.showmol"] {
                display: block; margin: 0 auto !important; width: 100% !important; max-width: 900px;
            }
            </style>""", 
            unsafe_allow_html=True
        )
        
        view = py3Dmol.view(width=900, height=600)
        if Path(rec_pdb).exists():
            view.addModel(Path(rec_pdb).read_text(), "pdb")
            
            if view_style == "Cartoon":
                view.setStyle({"model":0},{"cartoon":{"color":"spectrum"},"stick":{"radius":0.07,"color":"#30363d"}})
            elif view_style == "Surface":
                view.setStyle({"model":0},{"stick":{"radius":0.07,"color":"#30363d"}})
                view.addSurface(py3Dmol.VDW, {"opacity": 0.85, "color": "white"}, {"model":0})
                
        if pose_sdf and Path(pose_sdf).exists():
            view.addModel(Path(pose_sdf).read_text(), "sdf")
            view.setStyle({"model":1},{"stick":{"radius":0.22,"colorscheme":"cyanCarbon"}})
            view.addSurface(py3Dmol.VDW,{"opacity":0.3,"color":"#388bfd"},{"model":1})
                
        view.zoomTo()
        view.setBackgroundColor("#0c0f1d")
        
        # Free from columns; the CSS rule centers the iframe wrapper naturally
        showmol(view, height=600, width=900)
    except Exception as e:
        st.error(f"Viewer error: {e}")

def render_box_preview(rec_bytes, boxes):
    try:
        import py3Dmol
        from stmol import showmol
        
        view = py3Dmol.view(width=900, height=500)
        view.addModel(rec_bytes.decode("utf-8", errors="ignore"), "pdb")
        view.setStyle({"model":0},{"cartoon":{"color":"spectrum"}})
        
        for box in boxes:
            bspec = {
                "center": {"x": box["cx"], "y": box["cy"], "z": box["cz"]},
                "dimensions": {"w": box["sx"], "h": box["sy"], "d": box["sz"]},
                "color": "#00c6ff",
                "opacity": 0.5,
                "wireframe": True
            }
            view.addBox(bspec)
            bsolid = bspec.copy()
            bsolid["wireframe"] = False
            bsolid["opacity"] = 0.15
            view.addBox(bsolid)
        
        view.zoomTo()
        view.setBackgroundColor("#0c0f1d")
        view.enableFog(True)
        
        st.markdown(
            """<style>
            iframe[title="stmol.showmol"] {
                display: block; margin: 0 auto !important; width: 100% !important; max-width: 900px;
                border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            }
            </style>""", 
            unsafe_allow_html=True
        )
        showmol(view, height=500, width=900)
    except Exception as e:
        st.error(f"Box preview error: {e}")


# =============================================================================
# ── PAGE 1: UPLOAD
# =============================================================================
def page_upload():
    render_stepper()
    st.markdown('<div class="page-title">Upload Files</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Add your receptor and ligands to get started. All fields below are required before running docking.</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        rec_label_col, demo_col = st.columns([3, 1.5])
        with rec_label_col:
            st.markdown("**Receptor(s)** — protein PDB file(s)")
        with demo_col:
            # ── Demo Mode ──
            demo_dir = os.path.join(os.path.dirname(__file__), "demo_data")
            demo_rec = os.path.join(demo_dir, "COVID_Mpro_6LU7.pdb")
            demo_lig = os.path.join(demo_dir, "test_drugs.smi")
            if os.path.exists(demo_rec) and os.path.exists(demo_lig):
                if st.button("🧪 Demo", key="demo_btn", use_container_width=True,
                             help="Load sample data: COVID-19 Main Protease (PDB: 6LU7) + 3 drugs (Aspirin, Ibuprofen, Paracetamol). Great for testing the pipeline without your own files."):
                    rec_bytes = open(demo_rec, "rb").read()
                    lig_bytes = open(demo_lig, "rb").read()
                    st.session_state.rec_files_data = [("COVID_Mpro_6LU7.pdb", rec_bytes)]
                    st.session_state.rec_bytes = rec_bytes
                    st.session_state.lig_files_data = [("test_drugs.smi", lig_bytes)]
                    st.rerun()
        rec_files = st.file_uploader("Receptor PDB(s)", type=["pdb"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed", key="rec_up")
        if rec_files:
            st.session_state.rec_files_data = [(f.name, f.read()) for f in rec_files]
            for f in rec_files: f.seek(0)
            # Set rec_bytes to first receptor for autobox compatibility
            st.session_state.rec_bytes = st.session_state.rec_files_data[0][1]
            st.success(f"Loaded {len(rec_files)} receptor(s)")
        elif getattr(st.session_state, "rec_files_data", None):
            for rname, _ in st.session_state.rec_files_data:
                st.success(f"Loaded: {rname}")
            if st.button("❌ Clear files", key="clear_rec"):
                st.session_state.rec_files_data = None
                st.session_state.rec_bytes = None
                st.session_state.lig_files_data = []
                st.session_state.boxes = {}
                st.session_state.results = []
                st.rerun()

        st.markdown("")
        st.markdown("**Reference Ligand** *(optional)*")
        st.caption("If provided, the docking box will be centred on this ligand's position rather than the full receptor.")
        ref_file = st.file_uploader("Reference ligand SDF", type=["sdf"],
                                     label_visibility="collapsed", key="ref_up")

    with col_r:
        st.markdown("**Ligands** — SDF, SMILES, or ZIP file(s)")
        lig_files = st.file_uploader("Ligand files", type=["sdf","smi","smiles","txt","zip"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed", key="lig_up")
        if lig_files:
            st.session_state.lig_files_data = [(f.name, f.read()) for f in lig_files]
            for f in lig_files: f.seek(0)
            st.success(f"{len(lig_files)} file(s) loaded")
        elif getattr(st.session_state, "lig_files_data", None):
            for lname, _ in st.session_state.lig_files_data:
                st.success(f"Loaded: {lname}")
            if st.button("❌ Clear ligands", key="clear_lig"):
                st.session_state.lig_files_data = []
                st.session_state.admet_data = {}
                st.session_state.results = []
                st.rerun()
            
            st.markdown("")
            if st.checkbox("👁️ Preview Ligands (2D Structure)"):
                try:
                    from rdkit.Chem import Draw
                    class F:
                        def __init__(self,n,b): self.name=n; self.b=b
                        def read(self): return self.b
                    prev_mols = load_ligands([F(n,b) for n,b in st.session_state.lig_files_data[:3]])[:12]
                    if prev_mols:
                        ms = [m for n,m in prev_mols]
                        ns = [n[:15] for n,m in prev_mols]
                        img = Draw.MolsToGridImage(ms, molsPerRow=4, subImgSize=(150, 150), legends=ns)
                        st.image(img, width=460)
                except Exception as e:
                    st.warning(f"Could not generate 2D previews: {e}")

    st.divider()
    st.markdown("**Docking Box**")
    st.caption("The box defines the 3D search space for docking. Click below to calculate it automatically from your receptor, or set it manually in Settings.")

    col_btn, col_pbtn, col_multi, col_pad = st.columns([1,1,1,1])
    with col_btn:
        if st.button("Compute Autobox", use_container_width=True):
            if not getattr(st.session_state, "rec_files_data", None):
                st.error("Upload at least one receptor PDB first.")
            else:
                all_boxes = {}
                for rname, rbytes in st.session_state.rec_files_data:
                    with tempfile.TemporaryDirectory() as td:
                        rp = os.path.join(td, rname)
                        Path(rp).write_bytes(rbytes)
                        rfp = None
                        if ref_file:
                            rfp = os.path.join(td, "ref.sdf")
                            Path(rfp).write_bytes(ref_file.read())
                            ref_file.seek(0)
                        try:
                            box = compute_autobox(rp, rfp, st.session_state.box_padding)
                            box["name"] = "Autobox"
                            all_boxes[rname] = [box]
                        except Exception as e:
                            st.error(f"Autoboxing failed for {rname}: {e}")
                if all_boxes:
                    st.session_state.boxes = all_boxes
                    st.rerun()
                        
    with col_multi:
        st.session_state.multi_pocket = st.toggle("Multi-Pocket (Top 5)", value=st.session_state.get("multi_pocket", False))

    with col_pbtn:
        if st.button("Suggest Pocket (fpocket)", use_container_width=True, help="Automatically detects the best binding cavity"):
            if not getattr(st.session_state, "rec_files_data", None):
                st.error("Upload at least one receptor PDB first.")
            else:
                import subprocess
                all_boxes = {}
                max_pockets = 5 if st.session_state.multi_pocket else 1
                for rname, rbytes in st.session_state.rec_files_data:
                    with tempfile.TemporaryDirectory() as td:
                        rp = os.path.join(td, rname)
                        Path(rp).write_bytes(rbytes)
                        try:
                            subprocess.run(["fpocket", "-f", rp], cwd=td, capture_output=True, check=True)
                            found_boxes = []
                            stem = Path(rname).stem
                            for i in range(1, max_pockets + 1):
                                p_file = Path(td) / f"{stem}_out" / "pockets" / f"pocket{i}_atm.pdb"
                                if p_file.exists():
                                    bx = compute_autobox(str(p_file), None, st.session_state.box_padding)
                                    bx["name"] = f"Pocket {i}"
                                    found_boxes.append(bx)
                            if found_boxes:
                                all_boxes[rname] = found_boxes
                            else:
                                st.warning(f"No pockets detected for {rname}.")
                        except Exception as e:
                            st.error(f"fpocket failed for {rname}: {e}")
                if all_boxes:
                    st.session_state.boxes = all_boxes
                    st.rerun()

    if st.session_state.boxes:
        for rname, rboxes in st.session_state.boxes.items():
            st.markdown(f"**{rname}** — {len(rboxes)} pocket(s)")
            if len(rboxes) <= 2:
                for b in rboxes:
                    st.markdown(f"*{b['name']}*")
                    st.markdown(
                        '<div class="box-grid" style="margin-bottom: 0.6rem;">'
                        + "".join(f'<div class="box-cell"><div class="box-val">{b[k]:.1f}</div>'
                                   f'<div class="box-lbl">{l}</div></div>'
                                   for k,l in [("cx","Center X"),("cy","Center Y"),("cz","Center Z"),
                                               ("sx","Size X"),("sy","Size Y"),("sz","Size Z")])
                        + "</div>",
                        unsafe_allow_html=True
                    )
            else:
                pocket_rows = ""
                for b in rboxes:
                    pocket_rows += f'''<tr>
                        <td style="font-weight:600;color:#EAEAEA;">{b["name"]}</td>
                        <td>{b["cx"]:.1f}, {b["cy"]:.1f}, {b["cz"]:.1f}</td>
                        <td>{b["sx"]:.0f} × {b["sy"]:.0f} × {b["sz"]:.0f}</td>
                    </tr>'''
                st.markdown(f'''
                <table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-top:0.5rem;">
                    <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
                        <th style="text-align:left;padding:6px 10px;color:#888;text-transform:uppercase;font-size:0.7rem;letter-spacing:0.06em;">Pocket</th>
                        <th style="text-align:left;padding:6px 10px;color:#888;text-transform:uppercase;font-size:0.7rem;letter-spacing:0.06em;">Center (X, Y, Z)</th>
                        <th style="text-align:left;padding:6px 10px;color:#888;text-transform:uppercase;font-size:0.7rem;letter-spacing:0.06em;">Size (Å)</th>
                    </tr></thead>
                    <tbody style="color:#D2D2D2;">{pocket_rows}</tbody>
                </table>
                ''', unsafe_allow_html=True)
        
        if getattr(st.session_state, "rec_files_data", None):
            show_3d = st.checkbox("Show 3D Box Preview", value=False, key="show_box_preview")
            if show_3d:
                if len(st.session_state.rec_files_data) > 1:
                    rec_names = [rd[0] for rd in st.session_state.rec_files_data]
                    sel_rec_name = st.selectbox("Receptor for preview", rec_names, key="preview_rec")
                    sel_rec_bytes = next(b for n, b in st.session_state.rec_files_data if n == sel_rec_name)
                else:
                    sel_rec_name = st.session_state.rec_files_data[0][0]
                    sel_rec_bytes = st.session_state.rec_files_data[0][1]
                render_box_preview(sel_rec_bytes, st.session_state.boxes.get(sel_rec_name, []))

    st.divider()
    _s1, c1, c2, c3, _s2 = st.columns([1, 1, 1, 1, 1])
    c1.metric("Receptor",  "Ready" if st.session_state.rec_bytes else "Optional")
    c2.metric("Ligands",   len(st.session_state.lig_files_data) or "Missing")
    _total_boxes = sum(len(v) for v in st.session_state.boxes.values()) if st.session_state.boxes else 0
    c3.metric("Box",       f"{_total_boxes} ready" if _total_boxes else "Not set")

    st.markdown("")
    ready = st.session_state.lig_files_data and st.session_state.boxes
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Next: Settings →" if ready else "Upload ligands & set box", 
                     key="nav_next_upload", use_container_width=True,
                     type="primary" if ready else "secondary",
                     disabled=not ready):
            st.session_state.page = "Settings"; st.rerun()


# =============================================================================
# ── PAGE 2: SETTINGS
# =============================================================================
def page_settings():
    render_stepper()
    st.markdown('<div class="page-title">Settings &amp; Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Configure the docking engine and preparation tools. Hover over any label for a tooltip. All defaults work well for most cases.</div>', unsafe_allow_html=True)

    # ── Engine cards ───────────────────────────────────────────────────────
    st.markdown("**Docking Engine**")
    st.caption("Choose the engine. AutoDock Vina works on every system. GNINA requires an NVIDIA GPU.")

    gpu_warn = gpu["mode"] != "nvidia"
    vina_selected  = st.session_state.engine == "AutoDock Vina"
    gnina_selected = not vina_selected

    eng_col1, eng_col2, _spc = st.columns([1, 1, 2])
    with eng_col1:
        v_border = "border: 2px solid #3fb950; box-shadow: 0 0 18px rgba(63,185,80,0.3);" if vina_selected else ""
        st.markdown(f"""
        <div class="eng-card" style="{v_border}">
            <div class="eng-title">AutoDock Vina <span class="eng-badge cpu">CPU</span></div>
            <div class="eng-sub">Fast, universal. Works on all CPU systems.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="sel_vina", use_container_width=True,
                     type="primary" if vina_selected else "secondary"):
            st.session_state.engine = "AutoDock Vina"; st.rerun()

    with eng_col2:
        g_border = "border: 2px solid #FF4B4B; box-shadow: 0 0 18px rgba(255,75,75,0.3);" if gnina_selected else ""
        g_sub = "AI CNN scoring. NVIDIA GPU not detected (CPU mode)." if gpu_warn else "AI CNN scoring. NVIDIA GPU detected."
        st.markdown(f"""
        <div class="eng-card" style="{g_border}">
            <div class="eng-title">GNINA <span class="eng-badge gpu">GPU / CPU</span></div>
            <div class="eng-sub">{g_sub}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="sel_gnina", use_container_width=True,
                     type="primary" if gnina_selected else "secondary"):
            st.session_state.engine = "GNINA"; st.rerun()

    engine = st.session_state.engine
    # ── 4-Tab Settings Layout ──────────────────────────────────────────────────
    tab_eng, tab_lig, tab_prot, tab_box = st.tabs([
        "⚙️ Engine", "🧪 Ligand Prep", "🧬 Protein Prep", "📦 Box & Parallel"
    ])

    with tab_eng:
        st.caption("Choose the docking engine. AutoDock Vina works on every system. GNINA requires an NVIDIA GPU for best performance.")
        if engine == "AutoDock Vina":
            st.markdown("#### AutoDock Vina settings")
            st.session_state.v_exhaustiveness = st.slider(
                "Exhaustiveness", 1, 32, st.session_state.v_exhaustiveness,
                help="How thoroughly Vina searches the binding site. Higher = better results, slower.")
            st.session_state.v_num_poses = st.slider(
                "Number of poses", 1, 20, st.session_state.v_num_poses,
                help="How many docked poses to return per ligand.")
            st.session_state.energy_range = st.slider(
                "Energy range (kcal/mol)", 1, 10, st.session_state.energy_range,
                help="Only output poses within this energy difference of the best pose.")
            st.session_state.scoring_fn = st.selectbox(
                "Scoring function", ["vina","vinardo","ad4"],
                index=["vina","vinardo","ad4"].index(st.session_state.scoring_fn),
                help="vina: default. vinardo: better for macrocycles. ad4: AutoDock 4 empirical scoring.")
            st.session_state.vina_seed = st.number_input(
                "Random seed", 0, 99999, st.session_state.vina_seed,
                help="Set to a non-zero value for reproducible results.")
            if st.checkbox("ℹ️ Learn more about Vina parameters"):
                st.info("""
                **Exhaustiveness:** The single most important parameter — controls how many independent physics simulations run inside the box.
                - `8`: Fast, great for screening. `32`: Slow but highly accurate for publications.

                **Energy Range:** Only shows poses within X kcal/mol of the best pose. Increase to see radically different binding angles.
                """, icon="⚡")
        else:
            st.markdown("#### GNINA settings")
            st.session_state.exhaustiveness = st.slider(
                "Exhaustiveness", 1, 64, st.session_state.exhaustiveness,
                help="Search thoroughness.")
            st.session_state.num_poses = st.slider(
                "Poses", 1, 20, st.session_state.num_poses)
            st.session_state.cnn_model = st.selectbox(
                "CNN model", ["default","dense","crossdocked_default2018"],
                index=["default","dense","crossdocked_default2018"].index(st.session_state.cnn_model),
                help="Neural network model for CNN scoring.")
            cnn_modes_available = ["none", "rescore", "refinement", "all"]
            if gpu_warn:
                cnn_modes_available = ["none", "rescore"]
                if st.session_state.cnn_mode in ("refinement", "all"):
                    st.session_state.cnn_mode = "rescore"
                st.info("⚠️ GPU not detected. CNN Refinement & All modes disabled.", icon="🖥️")
            st.session_state.cnn_mode = st.selectbox(
                "CNN Mode", cnn_modes_available,
                index=cnn_modes_available.index(st.session_state.cnn_mode),
                help="none: skip AI. rescore: fast Vina then AI grading. refinement: AI guides minimization (10x slower, GPU required).")
            st.session_state.cnn_weight = st.slider(
                "CNN scoring weight", 0.0, 1.0, st.session_state.cnn_weight, 0.05,
                help="1.0 = pure CNN, 0.0 = pure Vina score.")
            st.session_state.min_rmsd = st.slider(
                "Min RMSD between poses (Å)", 0.5, 3.0, st.session_state.min_rmsd, 0.1)
            c1g, c2g = st.columns(2)
            st.session_state.gnina_seed = c1g.number_input("Seed", 0, 99999, st.session_state.gnina_seed)
            st.session_state.gpu_device = c2g.selectbox(
                "GPU device", ["auto","0","1","cpu"],
                index=["auto","0","1","cpu"].index(st.session_state.gpu_device))
            st.session_state.flexdist = st.toggle(
                "Flexible Docking (Induced Fit)", st.session_state.flexdist,
                help="Allows receptor side-chains within 3.5 Å of the pocket to flex.")
            if st.session_state.flexdist:
                st.session_state.max_flex_res = st.slider(
                    "Max flexible residues", 1, 5, st.session_state.get("max_flex_res", 4),
                    help="Hard cap — more than 4 causes exponentially long compute times.")
                st.warning("⚡ Each additional residue multiplies docking time ~3x. Keep ≤4.", icon="⏳")
            if st.checkbox("ℹ️ Learn more about GNINA parameters"):
                st.info("""
                **CNN Weight:** 0.0 = pure physics, 1.0 = pure AI. Values around 0.5 balance both.

                **CNN Mode:** `rescore` is the default (fast). `refinement` actively reshapes poses using AI (>10x slower, GPU required).

                **Flexible Docking:** Protein side-chains physically bend to accommodate the ligand. Vina scores may look worse due to deformation penalties — use CNN Probability instead.
                """, icon="🤖")

    with tab_lig:
        st.caption("Controls how ligand files are converted into 3D structures ready for docking.")
        c_a, c_b = st.columns(2, gap="large")
        with c_a:
            st.markdown("**3D Generation (RDKit)**")
            st.session_state.target_ph = st.number_input(
                "Target pH", min_value=0.0, max_value=14.0,
                value=st.session_state.get("target_ph", 7.4), step=0.1,
                help="If input is 1D SMILES, OpenBabel protonates the molecule at this pH before 3D embedding.")
            st.session_state.embed_method = st.selectbox(
                "Embedding method", ["ETKDGv3","ETKDGv2","ETKDG"],
                index=["ETKDGv3","ETKDGv2","ETKDG"].index(st.session_state.embed_method),
                help="Algorithm for generating the 3D shape. ETKDGv3 is the most accurate.")
            st.session_state.max_attempts = st.slider(
                "Max embedding attempts", 10, 5000, st.session_state.max_attempts, step=50,
                help="How many times RDKit will retry if 3D generation fails.")
            st.session_state.force_field = st.selectbox(
                "Force field", ["MMFF94","MMFF94s","UFF"],
                index=["MMFF94","MMFF94s","UFF"].index(st.session_state.force_field),
                help="Energy function used to optimise the 3D geometry.")
            st.session_state.ff_steps = st.slider(
                "Force field steps", 0, 2000, st.session_state.ff_steps, step=100,
                help="Geometry optimisation steps. More steps = more accurate but slower.")
            st.session_state.add_hs = st.toggle("Add hydrogens", st.session_state.add_hs,
                help="Adds explicit hydrogen atoms before 3D embedding.")
            st.session_state.keep_chirality = st.toggle("Preserve chirality", st.session_state.keep_chirality,
                help="Maintains stereocenters (R/S) from the input molecule.")
        with c_b:
            st.markdown("**PDBQT Conversion (Meeko)**")
            st.session_state.merge_nphs = st.toggle(
                "Merge non-polar H", st.session_state.merge_nphs,
                help="Absorbs non-polar hydrogens into their parent heavy atom. Standard for AD4 scoring.")
            st.session_state.hydrate = st.toggle(
                "Add hydration waters", st.session_state.hydrate,
                help="Adds explicit water molecules around the ligand for hydrated docking.")
            st.markdown("")
            if st.checkbox("ℹ️ Learn more", key="learn_more_lig"):
                st.info("""
                **ETKDGv3:** Latest RDKit 3D conformer algorithm — uses experimental torsion knowledge for drug-like molecules.

                **MMFF94:** Standard force field for organic molecules. Minimises internal strain before docking.

                **Merge non-polar H:** Required for AutoDock PDBQT format. Non-polar H atoms are absorbed into their parent carbons.
                """, icon="🧠")

    with tab_prot:
        st.caption("Automated receptor cleaning and repair pipeline. Runs automatically before docking on every uploaded receptor.")
        c_p1, c_p2 = st.columns(2, gap="large")
        with c_p1:
            st.markdown("**Cleaning**")
            st.session_state.prep_remove_water = st.toggle(
                "Remove water (HOH)", st.session_state.prep_remove_water,
                help="Strip all crystallographic water molecules from the PDB file.")
            st.session_state.prep_remove_het = st.toggle(
                "Remove heteroatoms", st.session_state.prep_remove_het,
                help="Remove co-crystallized ligands, buffer molecules, and non-protein atoms.")
            if st.session_state.prep_remove_het:
                st.session_state.prep_keep_metals = st.toggle(
                    "  ↳ Keep metal ions", st.session_state.prep_keep_metals,
                    help="Preserve metal ions (Zn²⁺, Mg²⁺, Ca²⁺) that may be important for catalysis.")
            st.markdown("**Repair**")
            st.session_state.prep_add_h = st.toggle(
                "Add hydrogens at pH", st.session_state.prep_add_h,
                help="Add missing hydrogen atoms at the target pH using PDBFixer.")
            st.session_state.prep_fix_atoms = st.toggle(
                "Fix missing atoms", st.session_state.prep_fix_atoms,
                help="Detect and rebuild missing heavy atoms in incomplete residues.")
            st.session_state.prep_fix_loops = st.toggle(
                "Model missing loops", st.session_state.prep_fix_loops,
                help="Identify and model missing residues/loops in the crystal structure.")
        with c_p2:
            st.markdown("**Advanced** *(slower)*")
            st.session_state.prep_propka = st.toggle(
                "propka protonation states", st.session_state.prep_propka,
                help="Run propka3 to predict pKa values for titratable residues (His, Glu, Asp, Lys, Cys).")
            st.session_state.prep_minimize = st.toggle(
                "Energy minimization (OpenMM)", st.session_state.prep_minimize,
                help="Quick Amber14 energy minimization to relax steric clashes in the receptor.")
            if st.session_state.prep_minimize:
                st.session_state.prep_min_steps = st.slider(
                    "Minimization steps", 50, 1000, st.session_state.prep_min_steps, step=50,
                    help="L-BFGS iterations. More steps = better but slower.")
            st.markdown("")
            if st.checkbox("ℹ️ Learn more", key="learn_more_prot"):
                st.info("""
                **Water/Heteroatom removal:** Crystal structures contain buffer salts, co-crystallized inhibitors, and ordered water — almost always removed before docking.

                **PDBFixer repairs:** Missing atoms and loops from unresolved crystal regions are modelled automatically.

                **propka:** Predicts protonation states of His, Glu, Asp residues at your target pH.

                **Minimization:** Amber14 relaxation resolves steric clashes from crystal packing or PDBFixer repairs.
                """, icon="🔬")

    with tab_box:
        c_bx, c_par = st.columns(2, gap="large")
        with c_bx:
            st.markdown("**Docking Box**")
            st.caption("The 3D search region. Set on the Upload page via Autobox or fpocket.")
            st.session_state.box_padding = st.slider(
                "Autobox padding (Å)", 1, 20, st.session_state.box_padding,
                help="Extra space added around the receptor when computing the box.")
            manual = st.toggle("Manual box override", False)
            if manual:
                cc, cs = st.columns(2)
                cbox = None
                if st.session_state.boxes:
                    first_boxes = list(st.session_state.boxes.values())[0]
                    cbox = first_boxes[0] if first_boxes else None
                cx = cc.number_input("Center X", value=cbox["cx"] if cbox else 0.0)
                cy = cc.number_input("Center Y", value=cbox["cy"] if cbox else 0.0)
                cz = cc.number_input("Center Z", value=cbox["cz"] if cbox else 0.0)
                sx = cs.number_input("Size X", value=cbox["sx"] if cbox else 20.0)
                sy = cs.number_input("Size Y", value=cbox["sy"] if cbox else 20.0)
                sz = cs.number_input("Size Z", value=cbox["sz"] if cbox else 20.0)
                manual_box = {"name": "Manual", "cx":cx,"cy":cy,"cz":cz,"sx":sx,"sy":sy,"sz":sz}
                if getattr(st.session_state, "rec_files_data", None):
                    st.session_state.boxes = {rn: [manual_box.copy()] for rn, _ in st.session_state.rec_files_data}
                else:
                    st.session_state.boxes = {"manual": [manual_box]}
        with c_par:
            st.markdown("**Parallelization**")
            st.caption("Run multiple docking jobs simultaneously.")
            max_cpu = os.cpu_count() or 4
            st.session_state.n_jobs = st.slider(
                "CPU cores", 1, max_cpu, st.session_state.n_jobs, key="njobs_slider",
                help="Number of ligands to dock in parallel. Set to 1 for live progress updates.")
            st.session_state.backend = st.selectbox(
                "Parallel backend", ["loky","threading","multiprocessing"],
                index=["loky","threading","multiprocessing"].index(st.session_state.backend),
                help="loky is stable. threading is faster for I/O-bound tasks.")


    st.divider()
    if st.button("Next: Run Docking →"):
        st.session_state.page = "Run"; st.rerun()

def build_pdf_report(df, engine, cfg=None):
    from fpdf import FPDF
    from datetime import datetime
    
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 22)
            self.set_text_color(30, 30, 30)
            self.cell(0, 15, "StrataDock Screening Report", ln=True, align="C")
            self.set_font("helvetica", "", 10)
            self.set_text_color(120, 120, 120)
            self.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            self.ln(3)
            # Thin line
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), self.w - 15, self.get_y())
            self.ln(5)

    pdf = PDF()
    pdf.add_page()

    # ── Run Configuration Section ──
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Run Configuration", ln=True)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)

    config_items = [("Engine", engine)]
    if cfg:
        if engine == "GNINA":
            cnn_mode = cfg.get("cnn_mode", "none")
            config_items.append(("CNN Mode", cnn_mode.capitalize()))
            config_items.append(("CNN Weight", f"{cfg.get('cnn_weight', 1.0):.2f}"))
            config_items.append(("CNN Model", cfg.get("cnn_model", "default")))
            if cfg.get("flexdist"):
                config_items.append(("Flexible Docking", f"Enabled (max {cfg.get('max_flex_res', 4)} residues)"))
            else:
                config_items.append(("Flexible Docking", "Disabled"))
            config_items.append(("Exhaustiveness", str(cfg.get("exhaustiveness", 16))))
            config_items.append(("Poses", str(cfg.get("num_poses", 9))))
        else:
            config_items.append(("Scoring Function", cfg.get("scoring_fn", "vina")))
            config_items.append(("Exhaustiveness", str(cfg.get("v_exhaustiveness", 8))))
            config_items.append(("Poses", str(cfg.get("v_num_poses", 9))))
            config_items.append(("Energy Range", f"{cfg.get('energy_range', 3)} kcal/mol"))
        _boxes = st.session_state.get("boxes", {})
        _npock = sum(len(v) for v in _boxes.values()) if isinstance(_boxes, dict) else len(_boxes)
        config_items.append(("Pockets", str(_npock)))
        config_items.append(("Force Field", cfg.get("force_field", "MMFF94")))

    for label, value in config_items:
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(45, 6, f"{label}:")
        pdf.set_font("helvetica", "", 9)
        pdf.cell(0, 6, str(value), ln=True)

    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(5)

    # ── Results Table ──
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Top Docking Hits", ln=True)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(50, 50, 50)
    col_w = [25, 55, 22, 25, 28, 30]
    headers = ["Receptor", "Ligand", "Pocket", "Vina Score", "CNN Score" if engine=="GNINA" else "Status", "CNN Mode" if engine=="GNINA" else ""]
    for i, h in enumerate(headers):
        if h:
            pdf.cell(col_w[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()
    
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    # Sort and take top 50
    df_top = df[df["Status"] == "success"].copy()
    if not df_top.empty:
        sort_col = "CNN Pose Probability (0-1)" if engine=="GNINA" else "Vina Forcefield (kcal/mol)"
        ascend = engine != "GNINA"
        df_top = df_top.sort_values(sort_col, ascending=ascend).head(50)
        
        for idx, (_, row) in enumerate(df_top.iterrows()):
            # Alternate row fill
            fill = idx % 2 == 0
            if fill:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w[0], 7, str(row["Receptor"])[:12], border=1, fill=fill)
            pdf.cell(col_w[1], 7, str(row["Ligand"])[:27], border=1, fill=fill)
            pdf.cell(col_w[2], 7, str(row.get("Pocket", ""))[:10], border=1, fill=fill)
            vina_val = f"{row['Vina Forcefield (kcal/mol)']:.2f}" if pd.notna(row['Vina Forcefield (kcal/mol)']) else "-"
            pdf.cell(col_w[3], 7, vina_val, border=1, align="C", fill=fill)
            
            if engine == "GNINA":
                cnn_val = f"{row['CNN Pose Probability (0-1)']:.3f}" if pd.notna(row['CNN Pose Probability (0-1)']) else "-"
                pdf.cell(col_w[4], 7, cnn_val, border=1, align="C", fill=fill)
                cnn_m = cfg.get("cnn_mode", "none").capitalize() if cfg else "-"
                pdf.cell(col_w[5], 7, cnn_m, border=1, align="C", fill=fill)
            else:
                pdf.cell(col_w[4], 7, str(row["Status"]), border=1, align="C", fill=fill)
            pdf.ln()
    
    # Footer note
    pdf.ln(8)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, "Report generated by StrataDock Alpha v0.7 | Prithvi Rajan, Freie Universitat Berlin", ln=True, align="C")
    if engine == "GNINA" and cfg and cfg.get("flexdist"):
        pdf.cell(0, 5, "Note: Flexible docking was enabled. Vina scores may appear less negative due to protein deformation penalties.", ln=True, align="C")
            
    return bytes(pdf.output())


def build_zip_archive(results, df, engine, session_state, cfg=None):
    import zipfile, io
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        # 1. Add CSV
        zf.writestr("stratadock_results.csv", df.to_csv(index=False))
        
        # 2. Add PDF Report
        try: zf.writestr("stratadock_report.pdf", build_pdf_report(df, engine, cfg=cfg))
        except: pass
        
        # 3. Add Inputs (Receptors)
        pml_recs = []
        if getattr(session_state, "rec_files_data", None):
            for rname, rbytes in session_state.rec_files_data:
                zf.writestr(f"inputs/{rname}", rbytes)
                pml_recs.append(rname)
        
        # 4. Add Outputs — grouped by receptor
        pml_poses = []
        ok_res = [r for r in results if r["status"] == "success" and r.get("pose_sdf") and Path(r["pose_sdf"]).exists()]
        for r in ok_res:
            rec_folder = r.get("Receptor", "unknown").replace(".pdb", "")
            safe_lname = r["name"]
            
            # Pose SDF
            sdf_name = f"outputs/{rec_folder}/{safe_lname}_pose.sdf"
            zf.writestr(sdf_name, Path(r["pose_sdf"]).read_bytes())
            pml_poses.append((rec_folder, f"{safe_lname}_pose.sdf"))
            
            # Complex PDB
            cpdb = r.get("complex_pdb")
            if cpdb and Path(cpdb).exists():
                zf.writestr(f"outputs/{rec_folder}/{safe_lname}_complex.pdb", Path(cpdb).read_bytes())
            
            # Binding site interactions CSV
            idf = r.get("interactions")
            if isinstance(idf, pd.DataFrame) and not idf.empty:
                zf.writestr(f"outputs/{rec_folder}/{safe_lname}_interactions.csv", idf.to_csv(index=False))
        
        # 5. Add terminal log
        term_log = session_state.get("terminal_log", "")
        if term_log:
            zf.writestr("run_log.txt", term_log)
        
        # 6. Generate summary.txt
        from datetime import datetime
        summary_lines = [
            "STRATADOCK EXPERIMENT SUMMARY",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Engine: {engine}",
            f"Total jobs: {len(results)}",
            f"Successful: {len(ok_res)}",
            f"Failed: {len(results) - len(ok_res)}",
            "",
        ]
        # Group results by receptor
        from collections import defaultdict
        by_rec = defaultdict(list)
        for r in results:
            by_rec[r.get('Receptor', 'unknown')].append(r)
        
        for rec_name, rec_results in by_rec.items():
            summary_lines.append("=" * 60)
            summary_lines.append(f"RECEPTOR: {rec_name}")
            summary_lines.append("=" * 60)
            ok_r = [r for r in rec_results if r['status'] == 'success']
            summary_lines.append(f"  Successful docks: {len(ok_r)}/{len(rec_results)}")
            summary_lines.append("")
            
            for r in rec_results:
                summary_lines.append(f"  --- {r.get('Ligand', r.get('name', '?'))} [{r.get('Pocket', '?')}] ---")
                summary_lines.append(f"    Status: {r['status']}")
                if r['status'] == 'success':
                    if r.get('vina_score') is not None:
                        summary_lines.append(f"    Vina Score: {r['vina_score']:.2f} kcal/mol")
                    if r.get('cnn_score') is not None:
                        summary_lines.append(f"    CNN Pose Prob: {r['cnn_score']:.4f}")
                    if r.get('cnn_affinity') is not None:
                        summary_lines.append(f"    CNN Affinity: {r['cnn_affinity']:.3f} pKd")
                    # Interaction summary
                    idf = r.get('interactions')
                    if isinstance(idf, pd.DataFrame) and not idf.empty:
                        summary_lines.append(f"    Interacting residues: {len(idf)}")
                        bond_counts = idf['Bond Type'].value_counts()
                        for btype, cnt in bond_counts.items():
                            summary_lines.append(f"      {btype}: {cnt}")
                        summary_lines.append("    Key residues:")
                        for _, row in idf.head(10).iterrows():
                            summary_lines.append(f"      {row['Residue']} {row['Chain']}{row['ResID']} "
                                                 f"({row['Receptor Atom']}) — {row['Distance (Å)']:.2f} Å [{row['Bond Type']}]")
                summary_lines.append("")
        
        summary_lines.append("=" * 60)
        summary_lines.append("End of Report")
        zf.writestr("summary.txt", "\n".join(summary_lines))
            
        # 6. Generate PyMOL Script
        if pml_recs and pml_poses:
            pml = [
                "# StrataDock Auto-Generated PyMOL Visualization Script",
                "bg_color white",
                "set depth_cue, 1",
                "set ray_trace_fog, 1",
                "set antialias, 2"
            ]
            for r in pml_recs:
                obj_name = r.replace(".pdb","")
                pml.append(f"load inputs/{r}, {obj_name}")
                pml.append(f"hide all, {obj_name}")
                pml.append(f"show cartoon, {obj_name}")
                pml.append(f"color gray80, {obj_name}")
                pml.append(f"set cartoon_transparency, 0.2, {obj_name}")
                pml.append(f"show surface, {obj_name}")
                pml.append(f"set transparency, 0.6, {obj_name}")
            
            for rec_folder, pose in pml_poses:
                obj_name = pose.replace(".sdf","")
                pml.append(f"load outputs/{rec_folder}/{pose}, {obj_name}")
                pml.append(f"show sticks, {obj_name}")
                pml.append(f"util.cbas {obj_name}")
            
            pml.append("center all")
            pml.append("zoom all, 5")
            pml.append("echo 'StrataDock Project Loaded Successfully!'")
            
            zf.writestr("visualize.pml", "\\n".join(pml))

    return zip_buffer.getvalue()


# =============================================================================
# ── PAGE 3: RUN & RESULTS
# =============================================================================
def page_run():
    render_stepper()
    st.markdown('<div class="page-title">Run &amp; Results</div>', unsafe_allow_html=True)

    ready = (getattr(st.session_state, "rec_files_data", None)
             and st.session_state.lig_files_data
             and st.session_state.boxes)

    if not ready:
        st.warning("Complete the **Upload** page first — receptor, ligands, and box must all be set.")
        if st.button("Go to Upload"): st.session_state.page = "Upload"; st.rerun()
        return

    cfg = {k: st.session_state[k] for k in [
        "engine","embed_method","max_attempts","force_field","ff_steps", "target_ph",
        "add_hs","keep_chirality","merge_nphs","hydrate",
        "exhaustiveness","num_poses","cnn_model","cnn_weight","min_rmsd","gnina_seed","gpu_device", "flexdist", "cnn_mode",
        "v_exhaustiveness","v_num_poses","energy_range","scoring_fn","vina_seed",
        "n_jobs","backend"
    ]}

    # ── Guardrail 3: Compute Cost Estimator ──
    n_ligs = len(st.session_state.lig_files_data)
    n_pockets = sum(len(v) for v in st.session_state.boxes.values()) if st.session_state.boxes else 0
    exh = st.session_state.v_exhaustiveness if cfg["engine"] == "AutoDock Vina" else st.session_state.exhaustiveness
    total_jobs = n_ligs * n_pockets
    est_seconds = total_jobs * (15 if cfg["engine"] == "AutoDock Vina" else 25)
    if cfg.get("cnn_mode") in ("refinement", "all"):
        est_seconds *= 3
    if cfg.get("flexdist"):
        est_seconds *= 4
    est_min = est_seconds / 60
    est_hrs = est_seconds / 3600

    c1, c2, c3_est, c4, c5 = st.columns([1.2, 1.2, 1.2, 1, 1])
    c1.metric("Engine", cfg["engine"])
    c2.metric("Ligand files", n_ligs)
    c3_est.metric("Est. Time", f"~{est_min:.0f} min" if est_min < 120 else f"~{est_hrs:.1f} hrs")

    massive_run = est_seconds > 7200  # > 2 hours
    run_allowed = True
    if massive_run:
        st.warning(f"⚠️ This run is estimated to take **~{est_hrs:.1f} hours** ({total_jobs} jobs × {n_pockets} pockets). Consider reducing ligands, pockets, or exhaustiveness.", icon="🕐")
        run_allowed = st.checkbox("I understand this is a massive run and may take several hours", key="massive_ack")

    with c4:
        go = st.button("Run Docking", use_container_width=True, type="primary", disabled=(massive_run and not run_allowed))
    with c5:
        st.button("Stop / Cancel", use_container_width=True, on_click=lambda: st.session_state.update(cancel_run=True))

    if go:
        work_dir = tempfile.mkdtemp(prefix="stratadock_")
        st.session_state.work_dir = work_dir
        
        status = st.empty()
        rec_items, failed_recs = [], []
        for i, (rname, rbytes) in enumerate(st.session_state.rec_files_data):
            status.caption(f"Preparing receptor {rname} ({i+1}/{len(st.session_state.rec_files_data)})…")
            rp = os.path.join(work_dir, rname)
            Path(rp).write_bytes(rbytes)
            
            # Run protein preparation pipeline (steps 1-7)
            prep_cfg = {k: v for k, v in st.session_state.items() if k.startswith("prep_") or k == "target_ph"}
            try:
                prepared_pdb, prep_log = prepare_protein(rp, prep_cfg)
                for msg in prep_log:
                    status.caption(f"  ↳ {rname}: {msg}")
                rp = prepared_pdb  # Use prepared PDB for PDBQT conversion
            except Exception as e:
                status.caption(f"  ↳ Protein prep failed ({e}), using raw PDB")
            
            try:
                rpdbqt = prepare_receptor(rp, work_dir)
                rec_items.append((rname, rpdbqt, rp))
            except Exception as e:
                import traceback
                failed_recs.append((rname, traceback.format_exc()))
                
        if failed_recs:
            st.warning(f"{len(failed_recs)} receptor(s) failed preparation")
            for rname, err in failed_recs:
                st.error(f"Error log: {rname}", icon="🚨")
                st.code(err, language="text")
        if not rec_items: st.error("No receptors prepared successfully."); st.stop()

        lig_mols = []
        for fname, fbytes in st.session_state.lig_files_data:
            class FakeFile:
                name = fname
                def read(self): return fbytes
            lig_mols.extend(load_ligands([FakeFile()]))
        if not lig_mols: st.error("No molecules found."); st.stop()

        # ── Guardrail 4: Weird Chemistry RDKit Filter ──
        from rdkit.Chem import Descriptors
        TRANSITION_METALS = {"Ru","Pt","Pd","Fe","Cu","Zn","Co","Ni","Mn","Cr","Mo","W","Rh","Ir","Os","Ag","Au","Hg","Cd"}
        flagged_ligs = []
        for name, mol in lig_mols:
            issues = []
            mw = Descriptors.MolWt(mol)
            if mw > 700:
                issues.append(f"MW={mw:.0f} Da (>700)")
            atom_symbols = {a.GetSymbol() for a in mol.GetAtoms()}
            metals_found = atom_symbols & TRANSITION_METALS
            if metals_found:
                issues.append(f"Contains: {', '.join(metals_found)}")
            if issues:
                flagged_ligs.append((name, "; ".join(issues)))
        if flagged_ligs:
            st.warning(f"⚠️ {len(flagged_ligs)} ligand(s) outside standard drug-like space. AI affinity predictions may be unreliable for these:", icon="🧪")
            flag_df = pd.DataFrame(flagged_ligs, columns=["Ligand", "Issue"])
            st.dataframe(flag_df, use_container_width=True, hide_index=True)

        prog   = st.progress(0.0)
        status = st.empty()

        lig_items, failed, admet_data = [], [], {}
        for i, (name, mol) in enumerate(lig_mols):
            status.caption(f"Preparing {name} ({i+1}/{len(lig_mols)})…")
            p, err = prepare_ligand(mol, name, cfg, work_dir)
            if p: 
                lig_items.append((name, p))
                admet_data[name] = compute_admet(mol)
            else: 
                failed.append((name, err))
            prog.progress((i+1)/len(lig_mols))
        st.session_state.admet_data = admet_data

        if failed:
            st.warning(f"{len(failed)} ligand(s) failed preparation")
            for fname, err in failed:
                st.error(f"Error log: {fname}", icon="🚨")
                st.code(err, language="text")
        if not lig_items: st.error("No ligands prepared."); st.stop()

        prog2, stat2 = st.progress(0.0), st.empty()
        results = batch_dock(rec_items, lig_items, st.session_state.boxes, cfg, work_dir, prog2, stat2)
        st.session_state.results = results
        ok = sum(1 for r in results if r["status"]=="success")
        
        with st.spinner("Compiling Analytics & Metrics..."):
            import time; time.sleep(1.2)
            
        stat2.success(f"Done — {ok}/{len(results)} ligands docked successfully")
        if ok: st.toast(f"Pipeline executed successfully! ({ok} ligands)", icon="🚀")
        
        # Auto-navigate to Results page
        st.session_state.page = "Results"
        st.rerun()

    # Show link to results if they exist from a previous run
    if st.session_state.results:
        st.divider()
        st.success(f"✅ Previous run completed with {len(st.session_state.results)} jobs. View your results below.")
        if st.button("📊 Go to Results", use_container_width=True):
            st.session_state.page = "Results"
            st.rerun()


# =============================================================================
# ── PAGE 4: RESULTS
# =============================================================================
def page_results():
    render_stepper()
    
    if not st.session_state.results:
        st.markdown('<div class="page-title">Results</div>', unsafe_allow_html=True)
        st.info("No results yet. Run a docking experiment first.")
        if st.button("▶ Go to Run"):
            st.session_state.page = "Run"
            st.rerun()
        return
    
    res = st.session_state.results
    admet_data = st.session_state.get("admet_data", {})
    engine = st.session_state.engine
    is_gnina = engine == "GNINA"
    
    cfg = {k: st.session_state.get(k) for k in [
        "engine","embed_method","max_attempts","force_field","ff_steps", "target_ph",
        "add_hs","keep_chirality","merge_nphs","hydrate",
        "exhaustiveness","num_poses","cnn_model","cnn_weight","min_rmsd","gnina_seed","gpu_device", "flexdist", "cnn_mode",
        "v_exhaustiveness","v_num_poses","energy_range","scoring_fn","vina_seed",
        "n_jobs","backend","max_flex_res"
    ]}
    
    # ── Hero Header ──
    ok_count = sum(1 for r in res if r.get("status") == "success")
    fail_count = len(res) - ok_count
    st.markdown(f"""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <div class="page-title" style="margin-bottom:0.3rem;">Screening Results</div>
        <div class="page-sub">{ok_count} successful docks · {engine} engine · {sum(len(v) for v in st.session_state.get('boxes',{}).values()) if isinstance(st.session_state.get('boxes'), dict) else len(st.session_state.get('boxes',[]))} pocket(s)</div>
    </div>
    """, unsafe_allow_html=True)

    # Merge ADMET data into results
    data_rows = []
    for r in res:
        row = {
            "Receptor": r["Receptor"], 
            "Ligand": r["Ligand"], 
            "Pocket": r.get("Pocket", "Autobox"), 
            "Vina Forcefield (kcal/mol)": r.get("vina_score"),
            "Status": r.get("status")
        }
        if is_gnina:
            row["CNN Pose Probability (0-1)"] = r.get("cnn_score")
            row["CNN Affinity (pKd)"] = r.get("cnn_affinity")
        if r["Ligand"] in admet_data:
            row.update(admet_data[r["Ligand"]])
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows)
    ok_df = df[df["Status"]=="success"]

    # ── Key Metrics Row ──
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Jobs", len(df))
    c2.metric("Successful", len(ok_df))
    c3.metric("Best Vina", f"{ok_df['Vina Forcefield (kcal/mol)'].min():.2f} kcal/mol" if ok_df["Vina Forcefield (kcal/mol)"].notna().any() else "—")
    if is_gnina and "CNN Pose Probability (0-1)" in ok_df.columns:
        c4.metric("Best CNN", f"{ok_df['CNN Pose Probability (0-1)'].max():.3f}" if ok_df["CNN Pose Probability (0-1)"].notna().any() else "—")
    else:
        c4.metric("Best CNN", "N/A (Vina)")

    # ── Sort & Filter Controls ──
    st.markdown("")
    col_s, col_f = st.columns([1, 1])
    with col_s:
        sort_by = st.selectbox("Sort", ["Vina Score (best first)","CNN Score (best first)","Ligand name"],
                               label_visibility="collapsed")
    with col_f:
        if not ok_df.empty and ok_df["Vina Forcefield (kcal/mol)"].notna().any():
            min_v, max_v = float(ok_df["Vina Forcefield (kcal/mol)"].min()), float(ok_df["Vina Forcefield (kcal/mol)"].max())
            if min_v == max_v: min_v -= 1.0; max_v += 1.0
            vina_filter = st.slider("Filter by Vina Score threshold:", 
                                    min_value=min_v, max_value=max_v, 
                                    value=max_v, step=0.1, label_visibility="collapsed")
        else:
            vina_filter = 999.0

    # ── Export Buttons & Tabs (Inline) ──
    try:
        zip_bytes = build_zip_archive(res, df, engine, st.session_state, cfg=cfg)
    except Exception:
        zip_bytes = None
    try:
        pdf_bytes = build_pdf_report(df, engine, cfg=cfg)
    except Exception:
        pdf_bytes = None

    import base64
    z_b64 = base64.b64encode(zip_bytes).decode('utf-8') if zip_bytes else ""
    p_b64 = base64.b64encode(pdf_bytes).decode('utf-8') if pdf_bytes else ""
    
    st.markdown(f"""
    <style>
    .results-header-row {{
        display: flex; justify-content: flex-end; align-items: center;
        gap: 10px; margin-top: -38px; margin-bottom: 5px; position: relative; z-index: 99;
    }}
    .dl-btn-mini {{
        background: #2e2e2e; color: #D2D2D2 !important;
        border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
        padding: 4px 12px; font-size: 0.8rem; font-weight: 600; text-decoration: none;
        transition: all 0.2s; letter-spacing: 0.03em;
    }}
    .dl-btn-mini:hover {{
        background: #FF4B4B; color: #1a1a1a !important; border-color: #FF4B4B;
        transform: translateY(-1px);
    }}
    </style>
    <div class="results-header-row">
        {"<a href='data:application/zip;base64," + z_b64 + "' download='stratadock_experiment.zip' class='dl-btn-mini'>📦 ZIP</a>" if z_b64 else ""}
        {"<a href='data:application/pdf;base64," + p_b64 + "' download='stratadock_report.pdf' class='dl-btn-mini'>📄 PDF</a>" if p_b64 else ""}
    </div>
    """, unsafe_allow_html=True)

    res_tab1, res_tab4, res_tab5, res_tab3 = st.tabs(["📊 Docking Scores", "💊 Pharmacokinetics", "🔬 Binding Site", "🧊 3D Viewer"])

    # Apply Dynamic Filter
    filtered_df = df[((df["Status"] == "success") & (df["Vina Forcefield (kcal/mol)"] <= vina_filter)) | (df["Status"] != "success")]

    # Base sorting logic
    sort_map = {
        "Vina Score (best first)":  ("Vina Forcefield (kcal/mol)", True),
        "CNN Score (best first)":   ("CNN Pose Probability (0-1)", False),
        "Ligand name":              ("Ligand", True),
    }
    sc, sa = sort_map.get(sort_by, ("Vina Forcefield (kcal/mol)", True))
    if sc in filtered_df.columns:
        sorted_df = filtered_df.sort_values(sc, ascending=sa, na_position="last")
    else:
        sorted_df = filtered_df

    with res_tab1:
        dock_cols = [c for c in ["Receptor", "Ligand", "Pocket", "Vina Forcefield (kcal/mol)", "CNN Pose Probability (0-1)", "CNN Affinity (pKd)", "Status"] if c in sorted_df.columns]
        dock_df = sorted_df[dock_cols]
        
        # Safe styling
        try:
            if dock_df["Vina Forcefield (kcal/mol)"].notna().any():
                styled = dock_df.style.background_gradient(
                    subset=["Vina Forcefield (kcal/mol)"],
                    cmap="RdYlGn_r",
                    vmin=-9.5,
                    vmax=-2.0
                )
        except Exception:
            styled = dock_df.style
            
        col_config = {"Receptor": st.column_config.TextColumn("Receptor", width="medium"),
                      "Ligand": st.column_config.TextColumn("Ligand", width="large")}
        if "CNN Pose Probability (0-1)" in dock_df.columns:
            col_config["CNN Pose Probability (0-1)"] = st.column_config.ProgressColumn("CNN Pose Prob.", min_value=0, max_value=1, format="%.4f")
        if "CNN Affinity (pKd)" in dock_df.columns:
            col_config["CNN Affinity (pKd)"] = st.column_config.NumberColumn("CNN Affinity", format="%.3f")
        
        st.dataframe(styled, use_container_width=True, height=450,
                     column_config=col_config)
        
        # Show error details for failed jobs
        failed_jobs = [r for r in res if r["status"] != "success"]
        if failed_jobs:
            st.warning(f"{len(failed_jobs)} docking job(s) returned an error")
            for r in failed_jobs:
                st.error(f"Error log: {r.get('Receptor','?')} ➤ {r.get('Ligand','?')} ({r.get('name','?')})", icon="🚨")
                st.code(r["status"], language="text")

    with res_tab4:
        # Show only ADMET columns
        admet_cols = [c for c in ["Ligand", "MW", "LogP", "TPSA", "HBD", "HBA", "QED", "Lipinski Fails"] if c in sorted_df.columns]
        if len(admet_cols) > 1:
            admet_df = sorted_df[admet_cols].copy()
            for col in ["MW", "LogP", "TPSA", "QED"]:
                if col in admet_df.columns:
                    admet_df[col] = admet_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            for col in ["HBD", "HBA", "Lipinski Fails"]:
                if col in admet_df.columns:
                    admet_df[col] = admet_df[col].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
            
            st.dataframe(admet_df, use_container_width=True, height=400)
            
            st.caption("""
            **Metrics Explained:** 
            - **MW**: Molecular Weight (< 500 Da ideal). 
            - **LogP**: Lipophilicity / fat-solubility (0–5 ideal). 
            - **TPSA**: Topological Polar Surface Area (< 90 Å² for blood-brain barrier penetration, < 140 Å² for general oral bioavailability).
            - **HBD**: Hydrogen Bond Donors (≤ 5 ideal).
            - **HBA**: Hydrogen Bond Acceptors (≤ 10 ideal).
            - **QED**: Quantitative Estimate of Drug-likeness (0.0 to 1.0, higher is more drug-like).
            - **Lipinski Fails**: Number of Lipinski's Rule of 5 violations (0 is ideal, ≤ 1 is acceptable).
            """)
        else:
            st.info("ADMET calculations are available after a successful run.")

    with res_tab3:
        ok_results = [r for r in res if r["status"]=="success" and r.get("pose_sdf") and Path(r["pose_sdf"]).exists()]
        if ok_results:
            s1, s2 = st.columns(2)
            with s1:
                sel_r = st.selectbox("Receptor", list({r["Receptor"] for r in ok_results}))
            with s2:
                sel_l = st.selectbox("Ligand", [r["name"] for r in ok_results if r["Receptor"]==sel_r])
            
            view_style = st.radio("Protein Style", ["Cartoon","Surface","Sticks","Spheres"], horizontal=True)
            
            r_match = next((r for r in ok_results if r["Receptor"]==sel_r and r["name"]==sel_l), None)
            if r_match:
                render_3d(r_match["rec_pdb"], r_match["pose_sdf"], view_style)
                
                if Path(r_match["pose_sdf"]).exists():
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button("Download pose SDF", Path(r_match["pose_sdf"]).read_bytes(), f"{sel_r}_{sel_l}_pose.sdf", use_container_width=True)
                    with dl2:
                        cpdb = r_match.get("complex_pdb")
                        if cpdb and Path(cpdb).exists():
                            st.download_button("Download complex PDB", Path(cpdb).read_bytes(), f"{sel_r}_{sel_l}_complex.pdb", use_container_width=True)
        else:
            st.info("No successful 3D poses generated.")

    with res_tab5:
        ok_results_bs = [r for r in res if r["status"]=="success" and isinstance(r.get("interactions"), pd.DataFrame) and not r["interactions"].empty]
        if ok_results_bs:
            bs1, bs2 = st.columns(2)
            with bs1:
                sel_r_bs = st.selectbox("Receptor", list({r["Receptor"] for r in ok_results_bs}), key="bs_rec")
            with bs2:
                sel_l_bs = st.selectbox("Ligand", [r["Ligand"] for r in ok_results_bs if r["Receptor"]==sel_r_bs], key="bs_lig")
            
            r_match_bs = next((r for r in ok_results_bs if r["Receptor"]==sel_r_bs and r["Ligand"]==sel_l_bs), None)
            if r_match_bs is not None:
                idf = r_match_bs["interactions"]
                st.markdown(f"**{len(idf)} interacting residues** within 4.0 Å of docked pose")
                st.dataframe(idf, use_container_width=True, height=400)
                
                # Summary stats
                bond_counts = idf["Bond Type"].value_counts()
                scols = st.columns(len(bond_counts))
                for col, (btype, count) in zip(scols, bond_counts.items()):
                    col.metric(btype, count)
                
                st.caption("Bond types are inferred from atom distances and element types. H-bonds: polar atoms < 3.5 Å. Hydrophobic: carbon-carbon contacts < 4.0 Å. Salt bridges: charged residue-polar ligand atom < 4.0 Å.")
        else:
            st.info("No binding site interaction data available. Run a docking experiment first.")
    
    # ── Re-run option ──
    st.markdown("")
    r_col, l_col = st.columns([1, 1])
    with r_col:
        if st.button("▶ Run Again with Different Settings", use_container_width=True):
            st.session_state.page = "Run"
            st.rerun()
    with l_col:
        term_log = st.session_state.get("terminal_log", "")
        if term_log:
            st.download_button("📋 Download Run Log", term_log.encode("utf-8"), "stratadock_run_log.txt", mime="text/plain", use_container_width=True)


# =============================================================================
# ── PAGE 5: HELP
# =============================================================================
def page_help():
    render_stepper()
    st.markdown('<div class="page-title">Help &amp; Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Learn how to use StrataDock and interpret advanced computational outputs.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Quick Start", "🧠 GNINA AI & Scoring", "🧬 Advanced Features (Flex)", "❓ FAQ", "💾 Save/Load Project"])

    with tab5:
        st.markdown("### 💾 Manage Project Session")
        st.caption("Save your progress, uploaded proteins, and tuning settings to a file, or resume a previous docking project.")
        
        export_keys = [
            "engine", "boxes", "rec_files_data", "lig_files_data", "box_padding", "multi_pocket",
            "embed_method", "max_attempts", "force_field", "ff_steps", "add_hs", 
            "keep_chirality", "merge_nphs", "hydrate", "exhaustiveness", "num_poses", 
            "cnn_model", "cnn_weight", "min_rmsd", "gnina_seed", "gpu_device", "flexdist", "cnn_mode",
            "v_exhaustiveness", "v_num_poses", "energy_range", "scoring_fn", "vina_seed", 
            "n_jobs", "backend",
            "prep_remove_water", "prep_remove_het", "prep_keep_metals", "prep_add_h",
            "prep_fix_atoms", "prep_fix_loops", "prep_propka", "prep_minimize", "prep_min_steps"
        ]
        export_state = {k: st.session_state[k] for k in export_keys if k in st.session_state}
        
        col_exp, col_imp = st.columns(2, gap="large")
        
        with col_exp:
            st.markdown("**Export Current Session**")
            import pickle, base64
            b64_str = base64.b64encode(pickle.dumps(export_state)).decode("utf-8")
            st.download_button("📥 Click here to download (.stratadock)", b64_str, "project.stratadock", use_container_width=True)
            
        with col_imp:
            st.markdown("**Restore Previous Session**")
            uploaded = st.file_uploader("Upload .stratadock session file", type=["stratadock"], label_visibility="collapsed")
            if uploaded:
                if st.button("Restore Session Now", type="primary", use_container_width=True):
                    try:
                        data = pickle.loads(base64.b64decode(uploaded.read()))
                        for k, v in data.items():
                            st.session_state[k] = v
                        st.success("Session restored! Reloading...")
                        import time; time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

    with tab1:
        st.markdown("""
### 🚀 How to run a docking job

**Step 1 — Upload**
- Upload your **receptor(s)** as `.pdb` files. 
- Upload your **ligand(s)** as `.sdf` (3D structures) or `.smi` / `.txt` (1D SMILES strings).
- **Defining the Box:** You must specify where the docking engine should search.
    - **Option A:** Upload a Reference Ligand and click **Compute Autobox**.
    - **Option B [NEW]:** Click **Suggest Pockets (Auto)**. StrataDock will run a pocket-detection algorithm (FPocket-inspired) to scan the receptor for the largest, deepest binding cavities and automatically snap 3D grid boxes around them!

**Step 2 — Settings**
- Choose your **docking engine**. AutoDock Vina is recommended for most users and works on all CPUs. GNINA utilizes neural-network scoring and works best with an NVIDIA/AMD GPU.
- Configure **Protein Preparation** — StrataDock can automatically:
    - Remove water molecules and heteroatoms from crystal structures
    - Add missing hydrogen atoms at physiological pH
    - Repair missing atoms and model missing loops (PDBFixer)
    - Assign protonation states via propka (advanced)
    - Run energy minimization via OpenMM Amber14 (advanced)
- Adjust molecular parameters if needed. **The default values are optimized for high-throughput screening.**

**Step 3 — Run & Results**
- Click **Run Docking**. The cinematic console will stream engine logs in real-time.
- Protein preparation runs automatically before docking (based on your settings).
- For bulk / N × M cross-docking (multiple receptors × multiple ligands), utilize the **Parallelization** settings to unleash your CPU cores.
- After running, view results across 4 tabs: **Docking Scores**, **Pharmacokinetics**, **Binding Site** (interacting residues), and **3D Viewer** (with complex PDB download).
- Download the **ZIP archive** — organized by receptor with poses, complex PDBs, interaction CSVs, and a comprehensive `summary.txt`.
        """)

    with tab2:
        st.markdown("### Understanding the scores")
        st.warning("**Crucial Note:** GNINA outputs completely different metrics than standard AutoDock. Do not mix them up!")

        st.markdown("""
**Vina Forcefield** *(kcal/mol)*
> The estimated binding free energy from the traditional empirical physics engine.
> **More negative = stronger binding.** A score below −7 kcal/mol is generally considered a good binder.
> *Note: If you enable Flexible Docking, this score may become positive due to deformation penalties (see the Advanced Features tab).*

**CNN Pose Probability** *(GNINA only)*
> A probability score from GNINA's convolutional neural network that estimates how likely a pose represents a **true biological binding pose**.
> **Range: 0.0 to 1.0 (Higher is objectively better).** A Probability above 0.5 is generally considered a highly confident binding pose regardless of what the Vina Forcefield score says.

**CNN Affinity** *(GNINA only)*
> The CNN's prediction of binding affinity (pKd).
> **Higher is better.** (Note: This is an empirical constant, not a kcal/mol energy).
        """)

    with tab3:
        st.markdown("### 🧬 Advanced Features & Flexibility")
        st.markdown("""
#### 1. Automated Receptor Flexibility (`--flexdist`)
By default, docking engines assume the target protein is entirely frozen. This is fast but biologically inaccurate. Turning on **Flexible Docking** in the settings fundamentally changes the calculation.
- GNINA will detect any of the protein's native amino acid side-chains that are within **3.5 Angstroms** of the pocket center.
- It will then temporarily "unfreeze" those atoms, allowing physical rotation and bending during the simulation to dynamically warp around your drug (Induced Fit).

**Why is my Vina Forcefield score positive when Flexible Docking is on?**
When you bend a physical protein out of its native relaxed state, the physics engine assigns a massive "Deformation Penalty" for steric clashing and torsional strain inside the receptor itself. Therefore, a perfectly placed ligand might generate an empirically terrible Vina score (e.g. `+1.85`). If you use Flexible Docking, **always trust the CNN Pose Probability instead of the Vina score!**

#### 2. Deep Learning Energy Minimization (CNN Modes)
GNINA has two brains: a fast physics equation (Vina) and a slow Deep Learning Neural Network (CNN).
- **none:** Completely bypass the Neural Network. Runs identically to classic AutoDock Vina.
- **rescore (Default):** Runs the fast Vina physics to generate the poses, and then simply asks the Neural Network to grade the final answers as a probability.
- **refinement:** Extremely slow but incredibly accurate. The Neural Network actively overrides the physics engine *during* the molecular relaxation phase, dynamically pulling the ligand into the most perfect biological pose. 
        """)

    with tab4:
        faqs = [
            ("My ligand failed preparation — what do I do?",
             "Some molecules cannot be embedded in 3D by RDKit (e.g. complex metal complexes, unusual valences). Try increasing 'Max embedding attempts' in Settings, or check that your SMILES/SDF is valid. You can validate SMILES at rdkit.org."),
            ("What is a good Vina score?",
             "Below −6 kcal/mol is considered a weak binder, −7 to −9 is a moderate binder, and below −9 is a strong binder. These are rough guidelines — always compare within your own screening set. (This only applies to rigid docking without flexdist)."),
            ("How does the 'Suggest Pocket (Auto)' feature work?",
             "When you click Suggest Pockets, StrataDock runs a custom geometric cavity-detection algorithm based on principles from `fpocket`. It constructs an alpha-shape of the receptor, maps out deep crevices using Voronoi vertices, filters them by hydrophobicity and volume, and automatically selects the largest, most druggable pockets as your target docking sites!"),
            ("Why does GNINA not work on my machine?",
             "GNINA relies heavily on massive convolutions. It can run on a CPU (using the CPU Fallback mode), but it is aggressively slow. If you don't have an NVIDIA GPU, set CNN Mode to `none` to speed things up!"),
            ("How do I run on an NVIDIA server?",
             "Run `bash setup.sh` on the server. StrataDock auto-detects NVIDIA GPUs and enables GNINA automatically. No manual configuration needed!"),
        ]
        faq_html = ""
        for q, a in faqs:
            faq_html += f"""
            <details class="faq-item">
                <summary>{q}</summary>
                <div class="faq-body">{a}</div>
            </details>"""
        st.markdown(faq_html, unsafe_allow_html=True)


# =============================================================================
# ROUTER
# =============================================================================
page = st.session_state.page
if page == "Upload":     page_upload()
elif page == "Settings": page_settings()
elif page == "Run":      page_run()
elif page == "Results":  page_results()
elif page == "Help":     page_help()

# ── Subtle footer credit ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; opacity:0.35; font-size:0.75rem; color:#8b949e; letter-spacing:0.03em;">
    StrataDock Alpha V0.7.02 · Developed by Prithvi Rajan · Department of Bioinformatics, Freie Universität Berlin · © 2026
</div>
""", unsafe_allow_html=True)
