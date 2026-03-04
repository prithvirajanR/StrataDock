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
        "prep_add_h": True, "prep_fix_atoms": False, "prep_fix_loops": False,
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
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, FilterCatalog
    try:
        mw   = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        qed  = QED.qed(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        chrg = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        fails = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        # PAINS filter
        pains_flag = None
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            entry = catalog.GetFirstMatch(mol)
            if entry:
                pains_flag = entry.GetDescription()
        except Exception:
            pass

        # Rule-based toxicity flags
        herg_risk    = (logp > 4.5 and mw > 450 and hba < 4)
        bbb_ok       = (mw < 450 and 1 <= logp <= 5 and tpsa < 90 and hbd <= 3)
        hepato_risk  = (logp > 4.0 and hbd < 2 and mw > 300)
        mutagen_risk = bool(pains_flag) or (arom >= 3 and chrg != 0)

        # Per-property traffic lights
        def _tl(val, green, yellow):
            if val is None: return "grey"
            glo, ghi = green; ylo, yhi = yellow
            if glo <= val <= ghi: return "green"
            if ylo <= val <= yhi: return "yellow"
            return "red"

        traffic = {
            "MW":   _tl(mw,   (0, 500),   (0, 600)),
            "LogP": _tl(logp, (0, 5),     (-1, 6)),
            "TPSA": _tl(tpsa, (0, 90),    (0, 140)),
            "HBD":  _tl(hbd,  (0, 5),     (0, 7)),
            "HBA":  _tl(hba,  (0, 10),    (0, 12)),
            "QED":  _tl(qed,  (0.5, 1.0), (0.3, 1.0)),
            "Fsp3": _tl(fsp3, (0.25, 1.0),(0.1, 1.0)),
        }

        return {
            "MW": mw, "LogP": logp, "TPSA": tpsa, "HBD": hbd, "HBA": hba,
            "QED": qed, "Lipinski Fails": fails,
            "Rotatable Bonds": rotb, "Aromatic Rings": arom,
            "Fsp3": fsp3, "Formal Charge": chrg,
            "PAINS Alert": pains_flag,
            "hERG Risk": herg_risk, "BBB Penetration": bbb_ok,
            "Hepatotoxicity Risk": hepato_risk, "Mutagenicity Risk": mutagen_risk,
            "_traffic": traffic,
        }
    except Exception:
        return {"MW": None, "LogP": None, "TPSA": None, "HBD": None, "HBA": None,
                "QED": None, "Lipinski Fails": None, "_traffic": {}}


def analyze_waters(pdb_path, box, cutoff=6.0):
    """Classify crystallographic waters inside the docking box.
    Returns list of {x, y, z, stable, hbond_count, dist}.
    Stable = H-bonded to >= 2 protein polar atoms within 3.5 Å.
    Pure-numpy approach — no NeighborSearch (avoids BioPython array shape bugs)."""
    waters = []
    try:
        from Bio.PDB import PDBParser, Selection
        import numpy as np
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("rec", pdb_path)

        cx  = float(box.get("cx", 0)); cy  = float(box.get("cy", 0)); cz  = float(box.get("cz", 0))
        sx  = float(box.get("sx", 20)); sy = float(box.get("sy", 20)); sz  = float(box.get("sz", 20))
        hx, hy, hz = sx / 2 + 2, sy / 2 + 2, sz / 2 + 2

        all_atoms = list(Selection.unfold_entities(struct, "A"))

        # Build numpy array of polar protein atom coords (N, O, S — not from water residues)
        polar_coords = []
        for a in all_atoms:
            try:
                rname = a.get_parent().get_resname().strip()
                if rname in ("HOH", "WAT", "SOL"): continue
                if a.element not in ("N", "O", "S"): continue
                c = a.get_coord()
                polar_coords.append([float(c[0]), float(c[1]), float(c[2])])
            except Exception:
                continue
        polar_arr = np.array(polar_coords) if polar_coords else np.empty((0, 3))

        # Find and classify waters inside the box
        for model in struct:
            for chain in model:
                for res in chain:
                    if res.get_resname().strip() not in ("HOH", "WAT", "SOL"):
                        continue
                    if "O" not in res:
                        continue
                    try:
                        c = res["O"].get_coord()
                        ox, oy, oz = float(c[0]), float(c[1]), float(c[2])
                    except Exception:
                        continue
                    # Box containment check
                    if not (abs(ox - cx) <= hx and abs(oy - cy) <= hy and abs(oz - cz) <= hz):
                        continue
                    dist = float(np.sqrt((ox-cx)**2 + (oy-cy)**2 + (oz-cz)**2))
                    # Count polar protein atoms within 3.5 Å (H-bond proxy)
                    hb = 0
                    if polar_arr.shape[0] > 0:
                        diffs = polar_arr - np.array([ox, oy, oz])
                        dists = np.sqrt((diffs**2).sum(axis=1))
                        hb = int((dists <= 3.5).sum())
                    waters.append({
                        "x": round(ox, 3), "y": round(oy, 3), "z": round(oz, 3),
                        "stable": hb >= 2,
                        "hbond_count": hb,
                        "dist": round(dist, 2),
                    })
    except Exception as e:
        return [{"_error": str(e)}]
    return waters





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
    """Tiered protein preparation pipeline. Simple steps use reliable line-filtering
    and obabel. PDBFixer is reserved for the optional complex steps only."""
    prep_log = []
    out_path = pdb_path.replace(".pdb", "_prepared.pdb")
    current = pdb_path  # track working file as we apply steps

    # ── Step 1: Remove water (HOH/WAT lines) ──────────────────────────────────
    if cfg.get("prep_remove_water", True):
        try:
            lines = Path(current).read_text(errors="replace").splitlines()
            kept = [l for l in lines
                    if not (l.startswith(("HETATM", "ATOM")) and l[17:20].strip() in ("HOH","WAT","SOL"))]
            Path(out_path).write_text("\n".join(kept) + "\n")
            current = out_path
            prep_log.append("Removed water molecules")
        except Exception as e:
            prep_log.append(f"[WARN] Water removal failed: {e}")

    # ── Step 2: Remove heteroatoms / co-crystallised ligands ──────────────────
    if cfg.get("prep_remove_het", True):
        try:
            keep_metals = cfg.get("prep_keep_metals", True)
            METALS = {"ZN","MG","CA","FE","MN","CU","CO","NI","NA","K","CL"}
            lines = Path(current).read_text(errors="replace").splitlines()
            kept = []
            for l in lines:
                if l.startswith("HETATM"):
                    res = l[17:20].strip()
                    atom = l[12:16].strip()
                    if res in ("HOH","WAT","SOL"):
                        kept.append(l); continue  # water already handled above
                    if keep_metals and atom.upper() in METALS:
                        kept.append(l); continue  # keep metal cofactors
                    # drop this HETATM
                else:
                    kept.append(l)
            Path(out_path).write_text("\n".join(kept) + "\n")
            current = out_path
            prep_log.append("Removed heteroatoms" + (" (kept metals)" if keep_metals else ""))
        except Exception as e:
            prep_log.append(f"[WARN] Heteroatom removal failed: {e}")

    # ── Step 3: Add hydrogens at physiological pH via obabel ──────────────────
    if cfg.get("prep_add_h", True):
        try:
            ph = cfg.get("target_ph", 7.4)
            ob_paths = [
                shutil.which("obabel"),
                os.path.expanduser("~/miniconda3/envs/stratadock/bin/obabel"),
                "/home/rajan/miniconda3/envs/stratadock/bin/obabel",
            ]
            ob_exe = next((p for p in ob_paths if p and Path(p).exists()), None)
            if ob_exe:
                h_out = out_path.replace("_prepared.pdb", "_h.pdb")
                r = subprocess.run([ob_exe, current, "-O", h_out, "-p", str(ph)],
                                   capture_output=True, text=True, timeout=60)
                if Path(h_out).exists():
                    current = h_out
                    out_path = h_out
                    prep_log.append(f"Added hydrogens at pH {ph} (obabel)")
                else:
                    prep_log.append(f"[WARN] obabel H addition produced no output: {r.stderr[:80]}")
            else:
                prep_log.append("[WARN] obabel not found — skipping hydrogen addition")
        except Exception as e:
            prep_log.append(f"[WARN] H addition failed: {e}")

    # ── Steps 4+5: PDBFixer — fix missing atoms and loops (optional, complex) ──
    if cfg.get("prep_fix_atoms", False) or cfg.get("prep_fix_loops", False):
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
            fixer = PDBFixer(filename=current)
            if cfg.get("prep_fix_loops", False):
                fixer.findMissingResidues()
                prep_log.append("Identified missing loops/residues")
            if cfg.get("prep_fix_atoms", False):
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                prep_log.append("Repaired missing heavy atoms")
            fixer_out = out_path.replace(".pdb", "_fixed.pdb")
            with open(fixer_out, "w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
            # Strip MODEL/ENDMDL records that confuse obabel
            lines = Path(fixer_out).read_text().splitlines()
            Path(fixer_out).write_text("\n".join(l for l in lines if not l.startswith(("MODEL","ENDMDL"))) + "\n")
            current = fixer_out
            out_path = fixer_out
            prep_log.append("PDBFixer structure repair complete")
        except ImportError:
            prep_log.append("[WARN] PDBFixer/OpenMM not installed — skipping structure repair")
        except Exception as e:
            prep_log.append(f"[WARN] PDBFixer error: {str(e)[:100]} — skipping structure repair")

    # ── Step 6: Protonation state analysis via propka (optional) ──────────────
    if cfg.get("prep_propka", False):
        try:
            import propka.run as propka_run
            mol_container = propka_run.single(current)
            pka_info = []
            for group in mol_container.conformations["AVR"].groups:
                if hasattr(group, "pka_value") and group.pka_value is not None:
                    pka_info.append(f"{group.residue_type}{group.atom.res_num}: pKa={group.pka_value:.2f}")
            prep_log.append(f"propka3: {len(pka_info)} titratable residues analyzed")
        except ImportError:
            prep_log.append("[WARN] propka not installed — skipping protonation analysis")
        except Exception as e:
            prep_log.append(f"[WARN] propka error: {str(e)[:100]}")

    # ── Step 7: Energy minimization via OpenMM (optional) ─────────────────────
    if cfg.get("prep_minimize", False):
        try:
            from openmm.app import PDBFile, ForceField, Modeller, Simulation
            from openmm import LangevinMiddleIntegrator, LocalEnergyMinimizer
            import openmm.unit as unit
            pdb = PDBFile(current)
            forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            modeller = Modeller(pdb.topology, pdb.positions)
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=0, constraints=None)
            integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            simulation = Simulation(modeller.topology, system, integrator)
            simulation.context.setPositions(modeller.positions)
            max_steps = cfg.get("prep_min_steps", 100)
            LocalEnergyMinimizer.minimize(simulation.context, maxIterations=max_steps)
            positions = simulation.context.getState(getPositions=True).getPositions()
            min_out = current.replace(".pdb", "_min.pdb")
            with open(min_out, "w") as f:
                PDBFile.writeFile(simulation.topology, positions, f)
            lines = Path(min_out).read_text().splitlines()
            Path(min_out).write_text("\n".join(l for l in lines if not l.startswith(("MODEL","ENDMDL"))) + "\n")
            current = min_out
            prep_log.append(f"Energy minimization complete ({max_steps} steps, Amber14)")
        except ImportError:
            prep_log.append("[WARN] OpenMM not installed — skipping minimization")
        except Exception as e:
            prep_log.append(f"[WARN] Minimization error: {str(e)[:100]} — skipping")

    return current, prep_log




def prepare_receptor(pdb_path, out_dir):
    basename = Path(pdb_path).stem
    out = os.path.join(out_dir, f"{basename}.pdbqt")
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
            # ── Extract score: parse PDBQT REMARK VINA RESULT (authoritative) ──
            # v.energies() can return 0.0 in some vina API versions; the PDBQT
            # REMARK line is always written correctly by vina itself.
            try:
                pdbqt_text = Path(out_pdbqt).read_text()
                m_score = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", pdbqt_text)
                if m_score:
                    res["vina_score"] = float(m_score.group(1))
                else:
                    # Fallback: v.energies() column 0 = affinity
                    en = v.energies(n_poses=1)
                    if en is not None and len(en) and en[0][0] != 0.0:
                        res["vina_score"] = float(en[0][0])
            except Exception:
                en = v.energies(n_poses=1)
                if en is not None and len(en):
                    res["vina_score"] = float(en[0][0])
            if res["vina_score"] is not None:
                update_term(f"> Optimization reached. Best Vina affinity: {res['vina_score']:.2f} kcal/mol")

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
        rec_boxes = boxes.get(rname)
        if not rec_boxes:
            # Try a case-insensitive / stem-only match in case keys differ slightly
            for key, val in boxes.items():
                if Path(key).stem.lower() == Path(rname).stem.lower():
                    rec_boxes = val
                    break
        if not rec_boxes:
            status_el.warning(f"No docking box found for **{rname}** — skipping. "
                              f"Run 'Suggest Pocket' or 'Compute Autobox' for all receptors first.")
            continue
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


def render_3d(rec_pdb, pose_sdf, view_style="Cartoon", waters=None):
    try:
        import py3Dmol
        from stmol import showmol
        
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

        # HydroDock water spheres
        if waters:
            for w in waters:
                color = "#22c55e" if w["stable"] else "#ef4444"
                opacity = 0.55 if w["stable"] else 0.8
                view.addSphere({
                    "center": {"x": w["x"], "y": w["y"], "z": w["z"]},
                    "radius": 0.6,
                    "color": color,
                    "opacity": opacity,
                })
                
        view.zoomTo()
        view.setBackgroundColor("#0c0f1d")
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
@st.dialog("🧬 Fold-to-Dock — AI Structure Prediction", width="large")
def _esm_dialog():
    st.caption("Paste a protein sequence. ESMFold (Meta AI) predicts a 3D structure in ~10–30 s. Free, no login.")
    _fasta_raw = st.text_area(
        "Sequence (FASTA or plain AA)", height=120, key="esm_dlg_fasta",
        placeholder="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    _dc1, _dc2 = st.columns([2, 1])
    with _dc1:
        _esm_name = st.text_input("Structure name", value="predicted_structure",
                                   key="esm_dlg_name", placeholder="Structure name…")
    with _dc2:
        _do_fold = st.button("🔮 Predict", key="esm_dlg_run", use_container_width=True, type="primary")

    if _do_fold:
        _seq = "".join(ln.strip() for ln in _fasta_raw.splitlines()
                       if ln.strip() and not ln.startswith(">"))
        if not _seq:
            st.error("Paste a sequence first.")
        else:
            if len(_seq) > 600:
                st.warning(f"{len(_seq)} AA — ESMFold works best under 400 AA, proceeding…")
            try:
                import requests as _req
                with st.spinner(f"Folding {len(_seq)} AA via ESMFold API…"):
                    _resp = _req.post(
                        "https://api.esmatlas.com/foldSequence/v1/pdb/",
                        data=_seq,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=120,
                    )
                if _resp.status_code != 200:
                    st.error(f"ESMFold API error {_resp.status_code}: {_resp.text[:200]}")
                else:
                    _pdb_text = _resp.text
                    _bfs = []
                    for _aln in _pdb_text.splitlines():
                        if _aln.startswith("ATOM") and len(_aln) >= 66:
                            try: _bfs.append(float(_aln[60:66].strip()))
                            except Exception: pass
                    if _bfs and max(_bfs) <= 1.0:
                        _bfs = [b * 100 for b in _bfs]
                    _plddt = round(sum(_bfs)/len(_bfs), 1) if _bfs else None
                    _pdb_name = (_esm_name.strip() or "predicted_structure").replace(" ", "_")
                    if not _pdb_name.endswith(".pdb"): _pdb_name += ".pdb"
                    st.session_state["esm_pdb_name"]  = _pdb_name
                    st.session_state["esm_pdb_bytes"] = _pdb_text.encode()
                    st.session_state["esm_plddt"]     = _plddt
            except Exception as _ex:
                st.error(f"ESMFold failed: {_ex}")

    if st.session_state.get("esm_pdb_bytes"):
        _plddt    = st.session_state.get("esm_plddt")
        _pdb_name = st.session_state.get("esm_pdb_name", "structure.pdb")
        if _plddt is not None:
            _pc  = "#22c55e" if _plddt >= 70 else ("#f59e0b" if _plddt >= 50 else "#ef4444")
            _lbl = "High" if _plddt >= 70 else ("Medium" if _plddt >= 50 else "Low")
            _pct = min(int(_plddt), 100)
            st.markdown(f"""
            <div style="background:#0d0d1a;border-radius:10px;padding:12px 16px;
                        margin-top:10px;border:1px solid {_pc}44">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#aaa;font-size:0.82rem">pLDDT Confidence · {_pdb_name}</span>
                <span style="color:{_pc};font-weight:800">{_plddt} / 100 · {_lbl}</span>
              </div>
              <div style="background:#1a1a2e;border-radius:6px;height:8px;overflow:hidden">
                <div style="background:linear-gradient(90deg,#7c3aed,{_pc});
                            width:{_pct}%;height:100%;border-radius:6px"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")
        if st.button("📥 Load as Receptor", key="esm_dlg_load", use_container_width=True, type="primary"):
            _pdb_bytes = st.session_state["esm_pdb_bytes"]
            existing = [r for r in (st.session_state.get("rec_files_data") or [])
                        if r[0] != _pdb_name]
            existing.append((_pdb_name, _pdb_bytes))
            st.session_state.rec_files_data = existing
            st.session_state.rec_bytes      = _pdb_bytes
            st.session_state["esm_pdb_bytes"] = None
            st.rerun()


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

        st.caption("Select multiple files at once with Ctrl/Cmd.")
        _rv = st.session_state.get("_rec_up_v", 0)
        rec_files = st.file_uploader("Receptor PDB(s)", type=["pdb"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed", key=f"rec_up_{_rv}")
        if rec_files:
            st.session_state.rec_files_data = [(f.name, f.read()) for f in rec_files]
            st.session_state.rec_bytes = st.session_state.rec_files_data[0][1]
        elif getattr(st.session_state, "rec_files_data", None):
            for idx, (rname, _) in enumerate(list(st.session_state.rec_files_data)):
                c1, c2 = st.columns([7, 1])
                c1.caption(rname)
                if c2.button("x", key=f"rm_r_{idx}", use_container_width=True, help=f"Remove {rname}"):
                    upd = [(n, b) for n, b in st.session_state.rec_files_data if n != rname]
                    st.session_state.rec_files_data = upd or None
                    st.session_state.rec_bytes = upd[0][1] if upd else None
                    if not upd:
                        st.session_state.boxes = {}; st.session_state.results = []
                    st.session_state["_rec_up_v"] = _rv + 1
                    st.rerun()
            if st.button("Clear all receptors", key="clear_rec"):
                st.session_state.rec_files_data = None; st.session_state.rec_bytes = None
                st.session_state.boxes = {}; st.session_state.results = []
                st.session_state["_rec_up_v"] = _rv + 1
                st.rerun()

        st.markdown("")

        st.markdown("**Reference Ligand** *(optional)*")
        st.caption("If provided, the docking box will be centred on this ligand's position rather than the full receptor.")
        ref_file = st.file_uploader("Reference ligand SDF", type=["sdf"],
                                     label_visibility="collapsed", key="ref_up")

    with col_r:
        st.markdown("**Ligands** - SDF, SMILES, or ZIP file(s)")
        st.caption("Hold Ctrl/Cmd to select multiple files at once.")
        _lv = st.session_state.get("_lig_up_v", 0)
        lig_files = st.file_uploader("Ligand files", type=["sdf","smi","smiles","txt","zip"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed", key=f"lig_up_{_lv}")
        if lig_files:
            st.session_state.lig_files_data = [(f.name, f.read()) for f in lig_files]
        elif st.session_state.get("lig_files_data"):
            for idx, (lname, _) in enumerate(list(st.session_state.lig_files_data)):
                c1, c2 = st.columns([7, 1])
                c1.caption(lname)
                if c2.button("x", key=f"rm_l_{idx}", use_container_width=True, help=f"Remove {lname}"):
                    upd = [(n, b) for n, b in st.session_state.lig_files_data if n != lname]
                    st.session_state.lig_files_data = upd
                    if not upd:
                        st.session_state.admet_data = {}; st.session_state.results = []
                    st.session_state["_lig_up_v"] = _lv + 1
                    st.rerun()
            if st.button("Clear all ligands", key="clear_lig"):
                st.session_state.lig_files_data = []
                st.session_state.admet_data = {}; st.session_state.results = []
                st.session_state["_lig_up_v"] = _lv + 1
                st.rerun()

        st.markdown("")
        if st.session_state.get("lig_files_data") and st.checkbox("Preview Ligands (2D)", key="lig_prev_chk"):
            try:
                from rdkit.Chem import Draw
                class F:
                    def __init__(self,n,b): self.name=n; self.b=b
                    def read(self): return self.b
                prev_mols = load_ligands([F(n,b) for n,b in st.session_state.lig_files_data[:3]])[:12]
                if prev_mols:
                    ms = [m for n,m in prev_mols]
                    ns = [n[:15] for n,m in prev_mols]
                    img = Draw.MolsToGridImage(ms, molsPerRow=4, subImgSize=(150,150), legends=ns)
                    st.image(img, width=460)
            except Exception as e:
                st.warning(f"Could not generate 2D previews: {e}")


        # ── ESMFold card (under ligand section) ─────────────────────
        st.markdown("")
        st.markdown("**Fold-to-Dock** — AI structure prediction")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f0c29,#302b63,#1a0533);
                    border-radius:10px;border:1px solid #7c3aed;padding:10px 14px;
                    box-shadow:0 0 14px #7c3aed33;
                    display:flex;align-items:center;gap:10px">
          <div style="font-size:1.4rem;filter:drop-shadow(0 0 6px #a78bfa)">🧬</div>
          <div>
            <div style="font-size:0.82rem;font-weight:800;color:#e9d5ff">
              Fold-to-Dock
              <span style="font-size:0.55rem;background:#7c3aed;color:#fff;
              border-radius:3px;padding:1px 5px;vertical-align:middle;margin-left:4px">AI</span>
            </div>
            <div style="font-size:0.68rem;color:#a78bfa;margin-top:2px">Sequence → 3D · ESMFold · free</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
          .esm-fold-btn + div .stButton > button {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            color: rgba(255,255,255,0.9) !important;
            -webkit-text-fill-color: rgba(255,255,255,0.9) !important;
            text-transform: none !important;
            box-shadow: none !important;
          }
          .esm-fold-btn + div .stButton > button:hover {
            background: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.4) !important;
          }
        </style>
        <div class="esm-fold-btn"></div>
        """, unsafe_allow_html=True)
        if st.button("🧬 Click to Fold", key="esm_open_dlg", use_container_width=True):
            _esm_dialog()



        if st.session_state.get("esm_pdb_bytes"):
            _ep = st.session_state.get("esm_plddt")
            _en = st.session_state.get("esm_pdb_name", "structure.pdb")
            if _ep is not None:
                _ec = "#22c55e" if _ep >= 70 else ("#f59e0b" if _ep >= 50 else "#ef4444")
                _el = "High" if _ep >= 70 else ("Medium" if _ep >= 50 else "Low")
                st.markdown(f"""
                <div style="background:#0d0d1a;border-radius:8px;padding:8px 12px;margin-top:6px;border:1px solid {_ec}44">
                  <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px">
                    <span style="color:#aaa">pLDDT · {_en[:20]}</span>
                    <span style="color:{_ec};font-weight:700">{_ep} · {_el}</span>
                  </div>
                  <div style="background:#1a1a2e;border-radius:4px;height:6px">
                    <div style="background:linear-gradient(90deg,#7c3aed,{_ec});width:{min(int(_ep),100)}%;height:100%;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            if st.button("📥 Load as Receptor", key="esm_card_load", use_container_width=True):
                _eb = st.session_state["esm_pdb_bytes"]
                _existing = [r for r in (st.session_state.get("rec_files_data") or []) if r[0] != _en]
                _existing.append((_en, _eb))
                st.session_state.rec_files_data = _existing
                st.session_state.rec_bytes      = _eb
                st.session_state["esm_pdb_bytes"] = None
                st.rerun()

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
                import subprocess, shutil
                # Resolve fpocket path — works even if not in Streamlit's inherit PATH
                fpocket_bin = (shutil.which("fpocket")
                               or "/home/rajan/miniconda3/envs/stratadock/bin/fpocket"
                               or "fpocket")
                all_boxes  = {}
                fp_errors  = []
                max_pockets = 5 if st.session_state.multi_pocket else 1
                for rname, rbytes in st.session_state.rec_files_data:
                    with tempfile.TemporaryDirectory() as td:
                        rp = os.path.join(td, rname)
                        Path(rp).write_bytes(rbytes)
                        try:
                            proc = subprocess.run(
                                [fpocket_bin, "-f", rp], cwd=td,
                                capture_output=True, timeout=120)
                            if proc.returncode != 0:
                                raise RuntimeError(proc.stderr.decode()[:300])
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
                                fp_errors.append(f"No pockets found in top-{max_pockets} for {rname}.")
                        except Exception as e:
                            fp_errors.append(f"fpocket failed for {rname}: {e}")
                # Persist errors so they survive the rerun below
                st.session_state["_fpocket_errors"] = fp_errors
                if all_boxes:
                    st.session_state.boxes = all_boxes
                    st.rerun()
                else:
                    for msg in fp_errors:
                        st.error(msg)


    # Show any fpocket errors that survived the rerun
    for _fpe in st.session_state.pop("_fpocket_errors", []):
        st.warning(_fpe)

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

        # 7. results.txt — clean score table
        try:
            import numpy as _np
            score_col = "Vina Forcefield (kcal/mol)" if "Vina Forcefield (kcal/mol)" in df.columns else \
                        "CNN Affinity (pKd)" if "CNN Affinity (pKd)" in df.columns else None
            receptors_in_df = df["Receptor"].unique().tolist() if "Receptor" in df.columns else []

            if score_col:
                if len(receptors_in_df) <= 1:
                    # ── Single receptor ──────────────────
                    sub = df[df[score_col].notna()].copy()
                    sub["Ligand_Name"] = sub.apply(lambda r: r.get("Ligand", r.get("name", "?")), axis=1)
                    sub = sub[["Ligand_Name", score_col]].rename(columns={score_col: "Score"})
                    sub = sub.sort_values("Score")
                    
                    sub["Score"] = sub["Score"].apply(lambda x: f"{x:.2f}")
                    
                    padding = 22
                    lines = ["".join(f"{str(c):<{padding}}" for c in sub.columns)]
                    for _, row in sub.iterrows():
                        lines.append("".join(f"{str(row[c]):<{padding}}" for c in sub.columns))
                    txt_content = "\n".join(lines)
                else:
                    # ── Multi receptor: pivot matrix + Average + SD ───────────
                    pivot = df.pivot_table(
                        index="Ligand", columns="Receptor",
                        values=score_col, aggfunc="first"
                    ).reset_index()
                    pivot.columns.name = None
                    rec_cols = [c for c in pivot.columns if c != "Ligand"]
                    pivot["Average"] = pivot[rec_cols].mean(axis=1, skipna=True).round(5)
                    # Pandas stdev crashes on single values (ddof=1). Use numpy with ddof=0.
                    pivot["SD"] = pivot[rec_cols].apply(lambda x: _np.std(x.dropna(), ddof=0), axis=1).round(5)
                    pivot = pivot.sort_values("Average")
                    
                    pivot["Average"] = pivot["Average"].apply(lambda x: f"{x:.5f}")
                    pivot["SD"]      = pivot["SD"].apply(lambda x: f"{x:.5f}")
                    
                    # Replace NaN with "0.00" for better readability in the table
                    for rc in rec_cols:
                        pivot[rc] = pivot[rc].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
                        pivot = pivot.rename(columns={rc: Path(rc).stem})
                    
                    pivot = pivot.rename(columns={"Ligand": "Ligand_Name"})
                    
                    padding = 22
                    lines = ["".join(f"{str(c):<{padding}}" for c in pivot.columns)]
                    for _, row in pivot.iterrows():
                        lines.append("".join(f"{str(row[c]):<{padding}}" for c in pivot.columns))
                    txt_content = "\n".join(lines)
                    
                zf.writestr("results.txt", txt_content)
        except Exception as e:
            print(f"Error generating results.txt: {e}")



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
            # Store original path for HydroDock water analysis (before stripping)
            if "orig_pdbs" not in st.session_state:
                st.session_state.orig_pdbs = {}
            st.session_state.orig_pdbs[rname] = rp

            
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
    
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st_md = f"""
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
        {"<a href='data:application/zip;base64," + z_b64 + f"' download='stratadock_results_{ts}.zip' class='dl-btn-mini'>📦 ZIP</a>" if z_b64 else ""}
        {"<a href='data:application/pdf;base64," + p_b64 + f"' download='stratadock_report_{ts}.pdf' class='dl-btn-mini'>📄 PDF</a>" if p_b64 else ""}
    </div>
    """
    st.markdown(st_md, unsafe_allow_html=True)

    res_tab1, res_tab4, res_tab5, res_tab3, res_tab6 = st.tabs(["📊 Docking Scores", "💊 Pharmacokinetics", "🔬 Binding Site", "🧊 3D Viewer", "🔧 Lead Opt"])


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
        _TL = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444", "grey": "#6b7280"}
        _SYM = {"green": "\u2713", "yellow": "~", "red": "\u2717", "grey": "?"}

        def _badge(color, label, val):
            c, s = _TL.get(color, "#6b7280"), _SYM.get(color, "?")
            return (f'<span style="background:{c};color:#fff;border-radius:4px;'
                    f'padding:1px 7px;font-size:0.75rem;font-weight:700;margin-right:4px">{s}</span>'
                    f'<span style="font-size:0.85rem">{label}: <b>{val}</b></span>')

        def _risk(flag, label, good_when_true=False):
            if flag is None: return ""
            bad = (not flag) if good_when_true else flag
            if bad:
                style = "background:#ef4444;color:#fff"
            else:
                style = "background:#052e16;color:#4ade80;border:1px solid #16a34a"
            return (f'<span style="{style};border-radius:4px;'
                    f'padding:2px 8px;font-size:0.75rem;font-weight:700;margin-right:6px">{label}</span>')


        _adm_cols = [c for c in ["Ligand", "MW", "LogP", "TPSA", "HBD", "HBA", "QED",
                                   "Lipinski Fails", "Rotatable Bonds", "Aromatic Rings", "Fsp3"]
                     if c in sorted_df.columns]
        if len(_adm_cols) > 1:
            _adf = sorted_df[_adm_cols].copy()
            for _c in ["MW", "LogP", "TPSA", "QED", "Fsp3"]:
                if _c in _adf.columns:
                    _adf[_c] = _adf[_c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "\u2014")
            for _c in ["HBD", "HBA", "Lipinski Fails", "Rotatable Bonds", "Aromatic Rings"]:
                if _c in _adf.columns:
                    _adf[_c] = _adf[_c].apply(lambda x: str(int(x)) if pd.notna(x) else "\u2014")
            st.dataframe(_adf, use_container_width=True, height=280)

            st.markdown("### \U0001f9ea Molecular Profile Inspector")
            _lig_opts_raw = [_r.get("Ligand", _r.get("name","?"))
                             for _r in res if _r["status"] == "success"
                             and admet_data.get(_r.get("Ligand", _r.get("name","?")))]
            _lig_opts = list(dict.fromkeys(_lig_opts_raw))  # Deduplicate while preserving order
            
            if _lig_opts:
                _sel = st.selectbox("Select molecule to inspect", _lig_opts, key="admet_sel")
                _ad  = admet_data.get(_sel, {})
                _tf  = _ad.get("_traffic", {})
                _r_match = next((_r for _r in res if _r.get("Ligand", _r.get("name")) == _sel), {})
                _vv  = _r_match.get("vina_score")
                _vs  = f"{_vv:.2f} kcal/mol" if _vv is not None else "\u2014"

                # Build property badges row
                _bh = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:0.8rem 0'>"
                for _k, _lbl in [("MW","MW"),("LogP","LogP"),("TPSA","TPSA"),
                                  ("HBD","HBD"),("HBA","HBA"),("QED","QED"),("Fsp3","Fsp3")]:
                    _v = _ad.get(_k)
                    _vstr = f"{_v:.2f}" if isinstance(_v, float) else str(_v if _v is not None else "\u2014")
                    _col = _TL.get(_tf.get(_k, "grey"), "#6b7280")
                    _sym = _SYM.get(_tf.get(_k, "grey"), "?")
                    _bh += (f"<div style='padding:6px 14px;background:#1a1a1a;border-radius:8px;"
                            f"border:2px solid {_col};min-width:90px;text-align:center'>"
                            f"<div style='color:{_col};font-size:1rem;font-weight:800'>{_sym}</div>"
                            f"<div style='font-size:0.7rem;color:#aaa;margin:1px 0'>{_lbl}</div>"
                            f"<div style='font-size:0.9rem;font-weight:600;color:#fff'>{_vstr}</div>"
                            f"</div>")
                _bh += "</div>"
                st.markdown(_bh, unsafe_allow_html=True)

                # Extra descriptor metrics
                _c1, _c2, _c3, _c4, _c5 = st.columns(5)
                _c1.metric("Vina Score",       _vs)
                _c2.metric("Rotatable Bonds",  _ad.get("Rotatable Bonds", "\u2014"))
                _c3.metric("Aromatic Rings",   _ad.get("Aromatic Rings",  "\u2014"))
                _c4.metric("Formal Charge",    _ad.get("Formal Charge",   "\u2014"))
                _c5.metric("Lipinski Fails",   _ad.get("Lipinski Fails",  "\u2014"))

                # Toxicity flags
                st.markdown("**Toxicity / Safety Flags**")
                _rh = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin:0.4rem 0'>"
                _rh += _risk(_ad.get("hERG Risk"),           "hERG Risk",       good_when_true=False)
                _rh += _risk(_ad.get("Hepatotoxicity Risk"), "Hepatotoxicity",  good_when_true=False)
                _rh += _risk(_ad.get("Mutagenicity Risk"),   "Mutagenicity",    good_when_true=False)
                _rh += _risk(_ad.get("BBB Penetration"),     "BBB Penetration", good_when_true=True)
                _rh += "</div>"
                if _ad.get("PAINS Alert"):
                    _rh += (f'<div style="background:#7f1d1d;color:#fca5a5;border-radius:6px;'
                            f'padding:6px 14px;font-size:0.82rem;margin-top:6px">'
                            f'\u26a0\ufe0f PAINS: <b>{_ad["PAINS Alert"]}</b></div>')
                st.markdown(_rh, unsafe_allow_html=True)

            st.caption("""
            \U0001f7e2 Ideal  \u00b7  \U0001f7e1 Marginal  \u00b7  \U0001f534 Out of range  \u00b7
            hERG: LogP>4.5 + MW>450  \u00b7  BBB: MW<450 + LogP 1\u20135 + TPSA<90  \u00b7  PAINS: Pan-Assay Interference
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

            ctrl1, ctrl2 = st.columns([3, 1])
            with ctrl1:
                view_style = st.radio("Protein Style", ["Cartoon","Surface","Sticks","Spheres"], horizontal=True)
            with ctrl2:
                hydro_on = st.toggle("💧 HydroDock Waters", value=False, key="hydro_toggle")

            r_match = next((r for r in ok_results if r["Receptor"]==sel_r and r["name"]==sel_l), None)
            if r_match:
                # HydroDock water analysis
                waters = None
                if hydro_on:
                    orig_pdbs = st.session_state.get("orig_pdbs", {})
                    orig_pdb  = orig_pdbs.get(sel_r)
                    boxes     = st.session_state.get("boxes", {})
                    box       = next(iter(boxes.get(sel_r, list(boxes.values()) if boxes else [{}])), {})


                    if orig_pdb and Path(orig_pdb).exists() and box:
                        with st.spinner("Analysing crystallographic waters..."):
                            waters = analyze_waters(orig_pdb, box)
                        # Surface any error
                        errors = [w["_error"] for w in (waters or []) if "_error" in w]
                        waters = [w for w in (waters or []) if "_error" not in w]
                        if errors:
                            st.error(f"analyze_waters error: {errors[0]}")
                        if waters:

                            n_stable   = sum(1 for w in waters if w["stable"])
                            n_unstable = len(waters) - n_stable
                            wc1, wc2, wc3 = st.columns(3)
                            wc1.metric("Waters in pocket", len(waters))
                            wc2.metric("🟢 Stable",       n_stable,   help="H-bonded ≥2 times — risky to displace")
                            wc3.metric("🔴 Displaceable", n_unstable, help="Weakly coordinated — ligands gain energy displacing these")
                            if n_unstable:
                                st.success(f"🔴 {n_unstable} displaceable water(s) found — ligands that occupy this space may gain binding affinity.")
                        else:
                            st.info("No crystallographic waters found inside the docking box. The PDB may have had waters removed during prep.")
                    else:
                        st.warning("Original PDB not available for water analysis. Run docking again to enable HydroDock.")


                render_3d(r_match["rec_pdb"], r_match["pose_sdf"], view_style, waters=waters)

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

    with res_tab6:
        st.markdown("### \U0001f9ec Analog Designer")
        st.caption("Edit the SMILES of a docked ligand, then run a fast mini-dock to compare its affinity against the original.")

        ok_lo = [r for r in res if r["status"] == "success"]
        if not ok_lo:
            st.info("Run a docking experiment first.")
        else:
            lo_ligs = list({r.get("Ligand", r.get("name","?")) for r in ok_lo})
            lo_sel  = st.selectbox("Select parent ligand", lo_ligs, key="lo_lig")
            lo_r    = next((r for r in ok_lo if r.get("Ligand", r.get("name")) == lo_sel), None)

            if lo_r:
                orig_score = lo_r.get("vina_score")
                orig_smiles = lo_r.get("smiles", "")

                # Try to get SMILES from the pose SDF if not stored
                if not orig_smiles and lo_r.get("pose_sdf") and Path(lo_r["pose_sdf"]).exists():
                    try:
                        from rdkit.Chem import SDMolSupplier, MolToSmiles
                        _supp = SDMolSupplier(lo_r["pose_sdf"], removeHs=True)
                        _m = next((m for m in _supp if m), None)
                        if _m: orig_smiles = MolToSmiles(_m)
                    except Exception:
                        pass

                # Structure preview
                if orig_smiles:
                    try:
                        from rdkit.Chem import MolFromSmiles
                        from rdkit.Chem.Draw import rdMolDraw2D
                        import base64
                        _pm = MolFromSmiles(orig_smiles)
                        if _pm:
                            _d = rdMolDraw2D.MolDraw2DSVG(300, 200)
                            _d.drawOptions().addStereoAnnotation = True
                            _d.DrawMolecule(_pm)
                            _d.FinishDrawing()
                            _svg = _d.GetDrawingText()
                            _b64 = base64.b64encode(_svg.encode()).decode()
                            st.markdown(
                                f'<div style="display:flex;justify-content:center;background:#f8f8ff;'
                                f'border-radius:10px;padding:10px;margin-bottom:8px">'
                                f'<img src="data:image/svg+xml;base64,{_b64}" width="300"/></div>',
                                unsafe_allow_html=True)
                    except Exception:
                        pass

                # Quick-swap substituent buttons
                st.markdown("**⚡ Quick Modifications** — appends fragment to current SMILES:")
                _frags = {
                    "-F": "F", "-Cl": "Cl", "-Br": "Br", "-OH": "O",
                    "-CH₃": "C", "-CF₃": "C(F)(F)F", "-OCH₃": "OC",
                    "-NH₂": "N", "-CN": "C#N", "-NO₂": "[N+](=O)[O-]",
                }
                # staging pattern: apply BEFORE widget renders to avoid post-render mutation error
                if "lo_smiles_staging" in st.session_state:
                    st.session_state["lo_smiles_box"] = st.session_state.pop("lo_smiles_staging")
                elif "lo_smiles_box" not in st.session_state:
                    st.session_state["lo_smiles_box"] = orig_smiles

                _fcols = st.columns(len(_frags))
                for _fc, (_flbl, _fsmi) in zip(_fcols, _frags.items()):
                    if _fc.button(_flbl, key=f"frag_{_flbl}", use_container_width=True):
                        st.session_state["lo_smiles_staging"] = st.session_state.get("lo_smiles_box", orig_smiles) + f".{_fsmi}"
                        st.rerun()

                # SMILES editor
                analog_smiles = st.text_area(
                    "Analog SMILES (edit freely)",
                    height=80, key="lo_smiles_box",
                    help="Modify the SMILES to design your analog. Must be a valid RDKit SMILES."
                )
                if st.button("\U0001f4a1 Reset to original", key="lo_reset"):
                    st.session_state["lo_smiles_staging"] = orig_smiles
                    st.rerun()


                # Validate
                analog_mol = None
                try:
                    from rdkit.Chem import MolFromSmiles
                    analog_mol = MolFromSmiles(analog_smiles)
                    if analog_mol:
                        st.success(f"Valid SMILES \u2713  ({analog_mol.GetNumAtoms()} heavy atoms)")
                    else:
                        st.error("Invalid SMILES — check syntax.")
                except Exception as ex:
                    st.error(f"SMILES parse error: {ex}")

                # Mini-dock button
                if analog_mol and st.button("\U0001f680 Dock Analog (fast)", type="primary", use_container_width=True, key="lo_dock"):
                    try:
                        import tempfile, subprocess
                        from rdkit.Chem import AllChem, MolToMolBlock
                        from rdkit.Chem import AddHs

                        with st.spinner("Preparing analog & running mini-dock..."):
                            # Sanitize: keep only the largest fragment
                            # (disconnected SMILES like 'Aspirin.F' crash Vina PDBQT parser)
                            from rdkit.Chem import rdmolops, MolToSmiles
                            _frags_mol = rdmolops.GetMolFrags(analog_mol, asMols=True)
                            if len(_frags_mol) > 1:
                                analog_mol = max(_frags_mol, key=lambda m: m.GetNumHeavyAtoms())
                                st.warning(f"Disconnected SMILES detected — keeping largest fragment "
                                           f"({analog_mol.GetNumHeavyAtoms()} heavy atoms) for docking. "
                                           f"Edit the SMILES directly to fuse substituents into the scaffold.")

                            # 3D embed
                            _am = AddHs(analog_mol)
                            AllChem.EmbedMolecule(_am, AllChem.ETKDGv3())
                            AllChem.MMFFOptimizeMolecule(_am)
                            _td = tempfile.mkdtemp(prefix="lo_")
                            _sdf_in  = os.path.join(_td, "analog.sdf")
                            _pdbqt_l = os.path.join(_td, "analog.pdbqt")
                            with open(_sdf_in, "w") as _f: _f.write(MolToMolBlock(_am))


                            # SDF → PDBQT via obabel
                            _ob_paths = [
                                shutil.which("obabel"),
                                "/home/rajan/miniconda3/envs/stratadock/bin/obabel",
                            ]
                            _ob = next((p for p in _ob_paths if p and Path(p).exists()), None)
                            if not _ob: raise RuntimeError("obabel not found")
                            subprocess.run([_ob, _sdf_in, "-O", _pdbqt_l, "--gen3d"],
                                           capture_output=True, timeout=30)
                            if not Path(_pdbqt_l).exists():
                                raise RuntimeError("obabel ligand conversion failed")

                            # Derive receptor PDBQT from rec_pdb's work directory
                            _work_dir  = os.path.dirname(lo_r.get("rec_pdb", ""))
                            _rec_pdbqt = os.path.join(_work_dir, "receptor.pdbqt") if _work_dir else None
                            if not _rec_pdbqt or not Path(_rec_pdbqt).exists():
                                raise RuntimeError(f"Receptor PDBQT not found at: {_rec_pdbqt}")
                            _boxes = st.session_state.get("boxes", {})
                            _box   = next(iter(_boxes.get(lo_r.get("Receptor",""), list(_boxes.values()) if _boxes else [{}])), {})
                            if not _box:
                                raise RuntimeError("Docking box not found in session state.")


                            # Run Vina
                            from vina import Vina
                            _v = Vina(sf_name="vina", verbosity=0)
                            _v.set_receptor(_rec_pdbqt)
                            _v.set_ligand_from_file(_pdbqt_l)
                            _v.compute_vina_maps(
                                center=[_box["cx"], _box["cy"], _box["cz"]],
                                box_size=[_box["sx"], _box["sy"], _box["sz"]]
                            )
                            _v.dock(exhaustiveness=4, n_poses=1)
                            _out = os.path.join(_td, "out.pdbqt")
                            _v.write_poses(_out, n_poses=1, overwrite=True)

                            # Parse score
                            _ascore = None
                            if Path(_out).exists():
                                for _ln in Path(_out).read_text().splitlines():
                                    if "REMARK VINA RESULT" in _ln:
                                        try: _ascore = float(_ln.split()[3]); break
                                        except Exception: pass
                            if _ascore is None:
                                _e = _v.energies(n_poses=1)
                                if _e and _e[0][0] != 0.0: _ascore = _e[0][0]

                        # Show comparison
                        _cc1, _cc2, _cc3 = st.columns(3)

                        _cc1.metric("Original (" + lo_sel + ")",
                                    f"{orig_score:.2f} kcal/mol" if orig_score else "—")
                        _cc2.metric("Analog affinity",
                                    f"{_ascore:.2f} kcal/mol" if _ascore else "—")
                        _delta = (_ascore - orig_score) if (_ascore and orig_score) else None
                        _cc3.metric("ΔAffinity",
                                    f"{_delta:+.2f} kcal/mol" if _delta is not None else "—",
                                    delta=f"{_delta:+.2f}" if _delta is not None else None,
                                    delta_color="inverse")
                        if _delta and _delta < 0:
                            st.success(f"\U0001f31f Analog is **{abs(_delta):.2f} kcal/mol stronger** than the parent!")
                        elif _delta and _delta > 0:
                            st.warning(f"\U0001f4c9 Analog is **{_delta:.2f} kcal/mol weaker** — try a different modification.")

                    except Exception as _ex:
                        st.error(f"Mini-dock failed: {_ex}")


    
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
    st.markdown('<div class="page-sub">Learn how to use StrataDock and interpret its advanced computational outputs.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Quick Start", "🧠 GNINA AI & Scoring", "⚙️ Features & Pipeline",
        "❓ FAQ", "💾 Save/Load Project"
    ])


    with tab1:
        st.markdown("""
### 🚀 How to run a docking job

**Step 1 — Upload**
- Upload your **receptor(s)** as `.pdb` files, OR use **🧬 Fold-to-Dock** (right column) to predict a 3D structure directly from a protein sequence — no PDB file needed.
- Upload your **ligand(s)** as `.sdf` (3D structures) or `.smi` / `.txt` (SMILES strings).
- **Defining the Docking Box:**
    - **Option A:** Upload a Reference Ligand and click **Compute Autobox** — the box centers on the ligand.
    - **Option B:** Click **Suggest Pockets (Auto)** — StrataDock runs geometric cavity detection (FPocket-inspired) to find the most druggable binding sites and snaps boxes around them automatically.

**Step 2 — Settings**
- Choose your **docking engine**: AutoDock Vina (CPU, recommended) or GNINA (CNN-based, GPU-accelerated).
- Configure **Protein Preparation** — StrataDock can automatically:
    - Remove water molecules and heteroatoms from crystal structures
    - Add missing hydrogen atoms at physiological pH
    - Repair missing atoms and model missing loops (PDBFixer)
    - Assign protonation states via propka
    - Run energy minimization via OpenMM Amber14
- Adjust molecular parameters. **Default values are optimized for high-throughput screening.**

**Step 3 — Run & Results**
- Click **Run Docking**. The cinematic console streams engine logs in real-time.
- Protein preparation runs automatically before docking.
- For bulk / N × M cross-docking, use **Parallelization** settings to distribute across CPU cores.
- After running, results appear across **5 tabs**:
    - 📊 **Docking Scores** — ranked poses, affinity table, downloadable poses
    - 💊 **ADMET** — pharmacokinetics and drug-likeness prediction
    - 🧫 **Binding Site** — interacting residues and contact map
    - 💧 **3D Viewer** — interactive 3D view with HydroDock water analysis overlay
    - 🔧 **Lead Opt** — R-group analog generator and mini-dock workbench
- Download the **ZIP archive** — organized by receptor with poses, complex PDBs, interaction CSVs, and `summary.txt`.
        """)

    with tab2:
        st.markdown("### Understanding the scores")
        st.warning("**Crucial Note:** GNINA outputs completely different metrics than standard AutoDock. Do not mix them up!")

        st.markdown("""
**Vina Forcefield** *(kcal/mol)*
> The estimated binding free energy from the empirical physics engine.
> **More negative = stronger binding.** A score below −7 kcal/mol is generally a good binder.
> *Note: If Flexible Docking is enabled, this score may go positive due to deformation penalties — trust CNN Pose Probability instead.*

**CNN Pose Probability** *(GNINA only)*
> A 0–1 probability from GNINA's convolutional neural network estimating how likely a pose represents a **true biological binding mode**.
> **Above 0.5 = highly confident.** This score is often more meaningful than Vina for flexible docking.

**CNN Affinity** *(GNINA only)*
> The CNN's empirical prediction of binding affinity (pKd). **Higher is better.**

**pLDDT Score** *(Fold-to-Dock)*
> Per-residue confidence from ESMFold. Range 0–100. ≥70 = high confidence (reliable structure), 50–70 = medium, <50 = low (disordered region).
        """)

    with tab3:
        st.markdown("### ⚙️ Features & Pipeline")
        st.markdown("""
#### 🧬 Fold-to-Dock — AI Structure Prediction
Located on the **Upload page** (right column). Click **🧬 Click to Fold** to open the dialog.
- Paste any protein sequence → ESMFold (Meta AI) predicts a 3D structure in ~10–30 s, for free
- The **pLDDT bar** shows per-residue confidence (≥70 = high, 50–70 = medium, <50 = disordered)
- Click **📥 Load as Receptor** to inject into the pipeline. Works best under 400 AA.
- *Example:* `LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLS` (Villin HP36, should score ~90)

---

#### 💊 ADMET — Pharmacokinetics & Drug-likeness
Available in **Results → Pharmacokinetics** tab. Computes Lipinski Rule of 5, TPSA, LogP, HBD/HBA, MW, rotatable bonds via RDKit/SwissADME. Color-coded pass/fail helps prioritize orally bioavailable candidates.

---

#### 💧 HydroDock — Binding Site Water Analysis
Available in **Results → 3D Viewer** — toggle **Show HydroDock Waters**.
- Classifies waters as **Stable** (structural H-bonds, blue spheres) or **Displaceable** (loose, orange spheres)
- Displacing unstable waters with ligand substituents is a classic affinity-boosting strategy

---

#### 🔧 Lead Optimization R-Group Workbench
Available in **Results → Lead Opt** after docking.
1. Select a ligand → view its 2D structure
2. Edit SMILES or click **Quick Mod** buttons (─F, ─CH₃, ─OH, ─NH₂, ─CF₃)
3. **Run Mini-Dock** — fast Vina job (exhaustiveness=4) on the analog
4. Side-by-side **Δ affinity comparison** vs. the original

*Tips: F improves metabolic stability · CH₃ adds steric selectivity · OH creates new H-bonds*

---

#### 🧫 Binding Site Analysis
Available in **Results → Binding Site**. Lists all residues within contact distance of each pose, annotated with interaction type (H-bond, hydrophobic, charged). Use to understand your SAR.

---

#### 🔩 Receptor Flexibility (`--flexdist`)
Enabling **Flexible Docking** unfreezes side-chains within 3.5 Å of the pocket for induced-fit simulation.
> ⚠️ With flexdist on, Vina scores often go positive due to deformation penalties — always use **CNN Pose Probability** as the primary metric instead.

---

#### 🤖 CNN Modes *(GNINA only)*
- **none** — pure Vina (fast)
- **rescore** — Vina poses, CNN grading (default)
- **refinement** — CNN steers pose search (slow, most accurate)

---

#### 🏗️ Protein Preparation Pipeline
Runs automatically before every dock: remove water/HETATM → add H at pH 7 → repair atoms/loops (PDBFixer) → propka protonation → OpenMM Amber14 minimization.
        """)

    with tab4:
        faqs = [
            ("My ligand failed preparation — what do I do?",
             "Some molecules cannot be embedded in 3D by RDKit (e.g. complex metal complexes, unusual valences). Try increasing 'Max embedding attempts' in Settings, or check that your SMILES/SDF is valid. You can validate SMILES at rdkit.org."),
            ("What is a good Vina score?",
             "Below −6 kcal/mol is a weak binder, −7 to −9 is moderate, and below −9 is strong. These are rough guidelines — always compare within your own screening set. (Does not apply to flexible docking with flexdist enabled.)"),
            ("How does the 'Suggest Pocket (Auto)' feature work?",
             "StrataDock runs a geometric cavity-detection algorithm based on fpocket principles. It constructs an alpha-shape of the receptor, maps deep crevices using Voronoi vertices, filters by hydrophobicity and volume, and auto-selects the largest, most druggable pockets."),
            ("Why does GNINA not work on my machine?",
             "GNINA is GPU-accelerated. It can fall back to CPU (very slow). If you don't have an NVIDIA GPU, set CNN Mode to `none` to use pure Vina scoring which is significantly faster."),
            ("How do I run on an NVIDIA server?",
             "Run `bash setup.sh` on the server. StrataDock auto-detects NVIDIA GPUs and enables GNINA automatically."),
            ("What does a low pLDDT score mean for Fold-to-Dock?",
             "pLDDT < 50 means ESMFold has low confidence — the region is likely intrinsically disordered and won't produce reliable docking results. Use well-folded proteins (enzymes, kinases, receptors typically score ≥ 70)."),
            ("When should I use HydroDock?",
             "HydroDock is most useful when your receptor PDB contains crystallographic waters (e.g. from RCSB PDB structures). It identifies displaceable waters — positions ideal for new ligand substituents to improve affinity."),
            ("Can I use Lead Opt without running a full dock first?",
             "No — Lead Opt requires a completed docking result to extract the ligand pose and box. Run at least one docking job first."),
        ]
        faq_html = ""
        for q, a in faqs:
            faq_html += f"""
            <details class="faq-item">
                <summary>{q}</summary>
                <div class="faq-body">{a}</div>
            </details>"""
        st.markdown(faq_html, unsafe_allow_html=True)

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
            st.download_button("📥 Click here to download (.stratadock)", b64_str, "project.stratadock", use_container_width=True, key="help_dl_btn")

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
