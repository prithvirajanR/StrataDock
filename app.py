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

# =============================================================================
# CSS — minimal, no expander overrides
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
*, html, body { 
    font-family: 'Outfit', sans-serif !important; 
    -webkit-font-smoothing: antialiased; 
    -moz-osx-font-smoothing: grayscale;
}

/* ── App Background ───────────────────────────────────────── */
.stApp { 
    background: #0a0c12;
    color: #e2e8f0;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
::-webkit-scrollbar-thumb { background: rgba(0, 198, 255, 0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 198, 255, 0.6); }

section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 2.5rem 2rem 4rem; max-width: 95% !important; 
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Logo bar ───────────────────────────────────────────── */
.logo-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 0 0; margin-bottom: 2.5rem;
}
.logo-left { display:flex; align-items:center; gap:20px; }
.logo-icon {
    width:78px; height:78px;
    background: linear-gradient(145deg, rgba(12, 18, 30, 0.8), rgba(6, 10, 18, 0.6));
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-top: 1px solid rgba(0, 198, 255, 0.5);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-radius: 20px; display:flex; align-items:center;
    justify-content:center; flex-shrink:0; font-size:2.4rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.4), inset 0 2px 10px rgba(0, 198, 255, 0.15);
    position: relative; overflow: hidden;
}
.logo-icon::after {
    content:''; position:absolute; top:0; left:-100%; width:30%; height:100%;
    background: linear-gradient(90deg, transparent, rgba(0, 198, 255, 0.4), transparent);
    animation: shineIcon 6s infinite ease-in;
}
@keyframes shineIcon { 
    0% { left: -100% } 
    15% { left: 100% } 
    100% { left: 100% } 
}
.logo-name { 
    font-size:2.1rem; font-weight:800; 
    background: linear-gradient(135deg, #ffffff 10%, #00c6ff 50%, #ffffff 90%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing:-0.03em; margin-bottom: 2px;
    animation: shineText 6s infinite linear;
}
@keyframes shineText {
    0%   { background-position: -50% center; }
    12%  { background-position: -50% center; }
    25%  { background-position: 200% center; }
    100% { background-position: 200% center; }
}
.logo-sub  { font-size:0.92rem; color:#94a3b8; letter-spacing:0.04em; font-weight:500; text-transform: uppercase; }
.logo-gpu {
    display:flex;align-items:center;gap:8px;
    padding:0.4rem 1.2rem;
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
}
.logo-gpu:hover { background: rgba(15, 23, 42, 0.65); border-color: rgba(255, 255, 255, 0.2); transform: translateY(-1px); }
.logo-gpu-dot { width:8px;height:8px;border-radius:50%; display:inline-block; box-shadow: 0 0 10px currentColor; }
.logo-gpu-text { font-size:0.8rem; font-weight: 600; color:#cbd5e1; }

.eng-badge.cpu { background:rgba(28, 36, 51, 0.8); color:#58a6ff; border:1px solid #2d4a7a; }
.eng-badge.gpu { background:rgba(13, 45, 31, 0.8); color:#3fb950; border:1px solid #1f4a2a; }

/* ── Typography headers ──────────────────────────────────── */
.page-title {
    font-size: 2rem; font-weight: 700; color: #ffffff;
    margin-bottom: 0.25rem; line-height: 1.2; letter-spacing: -0.01em;
}
.page-sub {
    font-size: 0.95rem; color: #94a3b8; margin-bottom: 2rem; font-weight: 300;
}

/* ── Ultra-Sleek Glassmorphism Cards ─────────────────────── */
.card {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.6), rgba(11, 15, 25, 0.8));
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-top: 1px solid rgba(0, 198, 255, 0.25);
    border-radius: 18px; padding: 1.8rem; margin-bottom: 1.4rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease, border-color 0.4s ease;
}
.card:hover { 
    transform: translateY(-5px); 
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.2); 
    border-top-color: rgba(0, 198, 255, 0.6); 
    border-color: rgba(255,255,255,0.12);
}
.card-title { font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.9rem; letter-spacing: 0.02em; }

/* ── Input elements styling ──────────────────────────────── */
/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.03em !important; font-size: 1rem !important; padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { 
    transform: translateY(-2px) scale(1.01) !important; 
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5) !important; 
    color: #ffffff !important; 
}
.stButton > button:active { transform: translateY(1px) !important; box-shadow: 0 2px 10px rgba(0, 114, 255, 0.3) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(21, 30, 47, 0.7), rgba(11, 16, 26, 0.9));
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-left: 3px solid #00cefc;
    border-radius: 16px; padding: 1.4rem 1.6rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
[data-testid="stMetric"]:hover { 
    transform: translateY(-5px) scale(1.02); 
    border-color: rgba(0, 198, 255, 0.4); 
    box-shadow: 0 12px 40px rgba(0, 198, 255, 0.15); 
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight:500 !important; }
[data-testid="stMetricValue"] { 
    font-size: 1.8rem !important; font-weight: 700 !important; 
    background: linear-gradient(135deg, #ffffff, #94a3b8); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
}

/* Upload zones */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.3) !important;
    backdrop-filter: blur(10px) !important; -webkit-backdrop-filter: blur(10px) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { 
    border-color: #00c6ff !important; 
    background: rgba(0, 198, 255, 0.05) !important; 
    transform: scale(1.02) !important;
    box-shadow: 0 0 20px rgba(0, 198, 255, 0.2) inset !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    border-radius: 6px !important;
}

/* Alerts */
.stAlert { 
    border-radius: 14px !important; 
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.7), rgba(220, 38, 38, 0.05)) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-left: 3px solid #f85149 !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2) !important;
}

/* ── Cinematic Terminal ─────────────────────────────────── */
.terminal-block {
    background: linear-gradient(180deg, rgba(5, 8, 15, 0.9), rgba(11, 16, 26, 0.95));
    border: 1px solid rgba(0, 198, 255, 0.15);
    border-top: 1px solid rgba(0, 198, 255, 0.4);
    border-radius: 16px;
    padding: 1.2rem;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 0.88rem;
    color: #3fb950;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 2px 20px rgba(0, 198, 255, 0.05);
    height: 320px;
    overflow-y: auto;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column-reverse;
}
.terminal-block .err { color: #f85149; }
.terminal-block .warn { color: #d29922; }
.terminal-block .info { color: #58a6ff; }
.terminal-block::-webkit-scrollbar { width: 6px; }
.terminal-block::-webkit-scrollbar-track { background: transparent; }
.terminal-block::-webkit-scrollbar-thumb { background: rgba(0, 198, 255, 0.5); border-radius: 3px; }

/* ── Engine choice cards ──────────────────────────────── */
.eng-card {
    border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.4rem;
    transition: all .35s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.5), rgba(11, 15, 25, 0.7)); 
    backdrop-filter: blur(16px);
    margin-bottom: 0.8rem; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.eng-card.selected { 
    border-color: rgba(0, 198, 255, 0.5); 
    border-top: 1px solid rgba(0, 198, 255, 0.8);
    background: linear-gradient(145deg, rgba(0, 114, 255, 0.15), rgba(0, 198, 255, 0.05)); 
    box-shadow: 0 12px 35px rgba(0, 114, 255, 0.25), inset 0 2px 10px rgba(0, 198, 255, 0.1); 
    transform: translateY(-2px);
}
.eng-card:hover:not(.selected) { 
    border-color: rgba(255,255,255,0.2); 
    transform: translateY(-3px); 
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}
.eng-title { font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:0.3rem; }
.eng-sub   { font-size:0.8rem; color:#94a3b8; }
.eng-badge {
    display:inline-block; font-size:0.7rem; font-weight:600; padding:3px 8px;
    border-radius:12px; margin-left:8px; vertical-align:middle;
}

/* Box display */
.box-grid { display: grid; grid-template-columns: repeat(6,1fr); gap: 10px; margin-top:1rem; }
.box-cell {
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 12px 6px; text-align: center;
    transition: all 0.2s ease;
}
.box-cell:hover { border-color: rgba(255,255,255,0.2); }
.box-val { font-size: 1.05rem; font-weight: 700; color: #00c6ff; }
.box-lbl { font-size: 0.68rem; color: #94a3b8; margin-top: 3px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── FAQ accordion ───────────────────────────────────────── */
details.faq-item {
    border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 0 1.2rem;
    margin-bottom: 0.6rem; background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(8px); transition: all 0.2s ease;
}
details.faq-item:hover { border-color: rgba(255,255,255,0.15); background: rgba(15, 23, 42, 0.6); }
details.faq-item summary {
    padding: 1rem 0; cursor: pointer; font-size: 0.9rem;
    font-weight: 600; color: #f1f5f9; list-style: none;
    display: flex; align-items: center; gap: 10px;
}
details.faq-item summary::-webkit-details-marker { display:none; }
details.faq-item summary::before { content:'▶'; font-size:0.65rem; color:#64748b; transition:transform .2s; flex-shrink:0; }
details.faq-item[open] summary::before { transform:rotate(90deg); color: #00c6ff; }
details.faq-item .faq-body { padding: 0 0 1rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.7; }

.sect {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin: 2rem 0 0.8rem;
}

/* Settings row */
.setting-row { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1.5rem; }
.setting-group { background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; }
.setting-group-title { font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 1.2rem; }

/* Help */
.help-card { background: rgba(15, 23, 42, 0.5); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.8rem; margin-bottom: 1.2rem; }
.help-title { font-size: 1.1rem; font-weight: 700; color: #00c6ff; margin-bottom: 0.8rem; }
.help-desc { font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; }
.param-row { display: flex; gap: 1rem; padding: 0.65rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.param-name { font-size: 0.85rem; font-weight: 600; color: #f8fafc; min-width: 170px; }
.param-desc { font-size: 0.85rem; color: #94a3b8; }
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
        "results": [], "boxes": [], "rec_path": None,
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
        "flexdist": False, "cnn_mode": "none",
        # Vina
        "v_exhaustiveness": 8, "v_num_poses": 9,
        "energy_range": 3, "scoring_fn": "vina", "vina_seed": 0,
        # Box
        "box_padding": 8,
        "box_padding": 8,
        # Default target pH
        "target_ph": 7.4,
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
    ("Upload",   "1", "Upload"),
    ("Settings", "2", "Settings"),
    ("Run",      "3", "Run & Results"),
    ("Help",     "?", "Help"),
]

def render_stepper():
    """Logo bar + pill-button nav row."""

    # Logo bar
    gpu_color = {"nvidia":"#3fb950","cpu":"#6e7681"}.get(gpu["mode"], "#6e7681")
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
                <div class="logo-name">StrataDock</div>
                <div class="logo-sub">Alpha v0.7</div>
            </div>
        </div>
        <div class="logo-gpu">
            <span class="logo-gpu-dot" style="background:{gpu_color}"></span>
            <span class="logo-gpu-text">{gpu["detail"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav pill buttons
    order   = [s[0] for s in STEPS]
    cur_idx = order.index(st.session_state.page) if st.session_state.page in order else 0
    done    = {"Upload":   st.session_state.rec_bytes is not None and bool(st.session_state.lig_files_data),
               "Settings": bool(st.session_state.boxes),
               "Run":      bool(st.session_state.results),
               "Help":     False}

    cols = st.columns(len(STEPS) + 3)   # extra spacer cols to left-align pills
    btn_cols = cols[:len(STEPS)]
    for col, (i, (key, num, label)) in zip(btn_cols, enumerate(STEPS)):
        with col:
            is_active = (key == st.session_state.page)
            is_done   = (i < cur_idx) or done.get(key, False)
            prefix = "✓ " if is_done and not is_active else f"{num}. "
            btn_label = prefix + label
            btn_type  = "primary" if is_active else "secondary"
            if st.button(btn_label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                st.session_state.page = key
                st.rerun()

    pct = ((cur_idx + 1) / (len(STEPS) + 3)) * 100
    st.markdown(f"""
    <div style="width: 100%; height: 3px; background: rgba(255,255,255,0.05); margin: 0.8rem 0 1.5rem; border-radius: 2px;">
        <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #00c6ff, #0072ff); 
             box-shadow: 0 0 15px rgba(0, 198, 255, 0.6); border-radius: 2px; 
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
            
        # Reverse the list so the newest line is at the bottom of the flex column-reverse
        display_lines = display_lines[::-1]
        html = f"<div class='terminal-block'>{'<br>'.join(display_lines)}</div>"
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
        for lname, lpdbqt in lig_items:
            for box in boxes:
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
            results.append(res)
            prog.progress((i+1)/len(tasks))
            
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
        st.markdown("**Receptor(s)** — protein PDB file(s)")
        rec_files = st.file_uploader("Receptor PDB(s)", type=["pdb"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed", key="rec_up")
        if rec_files:
            st.session_state.rec_files_data = [(f.name, f.read()) for f in rec_files]
            for f in rec_files: f.seek(0)
            st.success(f"Loaded {len(rec_files)} receptor(s)")

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
                        img = Draw.MolsToGridImage(ms, molsPerRow=4, subImgSize=(200, 200), legends=ns)
                        st.image(img, use_container_width=True)
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
                with tempfile.TemporaryDirectory() as td:
                    rp = os.path.join(td,"rec.pdb")
                    Path(rp).write_bytes(st.session_state.rec_files_data[0][1])
                    rfp = None
                    if ref_file:
                        rfp = os.path.join(td,"ref.sdf")
                        Path(rfp).write_bytes(ref_file.read())
                    try:
                        box = compute_autobox(rp, rfp, st.session_state.box_padding)
                        box["name"] = "Autobox"
                        st.session_state.boxes = [box]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Autoboxing failed: {e}")
                        
    with col_multi:
        st.session_state.multi_pocket = st.toggle("Multi-Pocket (Top 5)", value=st.session_state.get("multi_pocket", False))

    with col_pbtn:
        if st.button("Suggest Pocket (fpocket)", use_container_width=True, help="Automatically detects the best binding cavity"):
            if not getattr(st.session_state, "rec_files_data", None):
                st.error("Upload at least one receptor PDB first.")
            else:
                import subprocess
                with tempfile.TemporaryDirectory() as td:
                    rp = os.path.join(td, "rec.pdb")
                    Path(rp).write_bytes(st.session_state.rec_files_data[0][1])
                    try:
                        subprocess.run(["fpocket", "-f", rp], cwd=td, capture_output=True, check=True)
                        found_boxes = []
                        max_pockets = 5 if st.session_state.multi_pocket else 1
                        
                        for i in range(1, max_pockets + 1):
                            p_file = Path(td) / "rec_out" / "pockets" / f"pocket{i}_atm.pdb"
                            if p_file.exists():
                                bx = compute_autobox(str(p_file), None, st.session_state.box_padding)
                                bx["name"] = f"Pocket {i}"
                                found_boxes.append(bx)
                                
                        if found_boxes:
                            st.session_state.boxes = found_boxes
                            st.rerun()
                        else:
                            st.error("No pockets detected by fpocket.")
                    except Exception as e:
                        st.error(f"fpocket failed: {e}. Is it installed via setup.sh?")

    if st.session_state.boxes:
        for b in st.session_state.boxes:
            st.markdown(f"**{b['name']}**")
            st.markdown(
                '<div class="box-grid" style="margin-bottom: 1rem;">'
                + "".join(f'<div class="box-cell"><div class="box-val">{b[k]:.1f}</div>'
                           f'<div class="box-lbl">{l}</div></div>'
                           for k,l in [("cx","Center X"),("cy","Center Y"),("cz","Center Z"),
                                       ("sx","Size X"),("sy","Size Y"),("sz","Size Z")])
                + "</div>",
                unsafe_allow_html=True
            )
        
        if getattr(st.session_state, "rec_files_data", None):
            st.markdown("<br>**3D Docking Box Preview**", unsafe_allow_html=True)
            render_box_preview(st.session_state.rec_files_data[0][1], st.session_state.boxes)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Receptor",  "Ready" if st.session_state.rec_bytes else "Missing")
    c2.metric("Ligands",   len(st.session_state.lig_files_data) or "Missing")
    c3.metric("Box",       f"{len(st.session_state.boxes)} ready" if st.session_state.boxes else "Not set")
    c4.metric("Engine",    st.session_state.engine)

    st.markdown("")
    if st.session_state.rec_bytes and st.session_state.lig_files_data and st.session_state.boxes:
        if st.button("Next: Settings →", use_container_width=False):
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
        v_cls = "eng-card selected" if vina_selected else "eng-card"
        st.markdown(f"""
        <div class="{v_cls}">
            <div class="eng-title">AutoDock Vina <span class="eng-badge cpu">CPU</span></div>
            <div class="eng-sub">Fast, universal. Works on all CPU systems.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="sel_vina", use_container_width=True,
                     type="primary" if vina_selected else "secondary"):
            st.session_state.engine = "AutoDock Vina"; st.rerun()

    with eng_col2:
        g_cls = "eng-card selected" if gnina_selected else "eng-card"
        g_sub = "AI CNN scoring. NVIDIA GPU not detected (CPU mode)." if gpu_warn else "AI CNN scoring. NVIDIA GPU detected."
        st.markdown(f"""
        <div class="{g_cls}">
            <div class="eng-title">GNINA <span class="eng-badge gpu">GPU / CPU</span></div>
            <div class="eng-sub">{g_sub}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="sel_gnina", use_container_width=True,
                     type="primary" if gnina_selected else "secondary"):
            st.session_state.engine = "GNINA"; st.rerun()

    engine = st.session_state.engine
    st.markdown('<hr style="border:none;border-top:1px solid #1e2530;margin:1.2rem 0 1rem">', unsafe_allow_html=True)

    # Three-column settings layout
    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown("#### Ligand Preparation (RDKit)")
        st.caption("Controls how ligands are converted to 3D structures.")

        st.session_state.target_ph = st.number_input(
            "Target pH (OpenBabel)", min_value=0.0, max_value=14.0, value=st.session_state.get("target_ph", 7.4), step=0.1,
            help="If input is 1D SMILES, OpenBabel will protonate the molecule to match this pH before 3D embedding.")
        st.session_state.embed_method = st.selectbox(
            "Embedding method",
            ["ETKDGv3","ETKDGv2","ETKDG"],
            index=["ETKDGv3","ETKDGv2","ETKDG"].index(st.session_state.embed_method),
            help="Algorithm for generating the 3D shape of the ligand. ETKDGv3 is the most accurate.")
        st.session_state.max_attempts = st.slider(
            "Max embedding attempts", 10, 5000, st.session_state.max_attempts, step=50,
            help="How many times RDKit will retry if 3D generation fails.")
        st.session_state.force_field = st.selectbox(
            "Force field", ["MMFF94","MMFF94s","UFF"],
            index=["MMFF94","MMFF94s","UFF"].index(st.session_state.force_field),
            help="Energy function used to optimise the 3D geometry.")
        st.session_state.ff_steps = st.slider(
            "Force field steps", 0, 2000, st.session_state.ff_steps, step=100,
            help="Number of geometry optimisation steps. Higher = more accurate, slower.")
        st.session_state.add_hs = st.toggle(
            "Add hydrogens", st.session_state.add_hs,
            help="Adds explicit hydrogen atoms before 3D embedding.")
        st.session_state.keep_chirality = st.toggle(
            "Preserve chirality", st.session_state.keep_chirality,
            help="Maintains stereocenters (R/S) from the input molecule.")
            
        if st.checkbox("ℹ️ Learn more about RDKit parameters"):
            st.info("""
            **Embedding Method:** Mathematical algorithm used to fold 2D drugs into 3D space. **ETKDGv3** is the latest industry standard combining distance geometry with experimental torsion-angle knowledge.  
            **Force Field:** A physics engine (like MMFF94) that relaxes the 3D molecule by subtly shifting its atoms to minimize internal strain energy before docking.
            """, icon="🧠")

        st.markdown('<hr style="border:none;border-top:1px solid #1e2530;margin:1.5rem 0 1rem">', unsafe_allow_html=True)
        st.markdown("**PDBQT Conversion** — Meeko")
        st.caption("Controls how the prepared ligand is written to the PDBQT format.")
        st.session_state.merge_nphs = st.toggle(
            "Merge non-polar H", st.session_state.merge_nphs,
            help="Absorbs non-polar hydrogens into their parent heavy atom. Standard practice for AD4 scoring.")
        st.session_state.hydrate = st.toggle(
            "Add hydration waters", st.session_state.hydrate,
            help="Adds explicit water molecules around the ligand for hydrated docking.")
            
        if st.checkbox("ℹ️ Learn more about Meeko parameters"):
            st.info("""
            **Merge non-polar H:** Traditional AutoDock format requires non-polar hydrogen atoms (like those attached to Carbons) to be logically absorbed into their parent atom to speed up computation. **Always leave this ON.**  
            **Add hydration waters:** Wraps the drug in a simulated shell of water molecules. Useful if you suspect the drug binds via a 'water bridge' to the target protein.
            """, icon="💧")

    with col_b:
        if engine == "AutoDock Vina":
            st.markdown("#### AutoDock Vina")
            st.caption("Controls the docking search and scoring.")
            st.session_state.v_exhaustiveness = st.slider(
                "Exhaustiveness", 1, 32, st.session_state.v_exhaustiveness,
                help="How thoroughly Vina searches the binding site. Higher = better results, slower. Default 8 is fine.")
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
                **Exhaustiveness:** The single most important parameter! It commands how many independent physics simulations run in parallel inside the box.  
                - `8`: Lightning fast, great for screening thousands of drugs.  
                - `32`: Extremely slow, but highly accurate for final publications.  
                
                **Energy Range:** Vina only outputs variant poses that are within X kcal/mol of the mathematically best pose. Default is 3. Increase this if you want Vina to show you "worse" but radically different binding angles.
                """, icon="⚡")
        else:
            st.markdown("#### GNINA")
            st.caption("GNINA adds CNN-based pose scoring on top of Vina search.")
            st.session_state.exhaustiveness = st.slider(
                "Exhaustiveness", 1, 64, st.session_state.exhaustiveness,
                help="Search thoroughness.")
            st.session_state.num_poses = st.slider(
                "Poses", 1, 20, st.session_state.num_poses)
            st.session_state.cnn_model = st.selectbox(
                "CNN model", ["default","dense","crossdocked_default2018"],
                index=["default","dense","crossdocked_default2018"].index(st.session_state.cnn_model),
                help="Neural network model for CNN scoring.")
            st.session_state.cnn_mode = st.selectbox(
                "CNN Mode", ["none", "rescore", "refinement", "all"],
                index=["none", "rescore", "refinement", "all"].index(st.session_state.cnn_mode),
                help="none: skip AI scoring. rescore: fast Vina search then AI grading. refinement: AI guides the physical energy minimization (10x slower but higher quality).")
            st.session_state.cnn_weight = st.slider(
                "CNN scoring weight", 0.0, 1.0, st.session_state.cnn_weight, 0.05,
                help="1.0 = pure CNN, 0.0 = pure Vina score for pose ranking.")
            st.session_state.min_rmsd = st.slider(
                "Min RMSD between poses (A)", 0.5, 3.0, st.session_state.min_rmsd, 0.1)
            st.session_state.gnina_seed = st.number_input("Seed", 0, 99999, st.session_state.gnina_seed)
            st.session_state.gpu_device = st.selectbox(
                "GPU device", ["auto","0","1","cpu"],
                index=["auto","0","1","cpu"].index(st.session_state.gpu_device))
            st.session_state.flexdist = st.toggle(
                "Flexible Docking (Induced Fit)", st.session_state.flexdist,
                help="Automatically allows receptor side-chains within 3.5 Å of the pocket center to move, simulating biological induced fit.")
                
            if st.checkbox("ℹ️ Learn more about GNINA parameters"):
                st.info("""
                **CNN Scoring Weight:** GNINA ranks drugs using two distinct brains: an old-school physics equation (Vina) and a modern Deep Learning Neural Network (CNN).  
                - `0.0`: The Neural Network is ignored. Traditional physics wins.  
                - `1.0`: Pure AI structural recognition.  
                
                **CNN Model:** The specific neural network weights loaded into your GPU. `default` is recommended. `dense` is slower but more thorough. `crossdocked2018` is trained exclusively on protein-ligand pairs that underwent structural shifts during binding.
                """, icon="🤖")

    with col_c:
        st.markdown("#### Docking Box")
        st.caption("The 3D search region. Calculated automatically on the Upload page.")
        st.session_state.box_padding = st.slider(
            "Autobox padding (A)", 1, 20, st.session_state.box_padding,
            help="Extra space (in Angstroms) added around the receptor or reference ligand when computing the box.")
        manual = st.toggle("Manual box override (Pocket 1)", False)
        if manual:
            cc, cs = st.columns(2)
            cbox = st.session_state.boxes[0] if st.session_state.boxes else None
            cx = cc.number_input("Center X", value=cbox["cx"] if cbox else 0.0)
            cy = cc.number_input("Center Y", value=cbox["cy"] if cbox else 0.0)
            cz = cc.number_input("Center Z", value=cbox["cz"] if cbox else 0.0)
            sx = cs.number_input("Size X", value=cbox["sx"] if cbox else 20.0)
            sy = cs.number_input("Size Y", value=cbox["sy"] if cbox else 20.0)
            sz = cs.number_input("Size Z", value=cbox["sz"] if cbox else 20.0)
            # Override all boxes with just 1 manual box
            st.session_state.boxes = [{"name": "Manual", "cx":cx,"cy":cy,"cz":cz,"sx":sx,"sy":sy,"sz":sz}]

        st.markdown('<hr style="border:none;border-top:1px solid #1e2530;margin:1.5rem 0 1rem">', unsafe_allow_html=True)
        st.markdown("**Parallelization**")
        max_cpu = os.cpu_count() or 4
        st.session_state.n_jobs = st.slider(
            "CPU cores", 1, max_cpu, st.session_state.n_jobs, key="njobs_slider",
            help="Number of ligands to dock in parallel. Set to 1 for live progress updates.")
        st.session_state.backend = st.selectbox(
            "Parallel backend", ["loky","threading","multiprocessing"],
            index=["loky","threading","multiprocessing"].index(st.session_state.backend),
            help="loky is the default and most stable. threading can be faster for I/O-bound tasks.")


    st.divider()
    if st.button("Next: Run Docking →"):
        st.session_state.page = "Run"; st.rerun()

def build_pdf_report(df, engine):
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 20)
            self.set_text_color(0, 114, 255)
            self.cell(0, 15, "StrataDock Screening Report", ln=True, align="C")
            self.set_font("helvetica", "", 10)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, f"Engine: {engine} | Top Hits Output", ln=True, align="C")
            self.ln(5)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(30,30,30)
    
    # Table Header
    col_w = [25, 60, 25, 25, 30]
    headers = ["Receptor", "Ligand", "Pocket", "Vina Score", "CNN Score" if engine=="GNINA" else "Status"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 10, h, border=1, align="C")
    pdf.ln()
    
    pdf.set_font("helvetica", "", 9)
    # Sort and take top 50
    df_top = df[df["Status"] == "success"].copy()
    if not df_top.empty:
        sort_col = "CNN Score" if engine=="GNINA" else "Vina Score"
        ascend = engine != "GNINA" # Vina ascends (more negative is better), CNN descends
        df_top = df_top.sort_values(sort_col, ascending=ascend).head(50)
        
        for _, row in df_top.iterrows():
            pdf.cell(col_w[0], 8, str(row["Receptor"])[:12], border=1)
            pdf.cell(col_w[1], 8, str(row["Ligand"])[:30], border=1)
            pdf.cell(col_w[2], 8, str(row.get("Pocket", ""))[:12], border=1)
            pdf.cell(col_w[3], 8, f"{row['Vina Score']:.2f}" if pd.notna(row['Vina Score']) else "-", border=1, align="C")
            
            if engine == "GNINA":
                v3 = f"{row['CNN Score']:.3f}" if pd.notna(row['CNN Score']) else "-"
            else:
                v3 = str(row["Status"])
            pdf.cell(col_w[4], 8, v3, border=1, align="C")
            pdf.ln()
            
    return bytes(pdf.output())


def build_zip_archive(results, df, engine, session_state):
    import zipfile, io
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        # 1. Add CSV
        zf.writestr("stratadock_results.csv", df.to_csv(index=False))
        
        # 2. Add PDF Report
        try: zf.writestr("stratadock_report.pdf", build_pdf_report(df, engine))
        except: pass
        
        # 3. Add Inputs (Receptors)
        pml_recs = []
        if getattr(session_state, "rec_files_data", None):
            for rname, rbytes in session_state.rec_files_data:
                zf.writestr(f"inputs/{rname}", rbytes)
                pml_recs.append(rname)
        
        # 4. Add Outputs (Poses)
        pml_poses = []
        ok_res = [r for r in results if r["status"] == "success" and r.get("pose_sdf") and Path(r["pose_sdf"]).exists()]
        for r in ok_res:
            safe_lname = r["name"]
            sdf_name = f"outputs/{safe_lname}_pose.sdf"
            zf.writestr(sdf_name, Path(r["pose_sdf"]).read_bytes())
            pml_poses.append(f"{safe_lname}_pose.sdf")
            
        # 5. Generate PyMOL Script
        if pml_recs and pml_poses:
            pml = [
                "# StrataDock Auto-Generated PyMOL Visualization Script",
                "bg_color white",
                "set depth_cue, 1",
                "set ray_trace_fog, 1",
                "set antialias, 2"
            ]
            # Load and format receptors
            for r in pml_recs:
                obj_name = r.replace(".pdb","")
                pml.append(f"load inputs/{r}, {obj_name}")
                pml.append(f"hide all, {obj_name}")
                pml.append(f"show cartoon, {obj_name}")
                pml.append(f"color gray80, {obj_name}")
                pml.append(f"set cartoon_transparency, 0.2, {obj_name}")
                pml.append(f"show surface, {obj_name}")
                pml.append(f"set transparency, 0.6, {obj_name}")
            
            # Load and format ligands
            for pose in pml_poses:
                obj_name = pose.replace(".sdf","")
                pml.append(f"load outputs/{pose}, {obj_name}")
                pml.append(f"show sticks, {obj_name}")
                pml.append(f"util.cbas {obj_name}") # Color by atom, carbon=cyan
            
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

    c1, c2, c3, c4 = st.columns([1.5,1.5,1,1])
    c1.metric("Engine", cfg["engine"])
    c2.metric("Ligand files", len(st.session_state.lig_files_data))
    with c3:
        go = st.button("Run Docking", use_container_width=True, type="primary")
    with c4:
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
        stat2.success(f"Done — {ok}/{len(results)} ligands docked successfully")
        if ok: st.snow()

    # ── Results table ─────────────────────────────────────────────────────────
    if st.session_state.results:
        st.divider()
        res = st.session_state.results
        
        # Merge ADMET data into results
        data_rows = []
        for r in res:
            row = {
                "Receptor": r["Receptor"], 
                "Ligand": r["Ligand"], 
                "Pocket": r.get("Pocket", "Autobox"), 
                "Vina Forcefield (kcal/mol)": r.get("vina_score"),
                "CNN Pose Probability (0-1)": r.get("cnn_score"), 
                "CNN Affinity (pKd)": r.get("cnn_affinity"),
                "Status": r.get("status")
            }
            if r["Ligand"] in admet_data:
                row.update(admet_data[r["Ligand"]])
            data_rows.append(row)
            
        df = pd.DataFrame(data_rows)
        ok_df = df[df["Status"]=="success"]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Jobs", len(df))
        c2.metric("Successful", len(ok_df))
        c3.metric("Best Vina", f"{ok_df['Vina Score'].min():.2f} kcal/mol" if ok_df["Vina Score"].notna().any() else "—")
        c4.metric("Best CNN",  f"{ok_df['CNN Score'].max():.3f}"            if ok_df["CNN Score"].notna().any()  else "—")

        st.markdown("")
        col_s, col_f, col_d1, col_d2 = st.columns([1.8, 1.8, 1.2, 1.2])
        with col_s:
            sort_by = st.selectbox("Sort", ["Vina Score (best first)","CNN Score (best first)","Ligand name"],
                                    label_visibility="collapsed")
        with col_f:
            if not ok_df.empty and ok_df["Vina Score"].notna().any():
                min_v, max_v = float(ok_df["Vina Score"].min()), float(ok_df["Vina Score"].max())
                if min_v == max_v: min_v -= 1.0; max_v += 1.0
                vina_filter = st.slider("Filter by Vina Score threshold:", 
                                        min_value=min_v, max_value=max_v, 
                                        value=max_v, step=0.1, label_visibility="collapsed")
            else:
                vina_filter = 999.0
        with col_d1:
            try:
                zip_bytes = build_zip_archive(res, df, st.session_state.engine, st.session_state)
                st.download_button("📦 Full Experiment (.zip)", zip_bytes, "stratadock_experiment.zip", mime="application/zip", use_container_width=True)
            except Exception as e:
                st.error(f"ZIP failed: {e}")
        with col_d2:
            try:
                pdf_bytes = build_pdf_report(df, st.session_state.engine)
                st.download_button("📄 PDF Report Only", pdf_bytes, "stratadock_report.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF failed: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        res_tab1, res_tab4, res_tab2, res_tab3 = st.tabs(["📊 Docking Scores", "💊 Pharmacokinetics", "📈 Binding Analytics", "🧊 3D Viewer"])
        
        # Apply Dynamic Filter
        filtered_df = df[
            ((df["Status"] == "success") & (df["Vina Score"] <= vina_filter)) |
            (df["Status"] != "success")
        ]

        # Base sorting logic
        sm = {"Vina Score (best first)":("Vina Score",True),
              "CNN Score (best first)":("CNN Score",False),
              "Ligand name":("Ligand",True)}
        sc, sa = sm[sort_by]
        sorted_df = filtered_df.sort_values(sc,ascending=sa,na_position="last").copy()

        with res_tab1:
            # Show only Docking columns
            dock_cols = [c for c in ["Receptor", "Ligand", "Pocket", "Vina Score", "CNN Score", "CNN Affinity", "Status"] if c in sorted_df.columns]
            dock_df = sorted_df[dock_cols]
            st.dataframe(
                dock_df.style.format({"Vina Score":"{:.3f}","CNN Score":"{:.4f}","CNN Affinity":"{:.3f}"},na_rep="—"),
                use_container_width=True, height=400
            )

            failed_jobs = [r for r in res if r["status"] != "success"]
            if failed_jobs:
                st.warning(f"{len(failed_jobs)} docking job(s) returned an error")
                for r in failed_jobs:
                    st.error(f"Error log: {r.get('Receptor','?')} \u2794 {r.get('Ligand','?')} ({r.get('name','?')})", icon="🚨")
                    st.code(r["status"], language="text")

        with res_tab4:
            # Show only ADMET columns
            admet_cols = [c for c in ["Ligand", "MW", "LogP", "TPSA", "HBD", "HBA", "QED", "Lipinski Fails"] if c in sorted_df.columns]
            if len(admet_cols) > 1: # More than just Ligand exists
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
                **MW**: Molecular Weight (<500 Da ideal). 
                **LogP**: Lipophilicity / fat-solubility (0-5 ideal). 
                **TPSA**: Topological Polar Surface Area (<90 Å² for blood-brain barrier penetration). 
                **QED**: Quantitative Estimate of Drug-likeness (0.0 to 1.0).
                """)
            else:
                st.info("ADMET prediction was not generated for these molecules.")

        with res_tab2:
            # Feature 3: Interactive Plotly Graph
            filt_ok = filtered_df[filtered_df["Status"] == "success"]
            if len(filt_ok) > 0:
                import plotly.express as px
                
                # If CNN scores exist, do a scatter plot CNN vs Vina
                if filt_ok["CNN Score"].notna().any():
                    fig = px.scatter(
                        filt_ok, x="Vina Score", y="CNN Score", color="Receptor", hover_name="Ligand",
                        size_max=15, title="Vina Empirical Energy vs. GNINA Neural Prediction",
                        template="plotly_dark", color_discrete_sequence=["#00cefc", "#7c3aed", "#3fb950"]
                    )
                else:
                    # Otherwise, do a bar chart of Vina scores by Ligand
                    fig = px.bar(
                        filt_ok.sort_values("Vina Score"), x="Ligand", y="Vina Score", color="Receptor",
                        title="Vina Binding Affinity (More negative is stronger)",
                        template="plotly_dark", color_discrete_sequence=["#00cefc", "#7c3aed", "#f85149"]
                    )
                    
                fig.update_layout(
                    plot_bgcolor="rgba(15, 23, 42, 0.4)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Outfit", color="#94a3b8"), margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No successful poses to plot.")

        with res_tab3:
            ok_res = [r for r in res if r["status"]=="success" and r.get("pose_sdf")]
            if ok_res:
                c_r, c_l = st.columns(2)
                sel_r = c_r.selectbox("Select Receptor", list(dict.fromkeys(r["Receptor"] for r in ok_res)))
                r_set = [r for r in ok_res if r["Receptor"]==sel_r]
                sel_l = c_l.selectbox("Select Ligand", [r["Ligand"] for r in r_set])
                r_match = next(x for x in r_set if x["Ligand"]==sel_l)
                
                # Feature 4: Advanced 3D Viewer Controls
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1,1,2,2])
                view_style = ctrl1.radio("Protein Style", ["Cartoon", "Surface"], horizontal=True, label_visibility="collapsed")
                
                sc1,sc2,sc3 = st.columns(3)
                sc1.metric("Vina Score",   f"{r_match['vina_score']:.3f} kcal/mol" if r_match.get("vina_score") else "—")
                sc2.metric("CNN Score",    f"{r_match['cnn_score']:.4f}"            if r_match.get("cnn_score")  else "—")
                sc3.metric("CNN Affinity", f"{r_match['cnn_affinity']:.3f}"         if r_match.get("cnn_affinity") else "—")
                
                render_3d(r_match["rec_pdb"], r_match["pose_sdf"], view_style)
                
                if Path(r_match["pose_sdf"]).exists():
                    st.download_button("Download pose SDF", Path(r_match["pose_sdf"]).read_bytes(), f"{sel_r}_{sel_l}_pose.sdf", use_container_width=True)
            else:
                st.info("No successful 3D poses generated.")


# =============================================================================
# ── PAGE 4: HELP
# =============================================================================
def page_help():
    render_stepper()
    st.markdown('<div class="page-title">Help &amp; Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Learn how to use StrataDock and understand what each parameter does.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["How to Use", "Scores Explained", "Parameters", "FAQ", "💾 Save/Load Project"])

    with tab5:
        st.markdown("### 💾 Manage Project Session")
        st.caption("Save your progress, uploaded proteins, and tuning settings to a file, or resume a previous docking project.")
        
        export_keys = [
            "engine", "boxes", "rec_files_data", "lig_files_data", "box_padding", "multi_pocket",
            "embed_method", "max_attempts", "force_field", "ff_steps", "add_hs", 
            "keep_chirality", "merge_nphs", "hydrate", "exhaustiveness", "num_poses", 
            "cnn_model", "cnn_weight", "min_rmsd", "gnina_seed", "gpu_device", "flexdist", "cnn_mode",
            "v_exhaustiveness", "v_num_poses", "energy_range", "scoring_fn", "vina_seed", 
            "n_jobs", "backend"
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

**Step 1 — Upload (Upload page)**
- Upload your **receptor(s)** as `.pdb` files. *NEW: You can upload multiple receptors at once!*
- Upload your **ligand(s)** as `.sdf` (3D structures) or `.smi` / `.txt` (1D SMILES strings). *NEW: You can upload entire folders compressed as a `.zip` archive!*
- **Defining the Box:** You must specify where the docking engine should search.
    - **Option A:** Upload a Reference Ligand and click **Compute Autobox**.
    - **Option B [NEW]:** Click **Suggest Pocket (Auto)**. StrataDock will run a pocket-detection algorithm (FPocket-inspired) to scan the receptor for the largest, deepest binding cavity and automatically snap the 3D grid box perfectly around it!
- Check the **3D Box Preview** to ensure the cyan box covers the intended active site.

**Step 2 — Settings (Settings page)**
- Choose your **docking engine**. AutoDock Vina is recommended for most users and works on all CPUs. GNINA utilizes neural-network scoring and works best with an NVIDIA/AMD GPU.
- Adjust molecular parameters if needed. **The default values are optimized for high-throughput screening.**

**Step 3 — Run (Run page)**
- Click **Run Docking**. The cinematic console will stream engine logs in real-time.
- For bulk / N x M cross-docking (multiple receptors x multiple ligands), utilize the **Parallelization** settings to unleash your CPU cores.

**Step 4 — Results**
- The interactive Results Table allows sorting by Vina/CNN scores.
- Click **Export PDF Report** to instantly generate a publication-ready document with embedded 3D pose visualizations and affinity metrics.

---
### 📁 File Formats

| Format | Use |
|---|---|
| `.pdb` | Receptor proteins (Single or Multiple) |
| `.zip` | Bulk folder uploads for high-throughput screening |
| `.sdf` | Ligand structures (3D or 2D — StrataDock natively generates 3D geometries) |
| `.smi` / `.txt` | SMILES strings, one per line. Format: `SMILES name` |
        """)

    with tab2:
        st.markdown("### Understanding the scores")

        st.markdown("""
**Vina Score** (kcal/mol)
> The estimated binding free energy from AutoDock Vina's empirical scoring function.
> **More negative = stronger binding.** A score below −7 kcal/mol is generally considered a good binder.
> This is the primary score when using AutoDock Vina as the engine.

**CNN Score** *(GNINA only)*
> A probability score from GNINA's convolutional neural network that estimates how likely a pose represents a **true binding pose**.
> **Range: 0 to 1. Higher is better.** A CNN Score above 0.5 is generally considered a good pose.

**CNN Affinity** *(GNINA only)*
> The CNN's prediction of binding affinity in kcal/mol, similar to the Vina Score but predicted by the neural network.
> **More negative = stronger predicted binding.**

---
### What to trust
- Use **Vina Score** for fast screening and ranking
- Use **CNN Score + CNN Affinity** to shortlist poses that are likely to be real binders
- Always visually check top-ranked poses in the 3D Viewer before drawing conclusions
        """)

    with tab3:
        params = [
            ("Exhaustiveness", "8–16 (Vina), 16–32 (GNINA)", "How thoroughly the binding site is searched. Higher = better results, much slower. Try 8 for initial screening, 16+ for final results."),
            ("Number of poses", "9", "How many docked conformations to generate per ligand."),
            ("Energy range", "3 kcal/mol", "Only Vina. Output poses within this range of the best score. Increase if you want more diverse poses."),
            ("Scoring function", "vina", "vina: default. vinardo: better for macrocycles and larger ligands. ad4: classic AutoDock 4 scoring."),
            ("Embedding method", "ETKDGv3", "How RDKit generates the 3D shape of 2D ligands. ETKDGv3 is the most accurate modern method."),
            ("Force field", "MMFF94", "Energy function for 3D geometry optimisation. MMFF94 is standard. UFF is a simpler fallback."),
            ("Force field steps", "500", "How many optimisation iterations to run. 500 is a good balance. Set to 0 to skip."),
            ("Add hydrogens", "ON", "Always leave on. Docking requires explicit H atoms."),
            ("Merge non-polar H", "ON", "Standard AD4 convention. Non-polar H atoms are merged into heavy atoms."),
            ("Autobox padding", "8 Å", "Extra space added around the receptor region. Increase if your binding site is near the surface."),
            ("CNN model", "default", "GNINA only. Different neural network architectures for CNN scoring. default works for most cases."),
            ("CNN scoring weight", "1.0", "GNINA only. 1.0 = rank poses purely by CNN score. 0.0 = rank by Vina score only."),
            ("CPU cores", "max-1", "Number of ligands to dock simultaneously. Set to 1 if you want to see per-ligand progress updates."),
            ("Random seed", "0", "Set to any non-zero integer for reproducible results across runs."),
        ]
        for name, default, desc in params:
            st.markdown(f"**{name}** *(default: {default})*")
            st.caption(desc)
            st.markdown("")

    with tab4:
        faqs = [
            ("My ligand failed preparation — what do I do?",
             "Some molecules cannot be embedded in 3D by RDKit (e.g. complex metal complexes, unusual valences). Try increasing 'Max embedding attempts' in Settings, or check that your SMILES/SDF is valid. You can validate SMILES at rdkit.org."),
            ("What is a good Vina score?",
             "Below −6 kcal/mol is considered a weak binder, −7 to −9 is a moderate binder, and below −9 is a strong binder. These are rough guidelines — always compare within your own screening set."),
            ("How does the 'Suggest Pocket (Auto)' feature work?",
             "When you click Suggest Pocket, StrataDock runs a custom geometric cavity-detection algorithm based on principles from fpocket. It constructs an alpha-shape of the receptor, maps out deep crevices using Voronoi vertices, filters them by hydrophobicity and volume, and automatically selects the largest, most druggable pocket as the target docking site!"),
            ("Why does GNINA not work on my machine?",
             "The pre-built GNINA binary requires NVIDIA CUDA libraries. If you have an AMD GPU or no GPU, use AutoDock Vina — it gives equivalent docking results without CNN scoring."),
            ("How do I know if my autobox is correct?",
             "After computing the autobox, check that the box size is reasonable for your binding site (typically 15–30 Å per side). If you know the binding site location, use a reference ligand for more accurate placement."),
            ("Can I use SMILES as input?",
             "Yes. Create a .smi or .txt file with one SMILES per line in the format: SMILES name. StrataDock will generate 3D conformers automatically using RDKit."),
            ("How do I run on an NVIDIA server?",
             "Run bash setup.sh on the server. StrataDock auto-detects NVIDIA GPU and enables GNINA automatically. No manual configuration needed."),
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
if page == "Upload":   page_upload()
elif page == "Settings": page_settings()
elif page == "Run":    page_run()
elif page == "Help":   page_help()

# ── Subtle footer credit ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; opacity:0.35; font-size:0.75rem; color:#8b949e; letter-spacing:0.03em;">
    StrataDock Alpha v0.7 · Developed by Prithvi Rajan · Bioinformatics, Freie Universität Berlin · © 2026
</div>
""", unsafe_allow_html=True)
