from __future__ import annotations

import base64
import html
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from stratadock.core.batch import run_batch_screen, run_ensemble_screen
from stratadock.core.boxes import box_from_center_size, box_from_ligand_file, validate_box
from stratadock.core.diagnostics import environment_diagnostics
from stratadock.core.gnina import locate_gnina
from stratadock.core.hardware import detect_hardware
from stratadock.core.history import import_session_zip, list_run_history, load_run
from stratadock.core.ligands import LigandPrepOptions, load_ligand_records_with_errors
from stratadock.core.models import NamedDockingBox
from stratadock.core.pdb_download import download_pdb
from stratadock.core.pockets import suggest_pockets_with_fpocket
from stratadock.core.reports import write_run_pdf_report
from stratadock.core.session import create_stop_file
from stratadock.core.structure_prediction import PredictionConfig, predict_structure_from_sequence
from stratadock.core.visualization import write_3dmol_viewer_html
from stratadock.ui.export import csv_bytes, zip_run_outputs, zip_selected_hit_outputs
from stratadock.ui.preview import ligand_preview_png
from stratadock.ui.state import normalize_uploaded_ligands, normalize_uploaded_receptors, selected_pocket_names
from stratadock.ui.tables import add_warning_flags, admet_columns, admet_summary, best_rows, filter_results, result_choice_labels, score_filter_bounds


st.set_page_config(
    page_title="StrataDock",
    page_icon="SD",
    layout="wide",
    initial_sidebar_state="collapsed",
)


STEPS = [
    ("Upload", "01", "Upload"),
    ("Settings", "02", "Settings"),
    ("Run", "03", "Run"),
    ("Results", "04", "Results"),
    ("Help", "05", "Help"),
]

ENGINE_VINA = "AutoDock Vina"
ENGINE_GNINA = "GNINA"
ENGINE_OPTIONS = [ENGINE_VINA, ENGINE_GNINA]


def init_state() -> None:
    defaults = {
        "page": "Upload",
        "receptor_name": None,
        "receptor_bytes": None,
        "receptor_uploads": [],
        "ligand_name": None,
        "ligand_bytes": None,
        "reference_name": None,
        "reference_bytes": None,
        "pdb_id_input": "",
        "fold_sequence": "",
        "fold_allow_network": False,
        "fold_timeout": 120,
        "box_mode": "Suggest pockets (fpocket)",
        "top_pockets": 5,
        "selected_pocket_ranks": [],
        "last_pocket_table": [],
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
        "exhaustiveness": 4,
        "seed": 1,
        "num_modes": 1,
        "energy_range": 3.0,
        "scoring": "vina",
        "dock_engine": ENGINE_VINA,
        "gnina_backend": "Auto",
        "gnina_device": "0",
        "prep_add_h": False,
        "prep_ph": 7.4,
        "prep_repair": False,
        "prep_minimize": False,
        "prep_remove_waters": True,
        "prep_remove_hetero": True,
        "prep_keep_metals": True,
        "prep_altloc": "A",
        "ligand_force_field": "MMFF94",
        "ligand_use_ph": False,
        "ligand_ph": 7.4,
        "ligand_strip_salts": False,
        "ligand_neutralize": False,
        "n_jobs": 1,
        "result_ligand_filter": "",
        "result_admet_filter": "all",
        "viewer_receptor_style": "cartoon",
        "viewer_ligand_style": "stick",
        "viewer_surface_opacity": 0.18,
        "viewer_show_interactions": True,
        "resume": False,
        "settings_ready": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.dock_engine not in ENGINE_OPTIONS:
        st.session_state.dock_engine = ENGINE_VINA


@st.cache_data(ttl=15)
def gnina_hardware_status() -> dict[str, object]:
    try:
        hardware = detect_hardware()
    except Exception as exc:
        return {
            "level": "warning",
            "title": "Hardware detection unavailable",
            "detail": f"Could not probe GPU hardware: {exc}. {ENGINE_VINA} remains the supported default.",
            "gnina_available": False,
            "nvidia_available": False,
            "amd_available": False,
            "recommended_gnina_cpu_only": True,
        }

    try:
        gnina_path = str(locate_gnina(project_root=ROOT))
    except FileNotFoundError:
        gnina_path = shutil.which("gnina")
    nvidia = hardware.nvidia
    amd = hardware.amd

    if gnina_path and nvidia.available:
        gpu = nvidia.name or "NVIDIA CUDA GPU"
        return {
            "level": "success",
            "title": "GNINA detected with NVIDIA CUDA",
            "detail": f"GNINA was found at {gnina_path}. {gpu} is detected; GNINA GPU use is supported when the backend is enabled.",
            "gnina_available": True,
            "nvidia_available": True,
            "amd_available": amd.available,
            "recommended_gnina_cpu_only": False,
        }
    if gnina_path and amd.available:
        gpu = amd.name or "AMD GPU"
        return {
            "level": "warning",
            "title": "GNINA detected, AMD GPU fallback",
            "detail": f"GNINA was found at {gnina_path}. {gpu} is detected, but official GNINA GPU acceleration is NVIDIA CUDA-only; GNINA will use CPU mode.",
            "gnina_available": True,
            "nvidia_available": False,
            "amd_available": True,
            "recommended_gnina_cpu_only": True,
        }
    if gnina_path:
        return {
            "level": "info",
            "title": "GNINA detected for CPU mode",
            "detail": f"GNINA was found at {gnina_path}. No NVIDIA CUDA GPU was detected, so GNINA will use CPU mode.",
            "gnina_available": True,
            "nvidia_available": False,
            "amd_available": amd.available,
            "recommended_gnina_cpu_only": True,
        }
    if nvidia.available:
        gpu = nvidia.name or "NVIDIA CUDA GPU"
        return {
            "level": "info",
            "title": "NVIDIA CUDA GPU detected",
            "detail": f"{gpu} is detected, but GNINA is not installed in the project tools folder or PATH. {ENGINE_VINA} remains runnable.",
            "gnina_available": False,
            "nvidia_available": True,
            "amd_available": amd.available,
            "recommended_gnina_cpu_only": False,
        }
    if amd.available:
        gpu = amd.name or "AMD GPU"
        return {
            "level": "warning",
            "title": "AMD GPU detected",
            "detail": f"{gpu} is detected. GNINA does not officially support AMD/ROCm GPU acceleration; use {ENGINE_VINA} or install GNINA for CPU mode.",
            "gnina_available": False,
            "nvidia_available": False,
            "amd_available": True,
            "recommended_gnina_cpu_only": True,
        }
    return {
        "level": "info",
        "title": "CPU docking path",
        "detail": f"No supported NVIDIA CUDA GPU was detected. {ENGINE_VINA} is runnable; GNINA needs its executable installed first.",
        "gnina_available": False,
        "nvidia_available": False,
        "amd_available": False,
        "recommended_gnina_cpu_only": True,
    }


def show_gnina_hardware_status() -> dict[str, object]:
    status = gnina_hardware_status()
    message = f"**{status['title']}**  \n{status['detail']}"
    if status["level"] == "success":
        st.success(message)
    elif status["level"] == "warning":
        st.warning(message)
    else:
        st.info(message)
    return status


def runtime_badge_text() -> str:
    status = gnina_hardware_status()
    if bool(status.get("nvidia_available")):
        return "NVIDIA GPU Available"
    if st.session_state.dock_engine == ENGINE_GNINA and not bool(status.get("gnina_available")):
        return "GNINA Missing"
    return "CPU Mode"


def selected_engine_key() -> str:
    return "gnina" if st.session_state.dock_engine == ENGINE_GNINA else "vina"


def selected_gnina_runtime() -> tuple[bool, str | None]:
    if st.session_state.dock_engine != ENGINE_GNINA:
        return False, None
    status = gnina_hardware_status()
    backend = st.session_state.gnina_backend
    if backend == "CPU only":
        return True, None
    if backend == "NVIDIA CUDA":
        return False, str(st.session_state.gnina_device or "0")
    return bool(status.get("recommended_gnina_cpu_only")), None


def asset_background() -> str:
    for path in [ROOT / "assets" / "molecular_bg.png", ROOT.parent / "assets" / "molecular_bg.png"]:
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    return ""


def apply_style() -> None:
    bg_b64 = asset_background()
    bg_img = ""
    if bg_b64:
        bg_img = f'<img src="data:image/png;base64,{bg_b64}" class="molecular-bg" />'
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body {{
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0;
}}
.stApp {{
    background: #242424;
    color: #d2d2d2;
}}
section[data-testid="stSidebar"], [data-testid="collapsedControl"] {{
    display: none !important;
}}
#MainMenu, footer, header {{
    visibility: hidden;
}}
.block-container {{
    padding: 2.35rem 2rem 4.2rem;
    max-width: 95% !important;
}}
.molecular-bg {{
    position: fixed;
    top: -30px;
    right: -30px;
    width: 650px;
    height: 650px;
    opacity: 0.20;
    pointer-events: none;
    z-index: 0;
    object-fit: cover;
    mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
}}
h1, h2, h3, h4, h5, h6, p, span, label, div, li {{
    color: #d2d2d2 !important;
}}
a, a:visited, a:hover {{
    color: #eaeaea !important;
}}
.logo-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 2.2rem;
    position: relative;
    z-index: 2;
}}
.logo-left {{
    display: flex;
    align-items: center;
    gap: 20px;
}}
.logo-name {{
    font-size: 2.2rem;
    font-weight: 850;
    color: #eaeaea !important;
    letter-spacing: -0.04em;
    text-transform: uppercase;
    transform: skewX(-8deg);
    transform-origin: left bottom;
}}
.logo-sub {{
    text-align: right;
    color: #888888 !important;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    font-weight: 600;
    text-transform: uppercase;
}}
.runtime-badge {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.45rem 1.2rem;
    background: #1a1a1a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}
.runtime-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #7acc7a;
    box-shadow: 0 0 7px #7acc7a;
}}
.runtime-text {{
    color: #999999 !important;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.page-title {{
    font-size: 2.4rem;
    font-weight: 850;
    color: #eaeaea !important;
    margin-bottom: 0.25rem;
    line-height: 1.15;
    letter-spacing: -0.03em;
    text-transform: uppercase;
    display: inline-block;
}}
.page-sub {{
    font-size: 0.9rem;
    color: #888888 !important;
    margin-bottom: 2rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}}
.card, .setting-group, .help-card {{
    background: #2e2e2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.45rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.30);
}}
.card-title, .setting-group-title {{
    font-size: 0.82rem;
    font-weight: 800;
    color: #eaeaea !important;
    margin-bottom: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.settings-section {{
    border-top: 1px solid rgba(255,255,255,0.09);
    padding-top: 1rem;
    margin-top: 1rem;
}}
.settings-section:first-child {{
    margin-top: 0;
}}
.settings-section-title {{
    color: #eaeaea !important;
    font-size: 0.86rem;
    font-weight: 850;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}}
.settings-section-note {{
    color: #8e8e8e !important;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-bottom: 1rem;
}}
.engine-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin: 0.3rem 0 0.6rem;
}}
.engine-card {{
    background: #2e2e2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.15rem;
    min-height: 126px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.24);
}}
.engine-card.selected {{
    border-color: rgba(255,75,75,0.86);
    box-shadow: 0 0 0 1px rgba(255,75,75,0.40), 0 8px 24px rgba(0,0,0,0.28);
}}
.engine-card-title {{
    color: #f0f0f0 !important;
    font-size: 1.0rem;
    font-weight: 850;
    margin-bottom: 0.45rem;
}}
.engine-card-body {{
    color: #9b9b9b !important;
    font-size: 0.84rem;
    line-height: 1.45;
}}
.engine-badge {{
    display: inline-block;
    margin-left: 0.4rem;
    padding: 0.12rem 0.42rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    color: #c9c9c9 !important;
    font-size: 0.62rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    vertical-align: middle;
}}
.admet-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 10px;
    margin: 0.4rem 0 1.2rem;
}}
.admet-tile {{
    background: #2e2e2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 4px solid #777;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    box-shadow: 0 3px 12px rgba(0,0,0,0.24);
}}
.admet-tile.good {{ border-left-color: #74d37a; }}
.admet-tile.warn {{ border-left-color: #e7c65f; }}
.admet-tile.bad {{ border-left-color: #ff5a5f; }}
.admet-tile.neutral {{ border-left-color: #8aa4ff; }}
.admet-value {{
    color: #eaeaea !important;
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1;
}}
.admet-label {{
    color: #8f8f8f !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.45rem;
}}
.warning-pill {{
    display: inline-block;
    padding: 0.16rem 0.45rem;
    border-radius: 999px;
    background: rgba(255,90,95,0.14);
    border: 1px solid rgba(255,90,95,0.35);
    color: #ffb5b8 !important;
    font-size: 0.68rem;
    font-weight: 800;
    margin: 0 0.2rem 0.2rem 0;
}}
.profile-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(130px, 1fr));
    gap: 10px;
    margin: 0.8rem 0 1.1rem;
}}
.profile-tile {{
    background: #252525;
    border: 1px solid rgba(255,255,255,0.07);
    border-top: 3px solid #777;
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
}}
.profile-tile.good {{ border-top-color: #74d37a; }}
.profile-tile.warn {{ border-top-color: #e7c65f; }}
.profile-tile.bad {{ border-top-color: #ff5a5f; }}
.profile-tile.neutral {{ border-top-color: #8aa4ff; }}
.profile-name {{
    color: #8f8f8f !important;
    font-size: 0.65rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.profile-number {{
    color: #eeeeee !important;
    font-size: 1.15rem;
    font-weight: 850;
    margin-top: 0.3rem;
}}
.risk-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 0.5rem 0 0.95rem;
}}
.risk-pill {{
    display: inline-block;
    border-radius: 6px;
    padding: 0.32rem 0.62rem;
    font-size: 0.78rem;
    font-weight: 750;
}}
.risk-pill.good {{
    background: rgba(116, 211, 122, 0.10);
    color: #9de7a1 !important;
    border: 1px solid rgba(116, 211, 122, 0.34);
}}
.risk-pill.bad {{
    background: rgba(255, 90, 95, 0.12);
    color: #ffb5b8 !important;
    border: 1px solid rgba(255, 90, 95, 0.34);
}}
.binding-overview {{
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 10px;
    margin: 0.55rem 0 1rem;
}}
.binding-tile {{
    background: #242424;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
}}
.binding-value {{
    color: #f2f2f2 !important;
    font-size: 1.18rem;
    font-weight: 850;
}}
.binding-label {{
    color: #8f8f8f !important;
    font-size: 0.66rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}}
.contact-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 0.35rem 0 0.8rem;
}}
.contact-chip {{
    display: inline-block;
    border-radius: 6px;
    padding: 0.28rem 0.58rem;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e5e5e5 !important;
    font-size: 0.78rem;
    font-weight: 750;
}}
.binding-note {{
    color: #b8b8b8 !important;
    font-size: 0.86rem;
    line-height: 1.45;
    margin: -0.2rem 0 0.9rem;
}}
.admet-caption {{
    color: #9d9d9d !important;
    font-size: 0.8rem;
    margin: -0.15rem 0 0.75rem;
}}
.muted {{
    color: #999999 !important;
    font-size: 0.86rem;
}}
.box-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 6px;
    margin-top: 0.6rem;
}}
.box-cell {{
    background: #262626;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 9px 5px;
    text-align: center;
}}
.box-val {{
    color: #eaeaea !important;
    font-size: 0.98rem;
    font-weight: 800;
}}
.box-lbl {{
    color: #8c8c8c !important;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.subtle-rule {{
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 0.8rem 0 0.7rem;
}}
.subtle-rule.compact {{
    margin: 0.65rem 0;
}}
.terminal-block {{
    background: #1a1a1a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.84rem;
    color: #8fbc8f !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
}}
.terminal-block.compact {{
    height: 220px;
}}
.terminal-block.autoscroll {{
    display: flex;
    flex-direction: column-reverse;
}}
.terminal-line {{
    min-height: 1.12rem;
}}
.forward-nav-spacer {{
    height: 1.35rem;
}}
.primary-action-spacer {{
    height: 0.75rem;
}}
.summary-line {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 1.2rem;
    align-items: center;
    margin: 0.55rem 0 1rem;
    padding: 0.55rem 0;
    border-top: 1px solid rgba(255,255,255,0.08);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.summary-line span {{
    color: #cfcfcf !important;
    font-size: 0.82rem;
    font-weight: 700;
}}
.summary-line b {{
    color: #ffffff !important;
}}
.table-section-title {{
    color: #f0f0f0 !important;
    font-size: 0.82rem;
    font-weight: 850;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0.25rem 0 0.55rem;
}}
.custom-footer {{
    margin: 2.2rem -2rem -2rem;
    background: #1a1a1a;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 0.55rem 2rem;
    text-align: center;
    font-size: 0.72rem;
    color: #6b6b6b !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.stButton > button, .stDownloadButton > button {{
    background: #d2d2d2 !important;
    color: #242424 !important;
    -webkit-text-fill-color: #242424 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    transition: all 0.22s ease !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px) !important;
    background: #ffffff !important;
}}
[data-testid="stMetric"] {{
    background: #2e2e2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #ff4b4b;
    border-radius: 10px;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}
[data-testid="stMetricLabel"] {{
    color: #888888 !important;
    font-size: 0.70rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="stMetricValue"] {{
    color: #eaeaea !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
}}
.metric-card {{
    background: #2e2e2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #ff4b4b;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    min-height: 64px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}
.metric-card .metric-label {{
    color: #888888 !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.32rem;
}}
.metric-card .metric-value {{
    color: #eaeaea !important;
    font-size: 1.25rem;
    font-weight: 850;
    line-height: 1.12;
    overflow-wrap: anywhere;
}}
.upload-status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 1.2rem;
    align-items: center;
    margin: 0.72rem 0 0.15rem;
    padding: 0.42rem 0;
    border-top: 1px solid rgba(255,255,255,0.08);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.upload-status-item {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    min-width: min(100%, 150px);
}}
.upload-status-dot {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #7acc7a;
    box-shadow: 0 0 8px rgba(122,204,122,0.55);
    flex: 0 0 auto;
}}
.upload-status-item.missing .upload-status-dot {{
    background: #ff5a5f;
    box-shadow: 0 0 8px rgba(255,90,95,0.45);
}}
.upload-status-label {{
    color: #9a9a9a;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0;
    text-transform: none;
}}
.upload-status-label::after {{
    content: ":";
}}
.upload-status-value {{
    color: #d0d0d0;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.2;
    overflow-wrap: anywhere;
}}
.upload-status-item.missing .upload-status-value {{
    color: #aaaaaa;
    font-weight: 600;
}}
.result-summary-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 0;
    margin: -0.6rem 0 1.3rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.result-summary-item {{
    padding: 0.9rem 1rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
}}
.result-summary-item:last-child {{
    border-right: 0;
}}
.result-summary-value {{
    color: #f1f1f1 !important;
    font-size: 1.15rem;
    font-weight: 850;
    line-height: 1.1;
    overflow-wrap: anywhere;
}}
.result-summary-label {{
    color: #8d8d8d !important;
    font-size: 0.66rem;
    font-weight: 850;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: #2e2e2e !important;
    border: 2px dashed rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: #d2d2d2 !important;
    background: #353535 !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background: #d2d2d2 !important;
    color: #242424 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-width: 122px !important;
    font-size: 0 !important;
}}
[data-testid="stFileUploaderDropzone"] button *,
.stButton > button *,
.stDownloadButton > button * {{
    color: #242424 !important;
    -webkit-text-fill-color: #242424 !important;
}}
[data-testid="stFileUploaderDropzone"] button * {{
    display: none !important;
}}
[data-testid="stFileUploaderDropzone"] button::after {{
    content: "BROWSE";
    font-size: 0.76rem !important;
    letter-spacing: 0.04em;
}}
[data-baseweb="select"] > div,
[data-baseweb="base-input"],
textarea {{
    background: #252631 !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #eaeaea !important;
    -webkit-text-fill-color: #eaeaea !important;
}}
[data-baseweb="select"] * ,
[data-baseweb="base-input"] *,
textarea,
input {{
    color: #eaeaea !important;
    -webkit-text-fill-color: #eaeaea !important;
}}
[data-baseweb="tag"] {{
    background: #ff4b4b !important;
    color: #ffffff !important;
}}
[data-baseweb="tag"] * {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: #888888 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.2rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: #2e2e2e !important;
    color: #eaeaea !important;
}}
.stAlert {{
    background: #2e2e2e !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-left: 3px solid #d2d2d2 !important;
    border-radius: 12px !important;
}}
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden;
}}
@media (max-width: 700px) {{
    .block-container {{
        padding: 2rem 1.1rem 3rem;
        max-width: 100% !important;
    }}
    .logo-bar {{
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.6rem;
    }}
    .logo-name {{
        font-size: 1.65rem;
        white-space: nowrap;
        transform: none;
    }}
    .logo-sub {{
        text-align: left;
        font-size: 0.72rem;
    }}
    .runtime-badge {{
        align-self: flex-start;
        padding: 0.45rem 0.9rem;
        max-width: 100%;
    }}
    .runtime-text {{
        font-size: 0.70rem;
        white-space: normal;
    }}
    .page-title {{
        font-size: 1.85rem;
    }}
    .molecular-bg {{
        width: 460px;
        height: 460px;
        right: -190px;
        opacity: 0.16;
    }}
    .engine-grid {{
        grid-template-columns: 1fr;
    }}
    .upload-status-strip {{
        grid-template-columns: 1fr;
    }}
    .result-summary-strip {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .result-summary-item:nth-child(2n) {{
        border-right: 0;
    }}
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #1a1a1a; }}
::-webkit-scrollbar-thumb {{ background: #4a4a4a; border-radius: 3px; }}
</style>
{bg_img}
""",
        unsafe_allow_html=True,
    )


def render_stepper() -> None:
    done = {
        "Upload": bool(st.session_state.receptor_bytes and st.session_state.ligand_bytes),
        "Settings": bool(st.session_state.settings_ready),
        "Run": bool(st.session_state.get("last_table")),
        "Results": bool(st.session_state.get("last_table")),
        "Help": False,
    }
    page_keys = [step[0] for step in STEPS]
    current_index = page_keys.index(st.session_state.page) if st.session_state.page in page_keys else 0

    st.markdown(
        f"""
<div class="logo-bar">
  <div class="logo-left">
    <div>
      <div class="logo-name">&gt;_ STRATA DOCK</div>
      <div class="logo-sub">StrataDock v 1.6.01</div>
    </div>
  </div>
  <div class="runtime-badge">
    <span class="runtime-dot"></span>
    <span class="runtime-text">{html.escape(runtime_badge_text())}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    columns = st.columns(len(STEPS))
    for index, (col, (key, number, label)) in enumerate(zip(columns, STEPS)):
        with col:
            active = key == st.session_state.page
            button_label = f"{number}. {label}"
            if st.button(button_label, key=f"nav_{key}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    pct = ((current_index + 1) / len(STEPS)) * 100
    st.markdown(
        f"""
<div style="width:100%; height:3px; background:rgba(255,255,255,0.05); margin:0.4rem 0 1.5rem; border-radius:2px;">
  <div style="width:{pct}%; height:100%; background:linear-gradient(90deg,#ff4b4b,#ff7b7b); box-shadow:0 0 12px rgba(255,75,75,0.4); border-radius:2px;"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="muted">{body}</div></div>', unsafe_allow_html=True)


def metric_card(target, label: str, value) -> None:
    target.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def terminal_block(lines: list[str] | str, *, limit: int = 36, compact: bool = False, autoscroll: bool = False) -> str:
    if isinstance(lines, str):
        raw_lines = lines.splitlines()
    else:
        raw_lines = list(lines)
    visible = raw_lines[-limit:] if raw_lines else ["> Waiting for run."]
    classes = ["terminal-block"]
    if compact:
        classes.append("compact")
    if autoscroll:
        classes.append("autoscroll")
        rendered_lines = reversed(visible)
        body = "".join(f'<div class="terminal-line">{html.escape(str(line))}</div>' for line in rendered_lines)
    else:
        body = "\n".join(html.escape(str(line)) for line in visible)
    return f'<div class="{" ".join(classes)}">{body}</div>'


def upload_status_strip(receptor_label: str, ligand_label: str) -> None:
    def status_item(label: str, value: str) -> str:
        status = "missing" if value.lower() == "missing" else "ready"
        return (
            f'<div class="upload-status-item {status}">'
            f'<span class="upload-status-dot"></span>'
            f'<span class="upload-status-label">{html.escape(label)}</span>'
            f'<span class="upload-status-value">{html.escape(value)}</span>'
            "</div>"
        )

    st.markdown(
        '<div class="upload-status-strip">'
        + status_item("Receptors", receptor_label)
        + status_item("Ligands", ligand_label)
        + "</div>",
        unsafe_allow_html=True,
    )


def result_summary_strip(items: list[tuple[str, object]]) -> None:
    html_items = "".join(
        '<div class="result-summary-item">'
        f'<div class="result-summary-value">{html.escape(str(value))}</div>'
        f'<div class="result-summary-label">{html.escape(label)}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="result-summary-strip">{html_items}</div>', unsafe_allow_html=True)


def section_header(title: str, note: str = "") -> None:
    note_html = f'<div class="settings-section-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="settings-section"><div class="settings-section-title">{html.escape(title)}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def engine_card_html(title: str, badge: str, body: str, selected: bool) -> str:
    selected_class = " selected" if selected else ""
    return (
        f'<div class="engine-card{selected_class}">'
        f'<div class="engine-card-title">{html.escape(title)}'
        f'<span class="engine-badge">{html.escape(badge)}</span></div>'
        f'<div class="engine-card-body">{html.escape(body)}</div>'
        "</div>"
    )


def set_upload_state(upload, name_key: str, bytes_key: str) -> None:
    if upload is not None:
        st.session_state[name_key] = upload.name
        st.session_state[bytes_key] = bytes(upload.getbuffer())


def set_receptor_uploads(uploads) -> None:
    if uploads:
        normalized = normalize_uploaded_receptors(
            {"name": upload.name, "bytes": bytes(upload.getbuffer())}
            for upload in uploads
        )
        st.session_state.receptor_uploads = normalized
        first = normalized[0] if normalized else None
        st.session_state.receptor_name = first["name"] if first else None
        st.session_state.receptor_bytes = first["bytes"] if first else None


def set_ligand_uploads(uploads) -> None:
    if uploads:
        normalized = normalize_uploaded_ligands(
            {"name": upload.name, "bytes": bytes(upload.getbuffer())}
            for upload in uploads
        )
        if normalized:
            st.session_state.ligand_name = normalized["name"]
            st.session_state.ligand_bytes = normalized["bytes"]


def write_upload_bytes(run_dir: Path) -> tuple[list[Path], Path, Path | None]:
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    receptors = list(st.session_state.get("receptor_uploads") or [])
    if not receptors and st.session_state.receptor_name and st.session_state.receptor_bytes:
        receptors = [{"name": st.session_state.receptor_name, "bytes": st.session_state.receptor_bytes}]
    receptor_paths: list[Path] = []
    for idx, receptor in enumerate(receptors, start=1):
        receptor_path = input_dir / f"{idx:02d}_{receptor['name']}" if len(receptors) > 1 else input_dir / receptor["name"]
        receptor_path.write_bytes(receptor["bytes"])
        receptor_paths.append(receptor_path)
    ligand_path = input_dir / st.session_state.ligand_name
    ligand_path.write_bytes(st.session_state.ligand_bytes)
    reference_path = None
    if st.session_state.reference_bytes and st.session_state.reference_name:
        reference_path = input_dir / st.session_state.reference_name
        reference_path.write_bytes(st.session_state.reference_bytes)
    return receptor_paths, ligand_path, reference_path


def selected_boxes(run_dir: Path, receptor_path: Path, reference_path: Path | None) -> list[NamedDockingBox]:
    mode = st.session_state.box_mode
    if mode == "Reference ligand PDB":
        if reference_path is None:
            raise ValueError("Upload a reference/native ligand PDB or choose another box source.")
        return [
            NamedDockingBox(
                name="reference_ligand",
                box=validate_box(box_from_ligand_file(reference_path)),
                source="reference_ligand",
            )
        ]
    if mode == "Suggest pockets (fpocket)":
        pockets = suggest_pockets_with_fpocket(receptor_path, run_dir / "fpocket", top_n=int(st.session_state.top_pockets))
        table_path = run_dir / "fpocket" / "pockets_table.json"
        if table_path.exists():
            try:
                st.session_state.last_pocket_table = json.loads(table_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                st.session_state.last_pocket_table = []
        selected = selected_pocket_names(st.session_state.last_pocket_table, list(st.session_state.selected_pocket_ranks))
        selected_pockets = [pocket for pocket in pockets if not selected or pocket.name in selected]
        if not selected_pockets:
            raise ValueError("No selected fpocket pockets are available. Increase the top-pocket count or clear the stale pocket selection.")
        return selected_pockets
    return [
        NamedDockingBox(
            name="manual",
            source="manual",
            box=box_from_center_size(
                center=[float(st.session_state.center_x), float(st.session_state.center_y), float(st.session_state.center_z)],
                size=[float(st.session_state.size_x), float(st.session_state.size_y), float(st.session_state.size_z)],
            ),
        )
    ]


def selected_boxes_for_run(
    run_dir: Path,
    receptor_paths: list[Path],
    reference_path: Path | None,
    log_callback=None,
) -> tuple[list[NamedDockingBox], list[list[NamedDockingBox]] | None]:
    if st.session_state.box_mode != "Suggest pockets (fpocket)" or len(receptor_paths) <= 1:
        return selected_boxes(run_dir, receptor_paths[0], reference_path), None

    selected_ranks = {int(rank) for rank in list(st.session_state.selected_pocket_ranks or [])}
    all_boxes: list[NamedDockingBox] = []
    boxes_by_receptor: list[list[NamedDockingBox]] = []
    pocket_rows: list[dict] = []

    for idx, receptor_path in enumerate(receptor_paths, start=1):
        pocket_dir = run_dir / f"receptor_{idx}_{receptor_path.stem}" / "fpocket"
        pockets = suggest_pockets_with_fpocket(receptor_path, pocket_dir, top_n=int(st.session_state.top_pockets))
        if selected_ranks:
            pockets = [pocket for pocket in pockets if pocket.rank in selected_ranks]
        if not pockets:
            raise ValueError(f"fpocket did not return selected pockets for {receptor_path.name}.")
        boxes_by_receptor.append(pockets)
        if log_callback:
            log_callback(f"> {receptor_path.name}: prepared {len(pockets)} receptor-specific fpocket box(es).")

        table_path = pocket_dir / "pockets_table.json"
        if table_path.exists():
            try:
                rows = json.loads(table_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                rows = [pocket.as_dict() for pocket in pockets]
        else:
            rows = [pocket.as_dict() for pocket in pockets]
        selected_names = {pocket.name for pocket in pockets}
        for row in rows:
            if row.get("name") not in selected_names:
                continue
            row = dict(row)
            row["receptor"] = receptor_path.name
            pocket_rows.append(row)

        for pocket in pockets:
            all_boxes.append(
                NamedDockingBox(
                    name=f"{receptor_path.stem}:{pocket.name}",
                    box=pocket.box,
                    source=pocket.source,
                    rank=pocket.rank,
                    score=pocket.score,
                    druggability_score=pocket.druggability_score,
                )
            )

    st.session_state.last_pocket_table = pocket_rows
    return all_boxes, boxes_by_receptor


def result_dataframe(batch_result) -> pd.DataFrame:
    rows = [asdict(result) for result in batch_result.results]
    df = pd.DataFrame(rows)
    columns = [
        "ligand_name",
        "receptor_name",
        "pocket_name",
        "prep_status",
        "docking_status",
        "vina_score",
        "vina_scores",
        "docking_engine",
        "cnn_score",
        "cnn_affinity",
        "cnn_scores",
        "cnn_affinities",
        "ligand_embedding_used",
        "ligand_force_field",
        "ligand_requested_force_field",
        "ligand_protonation_ph",
        "ligand_protonation_status",
        "ligand_salt_stripping_status",
        "ligand_neutralization_status",
        "molecular_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "qed",
        "lipinski_failures",
        "rotatable_bonds",
        "heavy_atom_count",
        "aromatic_rings",
        "formal_charge",
        "fraction_csp3",
        "molar_refractivity",
        "rule_of_five_pass",
        "veber_pass",
        "bbb_penetration",
        "herg_risk",
        "hepatotoxicity_risk",
        "mutagenicity_risk",
        "pains_alert_count",
        "brenk_alert_count",
        "structural_alert_count",
        "pose_pdbqt",
        "complex_pdb",
        "interactions_csv",
        "interactions_json",
        "pymol_script",
        "viewer_html",
        "error",
    ]
    return df[[col for col in columns if col in df.columns]]


def load_last_df() -> pd.DataFrame:
    if "last_table" not in st.session_state:
        return pd.DataFrame()
    return pd.read_json(st.session_state.last_table, orient="records")


def active_run_control_file() -> Path:
    return ROOT / "runs" / "_active_run.json"


def write_active_run_control(*, run_dir: Path, stop_file: Path, engine: str, box_mode: str) -> None:
    control_path = active_run_control_file()
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "stop_file": str(stop_file),
                "engine": engine,
                "box_mode": box_mode,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_active_run_control() -> dict[str, str] | None:
    path = active_run_control_file()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    stop_file = payload.get("stop_file")
    run_dir = payload.get("run_dir")
    if not stop_file or not run_dir:
        return None
    return {str(key): str(value) for key, value in payload.items()}


def clear_active_run_control(run_dir: Path) -> None:
    path = active_run_control_file()
    if not path.exists():
        return
    active = read_active_run_control()
    if not active or Path(active["run_dir"]).resolve() == run_dir.resolve():
        path.unlink(missing_ok=True)


def render_stop_control(container) -> None:
    active = read_active_run_control()
    last_stop = st.session_state.get("last_paths", {}).get("stop_file")
    with container.container():
        if active:
            run_dir = Path(active["run_dir"])
            stop_file = Path(active["stop_file"])
            st.caption(f"Active run: {run_dir.name} / {active.get('engine', 'unknown')}")
            if st.button("Request Safe Stop", use_container_width=True, type="secondary", key="request_active_stop"):
                create_stop_file(stop_file, reason="ui_stop_button")
                st.warning("Stop requested. The current docking subprocess will finish, then no new ligand/pocket jobs will start.")
            return
        if last_stop:
            stop_file = Path(last_stop)
            if st.button("Request Stop For Last Run", use_container_width=True, key="request_last_stop"):
                create_stop_file(stop_file, reason="ui_stop_button")
                st.warning(f"Stop requested: {stop_file}")


def display_box_preview() -> None:
    vals = [
        ("CX", st.session_state.center_x),
        ("CY", st.session_state.center_y),
        ("CZ", st.session_state.center_z),
        ("SX", st.session_state.size_x),
        ("SY", st.session_state.size_y),
        ("SZ", st.session_state.size_z),
    ]
    html = '<div class="box-grid">'
    for label, value in vals:
        html += f'<div class="box-cell"><div class="box-val">{float(value):.1f}</div><div class="box-lbl">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def page_upload() -> None:
    page_title("Upload Files", "Load receptor and ligand inputs before moving through the docking flow.")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card-title">Protein And Ligand Inputs</div>', unsafe_allow_html=True)
        receptor_upload = st.file_uploader("Receptor PDB", type=["pdb"], key="receptor_upload", accept_multiple_files=True)
        ligand_upload = st.file_uploader(
            "Ligands",
            type=["sdf", "smi", "smiles", "txt", "zip"],
            key="ligand_upload",
            accept_multiple_files=True,
        )
        set_receptor_uploads(receptor_upload)
        set_ligand_uploads(ligand_upload)

        show_pdb_fetch = st.checkbox("Fetch receptor by PDB ID", value=False)
        if show_pdb_fetch:
            st.session_state.pdb_id_input = st.text_input("RCSB PDB ID", value=str(st.session_state.pdb_id_input), max_chars=4)
            if st.button("Download PDB", use_container_width=True):
                try:
                    downloaded = download_pdb(st.session_state.pdb_id_input, ROOT / "runs" / "_pdb_downloads")
                    data = downloaded.read_bytes()
                    st.session_state.receptor_uploads = [{"name": downloaded.name, "bytes": data}]
                    st.session_state.receptor_name = downloaded.name
                    st.session_state.receptor_bytes = data
                    st.success(f"Loaded {downloaded.name}.")
                except Exception as exc:
                    st.error(str(exc))

        show_fold = st.checkbox("Fold protein sequence", value=False)
        if show_fold:
            st.session_state.fold_sequence = st.text_area(
                "Protein sequence",
                value=str(st.session_state.fold_sequence),
                height=120,
                placeholder="Paste FASTA or plain amino-acid sequence",
            )
            fc1, fc2 = st.columns([1, 1])
            fc1.checkbox(
                "Allow ESMFold network call",
                key="fold_allow_network",
            )
            st.session_state.fold_timeout = fc2.number_input(
                "Timeout seconds",
                min_value=15,
                max_value=600,
                value=int(st.session_state.fold_timeout),
                step=15,
            )
            if st.button("Predict And Load PDB", use_container_width=True):
                out_dir = ROOT / "runs" / "_fold_to_dock"
                out_path = out_dir / f"predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb"
                result = predict_structure_from_sequence(
                    st.session_state.fold_sequence,
                    out_path,
                    config=PredictionConfig(
                        allow_network=bool(st.session_state.fold_allow_network),
                        timeout_seconds=int(st.session_state.fold_timeout),
                        report_path=out_path.with_suffix(".json"),
                    ),
                )
                if result.ok:
                    data = result.output_path.read_bytes()
                    st.session_state.receptor_uploads = [{"name": result.output_path.name, "bytes": data}]
                    st.session_state.receptor_name = result.output_path.name
                    st.session_state.receptor_bytes = data
                    st.success(f"Loaded predicted receptor. Mean pLDDT: {result.mean_plddt or 'n/a'}")
                else:
                    st.error(result.error or result.status)

        receptor_count = len(st.session_state.get("receptor_uploads") or [])
        receptor_label = f"{receptor_count} receptor(s)" if receptor_count else "Missing"
        upload_status_strip(receptor_label, st.session_state.ligand_name or "Missing")

        if st.session_state.ligand_bytes and st.session_state.ligand_name:
            preview_dir = ROOT / "runs" / "_ui_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_file = preview_dir / st.session_state.ligand_name
            preview_file.write_bytes(st.session_state.ligand_bytes)
            if preview_file.suffix.lower() in {".sdf", ".smi", ".smiles", ".txt", ".zip"}:
                try:
                    if preview_file.suffix.lower() != ".zip":
                        st.image(ligand_preview_png(preview_file), caption="Ligand preview", use_container_width=True)
                    records, errors = load_ligand_records_with_errors(preview_file)
                    st.caption(f"{len(records)} valid ligand(s), {len(errors)} invalid record(s).")
                except Exception as exc:
                    st.caption(f"Preview unavailable: {exc}")

    with right:
        st.markdown('<div class="card-title">Run History</div>', unsafe_allow_html=True)
        history = list_run_history(ROOT / "runs", limit=10)
        if history:
            labels = [
                f"{item.name} | {item.counts.get('success', 0)}/{item.counts.get('total', 0)} docked | {item.created_at_utc or 'unknown time'}"
                for item in history
            ]
            selected_history = st.selectbox("Previous runs", list(range(len(history))), format_func=lambda idx: labels[idx])
            if st.button("Open Selected Run", use_container_width=True):
                loaded = load_run(history[selected_history].path)
                st.session_state.last_batch_dir = str(loaded.path)
                st.session_state.last_paths = {key: str(value) for key, value in loaded.paths.items()}
                st.session_state.last_table = pd.DataFrame(loaded.results).to_json(orient="records")
                st.session_state.last_counts = loaded.counts
                st.session_state.last_zip = zip_run_outputs(loaded.path)
                st.session_state.page = "Results"
                st.rerun()
        else:
            st.caption("No previous completed runs found.")

        session_zip = st.file_uploader("Import session ZIP", type=["zip"], key="session_zip_import")
        if session_zip is not None and st.button("Import Session", use_container_width=True):
            import_dir = ROOT / "runs" / f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_path = import_dir.with_suffix(".zip")
            archive_path.write_bytes(session_zip.getbuffer())
            extracted = import_session_zip(archive_path, import_dir)
            loaded = load_run(extracted)
            st.session_state.last_batch_dir = str(loaded.path)
            st.session_state.last_paths = {key: str(value) for key, value in loaded.paths.items()}
            st.session_state.last_table = pd.DataFrame(loaded.results).to_json(orient="records")
            st.session_state.last_counts = loaded.counts
            st.session_state.last_zip = zip_run_outputs(loaded.path)
            st.session_state.page = "Results"
            st.rerun()

        st.markdown('<div class="forward-nav-spacer"></div>', unsafe_allow_html=True)
        if st.button("Continue To Settings", type="primary", use_container_width=True):
            st.session_state.page = "Settings"
            st.rerun()


def page_settings() -> None:
    page_title("Docking Settings", "Choose the binding-site strategy, docking parameters, and protein prep behavior.")
    gnina_status = gnina_hardware_status()
    vina_selected = st.session_state.dock_engine == ENGINE_VINA
    gnina_selected = st.session_state.dock_engine == ENGINE_GNINA

    st.markdown(
        '<div class="engine-grid">'
        + engine_card_html("AutoDock Vina", "CPU", "Fast, stable, and works on every Ubuntu/WSL machine. Best default for screening.", vina_selected)
        + engine_card_html("GNINA", "CUDA / CPU", "CNN scoring engine installed by setup. NVIDIA CUDA when available, CPU mode otherwise.", gnina_selected)
        + "</div>",
        unsafe_allow_html=True,
    )
    c_engine_1, c_engine_2 = st.columns(2)
    if c_engine_1.button("Use AutoDock Vina", type="primary" if vina_selected else "secondary", use_container_width=True):
        st.session_state.dock_engine = ENGINE_VINA
        st.rerun()
    if c_engine_2.button("Use GNINA", type="primary" if gnina_selected else "secondary", use_container_width=True):
        st.session_state.dock_engine = ENGINE_GNINA
        st.rerun()

    if st.session_state.dock_engine == ENGINE_GNINA:
        show_gnina_hardware_status()

    tab_dock, tab_site, tab_protein, tab_ligand, tab_runtime = st.tabs(
        ["Docking", "Binding Site", "Protein Prep", "Ligand Prep", "Runtime"]
    )

    with tab_dock:
        if st.session_state.dock_engine == ENGINE_VINA:
            section_header("AutoDock Vina Parameters", "These control the search depth, pose count, and scoring function for the default engine.")
            c1, c2, c3 = st.columns(3)
            st.session_state.exhaustiveness = c1.slider("Exhaustiveness", 1, 32, int(st.session_state.exhaustiveness))
            st.session_state.num_modes = c2.slider("Number of modes", 1, 9, int(st.session_state.num_modes))
            st.session_state.energy_range = c3.slider("Energy range", 1.0, 10.0, float(st.session_state.energy_range), 0.5)
            c4, c5 = st.columns([1, 1])
            st.session_state.scoring = c4.selectbox(
                "Scoring function",
                ["vina", "vinardo"],
                index=["vina", "vinardo"].index(st.session_state.scoring),
            )
            st.session_state.seed = c5.number_input("Seed", min_value=0, max_value=999999, value=int(st.session_state.seed))
        else:
            section_header("GNINA Parameters", "GNINA uses affinity plus CNN outputs. The executable must be installed before this engine can run.")
            c1, c2, c3 = st.columns(3)
            st.session_state.exhaustiveness = c1.slider("Exhaustiveness", 1, 32, int(st.session_state.exhaustiveness))
            st.session_state.num_modes = c2.slider("Number of modes", 1, 9, int(st.session_state.num_modes))
            st.session_state.seed = c3.number_input("Seed", min_value=0, max_value=999999, value=int(st.session_state.seed))
            c4, c5 = st.columns([1, 1])
            st.session_state.gnina_backend = c4.selectbox(
                "GNINA backend",
                ["Auto", "CPU only", "NVIDIA CUDA"],
                index=["Auto", "CPU only", "NVIDIA CUDA"].index(st.session_state.gnina_backend),
            )
            if st.session_state.gnina_backend == "NVIDIA CUDA":
                st.session_state.gnina_device = c5.text_input("CUDA device", value=str(st.session_state.gnina_device))
            else:
                c5.caption("CUDA device is only used when NVIDIA CUDA is selected.")
            st.session_state.scoring = "vina"
            if not bool(gnina_status.get("gnina_available")):
                st.error("GNINA is not installed in the project tools folder or PATH. Select AutoDock Vina or install GNINA first.")
            elif st.session_state.gnina_backend == "NVIDIA CUDA" and not bool(gnina_status.get("nvidia_available")):
                st.warning("No NVIDIA CUDA GPU was detected. Switch GNINA backend to Auto or CPU only.")

    with tab_site:
        section_header("Binding Site", "Choose how the docking box is defined before the run starts.")
        st.session_state.box_mode = st.radio(
            "Docking box source",
            ["Suggest pockets (fpocket)", "Reference ligand PDB", "Manual"],
            index=["Suggest pockets (fpocket)", "Reference ligand PDB", "Manual"].index(st.session_state.box_mode),
            horizontal=True,
        )
        if st.session_state.box_mode == "Reference ligand PDB":
            reference_upload = st.file_uploader("Reference/native ligand", type=["pdb", "sdf", "mol", "mol2"], key="reference_upload")
            set_upload_state(reference_upload, "reference_name", "reference_bytes")
            st.caption(st.session_state.reference_name or "Upload a co-crystallized ligand as PDB, SDF, MOL, or MOL2.")
        elif st.session_state.box_mode == "Suggest pockets (fpocket)":
            c1, c2 = st.columns([1, 1])
            st.session_state.top_pockets = c1.slider("Top fpocket pockets to dock", 1, 10, int(st.session_state.top_pockets))
            c2.caption("Use top 3 for speed; top 5 when runtime is acceptable.")
            if st.button("Preview fpocket pockets", use_container_width=True, disabled=not bool(st.session_state.receptor_bytes)):
                try:
                    preview_dir = ROOT / "runs" / "_ui_fpocket_preview"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    rows = []
                    receptors = list(st.session_state.get("receptor_uploads") or [])
                    if not receptors and st.session_state.receptor_name and st.session_state.receptor_bytes:
                        receptors = [{"name": st.session_state.receptor_name, "bytes": st.session_state.receptor_bytes}]
                    for idx, receptor in enumerate(receptors, start=1):
                        receptor_path = preview_dir / f"{idx:02d}_{receptor['name']}"
                        receptor_path.write_bytes(receptor["bytes"])
                        pocket_dir = preview_dir / f"fpocket_{idx}_{Path(receptor['name']).stem}"
                        pockets = suggest_pockets_with_fpocket(receptor_path, pocket_dir, top_n=int(st.session_state.top_pockets))
                        table_path = pocket_dir / "pockets_table.json"
                        receptor_rows = json.loads(table_path.read_text(encoding="utf-8")) if table_path.exists() else [pocket.as_dict() for pocket in pockets]
                        for row in receptor_rows:
                            row = dict(row)
                            row["receptor"] = receptor["name"]
                            rows.append(row)
                    st.session_state.last_pocket_table = rows
                except Exception as exc:
                    st.error(str(exc))
            if st.session_state.last_pocket_table:
                st.dataframe(pd.DataFrame(st.session_state.last_pocket_table), use_container_width=True, hide_index=True)
                ranks = [int(row["rank"]) for row in st.session_state.last_pocket_table if row.get("rank") is not None]
                pocket_mode = st.selectbox(
                    "Dock pocket ranks",
                    ["Top requested pockets", "All previewed pockets", "Single pocket"],
                )
                if pocket_mode == "All previewed pockets":
                    st.session_state.selected_pocket_ranks = ranks
                elif pocket_mode == "Single pocket":
                    selected_rank = st.selectbox("Pocket rank", ranks)
                    st.session_state.selected_pocket_ranks = [int(selected_rank)]
                else:
                    st.session_state.selected_pocket_ranks = ranks[: min(len(ranks), int(st.session_state.top_pockets))]
        else:
            c1, c2, c3 = st.columns(3)
            st.session_state.center_x = c1.number_input("Center X", value=float(st.session_state.center_x))
            st.session_state.center_y = c2.number_input("Center Y", value=float(st.session_state.center_y))
            st.session_state.center_z = c3.number_input("Center Z", value=float(st.session_state.center_z))
            s1, s2, s3 = st.columns(3)
            st.session_state.size_x = s1.number_input("Size X", min_value=1.0, value=float(st.session_state.size_x))
            st.session_state.size_y = s2.number_input("Size Y", min_value=1.0, value=float(st.session_state.size_y))
            st.session_state.size_z = s3.number_input("Size Z", min_value=1.0, value=float(st.session_state.size_z))
            display_box_preview()

    with tab_protein:
        section_header("Protein Preparation", "Cleaning and repair steps applied to every uploaded receptor.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.checkbox("Remove crystallographic waters", key="prep_remove_waters")
            st.checkbox("Remove non-protein heteroatoms/cofactors", key="prep_remove_hetero")
            st.checkbox("Keep metals and simple ions", key="prep_keep_metals")
            st.session_state.prep_altloc = st.text_input("Default alternate location", value=str(st.session_state.prep_altloc or "A"), max_chars=1)
        with c2:
            st.checkbox("Add receptor hydrogens with OpenBabel", key="prep_add_h")
            if st.session_state.prep_add_h:
                st.session_state.prep_ph = st.number_input("Target pH", min_value=0.0, max_value=14.0, value=float(st.session_state.prep_ph), step=0.1)
            st.checkbox("Repair with PDBFixer if installed", key="prep_repair")
            st.checkbox("Minimize with OpenMM if installed", key="prep_minimize")

    with tab_ligand:
        section_header("Ligand Preparation", "Controls for 3D embedding, cleanup, protonation, and force-field minimization.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.session_state.ligand_force_field = st.selectbox(
                "Ligand force field",
                ["MMFF94", "MMFF94s", "UFF"],
                index=["MMFF94", "MMFF94s", "UFF"].index(st.session_state.ligand_force_field),
            )
            st.checkbox("Strip salts / keep largest fragment", key="ligand_strip_salts")
            st.checkbox("Neutralize simple formal charges", key="ligand_neutralize")
        with c2:
            st.checkbox("Protonate ligands with OpenBabel pH", key="ligand_use_ph")
            if st.session_state.ligand_use_ph:
                st.session_state.ligand_ph = st.number_input("Ligand pH", min_value=0.0, max_value=14.0, value=float(st.session_state.ligand_ph), step=0.1)
            else:
                st.caption("pH protonation is off. Existing ligand protonation is preserved where possible.")

    with tab_runtime:
        section_header("Runtime", "Controls that affect how work is scheduled and whether previous pose files are reused.")
        c1, c2 = st.columns(2, gap="large")
        st.session_state.n_jobs = c1.slider("Parallel jobs", 1, 8, int(st.session_state.n_jobs))
        c2.checkbox("Resume existing output directory when possible", key="resume")

    can_continue = st.session_state.dock_engine == ENGINE_VINA or (
        st.session_state.dock_engine == ENGINE_GNINA
        and bool(gnina_status.get("gnina_available"))
        and not (st.session_state.gnina_backend == "NVIDIA CUDA" and not bool(gnina_status.get("nvidia_available")))
    )
    if st.button("Continue To Run", type="primary", use_container_width=True, disabled=not can_continue):
        st.session_state.settings_ready = True
        st.session_state.page = "Run"
        st.rerun()


def config_rows() -> list[str]:
    ligand_count = "unknown"
    invalid_count = "unknown"
    if st.session_state.ligand_bytes and st.session_state.ligand_name:
        tmp = ROOT / "runs" / "_ui_preview" / st.session_state.ligand_name
        if tmp.exists():
            try:
                records, errors = load_ligand_records_with_errors(tmp)
                ligand_count = str(len(records))
                invalid_count = str(len(errors))
            except Exception:
                pass
    pocket_count = st.session_state.top_pockets if st.session_state.box_mode == "Suggest pockets (fpocket)" else 1
    if st.session_state.box_mode == "Suggest pockets (fpocket)" and st.session_state.selected_pocket_ranks:
        pocket_count = len(st.session_state.selected_pocket_ranks)
    receptor_count = len(st.session_state.get("receptor_uploads") or [])
    workload = "unknown" if ligand_count == "unknown" else str(int(ligand_count) * int(pocket_count))
    if workload != "unknown" and receptor_count:
        workload = str(int(workload) * receptor_count)
    return [
        f"Receptors: {receptor_count or 0}",
        f"Ligands: {st.session_state.ligand_name or 'missing'}",
        f"Valid ligand records: {ligand_count}",
        f"Invalid ligand records: {invalid_count}",
        f"Box mode: {st.session_state.box_mode}",
        f"Pockets: {pocket_count if st.session_state.box_mode == 'Suggest pockets (fpocket)' else 'n/a'}",
        f"Engine: {st.session_state.dock_engine}",
        f"GNINA backend: {st.session_state.gnina_backend if st.session_state.dock_engine == ENGINE_GNINA else 'n/a'}",
        f"Estimated docking jobs: {workload}",
        f"Scoring: {st.session_state.scoring if st.session_state.dock_engine == ENGINE_VINA else 'GNINA affinity + CNN'}",
        f"Exhaustiveness: {st.session_state.exhaustiveness}",
        f"Modes: {st.session_state.num_modes}",
        f"Parallel jobs: {st.session_state.n_jobs}",
        f"Seed: {st.session_state.seed}",
        f"Ligand prep: {st.session_state.ligand_force_field}, "
        f"pH {st.session_state.ligand_ph if st.session_state.ligand_use_ph else 'off'}, "
        f"salts {'strip' if st.session_state.ligand_strip_salts else 'keep'}, "
        f"neutralize {'on' if st.session_state.ligand_neutralize else 'off'}",
        f"Prep: waters {'removed' if st.session_state.prep_remove_waters else 'kept'}, "
        f"hetero {'removed' if st.session_state.prep_remove_hetero else 'kept'}, "
        f"metals {'kept' if st.session_state.prep_keep_metals else 'removed'}",
    ]


def run_pipeline(log_callback=None) -> None:
    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    if not st.session_state.receptor_bytes or not st.session_state.ligand_bytes:
        raise ValueError("Upload receptor and ligand files first.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / f"ui_{timestamp}"
    stop_file = run_dir / "STOP"
    log(f"> Creating run folder: {run_dir.name}")
    engine_key = selected_engine_key()
    receptor_paths, ligand_path, reference_path = write_upload_bytes(run_dir)
    if not receptor_paths:
        raise ValueError("Upload at least one receptor PDB.")
    log(f"> Wrote {len(receptor_paths)} receptor file(s) and ligand library.")
    log(f"> Resolving binding-site boxes using: {st.session_state.box_mode}")
    boxes, boxes_by_receptor = selected_boxes_for_run(run_dir, receptor_paths, reference_path, log_callback=log)
    if not boxes:
        raise ValueError("No docking boxes were selected. Check binding-site settings.")
    log(f"> Prepared {len(boxes)} docking box(es).")
    write_active_run_control(run_dir=run_dir, stop_file=stop_file, engine=engine_key, box_mode=str(st.session_state.box_mode))
    log("> Safe stop is available from the Run page in another browser tab.")
    if stop_file.exists():
        log("> Stop requested before docking jobs started.")
        clear_active_run_control(run_dir)
        raise RuntimeError("Run stopped before docking jobs started.")
    ligand_prep_options = LigandPrepOptions(
        force_field=str(st.session_state.ligand_force_field),
        protonation_ph=float(st.session_state.ligand_ph) if st.session_state.ligand_use_ph else None,
        strip_salts=bool(st.session_state.ligand_strip_salts),
        neutralize=bool(st.session_state.ligand_neutralize),
    )
    gnina_cpu_only, gnina_device = selected_gnina_runtime()
    common_kwargs = dict(
        project_root=ROOT,
        ligand_file=ligand_path,
        box=boxes[0].box,
        boxes=boxes,
        output_dir=run_dir,
        exhaustiveness=int(st.session_state.exhaustiveness),
        seed=int(st.session_state.seed),
        num_modes=int(st.session_state.num_modes),
        energy_range=float(st.session_state.energy_range),
        scoring=st.session_state.scoring,
        engine=engine_key,
        gnina_cpu_only=gnina_cpu_only,
        gnina_device=gnina_device,
        add_hydrogens_ph=float(st.session_state.prep_ph) if st.session_state.prep_add_h else None,
        repair_with_pdbfixer=bool(st.session_state.prep_repair),
        minimize_with_openmm=bool(st.session_state.prep_minimize),
        remove_waters=bool(st.session_state.prep_remove_waters),
        remove_non_protein_heteroatoms=bool(st.session_state.prep_remove_hetero),
        keep_metals=bool(st.session_state.prep_keep_metals),
        default_altloc=str(st.session_state.prep_altloc or "A")[:1],
        n_jobs=int(st.session_state.n_jobs),
        resume=bool(st.session_state.resume),
        stop_file=stop_file,
        ligand_prep_options=ligand_prep_options,
        progress_callback=log,
    )
    try:
        if engine_key == ENGINE_GNINA:
            backend = "CPU only" if gnina_cpu_only else f"CUDA device {gnina_device or 'auto'}"
            log(f"> Starting GNINA pipeline ({backend}).")
        else:
            log("> Starting AutoDock Vina pipeline.")
        if len(receptor_paths) == 1:
            batch_result = run_batch_screen(receptor_pdb=receptor_paths[0], **common_kwargs)
        else:
            batch_result = run_ensemble_screen(receptor_pdbs=receptor_paths, boxes_by_receptor=boxes_by_receptor, **common_kwargs)
    finally:
        clear_active_run_control(run_dir)
    log("> Docking run finished. Building UI result state.")
    st.session_state.last_batch_dir = str(run_dir)
    receptor_reports = []
    if hasattr(batch_result, "receptor"):
        receptor_reports = [str(batch_result.receptor.report_json)]
    elif hasattr(batch_result, "runs"):
        receptor_reports = [str(run.receptor.report_json) for run in batch_result.runs]
    st.session_state.last_paths = {
        "results_csv": str(batch_result.results_csv),
        "results_json": str(batch_result.results_json),
        "summary_txt": str(batch_result.summary_txt),
        "box_json": str(getattr(batch_result, "box_json", "")),
        "boxes_json": str(batch_result.boxes_json),
        "best_by_ligand_csv": str(batch_result.best_by_ligand_csv),
        "best_by_pocket_csv": str(batch_result.best_by_pocket_csv),
        "manifest_json": str(batch_result.manifest_json),
        "receptor_report_json": receptor_reports[0] if receptor_reports else "",
        "receptor_report_jsons": json.dumps(receptor_reports),
        "html_report": str(batch_result.html_report),
        "pdf_report": str(batch_result.pdf_report),
        "events_log": str(batch_result.events_log),
        "stop_file": str(stop_file),
    }
    st.session_state.last_zip = zip_run_outputs(run_dir)
    st.session_state.last_table = result_dataframe(batch_result).to_json(orient="records")
    st.session_state.last_counts = {
        "total": len(batch_result.results),
        "success": sum(1 for result in batch_result.results if result.docking_status == "success"),
        "failed": sum(1 for result in batch_result.results if result.docking_status != "success"),
    }
    log(f"> Complete: {st.session_state.last_counts['success']} success / {st.session_state.last_counts['total']} total.")


def page_run() -> None:
    page_title("Run Docking", "Review the job setup, then execute the validated docking pipeline.")
    left, right = st.columns([0.86, 1.14], gap="large")
    with right:
        st.markdown('<div class="card-title">Live Terminal</div>', unsafe_allow_html=True)
        terminal_box = st.empty()
        stop_box = st.empty()

    run_clicked = False
    with left:
        st.markdown('<div class="card-title">Run Configuration</div>', unsafe_allow_html=True)
        st.markdown(terminal_block(config_rows(), limit=28, compact=True), unsafe_allow_html=True)
        gnina_status = gnina_hardware_status()
        if st.session_state.dock_engine == ENGINE_GNINA and not bool(gnina_status.get("gnina_available")):
            st.warning("GNINA is selected but the executable is missing. Return to Settings and select AutoDock Vina or install GNINA.")
        ready = bool(st.session_state.receptor_bytes and st.session_state.ligand_bytes)
        if not ready:
            st.warning("Upload receptor and ligand files first.")
        runnable = ready and (
            st.session_state.dock_engine == ENGINE_VINA
            or (
                st.session_state.dock_engine == ENGINE_GNINA
                and bool(gnina_status.get("gnina_available"))
                and not (st.session_state.gnina_backend == "NVIDIA CUDA" and not bool(gnina_status.get("nvidia_available")))
            )
        )
        st.markdown('<div class="primary-action-spacer"></div>', unsafe_allow_html=True)
        run_clicked = st.button("Run Docking", type="primary", use_container_width=True, disabled=not runnable)

    if run_clicked:
        try:
            terminal_lines = ["> StrataDock run requested."]

            def update_terminal(message: str) -> None:
                terminal_lines.append(message)
                terminal_box.markdown(terminal_block(terminal_lines, limit=80, autoscroll=True), unsafe_allow_html=True)

            update_terminal("> Validating inputs and settings.")
            render_stop_control(stop_box)
            with st.spinner("Preparing receptor, preparing ligands, and running docking..."):
                run_pipeline(log_callback=update_terminal)
            if st.session_state.get("last_paths", {}).get("events_log"):
                events_path = Path(st.session_state.last_paths["events_log"])
                if events_path.exists():
                    update_terminal("> Event log written.")
                    combined = terminal_lines[-24:] + events_path.read_text(encoding="utf-8").splitlines()[-24:]
                    terminal_box.markdown(terminal_block(combined, limit=80, autoscroll=True), unsafe_allow_html=True)
            st.success("Docking complete.")
            st.session_state.page = "Results"
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    else:
        terminal_text = "> Waiting for run.\n> Live progress appears here while docking is active."
        if st.session_state.get("last_paths", {}).get("events_log"):
            log_path = Path(st.session_state.last_paths["events_log"])
            if log_path.exists():
                terminal_text = log_path.read_text(encoding="utf-8")[-6000:]
        terminal_box.markdown(terminal_block(terminal_text, limit=80, autoscroll=True), unsafe_allow_html=True)
        render_stop_control(stop_box)


def available_run_artifacts(paths: dict[str, str]) -> list[dict[str, object]]:
    specs = [
        ("CSV", "results_csv", "stratadock_results.csv", "text/csv"),
        ("JSON", "results_json", "stratadock_results.json", "application/json"),
        ("Summary", "summary_txt", "run_summary.txt", "text/plain"),
        ("Boxes", "boxes_json", "boxes.json", "application/json"),
        ("Best Ligand", "best_by_ligand_csv", "best_by_ligand.csv", "text/csv"),
        ("Best Pocket", "best_by_pocket_csv", "best_by_pocket.csv", "text/csv"),
        ("HTML", "html_report", "stratadock_report.html", "text/html"),
        ("PDF Report", "pdf_report", "stratadock_report.pdf", "application/pdf"),
        ("Events", "events_log", "run_events.jsonl", "application/jsonl"),
    ]
    artifacts: list[dict[str, object]] = []
    for label, key, filename, mime in specs:
        raw = paths.get(key)
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            artifacts.append({"label": label, "path": path, "filename": filename, "mime": mime})
    return artifacts


def refresh_pdf_artifact(paths: dict[str, str]) -> bool:
    pdf_raw = paths.get("pdf_report")
    if not pdf_raw or "last_table" not in st.session_state:
        return False
    pdf_path = Path(pdf_raw)
    try:
        df = load_last_df()
        if df.empty:
            return False
        manifest_path = Path(paths.get("manifest_json") or pdf_path.with_name("run_manifest.json"))
        summary_path = Path(paths.get("summary_txt") or pdf_path.with_name("run_summary.txt"))
        title = "StrataDock Screening Report"
        write_run_pdf_report(
            title=title,
            results=df.to_dict("records"),
            output_path=pdf_path,
            manifest_json=manifest_path,
            summary_txt=summary_path,
        )
        return True
    except Exception as exc:
        st.warning(f"Could not refresh PDF report: {exc}")
        return False


def artifact_download_picker(artifacts: list[dict[str, object]], *, key_prefix: str) -> None:
    if not artifacts:
        st.info("No downloadable artifacts are available for this selection.")
        return
    st.markdown('<div class="subtle-rule"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.8, 0.8])
    selected = c1.selectbox(
        "Artifact",
        list(range(len(artifacts))),
        format_func=lambda idx: str(artifacts[idx]["label"]),
        key=f"{key_prefix}_artifact_choice",
        label_visibility="collapsed",
    )
    artifact = artifacts[selected]
    data = artifact.get("data")
    if data is None:
        data = Path(artifact["path"]).read_bytes()
    c2.download_button(
        "Download",
        data,
        str(artifact["filename"]),
        mime=str(artifact["mime"]),
        use_container_width=True,
        key=f"{key_prefix}_artifact_download",
    )


def download_grid(paths: dict[str, str]) -> None:
    st.markdown("**Run Artifacts**")
    pdf_refreshed = refresh_pdf_artifact(paths)
    if pdf_refreshed and st.session_state.get("last_batch_dir"):
        st.session_state.last_zip = zip_run_outputs(Path(st.session_state.last_batch_dir))
    artifacts = available_run_artifacts(paths)
    if st.session_state.get("last_zip"):
        artifacts.append(
            {
                "label": "Full Session ZIP",
                "data": st.session_state.last_zip,
                "filename": "stratadock_run.zip",
                "mime": "application/zip",
            }
        )
    artifact_download_picker(artifacts, key_prefix="run_downloads")


def _score_column_style(series: pd.Series) -> list[str]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return [""] * len(series)
    best = float(finite.min())
    worst = float(finite.max())
    span = worst - best
    styles: list[str] = []
    for value in values:
        if pd.isna(value):
            styles.append("")
            continue
        ratio = 0.0 if span == 0 else max(0.0, min(1.0, (float(value) - best) / span))
        red = int(58 + ratio * 142)
        green = int(168 - ratio * 86)
        styles.append(
            f"background-color: rgba({red}, {green}, 96, 0.34); color: #f8f8f8; font-weight: 800;"
        )
    return styles


def styled_score_table(df: pd.DataFrame, *, score_col: str = "docking_score"):
    if df.empty or score_col not in df.columns:
        return df
    return df.style.apply(_score_column_style, subset=[score_col])


def compact_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "ligand_name",
        "receptor_name",
        "pocket_name",
        "docking_score",
        "docking_status",
        "rule_of_five_pass",
        "warnings",
        "docking_engine",
        "exhaustiveness",
        "num_modes",
    ]
    return format_bool_columns(df[[column for column in preferred if column in df.columns]])


def compact_best_rows(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    rows = best_rows(df, group_field).rename(columns={"vina_score": "docking_score"})
    preferred = [group_field, "ligand_name", "pocket_name", "docking_score", "rule_of_five_pass", "warnings"]
    columns = []
    for column in preferred:
        if column in rows.columns and column not in columns:
            columns.append(column)
    return format_bool_columns(rows[columns]) if columns else rows


def _bool_text(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return "Pass" if bool(value) else "Fail"


def format_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["rule_of_five_pass", "veber_pass"]:
        if column in formatted:
            formatted[column] = formatted[column].map(_bool_text)
    return formatted


def compact_admet_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ligand_name",
        "vina_score",
        "molecular_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "qed",
        "lipinski_failures",
        "rotatable_bonds",
        "aromatic_rings",
        "fraction_csp3",
        "rule_of_five_pass",
        "veber_pass",
        "bbb_penetration",
        "herg_risk",
        "hepatotoxicity_risk",
        "mutagenicity_risk",
        "structural_alert_count",
    ]
    out = df[[column for column in columns if column in df.columns]].copy()
    rename = {
        "ligand_name": "Ligand",
        "vina_score": "Score",
        "molecular_weight": "MW",
        "logp": "LogP",
        "tpsa": "TPSA",
        "hbd": "HBD",
        "hba": "HBA",
        "qed": "QED",
        "lipinski_failures": "Lipinski Fails",
        "rotatable_bonds": "Rot Bonds",
        "aromatic_rings": "Aromatic",
        "fraction_csp3": "Fsp3",
        "rule_of_five_pass": "Rule of Five",
        "veber_pass": "Veber",
        "bbb_penetration": "BBB",
        "herg_risk": "hERG",
        "hepatotoxicity_risk": "Hepatotoxicity",
        "mutagenicity_risk": "Mutagenicity",
        "structural_alert_count": "Alerts",
    }
    out = out.rename(columns=rename)
    for column in ["Score", "MW", "LogP", "TPSA", "QED", "Fsp3"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda value: "n/a" if pd.isna(value) else f"{value:.3f}" if column in {"Score", "QED"} else f"{value:.2f}")
    for column in ["HBD", "HBA", "Lipinski Fails", "Rot Bonds", "Aromatic", "Alerts"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(lambda value: "n/a" if pd.isna(value) else str(int(value)))
    for column in ["Rule of Five", "Veber"]:
        if column in out:
            out[column] = out[column].map(_bool_text)
    return out


def admet_summary_tiles(df: pd.DataFrame) -> None:
    summary = admet_summary(df)
    html = f"""
<div class="summary-line">
  <span><b>{summary['pass']}</b> pass</span>
  <span><b>{summary['warn']}</b> warning</span>
  <span><b>{summary['fail']}</b> fail</span>
  <span><b>{summary['total']}</b> ligand(s)</span>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def warning_pills(row: pd.Series) -> None:
    warnings = str(row.get("warnings", "") or "")
    if not warnings:
        st.caption("No automatic warning flags for this row.")
        return
    pills = "".join(f'<span class="warning-pill">{item.strip()}</span>' for item in warnings.split(";") if item.strip())
    st.markdown(pills, unsafe_allow_html=True)


def _traffic_class(label: str, value: object) -> str:
    if pd.isna(value):
        return "warn"
    if isinstance(value, bool):
        return "good" if value else "bad"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "good" if bool(value) else "bad"
    if label == "MW":
        return "good" if number <= 500 else "bad"
    if label == "LogP":
        return "good" if number <= 5 else "bad"
    if label == "TPSA":
        return "good" if number <= 140 else "warn"
    if label == "QED":
        return "good" if number >= 0.5 else "warn"
    if label in {"Lipinski", "Alerts"}:
        return "good" if number == 0 else "bad"
    if label == "Rot Bonds":
        return "good" if number <= 10 else "warn"
    if label == "Veber":
        return "good" if bool(value) else "bad"
    return "good"


def _format_profile_value(value: object, fmt: str) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, bool):
        return "Pass" if value else "Fail"
    try:
        return fmt.format(float(value) if "." in fmt else int(value))
    except (TypeError, ValueError):
        return str(value)


def molecular_profile_inspector(row: pd.Series) -> None:
    specs = [
        ("MW", row.get("molecular_weight"), "{:.1f}"),
        ("LogP", row.get("logp"), "{:.2f}"),
        ("TPSA", row.get("tpsa"), "{:.1f}"),
        ("HBD", row.get("hbd"), "{}"),
        ("HBA", row.get("hba"), "{}"),
        ("QED", row.get("qed"), "{:.3f}"),
        ("Lipinski", row.get("lipinski_failures"), "{}"),
        ("Rot Bonds", row.get("rotatable_bonds"), "{}"),
        ("Aromatic", row.get("aromatic_rings"), "{}"),
        ("Fsp3", row.get("fraction_csp3"), "{:.2f}"),
        ("Charge", row.get("formal_charge"), "{}"),
        ("Alerts", row.get("structural_alert_count"), "{}"),
        ("Veber", row.get("veber_pass"), "{}"),
    ]
    tiles: list[str] = []
    for label, value, fmt in specs:
        status = _traffic_class(label, value)
        text = _format_profile_value(value, fmt)
        tiles.append(
            f'<div class="profile-tile {status}">'
            f'<div class="profile-name">{html.escape(label)}</div>'
            f'<div class="profile-number">{html.escape(text)}</div>'
            f'</div>'
        )
    st.markdown('<div class="profile-grid">' + "".join(tiles) + "</div>", unsafe_allow_html=True)


def risk_pill_html(row: pd.Series) -> str:
    def pill(label: str, value: object, good_values: set[str]) -> str:
        text = "n/a" if pd.isna(value) else str(value)
        good = text.lower() in good_values
        klass = "good" if good else "bad"
        return f'<span class="risk-pill {klass}">{html.escape(label)}: <b>{html.escape(text)}</b></span>'

    return (
        '<div class="risk-row">'
        + pill("BBB", row.get("bbb_penetration"), {"likely"})
        + pill("hERG", row.get("herg_risk"), {"low"})
        + pill("Hepatotoxicity", row.get("hepatotoxicity_risk"), {"low"})
        + pill("Mutagenicity", row.get("mutagenicity_risk"), {"low"})
        + "</div>"
    )


def interaction_summary(interactions_df: pd.DataFrame) -> None:
    if interactions_df.empty:
        return
    type_col = "interaction_type" if "interaction_type" in interactions_df.columns else "type" if "type" in interactions_df.columns else interactions_df.columns[0]
    counts = interactions_df[type_col].fillna("interaction").astype(str).value_counts()
    chips = "".join(
        f'<span class="contact-chip"><b>{int(count)}</b> {html.escape(str(name).replace("_", " ").title())}</span>'
        for name, count in counts.items()
    )
    st.markdown(f'<div class="contact-chip-row">{chips}</div>', unsafe_allow_html=True)


def _residue_label(row: pd.Series) -> str:
    residue = str(row.get("residue_name", "") or "").strip()
    chain = str(row.get("chain_id", "") or "").strip()
    seq = str(row.get("residue_seq", "") or "").strip()
    return f"{residue} {chain}:{seq}".strip()


def binding_site_panel(interactions_df: pd.DataFrame, selected_row: pd.Series) -> None:
    if interactions_df.empty:
        st.info("No binding-site contacts were detected for this pose.")
        return
    df = interactions_df.copy()
    df["residue"] = df.apply(_residue_label, axis=1)
    df["interaction_type"] = df.get("interaction_type", pd.Series(["interaction"] * len(df))).fillna("interaction").astype(str)
    df["distance_angstrom"] = pd.to_numeric(df.get("distance_angstrom"), errors="coerce")
    residues = df["residue"].replace("", pd.NA).dropna()
    closest = df["distance_angstrom"].dropna().min()
    polar_count = int(df["interaction_type"].str.contains("polar|hydrogen", case=False, regex=True).sum())
    hydrophobic_count = int(df["interaction_type"].str.contains("hydrophobic", case=False, regex=False).sum())
    overview = [
        ("Contacts", len(df)),
        ("Residues", residues.nunique()),
        ("Closest", f"{closest:.2f} A" if pd.notna(closest) else "n/a"),
        ("Polar / Hydrophobic", f"{polar_count} / {hydrophobic_count}"),
    ]
    tiles = "".join(
        f'<div class="binding-tile"><div class="binding-value">{html.escape(str(value))}</div><div class="binding-label">{html.escape(label)}</div></div>'
        for label, value in overview
    )
    st.markdown('<div class="binding-overview">' + tiles + "</div>", unsafe_allow_html=True)
    interaction_summary(df)

    top_residues = (
        df.groupby("residue", dropna=True)
        .agg(
            contacts=("residue", "size"),
            closest_A=("distance_angstrom", "min"),
            interaction_types=("interaction_type", lambda values: ", ".join(sorted(set(str(v).replace("_", " ") for v in values)))),
            atoms=("receptor_atom", lambda values: ", ".join(sorted(set(str(v) for v in values if str(v) != "nan"))[:5])),
        )
        .reset_index()
        .sort_values(["contacts", "closest_A"], ascending=[False, True])
    )
    if not top_residues.empty:
        anchor = top_residues.iloc[0]
        st.markdown(
            f'<div class="binding-note">Main contact hotspot: <b>{html.escape(str(anchor["residue"]))}</b> '
            f'with {int(anchor["contacts"])} contact(s). Lower distances usually indicate tighter local contact geometry, '
            f'but final ranking should still use the docking/CNN scores across pockets.</div>',
            unsafe_allow_html=True,
        )
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown('<div class="table-section-title">Residue Hotspots</div>', unsafe_allow_html=True)
        st.dataframe(top_residues.head(12), use_container_width=True, hide_index=True)
    with c2:
        closest_contacts = df.sort_values("distance_angstrom").head(14)
        closest_contacts = closest_contacts[
            [
                column
                for column in [
                    "interaction_type",
                    "residue",
                    "receptor_atom",
                    "ligand_element",
                    "ligand_atom_index",
                    "distance_angstrom",
                ]
                if column in closest_contacts.columns
            ]
        ].rename(columns={"distance_angstrom": "distance_A"})
        st.markdown('<div class="table-section-title">Closest Contacts</div>', unsafe_allow_html=True)
        st.dataframe(closest_contacts, use_container_width=True, hide_index=True)

    with st.expander("Raw interaction table"):
        st.dataframe(interactions_df, use_container_width=True, hide_index=True)


def viewer_html_for_row(row: pd.Series) -> str:
    complex_path = Path(str(row.get("complex_pdb") or ""))
    if not complex_path.is_file():
        viewer_path = Path(str(row.get("viewer_html") or ""))
        return viewer_path.read_text(encoding="utf-8") if viewer_path.is_file() else ""
    interactions: list[dict[str, object]] = []
    interactions_path = Path(str(row.get("interactions_csv") or ""))
    if interactions_path.exists():
        interactions = pd.read_csv(interactions_path).to_dict("records")
    custom_path = complex_path.with_name(
        f"{complex_path.stem}_{st.session_state.viewer_receptor_style}_{st.session_state.viewer_ligand_style}_viewer.html"
    )
    write_3dmol_viewer_html(
        complex_pdb=complex_path,
        output_html=custom_path,
        title=f"{row.get('receptor_name', 'receptor')} / {row.get('pocket_name', 'pocket')} / {row.get('ligand_name', 'ligand')}",
        receptor_style=str(st.session_state.viewer_receptor_style),
        ligand_style=str(st.session_state.viewer_ligand_style),
        surface_opacity=float(st.session_state.viewer_surface_opacity),
        show_interactions=bool(st.session_state.viewer_show_interactions),
        interactions=interactions,
        title_metadata={"score": row.get("vina_score", "n/a"), "status": row.get("docking_status", "n/a")},
    )
    return custom_path.read_text(encoding="utf-8")


def results_filter_panel(df: pd.DataFrame) -> pd.DataFrame:
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 0.8])
        ligand_query = c1.text_input("Filter ligand", value=str(st.session_state.result_ligand_filter))
        st.session_state.result_ligand_filter = ligand_query
        pockets = sorted(str(value) for value in df.get("pocket_name", pd.Series(dtype=str)).dropna().unique())
        statuses = sorted(str(value) for value in df.get("docking_status", pd.Series(dtype=str)).dropna().unique())
        pocket_choice = c2.selectbox("Pocket", ["All pockets"] + pockets)
        status_choice = c3.selectbox("Status", ["All statuses"] + statuses)
        selected_pockets = pockets if pocket_choice == "All pockets" else [pocket_choice]
        selected_statuses = statuses if status_choice == "All statuses" else [status_choice]
        admet_mode = c4.selectbox(
            "ADMET",
            ["all", "pass", "fail"],
            index=["all", "pass", "fail"].index(st.session_state.result_admet_filter),
        )
        st.session_state.result_admet_filter = admet_mode
        max_score = None
        if "vina_score" in df:
            scores = pd.to_numeric(df["vina_score"], errors="coerce").dropna()
            if not scores.empty:
                bounds = score_filter_bounds(scores)
                if bounds:
                    max_score = st.slider("Maximum docking score", bounds[0], bounds[1], bounds[1], 0.1)
                else:
                    max_score = float(scores.iloc[0])
                    st.caption(f"Docking score filter fixed at {max_score:.3f}; all visible scores are identical.")
        return filter_results(
            df,
            ligand_query=ligand_query,
            pockets=selected_pockets,
            statuses=selected_statuses,
            max_score=max_score,
            admet_mode=admet_mode,
        )


def page_results() -> None:
    if "last_table" not in st.session_state:
        page_title("Results", "Ranked docking outputs, ADMET descriptors, binding contacts, viewer files, and reports.")
        st.warning("No run results yet. Run docking first.")
        if st.button("Go To Run", type="primary"):
            st.session_state.page = "Run"
            st.rerun()
        return

    counts = st.session_state.last_counts
    df = load_last_df()
    engine_label = "Docking"
    if not df.empty and "docking_engine" in df:
        engines = [str(value) for value in df["docking_engine"].dropna().unique() if str(value)]
        if engines:
            engine_label = " / ".join(sorted(engines))
    pocket_count = 0
    if not df.empty and "pocket_name" in df:
        pocket_count = len([value for value in df["pocket_name"].dropna().unique() if str(value)])
    page_title(
        "Screening Results",
        f"{counts['success']} successful docks - {engine_label} engine - {pocket_count or 'no'} pocket(s)",
    )
    best_score = "n/a"
    if not df.empty and "vina_score" in df:
        scores = pd.to_numeric(df["vina_score"], errors="coerce").dropna()
        if not scores.empty:
            best_score = f"{scores.min():.3f}"
    result_summary_strip(
        [
            ("Total Jobs", counts["total"]),
            ("Successful", counts["success"]),
            ("Failed / Skipped", counts["failed"]),
            ("Best Vina", best_score),
        ]
    )
    df = add_warning_flags(df)
    filtered_df = results_filter_panel(df)
    st.caption(f"Showing {len(filtered_df)} of {len(df)} result rows.")

    tab_scores, tab_admet, tab_binding, tab_viewer, tab_downloads = st.tabs(
        ["Docking Scores", "Pharmacokinetics", "Binding Site", "3D Viewer", "Downloads"]
    )
    with tab_scores:
        display_df = filtered_df.rename(columns={"vina_score": "docking_score", "vina_scores": "docking_scores"})
        st.dataframe(styled_score_table(compact_results_columns(display_df)), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "Export Filtered CSV",
            csv_bytes(filtered_df.to_dict("records")),
            "stratadock_filtered_results.csv",
            "text/csv",
            use_container_width=True,
        )
        if st.session_state.get("last_batch_dir"):
            c1.download_button(
                "Export Filtered Hits ZIP",
                zip_selected_hit_outputs(Path(st.session_state.last_batch_dir), filtered_df.to_dict("records")),
                "stratadock_filtered_hits.zip",
                "application/zip",
                use_container_width=True,
            )
        st.markdown('<div class="subtle-rule"></div>', unsafe_allow_html=True)
        c2a, c2b = st.columns(2)
        if "ligand_name" in filtered_df:
            c2a.markdown('<div class="table-section-title">Best Pose Per Ligand</div>', unsafe_allow_html=True)
            c2a.dataframe(
                styled_score_table(compact_best_rows(filtered_df, "ligand_name")),
                use_container_width=True,
                hide_index=True,
            )
        if "pocket_name" in filtered_df:
            c2b.markdown('<div class="table-section-title">Best Ligand Per Pocket</div>', unsafe_allow_html=True)
            c2b.dataframe(
                styled_score_table(compact_best_rows(filtered_df, "pocket_name")),
                use_container_width=True,
                hide_index=True,
            )
    with tab_admet:
        admet_summary_tiles(filtered_df)
        admet_df = filtered_df[admet_columns(filtered_df)]
        st.dataframe(compact_admet_table(filtered_df), use_container_width=True, hide_index=True, height=280)
        st.download_button(
            "Download ADMET CSV",
            csv_bytes(admet_df.to_dict("records")),
            "stratadock_admet.csv",
            "text/csv",
            use_container_width=True,
        )
        if not admet_df.empty:
            labels = result_choice_labels(filtered_df)
            selected = st.selectbox("ADMET ligand detail", list(range(len(filtered_df))), format_func=lambda idx: labels[idx])
            row = filtered_df.reset_index(drop=True).iloc[selected]
            st.markdown('<div class="table-section-title">Molecular Profile Inspector</div>', unsafe_allow_html=True)
            st.markdown('<div class="admet-caption">Traffic-light descriptors follow the old StrataDock pharmacokinetics layout: green is acceptable, yellow is marginal, red needs attention.</div>', unsafe_allow_html=True)
            molecular_profile_inspector(row)
            st.markdown('<div class="table-section-title">Toxicity / Safety Flags</div>', unsafe_allow_html=True)
            st.markdown(risk_pill_html(row), unsafe_allow_html=True)
            warning_pills(row)
    with tab_binding:
        success_df = filtered_df[filtered_df["interactions_csv"].notna()] if "interactions_csv" in filtered_df else pd.DataFrame()
        if success_df.empty:
            st.info("No interaction report available yet.")
        else:
            indexed = success_df.reset_index(drop=True)
            labels = result_choice_labels(indexed)
            selected = st.selectbox("Binding site ligand", list(range(len(indexed))), format_func=lambda idx: labels[idx])
            row = indexed.iloc[selected]
            interactions_path = Path(row["interactions_csv"])
            st.markdown(f"**{row.get('ligand_name', 'ligand')}** in **{row.get('pocket_name', 'pocket')}** / docking score `{row.get('vina_score', 'n/a')}`")
            if interactions_path.exists():
                interactions_df = pd.read_csv(interactions_path)
                binding_site_panel(interactions_df, row)
    with tab_viewer:
        success_df = filtered_df[filtered_df["viewer_html"].notna()] if "viewer_html" in filtered_df else pd.DataFrame()
        if success_df.empty:
            st.info("No viewer file available yet.")
        else:
            vc1, vc2, vc3, vc4 = st.columns(4)
            st.session_state.viewer_receptor_style = vc1.selectbox("Receptor style", ["cartoon", "stick", "surface"], index=["cartoon", "stick", "surface"].index(st.session_state.viewer_receptor_style))
            st.session_state.viewer_ligand_style = vc2.selectbox("Ligand style", ["stick", "sphere"], index=["stick", "sphere"].index(st.session_state.viewer_ligand_style))
            st.session_state.viewer_surface_opacity = vc3.slider("Surface opacity", 0.0, 0.6, float(st.session_state.viewer_surface_opacity), 0.05)
            vc4.checkbox("Show interactions", key="viewer_show_interactions")
            indexed = success_df.reset_index(drop=True)
            labels = result_choice_labels(indexed)
            selected = st.selectbox(
                "Viewer ligand",
                list(range(len(indexed))),
                format_func=lambda idx: labels[idx],
                key="viewer_ligand",
            )
            row = indexed.iloc[selected]
            st.markdown(f"**{row.get('ligand_name', 'ligand')}** / {row.get('pocket_name', 'pocket')} / docking score `{row.get('vina_score', 'n/a')}`")
            warning_pills(row)
            html_text = viewer_html_for_row(row)
            if html_text:
                components.html(html_text, height=680, scrolling=False)
            viewer_artifacts = []
            if row.get("complex_pdb"):
                cpx = Path(row["complex_pdb"])
                if cpx.exists():
                    viewer_artifacts.append({"label": "Complex PDB", "path": cpx, "filename": cpx.name, "mime": "chemical/x-pdb"})
            if row.get("pose_pdbqt"):
                pose = Path(row["pose_pdbqt"])
                if pose.exists():
                    viewer_artifacts.append({"label": "Pose PDBQT", "path": pose, "filename": pose.name, "mime": "text/plain"})
            if row.get("pymol_script"):
                pml = Path(row["pymol_script"])
                if pml.exists():
                    viewer_artifacts.append({"label": "PyMOL Script", "path": pml, "filename": pml.name, "mime": "text/plain"})
            artifact_download_picker(viewer_artifacts, key_prefix="viewer_downloads")
    with tab_downloads:
        download_grid(st.session_state.last_paths)

    c1, c2 = st.columns(2)
    if c1.button("Run Again With Different Settings", use_container_width=True):
        st.session_state.page = "Run"
        st.rerun()
    if c2.button("Back To Upload", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()


def page_help() -> None:
    page_title("Help And Documentation", "Quick operational notes for the lean v 1.6.01 workflow.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Quick Start", "Binding Sites", "Outputs", "Diagnostics", "Deferred"])
    with tab1:
        st.markdown(
            """
### How to run

1. Upload one receptor `.pdb` and ligand file `.sdf`, `.smi`, `.txt`, or `.zip`.
2. Use fpocket when you do not have a reference ligand.
3. Use reference ligand mode when you have a known co-crystal ligand.
4. Keep docking defaults for first-pass screening, then increase exhaustiveness for final checks.
5. Review results and download the full session ZIP.
"""
        )
    with tab2:
        st.markdown(
            """
### Defining the binding site

- **fpocket** predicts geometric pockets from the protein alone.
- **Reference ligand** builds the box around known ligand coordinates.
- **Manual** is for known center and size coordinates.

For no-reference workflows, top 3 fpocket pockets is the practical default; top 5 is safer but slower.
"""
        )
    with tab3:
        st.markdown(
            """
### Generated outputs

- Ranked docking table: `results.csv` and `results.json`
- Best summaries: `best_by_ligand.csv`, `best_by_pocket.csv`
- Complex structures: merged receptor + ligand PDB files
- Interaction reports: JSON and CSV
- Visualization: PyMOL script and browser 3Dmol HTML
- Reports: HTML and PDF
- Session archive: full ZIP with `session_manifest.json`
"""
        )
    with tab4:
        diagnostics = environment_diagnostics(ROOT)
        st.markdown("### Environment diagnostics")
        show_gnina_hardware_status()
        s1, s2, s3 = st.columns(3)
        metric_card(s1, "OK", diagnostics["summary"]["ok"])
        metric_card(s2, "Warnings", diagnostics["summary"]["warn"])
        metric_card(s3, "Errors", diagnostics["summary"]["error"])
        st.dataframe(pd.DataFrame(diagnostics["checks"]), use_container_width=True, hide_index=True)
    with tab5:
        st.markdown(
            """
### Engine notes

- AutoDock Vina is the default CPU-safe engine.
- GNINA is available when the `gnina` executable is installed in the project tools folder or PATH.
- GNINA GPU acceleration expects NVIDIA CUDA hardware.
- AMD GPUs are detected and explained, but official GNINA GPU acceleration is NVIDIA CUDA-only; use AutoDock Vina or GNINA CPU mode on AMD systems.
"""
        )


def render_footer() -> None:
    st.markdown(
        """
<div class="custom-footer">
  StrataDock v 1.6.01
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    apply_style()
    render_stepper()
    page = st.session_state.page
    if page == "Upload":
        page_upload()
    elif page == "Settings":
        page_settings()
    elif page == "Run":
        page_run()
    elif page == "Results":
        page_results()
    elif page == "Help":
        page_help()
    render_footer()


if __name__ == "__main__":
    main()
