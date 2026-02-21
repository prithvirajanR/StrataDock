#!/usr/bin/env bash
# =============================================================================
# StrataDock Setup Script — One-Click Installer
# Supports: Ubuntu/Debian on WSL2, native Linux, NVIDIA, or CPU-only
# Usage: bash setup.sh
# =============================================================================

set -e  # Exit on any error

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}\n"; }

CONDA_ENV="stratadock"
INSTALL_DIR="$HOME/stratadock"

# =============================================================================
echo -e "${CYAN}${BOLD}"
cat << 'SPLASH'

        ╭──────────────────────────────────────────╮
        │                                          │
        │      ⚗️  S T R A T A D O C K  ⚗️         │
        │                                          │
        │      Because drugs don't dock             │
        │      themselves.                          │
        │                                          │
        │            O                              │
        │           / \                             │
        │      HO--C   C==O                         │
        │           \ /                             │
        │            C                              │
        │           / \                             │
        │          H   OH                           │
        │                                          │
        │      Alpha v0.7                           │
        │      Prithvi Rajan                        │
        │      Freie Universität Berlin             │
        │                                          │
        ╰──────────────────────────────────────────╯

SPLASH
echo -e "${NC}"
# =============================================================================

# ── 1. Detect GPU ────────────────────────────────────────────────────────────
header "Step 1/6 — Sniffing for GPUs 🔍"

GPU_MODE="cpu"
GNINA_URL="https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_MODE="nvidia"
    GNINA_URL="https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8"
    success "NVIDIA GPU detected — GNINA will use CUDA"
else
    warn "No NVIDIA GPU detected — GNINA will run on CPU"
fi
echo "GPU_MODE=$GPU_MODE" > "$HOME/.stratadock_gpu"

# ── 2. System packages ───────────────────────────────────────────────────────
header "Step 2/6 — Waking up the package manager ☕"
info "You may be prompted for your Linux password once..."
sudo apt-get update -qq
sudo apt-get install -y -qq wget curl git build-essential libxrender1 libxtst6 \
    libxi6 libglu1-mesa libgl1 libglib2.0-0 2>/dev/null || true
success "System packages ready"

# ── 3. Conda check ───────────────────────────────────────────────────────────
header "Step 3/6 — Summoning the snake 🐍"

if ! command -v conda &>/dev/null; then
    info "Conda not found — installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    "$HOME/miniconda3/bin/conda" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    source "$HOME/.bashrc"
    CONDA_BIN="$HOME/miniconda3/bin/conda"
    success "Miniconda installed"
else
    CONDA_BIN="$(which conda)"
    success "Conda found: $(conda --version)"
fi

# ── 4. Create conda env + install packages ───────────────────────────────────
header "Step 4/6 — Brewing the chemistry toolkit 🧪"

CONDA_BASE=$("$CONDA_BIN" info --base)

if "$CONDA_BIN" env list | grep -q "^$CONDA_ENV "; then
    info "Updating existing environment '$CONDA_ENV'..."
    "$CONDA_BIN" install -n "$CONDA_ENV" -c conda-forge python=3.10 rdkit openbabel fpocket numpy scipy pandas biopython pillow gemmi joblib -y -q
    "$CONDA_BIN" install -n "$CONDA_ENV" -c nvidia cuda-toolkit=12.8 cudnn cuda-nvtx -y -q
    success "Environment updated"
else
    info "Creating conda environment '$CONDA_ENV' with Python 3.10..."
    "$CONDA_BIN" create -n "$CONDA_ENV" python=3.10 -y
    "$CONDA_BIN" install -n "$CONDA_ENV" -c conda-forge rdkit openbabel fpocket numpy scipy pandas biopython pillow gemmi joblib -y -q
    "$CONDA_BIN" install -n "$CONDA_ENV" -c nvidia cuda-toolkit=12.8 cudnn cuda-nvtx -y -q
    success "Conda environment created and packages installed"
fi

PIP="$CONDA_BASE/envs/$CONDA_ENV/bin/pip"
PYTHON="$CONDA_BASE/envs/$CONDA_ENV/bin/python"

info "Installing pure Python packages via Pip..."
"$PIP" install --quiet \
    meeko \
    vina \
    streamlit \
    stmol \
    py3Dmol \
    ipython_genutils \
    plotly \
    fpdf2

success "All Python packages installed"

# ── 5. GNINA binary ─────────────────────────────────────────────────────────
header "Step 5/6 — Downloading the AI brain 🧠"

GNINA_BIN="$HOME/.local/bin/gnina"
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_BASE/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH"

if [ -x "$GNINA_BIN" ] && "$GNINA_BIN" --version &>/dev/null; then
    warn "GNINA already installed at $GNINA_BIN — skipping download"
else
    info "Downloading GNINA v1.3.2 static binary from GitHub (~1.4 GB)..."
    info "This may take a few minutes depending on your internet speed."
    curl -L --progress-bar "$GNINA_URL" -o "$GNINA_BIN"
    chmod +x "$GNINA_BIN"
    success "GNINA installed at $GNINA_BIN"
fi

if "$GNINA_BIN" --version &>/dev/null; then
    success "GNINA binary verified: $($GNINA_BIN --version 2>&1 | head -1)"
else
    warn "GNINA verification failed — it may still work at runtime with LD_LIBRARY_PATH"
fi

# ── 6. Project dir + launcher ────────────────────────────────────────────────
header "Step 6/6 — Assembling the launchpad 🚀"

mkdir -p "$INSTALL_DIR"

cat > "$INSTALL_DIR/run.sh" << RUNSCRIPT
#!/usr/bin/env bash
# StrataDock launcher
CONDA_BASE=\$(conda info --base 2>/dev/null || echo "\$HOME/miniconda3")
source "\$CONDA_BASE/etc/profile.d/conda.sh"
conda activate stratadock

# Export paths so GNINA binary and CUDA libs can be found
export PATH="\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="\$CONDA_BASE/envs/stratadock/lib:\$LD_LIBRARY_PATH"

cd "\$HOME/stratadock"
echo ""
echo "  StrataDock is starting..."
echo "  Open your browser at: http://localhost:8501"
echo ""
"\$CONDA_BASE/envs/stratadock/bin/streamlit" run app.py \\
    --server.port 8501 \\
    --server.address 0.0.0.0 \\
    --server.headless true \\
    --browser.gatherUsageStats false
RUNSCRIPT
chmod +x "$INSTALL_DIR/run.sh"
success "Launcher script created at $INSTALL_DIR/run.sh"

# ── Health check ─────────────────────────────────────────────────────────────
header "Health Check"

"$PYTHON" - << 'PYCHECK'
checks = {
    "RDKit":      "from rdkit import Chem; from rdkit.Chem import AllChem",
    "Meeko":      "import meeko",
    "Streamlit":  "import streamlit",
    "py3Dmol":    "import py3Dmol",
    "stmol":      "import stmol",
    "Pandas":     "import pandas",
    "Joblib":     "import joblib",
    "BioPython":  "import Bio",
    "Vina":       "import vina",
    "Plotly":     "import plotly",
    "FPDF2":      "import fpdf",
}
all_ok = True
for name, stmt in checks.items():
    try:
        exec(stmt)
        print(f"  ✅  {name}")
    except ImportError as e:
        print(f"  ❌  {name}: {e}")
        all_ok = False
if all_ok:
    print("\n  All packages OK!")
else:
    print("\n  Some packages failed — re-run setup.sh to retry.")
PYCHECK

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   ⚗️  StrataDock Alpha v0.7 — Setup Complete!            ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   Developed by Prithvi Rajan                             ║${NC}"
echo -e "${GREEN}${BOLD}║   Bioinformatics, Freie Universität Berlin               ║${NC}"
echo -e "${GREEN}${BOLD}║   © 2026                                                 ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   GPU Mode: $(printf '%-45s' "${GPU_MODE^^}")║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   To start the app:                                      ║${NC}"
echo -e "${GREEN}${BOLD}║     bash ~/stratadock/run.sh                              ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   Then open in browser:                                  ║${NC}"
echo -e "${GREEN}${BOLD}║     http://localhost:8501                                 ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║   May your binding energies always be negative.      ║${NC}"
echo -e "${GREEN}${BOLD}║   Now go find a cure. 🧬                              ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
