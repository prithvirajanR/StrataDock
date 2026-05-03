#!/usr/bin/env bash
# =============================================================================
# StrataDock v 1.6.01 Setup Script - One-command installer for Ubuntu / WSL
# Installs the Vina and GNINA CPU docking runtimes, fpocket, OpenBabel,
# validation data, the Python environment, and a Streamlit launcher.
# Usage: bash setup_ubuntu.sh
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
PORT="${STRATADOCK_PORT:-8502}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}== $* ==${NC}\n"; }

cat <<'SPLASH'

        ==================================================
             >_ STRATA DOCK v 1.6.01
             Linux docking workstation setup
             Ubuntu / WSL one-command setup
        ==================================================

SPLASH

if [[ "$ROOT" == /mnt/* ]]; then
  warn "Running from a Windows-mounted path: $ROOT"
  warn "This works, but Linux-native paths such as ~/StrataDock are faster and less weird."
fi

info "Runtime target: AutoDock Vina CPU plus GNINA CPU."
info "GNINA GPU acceleration remains NVIDIA CUDA-specific; this installer verifies GNINA CPU mode everywhere."

wait_for_apt_locks() {
  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi
  while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    warn "Ubuntu package manager is busy. Waiting 10 seconds..."
    sleep 10
  done
}

detect_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO_CMD=""
  elif command -v sudo >/dev/null 2>&1; then
    SUDO_CMD="sudo"
  else
    SUDO_CMD=""
  fi
}

apt_install_required() {
  wait_for_apt_locks
  $SUDO_CMD apt-get install -y "$@"
}

apt_install_optional() {
  wait_for_apt_locks
  $SUDO_CMD apt-get install -y "$@" || return 1
}

step "Step 1/7 - System packages"
detect_sudo
if command -v apt-get >/dev/null 2>&1 && { [ "$(id -u)" -eq 0 ] || [ -n "$SUDO_CMD" ]; }; then
  info "Updating apt package index..."
  wait_for_apt_locks
  $SUDO_CMD apt-get update
  info "Installing required Linux packages..."
  apt_install_required python3 python3-venv python3-pip build-essential git ca-certificates curl openbabel
  if apt_install_optional fpocket; then
    ok "fpocket installed from apt."
  else
    warn "fpocket is not available from this apt repo. StrataDock will use/build a local copy."
  fi
else
  warn "apt-get/sudo unavailable. Skipping system package installation."
  warn "You must already have python3, python3-venv, build tools, git, OpenBabel, and fpocket/local build support."
fi

step "Step 2/7 - Python environment"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "python3 is required."
"$PYTHON_BIN" -m venv --help >/dev/null 2>&1 || fail "python3-venv is required."

if [ -d .venv ] && [ ! -f .venv/bin/activate ]; then
  warn "Existing .venv is not a Linux virtualenv. Recreating it for Ubuntu/WSL..."
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  info "Creating .venv..."
  "$PYTHON_BIN" -m venv .venv
else
  info "Using existing .venv."
fi
source .venv/bin/activate
ok "Virtual environment active."

step "Step 3/7 - fpocket"
ensure_fpocket() {
  if command -v fpocket >/dev/null 2>&1; then
    ok "fpocket found: $(command -v fpocket)"
    return 0
  fi

  mkdir -p tools .venv/bin
  if [ -x tools/fpocket_src/bin/fpocket ]; then
    ln -sf "$(pwd)/tools/fpocket_src/bin/fpocket" .venv/bin/fpocket
    ok "Using bundled fpocket at tools/fpocket_src/bin/fpocket"
    return 0
  fi

  command -v git >/dev/null 2>&1 || fail "git is required to fetch fpocket source."
  command -v make >/dev/null 2>&1 || fail "make/build-essential is required to build fpocket."

  info "Building fpocket locally..."
  if [ ! -d tools/fpocket_src ]; then
    git clone --depth 1 https://github.com/Discngine/fpocket.git tools/fpocket_src
  fi
  make -C tools/fpocket_src
  [ -x tools/fpocket_src/bin/fpocket ] || fail "fpocket build failed."
  ln -sf "$(pwd)/tools/fpocket_src/bin/fpocket" .venv/bin/fpocket
  ok "Built fpocket and linked it into .venv/bin/fpocket"
}
ensure_fpocket

step "Step 4/7 - Python packages"
python -m pip install --upgrade pip pytest
python -m pip install -e ".[ui]"
for script in mk_prepare_ligand mk_prepare_receptor; do
  if [ ! -e ".venv/bin/$script" ] && [ -e ".venv/bin/$script.py" ]; then
    ln -sf "$script.py" ".venv/bin/$script"
  fi
done
ok "Python packages installed."

step "Step 5/7 - Validation data and docking engines"
python scripts/install/download_validation_data.py
python scripts/install/download_vina.py
python scripts/install/download_gnina.py
ok "Validation data, AutoDock Vina, and GNINA are ready."
if command -v nvidia-smi >/dev/null 2>&1; then
  info "NVIDIA CUDA hardware probe found. GNINA CPU is verified; CUDA acceleration can be enabled from settings."
elif command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1 || command -v clinfo >/dev/null 2>&1; then
  warn "AMD GPU probe found. Official GNINA GPU acceleration is NVIDIA CUDA-only; GNINA CPU and Vina CPU are supported."
else
  info "No NVIDIA CUDA GPU probe found. GNINA CPU and Vina CPU are supported."
fi

step "Step 6/7 - Runtime smoke tests"
python scripts/install/check_environment.py
python scripts/validation/run_validation_case.py hiv_protease_1hsg
rm -rf runs/install_gnina_cpu_smoke
python scripts/user/run_batch.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --ligands data/validation/hiv_protease_1hsg/ligand_input.sdf \
  --box-from-ligand data/validation/hiv_protease_1hsg/native_ligand.pdb \
  --out-dir runs/install_gnina_cpu_smoke \
  --engine gnina \
  --gnina-cpu-only \
  --exhaustiveness 1 \
  --num-modes 1 \
  --seed 1 \
  --n-jobs 1
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("runs/install_gnina_cpu_smoke/results.json").read_text())
ok = rows and rows[0].get("docking_status") == "success" and rows[0].get("docking_engine") == "gnina"
if not ok:
    raise SystemExit("GNINA CPU smoke test failed.")
print("GNINA CPU smoke score:", rows[0].get("vina_score"))
print("GNINA CNNscore:", rows[0].get("cnn_score"))
print("GNINA CNNaffinity:", rows[0].get("cnn_affinity"))
PY
ok "Vina validation and GNINA CPU smoke test passed."

step "Step 7/7 - Launcher"
cat > run_stratadock.sh <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$ROOT"
source .venv/bin/activate
export PATH="\$ROOT/.venv/bin:\$PATH"
echo ""
echo "StrataDock v 1.6.01 is starting..."
echo "Open this URL in your browser:"
echo "  http://localhost:${PORT}"
echo ""
streamlit run streamlit_app.py \\
  --server.port ${PORT} \\
  --server.address 0.0.0.0 \\
  --server.headless true \\
  --server.fileWatcherType none \\
  --browser.gatherUsageStats false
RUNNER
chmod +x run_stratadock.sh
ok "Launcher created: ./run_stratadock.sh"

cat <<DONE

==================================================
  StrataDock v 1.6.01 setup complete.

  Start the app with:
    bash run_stratadock.sh

  Then open:
    http://localhost:${PORT}

  Useful checks:
    source .venv/bin/activate
    python scripts/install/check_environment.py
    python scripts/validation/run_validation_all.py
==================================================

DONE
