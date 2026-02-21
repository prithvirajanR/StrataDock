#!/usr/bin/env bash
# StrataDock launcher
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate stratadock

# Export the Conda environment library path so GNINA can find libcudnn.so.9
export LD_LIBRARY_PATH="$CONDA_BASE/envs/stratadock/lib:$LD_LIBRARY_PATH"

cd "$HOME/stratadock"
echo ""
echo "  StrataDock is starting..."
echo "  Open your browser at: http://localhost:8501"
echo ""
"$CONDA_BASE/envs/stratadock/bin/streamlit" run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
