#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PATH="$ROOT/.venv/bin:$PATH"
echo ""
echo "StrataDock v 1.6.01 is starting..."
echo "Open this URL in your browser:"
echo "  http://127.0.0.1:8502"
echo "If localhost shows only a loading skeleton, use 127.0.0.1 on Windows/WSL."
echo ""
streamlit run streamlit_app.py \
  --server.port 8502 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
