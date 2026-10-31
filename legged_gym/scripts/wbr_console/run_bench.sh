#!/usr/bin/env bash
# 启动 WBR Console（后端 + 静态站点）。默认 http://127.0.0.1:8770
# 用法：./run_bench.sh [--port 8770] [--bridge-port 8771] [--no-browser]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 用带 torch 的解释器（探测 checkpoint 维度要用）：WBR_PYTHON → $CONDA_PREFIX → python3
PY="${WBR_PYTHON:-}"
if [[ -z "$PY" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PY="$CONDA_PREFIX/bin/python"
fi
if [[ -z "$PY" ]]; then PY="python3"; fi

exec "$PY" "$HERE/wbr_bench.py" "$@"
