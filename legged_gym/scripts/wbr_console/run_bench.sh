#!/usr/bin/env bash
# 启动 WBR Console（后端 + 静态站点）。默认 http://127.0.0.1:8770
# 用法：./run_bench.sh [--port 8770] [--bridge-port 8771] [--no-browser]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解释器选择：WBR_PYTHON → 已激活的 CONDA_PREFIX（须能 import isaacgym）→
# 自动探测常见 conda 环境里带 isaacgym 的那个 → python3。
# 后端本身是纯标准库，但它会用该解释器探测 checkpoint 维度、并作为仿真
# 子进程的解释器——base 环境没有 isaacgym，选中它必然启动失败。
PY="${WBR_PYTHON:-}"
probe_isaac() {  # $1=python 路径；isaacgym 绑定依赖该环境的 libpython，探测时临时补上
  local root="$(dirname "$(dirname "$1")")"
  LD_LIBRARY_PATH="$root/lib:${LD_LIBRARY_PATH:-}" "$1" -c "import isaacgym" >/dev/null 2>&1
}
if [[ -z "$PY" ]]; then
  # 已激活的环境优先，但要求里面真能 import isaacgym
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] \
     && probe_isaac "$CONDA_PREFIX/bin/python"; then
    PY="$CONDA_PREFIX/bin/python"
  else
    # 探测 conda envs 下的环境（isaacgym01 优先），谁带 isaacgym 用谁
    for d in "$HOME/miniconda3/envs"/*/bin/python \
             "$HOME/anaconda3/envs"/*/bin/python; do
      [[ -x "$d" ]] || continue
      if probe_isaac "$d"; then
        PY="$d"
        # isaacgym01 命名命中即停（与训练环境一致）
        [[ "$d" == *isaacgym01* ]] && break
      fi
    done
  fi
fi
if [[ -z "$PY" ]]; then PY="python3"; fi

# isaacgym 绑定需要 libpython3.8.so：把所选解释器所在环境的 lib 补进加载路径
PYROOT="$(dirname "$(dirname "$PY")")"
if [[ -d "$PYROOT/lib" ]]; then
  export LD_LIBRARY_PATH="$PYROOT/lib:${LD_LIBRARY_PATH:-}"
fi

# 仿真子进程也用同一解释器（wbr_bench.py 会读 WBR_PYTHON）
export WBR_PYTHON="$PY"

echo "[run_bench] 解释器: $PY"
exec "$PY" "$HERE/wbr_bench.py" "$@"
