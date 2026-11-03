#!/usr/bin/env bash
# VidSum 启动脚本：自动定位 Python 环境、注入 CUDA 库路径（GPU 转写必需）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) 找 Python：优先带 faster-whisper 的 conda 环境
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in \
    "$HOME/miniconda3/envs/20260930/bin/python" \
    "$HOME/anaconda3/envs/20260930/bin/python" \
    "$HOME/miniconda3/bin/python" \
    "$(command -v python3 || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "错误：未找到 Python，可用 PYTHON=/path/to/python 启动" >&2; exit 1; }

# 2) 注入 pip nvidia 包里的 CUDA/cuDNN 库路径（ctranslate2 需要）
SP="$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || true)"
if [ -n "$SP" ] && [ -d "$SP/nvidia" ]; then
  NV_LIBS="$(find "$SP/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')"
  if [ -n "$NV_LIBS" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *"$SP/nvidia/cublas/lib"*) ;;  # 已包含则不重复加
      *) export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
fi

# 3) 默认完全离线（模型已在本地）；首次在新机器上下载模型时：
#    HF_HUB_OFFLINE=0 ./run.sh
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

cd "$DIR"
exec "$PY" server.py "$@"
