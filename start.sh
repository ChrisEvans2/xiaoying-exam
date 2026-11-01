#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo '尚未安装 Python 环境，请先按 README「安装」执行 uv sync --frozen --extra models。' >&2
  exit 1
fi
exec .venv/bin/python -m xiaoying.launcher "$@"
