#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="$PROJECT_DIR/.venv/bin/automotive-graphrag-admin"

if [[ ! -x "$APP" ]]; then
    printf '找不到管理後台啟動程式：%s\n請先在專案目錄建立 .venv 並安裝依賴。\n' "$APP" >&2
    exit 1
fi

export PROJECTS_ROOT="${PROJECTS_ROOT:-$PROJECT_DIR/projects}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

cd "$PROJECT_DIR"
exec "$APP"
