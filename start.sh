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

if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
    APP_URL="http://127.0.0.1:${GRADIO_SERVER_PORT}"
    (
        for ((attempt = 0; attempt < 60; attempt++)); do
            if python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1)' "$APP_URL" >/dev/null 2>&1; then
                if command -v wslview >/dev/null 2>&1; then
                    wslview "$APP_URL" >/dev/null 2>&1 || true
                elif command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "$APP_URL" >/dev/null 2>&1 || true
                elif command -v open >/dev/null 2>&1; then
                    open "$APP_URL" >/dev/null 2>&1 || true
                fi
                exit 0
            fi
            sleep 1
        done
    ) &
fi

cd "$PROJECT_DIR"
exec "$APP"
