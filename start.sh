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

if [[ "${DEV_RELOAD:-0}" != "1" ]]; then
    exec "$APP"
fi

# 開發模式：src/ 下的 .py 檔有變更時自動重啟管理後台
SOURCE_DIR="$PROJECT_DIR/src"
source_snapshot() {
    find "$SOURCE_DIR" -name '*.py' -printf '%p %T@\n' | sort
}

APP_PID=""
stop_app() {
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
}
trap 'stop_app; exit 0' INT TERM

while true; do
    "$APP" &
    APP_PID=$!
    snapshot="$(source_snapshot)"
    while kill -0 "$APP_PID" 2>/dev/null; do
        sleep 1
        if [[ "$(source_snapshot)" != "$snapshot" ]]; then
            printf '偵測到原始碼變更，重新啟動管理後台…\n' >&2
            sleep 1
            break
        fi
    done
    stop_app
    if [[ "$(source_snapshot)" == "$snapshot" ]]; then
        # 程式自行結束（例如啟動錯誤），等待原始碼變更後再重試
        printf '管理後台已結束，等待原始碼變更後重新啟動…\n' >&2
        while [[ "$(source_snapshot)" == "$snapshot" ]]; do
            sleep 1
        done
    fi
done
