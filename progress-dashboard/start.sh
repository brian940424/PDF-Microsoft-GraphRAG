#!/usr/bin/env bash
set -euo pipefail

DASHBOARD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 埠號與綁定位址固定寫在 package.json 的 dev 指令
HOST="127.0.0.1"
PORT="3100"
APP_URL="http://${HOST}:${PORT}"

# nvm 不一定載入到目前的 shell（例如 VS Code 工作），找不到 node 時改用 nvm 內最新版本
if ! command -v node >/dev/null 2>&1; then
    NODE_BIN="$(find "${NVM_DIR:-$HOME/.nvm}/versions/node" -mindepth 2 -maxdepth 2 -type d -name bin 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "$NODE_BIN" ]]; then
        export PATH="$NODE_BIN:$PATH"
    fi
fi

if ! command -v node >/dev/null 2>&1; then
    printf '找不到 Node.js，請先安裝 Node.js 20 以上版本。\n' >&2
    exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if ((NODE_MAJOR < 20)); then
    printf '需要 Node.js 20 以上版本，目前為 %s。\n' "$(node -v)" >&2
    exit 1
fi

port_open() {
    (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null
}

open_browser() {
    if [[ "${OPEN_BROWSER:-1}" != "1" ]]; then
        return 0
    fi
    # VS Code 遠端終端機會設定 BROWSER，由它在本機（Windows）開啟
    if [[ -n "${BROWSER:-}" ]] && command -v "$BROWSER" >/dev/null 2>&1; then
        "$BROWSER" "$APP_URL" >/dev/null 2>&1 || true
    elif command -v wslview >/dev/null 2>&1; then
        wslview "$APP_URL" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$APP_URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "$APP_URL" >/dev/null 2>&1 || true
    fi
}

if port_open; then
    printf '專案儀表板已在執行：%s\n' "$APP_URL"
    open_browser
    exit 0
fi

cd "$DASHBOARD_DIR"

if [[ ! -x node_modules/.bin/next ]]; then
    printf '首次啟動，安裝相依套件…\n'
    npm install
fi

(
    for ((attempt = 0; attempt < 60; attempt++)); do
        if port_open; then
            open_browser
            exit 0
        fi
        sleep 1
    done
) &

exec npm run dev
