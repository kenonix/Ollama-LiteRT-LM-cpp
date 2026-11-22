#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${1:-$REPO_ROOT/gemma-4-E2B-it.litertlm}"
PORT="${PORT:-11434}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ] && [ -d "$REPO_ROOT/lib/x86_64" ]; then
    export LD_LIBRARY_PATH="$REPO_ROOT/lib/x86_64:$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
elif [ "$ARCH" = "aarch64" ] && [ -d "$REPO_ROOT/lib/aarch64" ]; then
    export LD_LIBRARY_PATH="$REPO_ROOT/lib/aarch64:$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

BIN="$SCRIPT_DIR/target/release/ollama-litert-lm-rs"
if [ ! -f "$BIN" ]; then
    echo "[빌드] Release 바이너리가 없어 빌드를 시작합니다..."
    (cd "$SCRIPT_DIR" && cargo build --release)
fi

exec "$BIN" --server --gpu --mtp --max-tokens "$MAX_TOKENS" --port "$PORT" "$MODEL"
