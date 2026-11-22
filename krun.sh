#!/bin/bash
MODEL="${1:-./raon-1-nano-Q8_0_MTP.gguf}"
PORT="${PORT:-11434}"
MAX_TOKENS="${MAX_TOKENS:-6192}"

BIN="./build/multimodal_cli"
if [ ! -f "$BIN" ]; then
    BIN="./bazel-bin/multimodal_cli"
fi

"$BIN" --server --gpu --mtp --no-embedding --max-tokens "$MAX_TOKENS" --port "$PORT" "$MODEL"
