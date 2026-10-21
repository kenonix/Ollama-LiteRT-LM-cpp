#!/bin/bash
MODEL="${1:-./gemma-4-12B-it.litertlm}"
PORT="${PORT:-11434}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

# 포트 충돌 사전 감지
if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo "⚠️  [경고] 포트 ${PORT} 번이 이미 다른 프로세스(예: 백그라운드 Ollama)에 의해 사용 중입니다!"
    echo "해결 방법:"
    echo "  1) 기존 Ollama 중지: sudo systemctl stop ollama"
    echo "  2) 다른 포트로 실행: PORT=11435 ./run.sh"
    echo ""
fi

BIN="./build/multimodal_cli"
if [ ! -f "$BIN" ]; then
    BIN="./bazel-bin/multimodal_cli"
fi

"$BIN" --server --gpu --mtp --max-tokens "$MAX_TOKENS" --port "$PORT" "$MODEL"
