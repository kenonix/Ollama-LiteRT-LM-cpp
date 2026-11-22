#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${1:-$REPO_ROOT/gemma-4-E4B-it.litertlm}"
PORT="${PORT:-11434}"
# Jetson Orin (7.4 GiB 통합 메모리) 실측 기준:
# - 8192: 멀티모달(Vision + Audio) 병용 시 최고 안정성 (WebGPU 버퍼 오류 0건)
# - 10240: 텍스트 위주 최대 권장치 (12288 이상은 WebGPU Invalid Buffer / 스왑 스래싱 발생)
MAX_TOKENS="${MAX_TOKENS:-8192}"
MODEL_NAME="${MODEL_NAME:-gemma4-e4b:latest}"
ENABLE_MTP="${ENABLE_MTP:-1}"
# E4B(3.5GB) + Vision + Audio + 8K KV 캐시 여유 메모리 확보를 위해 기본 비활성화 (필요 시 ENABLE_EMBEDDING=1)
ENABLE_EMBEDDING="${ENABLE_EMBEDDING:-0}"

if [ ! -f "$MODEL" ]; then
    echo "❌ [오류] 모델 파일을 찾을 수 없습니다: $MODEL"
    exit 1
fi

if [ "$MAX_TOKENS" -gt 10240 ] 2>/dev/null; then
    echo "⚠️  [경고] MAX_TOKENS=${MAX_TOKENS}는 Gemma 4 E4B 권장 상한(10240)을 초과합니다."
    echo "    7.4 GiB 환경에서는 WebGPU Invalid Buffer 오류 및 스왑 스래싱이 발생할 수 있습니다."
fi

# 포트 충돌 사전 감지
if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo "⚠️  [경고] 포트 ${PORT} 번이 이미 사용 중입니다!"
    echo "    1) 기존 Ollama 중지: sudo systemctl stop ollama"
    echo "    2) 다른 포트로 실행: PORT=11435 $0"
    exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] && [ -d "$REPO_ROOT/lib/aarch64" ]; then
    export LD_LIBRARY_PATH="$REPO_ROOT/lib/aarch64:$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
elif [ "$ARCH" = "x86_64" ] && [ -d "$REPO_ROOT/lib/x86_64" ]; then
    export LD_LIBRARY_PATH="$REPO_ROOT/lib/x86_64:$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$REPO_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

BIN="$SCRIPT_DIR/target/release/ollama-litert-lm-rs"
if [ ! -f "$BIN" ]; then
    echo "[빌드] Release 바이너리가 없어 cargo build --release를 실행합니다..."
    (cd "$SCRIPT_DIR" && cargo build --release)
fi

ARGS=(
    --server
    --gpu
    --max-tokens "$MAX_TOKENS"
    --port "$PORT"
    --model-name "$MODEL_NAME"
)

if [ "$ENABLE_MTP" = "1" ]; then
    ARGS+=(--mtp)
fi

if [ "$ENABLE_EMBEDDING" != "1" ]; then
    ARGS+=(--no-embedding)
fi

ARGS+=("$MODEL")

echo "============================================================"
echo " Gemma 4 E4B (LiteRT-LM) Rust 멀티모달 서버 시작"
echo "------------------------------------------------------------"
echo "  모델       : $MODEL"
echo "  모델명     : $MODEL_NAME"
echo "  포트       : $PORT"
echo "  컨텍스트   : $MAX_TOKENS 토큰 (Vision + Audio 활성화)"
echo "  MTP 가속   : $([ "$ENABLE_MTP" = "1" ] && echo 'ON' || echo 'OFF')"
echo "  임베딩     : $([ "$ENABLE_EMBEDDING" = "1" ] && echo 'ON' || echo 'OFF (VRAM/RAM 절약)')"
echo "============================================================"

exec "$BIN" "${ARGS[@]}"
