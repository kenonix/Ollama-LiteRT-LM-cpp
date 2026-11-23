#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/gemma-4-E4B-it-gpu.litertlm" ]; then
    DEFAULT_MODEL="$REPO_ROOT/gemma-4-E4B-it-gpu.litertlm"
else
    DEFAULT_MODEL="$REPO_ROOT/gemma-4-E4B-it.litertlm"
fi

MODEL="$DEFAULT_MODEL"
PORT="${PORT:-11434}"
# GPU 전용 모델(gemma-4-E4B-it-gpu.litertlm) 실측 기준:
# - 8192 / 10240 / 12288 / 16384 모두 WebGPU 오류 0건, 가용 RAM ~1.66 GiB, 한국어 품질 100% 정상
MAX_TOKENS="${MAX_TOKENS:-16384}"
MODEL_NAME="${MODEL_NAME:-gemma4-e4b:latest}"
ENABLE_MTP="${ENABLE_MTP:-1}"
ENABLE_EMBEDDING="${ENABLE_EMBEDDING:-0}"

usage() {
    cat <<EOF
사용법: $0 [옵션] [모델_경로]

옵션:
  -t, --max-tokens, --num-ctx N   KV 캐시 최대 토큰 할당량 (기본: ${MAX_TOKENS}, 환경변수 MAX_TOKENS 지원)
  -p, --port PORT                 HTTP 서버 포트 (기본: ${PORT}, 환경변수 PORT 지원)
  -n, --model-name NAME           API 노출 모델명 (기본: ${MODEL_NAME})
  -m, --model PATH                모델 파일 경로 (기본: ${DEFAULT_MODEL})
      --mtp / --no-mtp            MTP 추측 디코딩 켜기/끄기 (기본: ON)
      --embedding / --no-embedding 임베딩 엔진 켜기/끄기 (기본: OFF)
  -h, --help                      도움말 출력

실행 중 외부에서 토큰 할당량 변경:
  GET  http://<host>:${PORT}/api/config
  POST http://<host>:${PORT}/api/config/max-tokens  -d '{"max_tokens": 8192}'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--max-tokens|--num-ctx)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-tokens=*|--num-ctx=*)
            MAX_TOKENS="${1#*=}"
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-name=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift
            ;;
        --mtp)
            ENABLE_MTP="1"
            shift
            ;;
        --no-mtp)
            ENABLE_MTP="0"
            shift
            ;;
        --embedding)
            ENABLE_EMBEDDING="1"
            shift
            ;;
        --no-embedding)
            ENABLE_EMBEDDING="0"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "❌ [오류] 알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

if [ ! -f "$MODEL" ]; then
    echo "❌ [오류] 모델 파일을 찾을 수 없습니다: $MODEL"
    exit 1
fi

# 포트 충돌 사전 감지
if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo "⚠️  [경고] 포트 ${PORT} 번이 이미 사용 중입니다!"
    echo "    1) 기존 Ollama 중지: sudo systemctl stop ollama"
    echo "    2) 다른 포트로 실행: $0 --port 11435"
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
if [ ! -f "$BIN" ] || [ -n "$(find "$SCRIPT_DIR/src" "$SCRIPT_DIR/Cargo.toml" -newer "$BIN" 2>/dev/null)" ]; then
    echo "[빌드] Release 바이너리가 없거나 소스가 변경되어 cargo build --release를 실행합니다..."
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
echo "  동적 변경  : POST http://0.0.0.0:${PORT}/api/config/max-tokens"
echo "============================================================"

exec "$BIN" "${ARGS[@]}"
