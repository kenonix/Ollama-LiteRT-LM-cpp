#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_LIB_DIR="$SCRIPT_DIR/third_party/cuda-13.5/lib64"
LLAMA_BIN_DIR="$SCRIPT_DIR/third_party/llama.cpp/build-cuda/bin"
RUST_BIN="$SCRIPT_DIR/rust-server/target/release/ollama-litert-lm-rs"

ARCH="$(uname -m)"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$LLAMA_BIN_DIR:$SCRIPT_DIR/lib/$ARCH:$SCRIPT_DIR/lib:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$SCRIPT_DIR/raon-1-mini-Q4_K_M.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-$SCRIPT_DIR/dflash-raon-1-mini-Q8_0.gguf}"
MODEL_NAME="${MODEL_NAME:-raon-1-mini:latest}"
PORT="${PORT:-11434}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-8}"
BACKEND="${BACKEND:-cuda}" # cuda | igpu | cpu
MODE="${MODE:-server}"     # server (rust-server: Ollama+OpenAI) | cli (llama-cli)
USE_DFLASH="${USE_DFLASH:-1}"

usage() {
    cat <<EOF
사용법: $0 [옵션]

KRAFTON Raon-1-mini (Q4_K_M) + DFlash2 (Q8_0) Ollama + OpenAI 통합 API 서버 (Rust + CUDA 13.5)

옵션:
  --cuda, --rtx         NVIDIA RTX 3060 (CUDA0) + CPU MoE 오프로드 + DFlash CUDA (기본값)
  --igpu                AMD Radeon iGPU (Vulkan0) 100% GPU 적재 (공유 메모리 24GB)
  --cpu                 CPU 전용 모드
  --server              Ollama + OpenAI 통합 API 서버 실행 (rust-server, 기본값)
  --cli                 터미널 대화형 CLI 모드로 실행 (llama-cli)
  -p, --port PORT       서버 포트 (기본값: ${PORT})
  -c, --ctx N           컨텍스트 길이 (기본값: ${CTX_SIZE})
  -n, --model-name NAME API 모델명 (기본값: ${MODEL_NAME})
  -t, --threads N       CPU 스레드 수 (기본값: ${THREADS})
  --no-dflash           DFlash 추측 디코딩 비활성화
  -h, --help            도움말 출력

예시:
  $0 --port 8000              # RTX 3060 CUDA + DFlash 통합 서버 (:8000)
  $0 --port 11434             # 기존 Ollama 중지 후 11434 포트로 실행
  $0 --igpu --ctx 16384       # AMD Radeon iGPU 100% GPU 통합 서버
  $0 --cuda --cli             # 터미널 대화형 실행
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda|--rtx) BACKEND="cuda"; shift ;;
        --igpu)       BACKEND="igpu"; shift ;;
        --cpu)        BACKEND="cpu"; shift ;;
        --server)     MODE="server"; shift ;;
        --cli)        MODE="cli"; shift ;;
        -p|--port)    PORT="$2"; shift 2 ;;
        --port=*)     PORT="${1#*=}"; shift ;;
        -c|--ctx)     CTX_SIZE="$2"; shift 2 ;;
        --ctx=*)      CTX_SIZE="${1#*=}"; shift ;;
        -n|--model-name) MODEL_NAME="$2"; shift 2 ;;
        --model-name=*)  MODEL_NAME="${1#*=}"; shift ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        --threads=*)  THREADS="${1#*=}"; shift ;;
        --no-dflash)  USE_DFLASH="0"; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "❌ 알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$MODEL" ]; then
    echo "❌ 메인 모델을 찾을 수 없습니다: $MODEL"
    exit 1
fi

if [ "$USE_DFLASH" = "1" ] && [ ! -f "$DRAFT_MODEL" ]; then
    echo "❌ DFlash 모델을 찾을 수 없습니다: $DRAFT_MODEL"
    exit 1
fi

if [ "$MODE" = "server" ]; then
    if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
        echo "⚠️  포트 ${PORT} 번이 이미 사용 중입니다!"
        echo "    1) 기존 Ollama 중지: sudo systemctl stop ollama"
        echo "    2) 다른 포트로 실행: $0 --port 8000"
        exit 1
    fi

    if [ ! -x "$RUST_BIN" ]; then
        echo "[빌드] rust-server release 바이너리를 빌드합니다..."
        (cd "$SCRIPT_DIR/rust-server" && cargo build --release)
    fi

    ARGS=(
        --server
        --no-embedding
        --port "$PORT"
        --max-tokens "$CTX_SIZE"
        --model-name "$MODEL_NAME"
        --threads "$THREADS"
    )

    if [ "$USE_DFLASH" = "1" ]; then
        ARGS+=(--draft-model "$DRAFT_MODEL")
    fi

    case "$BACKEND" in
        cuda)
            BACKEND_DESC="NVIDIA RTX 3060 (CUDA0, sm_86) + CPU MoE 오프로드 + DFlash CUDA"
            ARGS+=(--gpu --device CUDA0 --cpu-moe --no-op-offload)
            ;;
        igpu)
            BACKEND_DESC="AMD Radeon iGPU (Vulkan0, 100% GPU 적재)"
            ARGS+=(--gpu --device Vulkan0)
            ;;
        cpu)
            BACKEND_DESC="CPU 전용"
            ;;
        *)
            echo "❌ 지원하지 않는 백엔드: $BACKEND"
            exit 1
            ;;
    esac

    ARGS+=("$MODEL")

    echo "============================================================"
    echo " Raon-1-mini + DFlash2 통합 API 서버 (Ollama + OpenAI)"
    echo "------------------------------------------------------------"
    echo "  메인 모델  : $MODEL"
    echo "  DFlash     : $([ "$USE_DFLASH" = "1" ] && echo "$DRAFT_MODEL" || echo "OFF")"
    echo "  모델명     : $MODEL_NAME"
    echo "  백엔드     : $BACKEND_DESC"
    echo "  컨텍스트   : $CTX_SIZE 토큰"
    echo "  Ollama API : http://127.0.0.1:${PORT}/api/tags, /api/chat"
    echo "  OpenAI API : http://127.0.0.1:${PORT}/v1/models, /v1/chat/completions"
    echo "============================================================"

    exec "$RUST_BIN" "${ARGS[@]}"
else
    BIN="$LLAMA_BIN_DIR/llama-cli"
    ARGS=(
        -m "$MODEL"
        -c "$CTX_SIZE"
        -t "$THREADS"
        -fa on
        --temp 1.0
        --top-p 0.95
        --top-k 20
        --presence-penalty 1.5
    )
    if [ "$USE_DFLASH" = "1" ]; then
        ARGS+=(-md "$DRAFT_MODEL" --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-sampling probabilistic)
    fi
    case "$BACKEND" in
        cuda)
            ARGS+=(-dev CUDA0 -ngl 99 --cpu-moe --no-op-offload)
            [ "$USE_DFLASH" = "1" ] && ARGS+=(-devd CUDA0 -ngld 99)
            ;;
        igpu)
            ARGS+=(-dev Vulkan0 -ngl 99)
            [ "$USE_DFLASH" = "1" ] && ARGS+=(-devd Vulkan0 -ngld 99)
            ;;
        cpu)
            ARGS+=(-ngl 0)
            [ "$USE_DFLASH" = "1" ] && ARGS+=(-ngld 0)
            ;;
    esac
    exec "$BIN" "${ARGS[@]}"
fi
