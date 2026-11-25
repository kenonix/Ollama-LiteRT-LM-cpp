#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/gemma-4-12B-it-gpu.litertlm" ]; then
    DEFAULT_MODEL="$REPO_ROOT/gemma-4-12B-it-gpu.litertlm"
else
    DEFAULT_MODEL="$REPO_ROOT/gemma-4-12B-it.litertlm"
fi
RADEON_ICD="/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"

MODEL=""
PORT="${PORT:-11434}"
# 노트북(Ryzen 9 5900HS / RAM 46 GiB / RTX 3060 6GB + Radeon iGPU) 실측 기준:
# - NVIDIA RTX 3060 (VRAM 6GB): 가중치 로드 시 VK_ERROR_OUT_OF_DEVICE_MEMORY (실행 불가)
# - AMD Radeon iGPU (공유 메모리, 8K, MTP ON): 로드 17.0s, TTFT 10.8s, 327자 생성 19.3s (~17.0 자/초, ~9 tok/s)
# - CPU 모드 (XNNPACK, 8K):                    로드  8.5s, TTFT  6.4s, 293자 생성 36.1s (~8.1 자/초, 4.38 tok/s)
MAX_TOKENS="${MAX_TOKENS:-8192}"
MODEL_NAME="${MODEL_NAME:-gemma4-12b:latest}"
BACKEND="${BACKEND:-igpu}"
ENABLE_MTP="${ENABLE_MTP:-1}"
ENABLE_EMBEDDING="${ENABLE_EMBEDDING:-0}"

usage() {
    cat <<EOF
사용법: $0 [옵션] [모델_경로]

옵션:
  -b, --backend MODE              실행 백엔드 선택: igpu | cpu | gpu (기본: ${BACKEND}, 환경변수 BACKEND 지원)
      --igpu                      AMD Radeon 내장 GPU 사용 (공유 메모리, 5.17 tok/s)
      --cpu                       CPU(XNNPACK) 사용 (빠른 로드 8.5s, 4.38 tok/s)
      --gpu                       기본 시스템 GPU(Vulkan) 사용 (VRAM 8GB+ 외장 GPU 권장)
  -t, --max-tokens, --num-ctx N   KV 캐시 최대 토큰 할당량 (기본: ${MAX_TOKENS}, 환경변수 MAX_TOKENS 지원)
  -p, --port PORT                 HTTP 서버 포트 (기본: ${PORT}, 환경변수 PORT 지원)
  -n, --model-name NAME           API 노출 모델명 (기본: ${MODEL_NAME})
  -m, --model PATH                모델 파일 경로 (기본: ${DEFAULT_MODEL})
      --mtp / --no-mtp            MTP 추측 디코딩 켜기/끄기 (기본: ON)
      --embedding / --no-embedding 임베딩 엔진 켜기/끄기 (기본: OFF)
  -h, --help                      도움말 출력

예시:
  $0                              # 기본 실행 (Radeon iGPU, 8192 토큰, 포트 11434)
  $0 --cpu                        # CPU 모드로 실행 (빠른 로딩/TTFT)
  $0 --igpu -t 16384 -p 11435     # Radeon iGPU, 16K 컨텍스트, 포트 11435

실행 중 외부에서 토큰 할당량 변경:
  GET  http://<host>:${PORT}/api/config
  POST http://<host>:${PORT}/api/config/max-tokens  -d '{"max_tokens": 8192}'
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--backend)
            BACKEND="$2"
            shift 2
            ;;
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        --igpu)
            BACKEND="igpu"
            shift
            ;;
        --cpu)
            BACKEND="cpu"
            shift
            ;;
        --gpu)
            BACKEND="gpu"
            shift
            ;;
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

# 백엔드 검증 및 폴백
BACKEND_DESC=""
USE_GPU=0
case "$BACKEND" in
    igpu)
        if [ -f "$RADEON_ICD" ]; then
            export VK_ICD_FILENAMES="$RADEON_ICD"
            USE_GPU=1
            BACKEND_DESC="AMD Radeon iGPU (Vulkan RADV, 공유 메모리)"
        else
            echo "⚠️  [경고] Radeon Vulkan ICD($RADEON_ICD)를 찾을 수 없어 CPU 모드로 전환합니다."
            BACKEND="cpu"
            USE_GPU=0
            BACKEND_DESC="CPU (XNNPACK)"
        fi
        ;;
    cpu)
        USE_GPU=0
        BACKEND_DESC="CPU (XNNPACK)"
        ;;
    gpu)
        USE_GPU=1
        BACKEND_DESC="시스템 기본 GPU (Vulkan/WebGPU)"
        ;;
    *)
        echo "❌ [오류] 지원하지 않는 백엔드입니다: $BACKEND (igpu | cpu | gpu 중 선택)"
        exit 1
        ;;
esac

if [ -z "$MODEL" ]; then
    if [ "$USE_GPU" = "1" ] && [ -f "$REPO_ROOT/gemma-4-12B-it-gpu.litertlm" ]; then
        MODEL="$REPO_ROOT/gemma-4-12B-it-gpu.litertlm"
    else
        MODEL="$REPO_ROOT/gemma-4-12B-it.litertlm"
    fi
fi

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
    --max-tokens "$MAX_TOKENS"
    --port "$PORT"
    --model-name "$MODEL_NAME"
)

if [ "$USE_GPU" = "1" ]; then
    ARGS+=(--gpu)
fi

if [ "$ENABLE_MTP" = "1" ]; then
    ARGS+=(--mtp)
fi

if [ "$ENABLE_EMBEDDING" != "1" ]; then
    ARGS+=(--no-embedding)
fi

ARGS+=("$MODEL")

echo "============================================================"
echo " Gemma 4 12B (LiteRT-LM) Rust 멀티모달 서버 시작"
echo "------------------------------------------------------------"
echo "  모델       : $MODEL"
echo "  모델명     : $MODEL_NAME"
echo "  백엔드     : $BACKEND_DESC"
echo "  포트       : $PORT"
echo "  컨텍스트   : $MAX_TOKENS 토큰 (Vision + Audio 활성화)"
echo "  MTP 가속   : $([ "$ENABLE_MTP" = "1" ] && echo 'ON' || echo 'OFF')"
echo "  임베딩     : $([ "$ENABLE_EMBEDDING" = "1" ] && echo 'ON' || echo 'OFF (VRAM/RAM 절약)')"
echo "  동적 변경  : POST http://0.0.0.0:${PORT}/api/config/max-tokens"
echo "============================================================"

exec "$BIN" "${ARGS[@]}"
