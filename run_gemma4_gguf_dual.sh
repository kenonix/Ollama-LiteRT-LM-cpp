#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_LIB_DIR="$SCRIPT_DIR/third_party/cuda-13.5/lib64"
LLAMA_SERVER="$SCRIPT_DIR/third_party/llama.cpp/build-cuda/bin/llama-server"

if [ -d "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# 기본 모델 경로 (/mnt/data)
E4B_MODEL="${E4B_MODEL:-/mnt/data/gemma4 e4b it qat/Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf}"
E4B_MMPROJ="${E4B_MMPROJ:-/mnt/data/gemma4 e4b it qat/mmproj-model-bf16 (1).gguf}"
E4B_MTP="${E4B_MTP:-/mnt/data/gemma4 e4b it qat/mtp-ggml-model-bf16 (1).gguf}"

G12B_MODEL="${G12B_MODEL:-/mnt/data/gemma4 12b it qat/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf}"
G12B_MMPROJ="${G12B_MMPROJ:-/mnt/data/gemma4 12b it qat/mmproj-model-bf16.gguf}"
G12B_MTP="${G12B_MTP:-/mnt/data/gemma4 12b it qat/mtp-ggml-model-bf16 (2).gguf}"

E4B_PORT="${E4B_PORT:-11434}"
E4B_CTX="${E4B_CTX:-16384}"
G12B_PORT="${G12B_PORT:-11435}"
G12B_CTX="${G12B_CTX:-16384}"
G12B_BACKEND="${G12B_BACKEND:-igpu}" # igpu | cpu
E4B_MMPROJ_GPU="${E4B_MMPROJ_GPU:-0}" # 0: CPU 오프로드(VRAM 절약/16K 안정), 1: CUDA0 적재
ENABLE_MTP="${ENABLE_MTP:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"

LOG_DIR="${LOG_DIR:-/tmp}"
E4B_LOG="${LOG_DIR}/gemma4_gguf_dual_e4b.log"
G12B_LOG="${LOG_DIR}/gemma4_gguf_dual_12b.log"

usage() {
    cat <<EOF
사용법: $0 [옵션]

Gemma 4 GGUF 듀얼 서버 동시 실행 스크립트 (mmproj 비전/오디오 + MTP 투기적 디코딩)
  - E4B : NVIDIA RTX 3060 (CUDA0, sm_86) — 실측 ~114 t/s, Prompt ~160 t/s
  - 12B : AMD Radeon iGPU (Vulkan0, 24GB 공유 메모리) — 실측 ~7.35 t/s
Ctrl+C 를 누르면 두 서버가 함께 깔끔하게 종료됩니다.

옵션:
  --e4b-port PORT           E4B 서버 포트 (기본: ${E4B_PORT})
  --e4b-ctx N               E4B 컨텍스트 크기 (기본: ${E4B_CTX})
  --e4b-mmproj-gpu          E4B mmproj를 RTX 3060 VRAM에 적재 (기본: CPU 오프로드)
  --12b-port PORT           12B 서버 포트 (기본: ${G12B_PORT})
  --12b-ctx N               12B 컨텍스트 크기 (기본: ${G12B_CTX})
  --12b-backend MODE        12B 백엔드: igpu | cpu (기본: ${G12B_BACKEND})
  --no-mtp                  MTP 투기적 디코딩 비활성화
  --no-thinking             사고(Thinking) 토큰 생성 비활성화 (--reasoning-budget 0)
  -h, --help                도움말 출력

예시:
  $0                                    # E4B(:11434, CUDA) + 12B(:11435, iGPU)
  $0 --e4b-port 8001 --12b-port 8002    # 기존 Ollama 유지하며 별도 포트로 실행
  $0 --no-thinking                      # 사고 과정 없이 바로 본문 답변
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --e4b-port)           E4B_PORT="$2"; shift 2 ;;
        --e4b-port=*)         E4B_PORT="${1#*=}"; shift ;;
        --e4b-ctx|--e4b-max-tokens)     E4B_CTX="$2"; shift 2 ;;
        --e4b-ctx=*|--e4b-max-tokens=*) E4B_CTX="${1#*=}"; shift ;;
        --e4b-mmproj-gpu)     E4B_MMPROJ_GPU="1"; shift ;;
        --12b-port)           G12B_PORT="$2"; shift 2 ;;
        --12b-port=*)         G12B_PORT="${1#*=}"; shift ;;
        --12b-ctx|--12b-max-tokens)     G12B_CTX="$2"; shift 2 ;;
        --12b-ctx=*|--12b-max-tokens=*) G12B_CTX="${1#*=}"; shift ;;
        --12b-backend)        G12B_BACKEND="$2"; shift 2 ;;
        --12b-backend=*)      G12B_BACKEND="${1#*=}"; shift ;;
        --no-mtp)             ENABLE_MTP="0"; shift ;;
        --no-thinking)        DISABLE_THINKING="1"; shift ;;
        -h|--help)            usage; exit 0 ;;
        *)
            echo "❌ [오류] 알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "❌ [오류] llama-server 바이너리를 찾을 수 없습니다: $LLAMA_SERVER"
    exit 1
fi

for f in "$E4B_MODEL" "$E4B_MMPROJ" "$G12B_MODEL" "$G12B_MMPROJ"; do
    if [ ! -f "$f" ]; then
        echo "❌ [오류] 필수 모델 파일을 찾을 수 없습니다: $f"
        exit 1
    fi
done

if [ "$ENABLE_MTP" = "1" ]; then
    for f in "$E4B_MTP" "$G12B_MTP"; do
        if [ ! -f "$f" ]; then
            echo "❌ [오류] MTP 모델 파일을 찾을 수 없습니다: $f"
            exit 1
        fi
    done
fi

if [ "$E4B_PORT" = "$G12B_PORT" ]; then
    echo "❌ [오류] E4B 포트(${E4B_PORT})와 12B 포트(${G12B_PORT})는 서로 달라야 합니다."
    exit 1
fi

for PORT in "$E4B_PORT" "$G12B_PORT"; do
    if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
        echo "⚠️  [경고] 포트 ${PORT} 번이 이미 사용 중입니다!"
        echo "    1) 기존 Ollama 중지: sudo systemctl stop ollama"
        echo "    2) 다른 포트로 실행: $0 --e4b-port 8001 --12b-port 8002"
        exit 1
    fi
done

E4B_PID=""
G12B_PID=""

cleanup() {
    echo ""
    echo "[종료] 두 Gemma 4 GGUF 서버를 종료합니다..."
    if [ -n "$E4B_PID" ] && kill -0 "$E4B_PID" 2>/dev/null; then
        kill "$E4B_PID" 2>/dev/null || true
    fi
    if [ -n "$G12B_PID" ] && kill -0 "$G12B_PID" 2>/dev/null; then
        kill "$G12B_PID" 2>/dev/null || true
    fi
    wait "$E4B_PID" 2>/dev/null || true
    wait "$G12B_PID" 2>/dev/null || true
    echo "[완료] 모든 서버가 정상 종료되었습니다."
}

trap cleanup INT TERM EXIT

# E4B 인자 구성 (RTX 3060 CUDA0)
E4B_ARGS=(
    -m "$E4B_MODEL"
    -mm "$E4B_MMPROJ"
    -dev CUDA0
    -ngl 99
    -c "$E4B_CTX"
    -fa on
    -np 1
    --host 0.0.0.0
    --port "$E4B_PORT"
    --alias gemma4-e4b:latest
)
if [ "$E4B_MMPROJ_GPU" = "1" ]; then
    E4B_ARGS+=(-mmdev CUDA0)
else
    E4B_ARGS+=(--no-mmproj-offload)
fi
if [ "$ENABLE_MTP" = "1" ]; then
    E4B_ARGS+=(-md "$E4B_MTP" --spec-type draft-mtp -devd CUDA0 -ngld 99)
fi
if [ "$DISABLE_THINKING" = "1" ]; then
    E4B_ARGS+=(--reasoning-budget 0)
fi

# 12B 인자 구성 (AMD Radeon iGPU Vulkan0 또는 CPU)
G12B_ARGS=(
    -m "$G12B_MODEL"
    -mm "$G12B_MMPROJ"
    -c "$G12B_CTX"
    -fa on
    -np 1
    --host 0.0.0.0
    --port "$G12B_PORT"
    --alias gemma4-12b:latest
)
if [ "$G12B_BACKEND" = "igpu" ]; then
    G12B_ARGS+=(-dev Vulkan0 -mmdev Vulkan0 -ngl 99)
    if [ "$ENABLE_MTP" = "1" ]; then
        G12B_ARGS+=(-md "$G12B_MTP" --spec-type draft-mtp -devd Vulkan0 -ngld 99)
    fi
elif [ "$G12B_BACKEND" = "cpu" ]; then
    G12B_ARGS+=(-ngl 0 --no-mmproj-offload -t 8)
    if [ "$ENABLE_MTP" = "1" ]; then
        G12B_ARGS+=(-md "$G12B_MTP" --spec-type draft-mtp -ngld 0)
    fi
else
    echo "❌ [오류] 지원하지 않는 12B 백엔드: $G12B_BACKEND (igpu | cpu)"
    exit 1
fi
if [ "$DISABLE_THINKING" = "1" ]; then
    G12B_ARGS+=(--reasoning-budget 0)
fi

echo "============================================================"
echo " Gemma 4 GGUF 듀얼 서버 동시 시작 (mmproj + MTP)"
echo "------------------------------------------------------------"
echo "  [1] E4B (RTX 3060 CUDA0) : http://0.0.0.0:${E4B_PORT} (ctx=${E4B_CTX}, MTP=$([ "$ENABLE_MTP" = "1" ] && echo ON || echo OFF))"
echo "      로그: ${E4B_LOG}"
echo "  [2] 12B (${G12B_BACKEND^^})          : http://0.0.0.0:${G12B_PORT} (ctx=${G12B_CTX}, MTP=$([ "$ENABLE_MTP" = "1" ] && echo ON || echo OFF))"
echo "      로그: ${G12B_LOG}"
echo "============================================================"

"$LLAMA_SERVER" "${E4B_ARGS[@]}" > "$E4B_LOG" 2>&1 &
E4B_PID=$!

"$LLAMA_SERVER" "${G12B_ARGS[@]}" > "$G12B_LOG" 2>&1 &
G12B_PID=$!

wait_ready() {
    local NAME="$1"
    local PORT="$2"
    local PID="$3"
    local LOG="$4"
    for i in $(seq 1 120); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "❌ [${NAME}] 서버가 예기치 않게 종료되었습니다. 로그 확인: ${LOG}"
            tail -n 20 "$LOG"
            return 1
        fi
        if curl -sf "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; then
            echo "✅ [${NAME}] 준비 완료! (http://127.0.0.1:${PORT}, ${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "❌ [${NAME}] 준비 타임아웃. 로그 확인: ${LOG}"
    return 1
}

wait_ready "E4B (RTX 3060 CUDA0)" "$E4B_PORT" "$E4B_PID" "$E4B_LOG"
wait_ready "12B (${G12B_BACKEND^^})" "$G12B_PORT" "$G12B_PID" "$G12B_LOG"

echo "------------------------------------------------------------"
echo " 두 서버 모두 정상 가동 중입니다. 종료하려면 Ctrl+C 를 누르세요."
echo "------------------------------------------------------------"

wait -n "$E4B_PID" "$G12B_PID"
