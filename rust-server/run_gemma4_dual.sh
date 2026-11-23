#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

E4B_PORT="${E4B_PORT:-11434}"
E4B_MAX_TOKENS="${E4B_MAX_TOKENS:-16384}"
G12B_PORT="${G12B_PORT:-11435}"
G12B_MAX_TOKENS="${G12B_MAX_TOKENS:-8192}"
G12B_BACKEND="${G12B_BACKEND:-igpu}"

LOG_DIR="${LOG_DIR:-/tmp}"
E4B_LOG="${LOG_DIR}/gemma4_dual_e4b.log"
G12B_LOG="${LOG_DIR}/gemma4_dual_12b.log"

usage() {
    cat <<EOF
사용법: $0 [옵션]

Gemma 4 E4B (RTX 3060 dGPU) + Gemma 4 12B (AMD Radeon iGPU) 동시 실행 스크립트
Ctrl+C 를 누르면 두 서버가 함께 깔끔하게 종료됩니다.

옵션:
  --e4b-port PORT           E4B 서버 포트 (기본: ${E4B_PORT})
  --e4b-ctx N               E4B 최대 컨텍스트 토큰 (기본: ${E4B_MAX_TOKENS})
  --12b-port PORT           12B 서버 포트 (기본: ${G12B_PORT})
  --12b-ctx N               12B 최대 컨텍스트 토큰 (기본: ${G12B_MAX_TOKENS})
  --12b-backend MODE        12B 백엔드: igpu | cpu (기본: ${G12B_BACKEND})
  -h, --help                도움말 출력

예시:
  $0                                    # E4B(:11434, 16K) + 12B(:11435, 8K)
  $0 --e4b-ctx 32768 --12b-ctx 16384    # 컨텍스트 확장 실행
  $0 --12b-backend cpu                  # 12B를 CPU 모드로 실행
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --e4b-port)
            E4B_PORT="$2"
            shift 2
            ;;
        --e4b-port=*)
            E4B_PORT="${1#*=}"
            shift
            ;;
        --e4b-ctx|--e4b-max-tokens)
            E4B_MAX_TOKENS="$2"
            shift 2
            ;;
        --e4b-ctx=*|--e4b-max-tokens=*)
            E4B_MAX_TOKENS="${1#*=}"
            shift
            ;;
        --12b-port)
            G12B_PORT="$2"
            shift 2
            ;;
        --12b-port=*)
            G12B_PORT="${1#*=}"
            shift
            ;;
        --12b-ctx|--12b-max-tokens)
            G12B_MAX_TOKENS="$2"
            shift 2
            ;;
        --12b-ctx=*|--12b-max-tokens=*)
            G12B_MAX_TOKENS="${1#*=}"
            shift
            ;;
        --12b-backend)
            G12B_BACKEND="$2"
            shift 2
            ;;
        --12b-backend=*)
            G12B_BACKEND="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "❌ [오류] 알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "$E4B_PORT" = "$G12B_PORT" ]; then
    echo "❌ [오류] E4B 포트(${E4B_PORT})와 12B 포트(${G12B_PORT})는 서로 달라야 합니다."
    exit 1
fi

for PORT in "$E4B_PORT" "$G12B_PORT"; do
    if ss -tulpn 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
        echo "⚠️  [경고] 포트 ${PORT} 번이 이미 사용 중입니다!"
        exit 1
    fi
done

E4B_PID=""
G12B_PID=""

cleanup() {
    echo ""
    echo "[종료] 두 Gemma 4 서버를 종료합니다..."
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

BIN="$SCRIPT_DIR/target/release/ollama-litert-lm-rs"
if [ ! -f "$BIN" ] || [ -n "$(find "$SCRIPT_DIR/src" "$SCRIPT_DIR/Cargo.toml" -newer "$BIN" 2>/dev/null)" ]; then
    echo "[빌드] Release 바이너리가 없거나 소스가 변경되어 cargo build --release를 실행합니다..."
    (cd "$SCRIPT_DIR" && cargo build --release)
fi

echo "============================================================"
echo " Gemma 4 듀얼 서버 동시 시작"
echo "------------------------------------------------------------"
echo "  [1] E4B (RTX 3060 dGPU) : http://0.0.0.0:${E4B_PORT}  (ctx=${E4B_MAX_TOKENS}, MTP ON)"
echo "      로그: ${E4B_LOG}"
echo "  [2] 12B (${G12B_BACKEND^^})         : http://0.0.0.0:${G12B_PORT}  (ctx=${G12B_MAX_TOKENS}, MTP ON)"
echo "      로그: ${G12B_LOG}"
echo "============================================================"

"$SCRIPT_DIR/run_gemma4_e4b.sh" --port "$E4B_PORT" --max-tokens "$E4B_MAX_TOKENS" > "$E4B_LOG" 2>&1 &
E4B_PID=$!

"$SCRIPT_DIR/run_gemma4_12b.sh" --backend "$G12B_BACKEND" --port "$G12B_PORT" --max-tokens "$G12B_MAX_TOKENS" > "$G12B_LOG" 2>&1 &
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
        if curl -sf "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
            echo "✅ [${NAME}] 준비 완료! (http://127.0.0.1:${PORT}, ${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "❌ [${NAME}] 준비 타임아웃. 로그 확인: ${LOG}"
    return 1
}

wait_ready "E4B (RTX 3060)" "$E4B_PORT" "$E4B_PID" "$E4B_LOG"
wait_ready "12B (${G12B_BACKEND^^})" "$G12B_PORT" "$G12B_PID" "$G12B_LOG"

echo "------------------------------------------------------------"
echo " 두 서버 모두 정상 가동 중입니다. 종료하려면 Ctrl+C 를 누르세요."
echo "------------------------------------------------------------"

wait -n "$E4B_PID" "$G12B_PID"
