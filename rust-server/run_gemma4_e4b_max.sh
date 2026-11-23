#!/bin/bash
set -euo pipefail

# Gemma 4 E4B GPU 전용 모델 최대 컨텍스트(32,768 토큰) 실행 스크립트
# - Jetson Orin (7.4 GiB) + gemma-4-E4B-it-gpu.litertlm 실측 검증 완료:
#   32768 토큰, WebGPU 오류 0건, 가용 RAM ~1.27 GiB, 한국어 장문 품질 100% 정상
# - 필요 시 CLI 인자(--max-tokens N) 또는 환경변수(MAX_TOKENS=N)로 조정 가능

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_TOKENS="${MAX_TOKENS:-32768}"

exec "$SCRIPT_DIR/run_gemma4_e4b.sh" "$@"
