#!/bin/bash
set -euo pipefail

# Gemma 4 듀얼 서버 최대 컨텍스트(E4B 32K + 12B 32K) 동시 실행 스크립트
# - RTX 3060 6GB (E4B 32K, VRAM ~3.87 GiB) + Radeon iGPU (12B 32K, 전체 RAM ~15/46 GiB) 실측 검증 완료
# - 필요 시 CLI 인자(--e4b-ctx N, --12b-ctx N 등) 또는 환경변수로 조정 가능

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export E4B_MAX_TOKENS="${E4B_MAX_TOKENS:-32768}"
export G12B_MAX_TOKENS="${G12B_MAX_TOKENS:-32768}"

exec "$SCRIPT_DIR/run_gemma4_dual.sh" "$@"
