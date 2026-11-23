#!/bin/bash
set -euo pipefail

# 실행 중인 Rust 서버의 토큰 할당량(max_tokens / num_ctx)을 외부에서 조회/변경하는 스크립트
# 사용법:
#   ./set_max_tokens.sh                  # 현재 설정 조회
#   ./set_max_tokens.sh 32768            # 32768 토큰으로 동적 변경
#   ./set_max_tokens.sh 16384 192.168.0.10:11434  # 원격 서버 변경

HOST_PORT="${2:-127.0.0.1:11434}"

if [ $# -lt 1 ]; then
    curl -s "http://${HOST_PORT}/api/config" | python3 -m json.tool
    exit 0
fi

NEW_TOKENS="$1"
if ! [[ "$NEW_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "❌ [오류] 토큰 수는 정수여야 합니다: $NEW_TOKENS"
    echo "사용법: $0 [토큰_수] [호스트:포트(기본 127.0.0.1:11434)]"
    exit 1
fi

echo "[요청] http://${HOST_PORT}/api/config/max-tokens -> max_tokens=${NEW_TOKENS} (재로드에 30~40초 소요)"
curl -s -X POST "http://${HOST_PORT}/api/config/max-tokens" \
    -H 'Content-Type: application/json' \
    -d "{\"max_tokens\": ${NEW_TOKENS}}" | python3 -m json.tool
