# 로컬 수정 내용 상세 보고서 (Modifications Summary)

**작성 일시**: 2026-10-28  
**대상 파일**: `llm_engine.h`, `llm_engine.cpp`, `main.cpp`, `server_mode.cpp`  
**패치 파일 백업**: `MODIFICATIONS.patch` (필요 시 `git apply MODIFICATIONS.patch`로 복원 가능)

---

## 1. 개요 및 수정 목적
본 수정 작업은 LiteRT-LM 기반 멀티모달 / LLM 서버 구동 시 발생하는 다음 문제들을 해결하고 안정성을 극대화하기 위해 수행되었습니다:
1. **첫 번째 추론 요청 시 발생하는 긴 지연(TTFT)**: WebGPU/Vulkan 셰이더 컴파일 및 세션 초기화로 인해 첫 요청이 30초 이상 지연되거나 클라이언트 타임아웃이 발생하는 문제 해결.
2. **KV 캐시 초과 및 메모리 부족 에러(Prefill/Decode Overflow)**: SillyTavern, Chatbox 등 외부 클라이언트가 긴 대화 기록이나 방대한 시스템 프롬프트(로어북)를 전송할 때 KV 캐시 한도를 초과하여 서버가 비정상 종료되거나 에러를 뱉는 현상 방지.
3. **불필요한 Thought 채널 생성으로 인한 성능 저하**: Gemma 4 모델 등에서 기본 활성화되는 사고(Thinking) 채널을 비활성화하고 KV 캐시에서 제외하여 추론 속도 및 메모리 확보.
4. **스트리밍 연결 끊김 및 타임아웃 방지**: Prefill 또는 지연 발생 시 클라이언트 소켓 타임아웃을 방지하기 위한 주기적 keep-alive 청크 및 초기 응답 전송.
5. **동시성 블로킹 개선**: 텍스트 토큰 카운트(`CountTokens`)가 무거운 추론 락(`engine_mutex_`)에 묶여 대기열이 블로킹되는 현상 개선.
6. **MTP (Multi-Token Prediction / 투기적 디코딩) 활성화 옵션 추가**.

---

## 2. 파일별 상세 변경 내역

### 2.1 `llm_engine.h` / `llm_engine.cpp`
- **MTP (Speculative Decoding) 지원**:
  - `MultimodalCliApp` 생성자에 `bool enable_mtp = false` 파라미터 추가.
  - `litert_lm_engine_settings_set_enable_speculative_decoding(settings, true)` 호출 추가.
- **시작 시 1-토큰 파이프라인/세션 워밍업 (Warmup)**:
  - 생성자 완료 단계에서 1토큰짜리 가상 대화(`{"role":"user","content":"hi"}`)를 실행하여 GPU 파이프라인 및 셰이더를 사전 컴파일.
  - 첫 번째 실제 사용자 요청 시 지연 없이 즉각 반응하도록 최적화.
- **토크나이저 뮤텍스 분리 (`tokenizer_mutex_`)**:
  - 기존에는 `CountTokens()`가 `engine_mutex_`를 잠그어 추론 도중에 토큰 계산 요청이 오면 블로킹되었음.
  - 별도의 `tokenizer_mutex_`를 도입하여 추론 중에도 즉시 토큰 수를 계산할 수 있도록 분리.
- **Thinking / 채널 필터링 비활성화**:
  - `GenerateForServer` 및 `StreamForServer`에서 `litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(..., true)` 설정.
  - `LiteRtLmThinkingConfig`를 생성하여 `enable_thinking = false`로 강제 비활성화하여 KV 캐시 낭비 방지 및 속도 향상.
  - `extract_text_from_chunk`: JSON 응답에서 content 필드가 없는 메타/채널 오브젝트를 빈 문자열로 필터링.
- **KV 캐시 모니터링 및 상태 로깅**:
  - 완료 시 `litert_lm_conversation_get_token_count`를 확인하여 최대 KV 토큰 도달 여부, 출력 토큰 수, 중단 사유(Stop sequence, Client Cancelled, Length limit)를 상세히 터미널에 로깅.
- **기본 옵션 조정**:
  - `GenerationOptions::max_tokens` 기본값을 `2048`에서 `-1`로 변경 (미지정 시 모델의 최대 컨텍스트를 자연스럽게 활용).
  - `GetMaxTokens()` 게터 메서드 추가.

---

### 2.2 `main.cpp`
- **CLI 옵션 확장**:
  - `--mtp`, `--speculative`, `--enable-speculative-decoding`, `--enable_speculative_decoding`, `--enable_speculative_decoding=true` 플래그 지원.
  - 도움말 메시지에 `--mtp` 옵션 가이드 추가.

---

### 2.3 `server_mode.cpp`
- **HTTP 소켓 타임아웃 상향**:
  - 긴 프롬프트 처리 중 소켓이 끊어지지 않도록 `set_read_timeout(600, 0)`, `set_write_timeout(600, 0)`, `set_keep_alive_timeout(600)` 적용.
- **텍스트 추출 유틸리티 (`extract_message_text`)**:
  - `content`가 일반 문자열인 경우와 OpenAI 멀티모달 포맷(`[{"type": "text", "text": "..."}]`)인 경우 모두에서 순수 텍스트를 안전하게 추출.
- **KV 캐시 보호 가드 (`Server KV Guard`)**:
  - `truncate_utf8_to_token_budget`: UTF-8 멀티바이트 경계를 안전하게 보존하며 토큰 한도에 맞게 텍스트를 절단하는 함수 구현.
  - **오래된 대화 기록 자동 트리밍**:
    - 모델의 최대 KV 캐시 토큰(`max_kv_tokens`) 중 최소 응답 생성 공간(최대 1024, 최소 256토큰)을 예약(`min_output_reserve`).
    - 남은 입력 예산을 초과하는 경우 가장 오래된 턴부터 순차적으로 제거하며, 어시스턴트 고아 메시지가 발생하지 않도록 턴 정렬 유지.
  - **초대형 시스템 프롬프트 / 사용자 메시지 Truncation**:
    - 대화 기록을 모두 비워도 시스템 프롬프트(로어북)나 단일 사용자 메시지가 예산을 초과할 경우, 뒷부분(시스템 프롬프트) 또는 앞부분(사용자 메시지 최근 내용 보존)을 절단하여 Prefill 초과 오류를 원천 방지.
- **스트리밍 클라이언트 타임아웃(TTFT) 완화**:
  - 스트림 개시 즉시 빈 델타 청크(`role: assistant, content: ""`)를 전송하여 클라이언트의 즉각적인 응답 수신 보장.
  - Prefill 단계 등 3초 이상 새로운 토큰이 생성되지 않을 경우 `: keep-alive\n\n`(또는 Ollama용 빈 청크)을 전송하여 클라이언트 연결 유지.
- **스트리밍 에러 복구 및 종료 처리**:
  - 토큰 한도 초과 등의 오류 발생 시에도 버퍼에 남아있는 UTF-8 잔여 데이터를 정상 flush하고, `finish_reason: "length"` 또는 `"stop"`이 포함된 정규 종료 청크 및 `[DONE]`을 클라이언트에 전송하여 클라이언트 앱 오류 방지.
