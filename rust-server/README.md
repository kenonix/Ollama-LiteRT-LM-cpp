# Ollama-LiteRT-LM-rs (Rust Server)

C++ 서버(`Ollama-LiteRT-LM-cpp`)의 품질 점검에서 발견된 모든 이슈를 해결하고 **Rust (Tokio + Axum)** 기반으로 재작성한 Ollama / OpenAI 호환 멀티모달 LLM 서버입니다.

## 주요 개선 및 수정 사항
- **LiteRT-LM 음성 입력 완전 지원**: `.litertlm` 헤더(`tf_lite_audio_encoder`) 사전 감지로 오디오 지원 모델(Gemma 4 E2B/E4B 등)에 `audio_backend="cpu"` 자동 활성화, 미지원 모델은 중복 로딩 없이 폴백.
- **멀티모달 API 포맷 지원**: OpenAI `input_audio`, `audio_url`, `audio`, `image_url`, `image` 및 Ollama `images`/`audios` 지원 + `webm`/`ogg`/`m4a` 등은 `ffmpeg`로 16kHz 모노 WAV 자동 변환.
- **GGUF 스트리밍 데드락 해결**: `CountTokens` 락 분리 및 콜백 전 추론 락 해제로 `stream_options.include_usage=true` 시 데드락 완전 제거.
- **GGUF 페널티 샘플러 순서 수정**: `llama_sampler_init_penalties`를 `top_k`/`top_p`/`temp`/`dist`/`greedy` 앞에 배치해 정상 적용.
- **시스템 프롬프트 2중 주입 제거**: LiteRT-LM `Conversation`의 네이티브 `system` 역할만 사용하여 KV 캐시 낭비 제거.
- **메모리·동시성·보안 안전성**:
  - `std::system()` 완전 제거(`std::process::Command` 직접 실행으로 쉘 인젝션 차단).
  - `chrono::Utc` / `rand::thread_rng()`로 스레드 안전성 확보.
  - `Arc` + `tokio::task::spawn_blocking` + Axum Graceful Shutdown으로 detached 스레드 Use-After-Free 제거.
  - 클라이언트 연결 종료 시 `Drop` 가드로 추론 자동 취소 및 임시 파일 RAII 자동 삭제.
- **WebGPU 샘플러 로딩**: `run.sh`에서 `LD_LIBRARY_PATH`에 `../lib/<arch>` 자동 설정.

## 빌드 및 실행
```bash
cd rust-server
cargo build --release

# 실행 (LiteRT-LM 모델)
./run.sh ../gemma-4-E2B-it.litertlm

# 실행 (GGUF 모델)
./run.sh ../raon-1-nano-Q8_0_MTP.gguf
```
