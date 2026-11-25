# Ollama-LiteRT-LM-rs (Rust 통합 AI API 서버) 기술 문서

**Ollama API**와 **OpenAI API**를 동시에 제공하는 Rust (`Tokio` + `Axum`) 기반 로컬 LLM/멀티모달 추론 서버입니다.  
단일 바이너리(`ollama-litert-lm-rs`)로 **LiteRT-LM (`.litertlm`)** 과 **llama.cpp (`.gguf` — CUDA 13.5 / Vulkan / DFlash2 / MTP / MoE CPU 오프로드)** 백엔드를 모델 확장자에 따라 자동 선택하여 실행합니다.

---

## 1. 아키텍처 개요

```text
Client (Ollama / OpenAI SDK / SillyTavern / Web UI)
        │
        ▼ HTTP (Axum, 기본 포트 11434 또는 --port)
┌──────────────────────────────────────────────────────────┐
│                 ollama-litert-lm-rs (Rust)               │
│  - Ollama API : /, /api/version, /api/tags, /api/show,   │
│                 /api/chat, /api/generate, /api/embed     │
│  - OpenAI API : /v1/models, /v1/chat/completions,        │
│                 /v1/completions, /v1/embeddings          │
│  - Runtime API: GET /api/config,                         │
│                 POST /api/config/max-tokens              │
└───────────────┬──────────────────────────┬───────────────┘
                │ (.litertlm)              │ (.gguf)
                ▼                          ▼
     LiteRT-LM Engine (C FFI)     LlamaEngine (csrc/llama_shim.cpp)
     - WebGPU / Vulkan / CPU      - CUDA 13.5 (sm_86) / Vulkan / CPU
     - Vision + Audio 입력         - DFlash2 외부 드래프트 모델 (-md)
     - 내장 MTP 투기적 디코딩        - 내장 MTP 투기적 디코딩
                                  - MoE 전문가 CPU 오프로드 (--cpu-moe)
```

### 주요 설계 특징
- **이중 API 동시 지원**: 동일한 포트에서 Ollama API와 OpenAI API를 동시에 처리합니다.
- **이중 추론 백엔드 자동 감지**:
  - `.gguf` 모델 → `llama.cpp` 백엔드 (`CUDA0` / `Vulkan0` / `CPU`, FlashAttention, DFlash2, MTP, MoE CPU 오프로드).
  - `.litertlm` / `.tflite` 모델 → `LiteRT-LM` 백엔드 (`WebGPU/Vulkan` GPU 또는 `XNNPACK` CPU, Vision/Audio 멀티모달).
- **안정성 및 동시성**:
  - 토크나이저(`CountTokens`) 락 분리로 `stream_options.include_usage=true` 시 데드락 방지.
  - 클라이언트 연결 종료 시 `Drop` 가드로 추론 자동 취소 및 임시 파일 정리.
  - 실행 중 `POST /api/config/max-tokens` 호출로 서버 재시작 없이 KV 캐시 한도 동적 리로드.

---

## 2. 빌드 방법

### 2.1 사전 요구사항
- **Rust 툴체인**: `cargo`, `rustc`
- **LiteRT-LM 라이브러리**: 저장소 루트의 `lib/<arch>/liblitert-lm.so`
- **(선택) GGUF + CUDA/Vulkan 지원 (`krafton-ai/llama.cpp`)**:
  - `build.rs`가 다음 순서로 `llama.cpp`를 자동 탐색하여 `has_llama_cpp` 기능을 활성화합니다:
    1. 환경변수 `LLAMA_CPP_DIR`
    2. `<repo_root>/third_party/llama.cpp` (`build-cuda/bin` 우선, 없으면 `build/bin`)
    3. `$HOME/llama.cpp`
  - `third_party/cuda-13.5/lib64`가 존재하면 자동으로 `rpath`에 포함됩니다.

### 2.2 `krafton-ai/llama.cpp` (CUDA + Vulkan) 로컬 빌드 예시
Raon-1-mini (`raon1`) + DFlash2 (`dflash`) 모델을 실행하려면 `krafton-ai/llama.cpp`의 `raon-1-mini` 브랜치가 필요합니다.

```bash
# 1. krafton-ai/llama.cpp 클론
git clone --depth 1 -b raon-1-mini https://github.com/krafton-ai/llama.cpp third_party/llama.cpp

# 2. CUDA (예: RTX 3060 sm_86) + Vulkan 동시 빌드
CUDA_ROOT="$PWD/third_party/cuda-13.5"
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_VULKAN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build third_party/llama.cpp/build-cuda --target llama-server llama-cli -j "$(nproc)"
```

### 2.3 Rust 서버 빌드
```bash
cd rust-server
cargo build --release
```
빌드 완료 후 바이너리는 `rust-server/target/release/ollama-litert-lm-rs`에 생성됩니다.

---

## 3. CLI 옵션 레퍼런스

```bash
./rust-server/target/release/ollama-litert-lm-rs [옵션] <MODEL_PATH>
```

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `<MODEL_PATH>` | `./models/multimodal_model.tflite` | 실행할 `.litertlm` 또는 `.gguf` 모델 경로 |
| `--server` | `false` | C++ 바이너리 호환용 플래그 (Rust 서버는 항상 서버 모드로 동작) |
| `--gpu` | `false` | GPU 가속 활성화 (`LiteRT-LM`: Vulkan/WebGPU, `llama.cpp`: CUDA/Vulkan) |
| `--mtp`, `--speculative` | `false` | 모델 내장 MTP(Multi-Token Prediction) 투기적 디코딩 활성화 |
| `--draft-model`, `--dflash-model <PATH>` | 없음 | 외부 DFlash2 드래프트 GGUF 모델 경로 (예: `dflash-raon-1-mini-Q8_0.gguf`) |
| `--device <DEV>` | 자동 | `llama.cpp` ggml 백엔드 디바이스 명시 지정 (예: `CUDA0`, `Vulkan0`) |
| `--cpu-moe` | `false` | MoE 라우팅 전문가 가중치(`.ffn_*_exps`)를 CPU RAM에 유지 (6GB VRAM GPU 필수) |
| `--no-op-offload` | `false` | 호스트(CPU) 텐서 연산의 GPU 오프로드 비활성화 (`--cpu-moe` 사용 시 권장) |
| `--threads <N>` | `8` | `llama.cpp` CPU 추론 스레드 수 |
| `--max-tokens <N>` | `2048` | 최대 컨텍스트(KV 캐시) 토큰 수 |
| `--port <PORT>` | `11434` | HTTP 서버 수신 포트 |
| `--model-name <NAME>` | `litert-lm:latest` | Ollama/OpenAI API 응답에 표시될 모델명 |
| `--embedding-model <PATH>` | 자동 탐색 | EmbeddingGemma 2 임베딩 모델 경로 |
| `--embedding-backend <B>` | `cpu` | 임베딩 백엔드 (`cpu` 또는 `gpu`) |
| `--no-embedding` | `false` | 임베딩 엔진 비활성화 (RAM/VRAM 절약) |

---

## 4. 모델별 실행 가이드

### 4.1 Raon-1-mini (GGUF) + DFlash2 (CUDA 13.5 / Vulkan)
저장소 루트의 [run_raon1_dflash.sh](../run_raon1_dflash.sh) 스크립트를 사용합니다.

```bash
# 1) NVIDIA RTX 3060 (CUDA0) + CPU MoE 오프로드 + DFlash2 통합 서버 (권장, ~15 t/s)
./run_raon1_dflash.sh --cuda --port 8000

# 2) 기존 시스템 Ollama 중지 후 기본 포트(11434)로 실행
sudo systemctl stop ollama
./run_raon1_dflash.sh --cuda --port 11434

# 3) AMD Radeon iGPU (Vulkan0) 100% GPU 적재 (긴 컨텍스트 16K+ 유리)
./run_raon1_dflash.sh --igpu --ctx 16384 --port 8000

# 4) 터미널 대화형 CLI 모드 (llama-cli)
./run_raon1_dflash.sh --cuda --cli
```

#### 노트북 실측 성능 (Ryzen 9 5900HS / 46 GiB RAM / RTX 3060 6GB + Radeon iGPU)
| 백엔드 | Prompt | 생성 속도 | 로딩 시간 | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **CUDA 13.5 (`CUDA0` + `--cpu-moe --no-op-offload`)** | **21.6~26.7 t/s** | **14.6~15.1 t/s** | **3.9s** | KRAFTON CUDA 퓨전 커널(`raon-hc-fuse.cu`) + DFlash2 활성, VRAM ~5.5 GiB (8K 기준) |
| Vulkan (`Vulkan0`, AMD Radeon iGPU) | 10.2 t/s | 3.5 t/s | ~12s | 100% GPU 적재 (공유 메모리 23.6 GiB) |
| Vulkan (`Vulkan1`, RTX 3060 + `--cpu-moe`) | 33.3 t/s | 3.2 t/s | ~6s | NVVM 셰이더 경고 발생 |

### 4.2 Gemma 4 (LiteRT-LM `.litertlm`)
> **참고**: `.litertlm` 모델은 LiteRT-LM(WebGPU/Vulkan) 엔진을 사용하므로 CUDA 13.5 설치의 영향을 받지 않습니다.

```bash
# Gemma 4 E4B (RTX 3060 dGPU, 기본 16K 컨텍스트)
./rust-server/run_gemma4_e4b.sh --port 11434

# Gemma 4 12B (AMD Radeon iGPU 공유 메모리 또는 CPU)
./rust-server/run_gemma4_12b.sh --igpu --port 11435

# Gemma 4 E4B + 12B 듀얼 서버 동시 실행
./rust-server/run_gemma4_dual.sh
```

---

## 5. API 사용 예시

### 5.1 Ollama API
```bash
# 서버 상태 및 모델 목록 확인
curl -s http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/tags

# 채팅 요청 (비스트리밍)
curl -s http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "raon-1-mini:latest",
    "messages": [{"role": "user", "content": "안녕하세요! 한 문장으로 인사해 주세요."}],
    "stream": false
  }'
```

### 5.2 OpenAI 호환 API
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "raon-1-mini:latest",
    "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
    "max_tokens": 256
  }'
```

### 5.3 런타임 컨텍스트(KV 캐시) 동적 변경
```bash
# 현재 설정 조회
curl -s http://127.0.0.1:8000/api/config

# 최대 토큰 한도 변경 (엔진 자동 리로드)
curl -s -X POST http://127.0.0.1:8000/api/config/max-tokens \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 8192}'
```
