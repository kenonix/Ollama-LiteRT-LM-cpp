# LiteRT-LM Multimodal CLI

LiteRT-LM 엔진을 사용하는 멀티모달(텍스트 및 이미지) AI API 서버입니다. Ollama 및 OpenAI 호환 API를 제공합니다.

## ✨ 주요 기능

- **멀티모달 지원**: 텍스트 질문뿐만 아니라 이미지 파일을 함께 전달하여 분석할 수 있습니다.
- **API 서버**: Ollama 및 OpenAI 호환 API를 통해 다른 앱에서 모델을 호출할 수 있습니다.
- **설정 관리**: 시스템 프롬프트 및 LLM 파라미터(Temperature, Top-P 등)를 파일로 저장하고 관리할 수 있습니다.

## 🛠 빌드 방법

이 프로젝트는 **Bazel**을 주 빌드 시스템으로 사용합니다.

```bash
# 프로젝트 빌드
./bazelisk build //:multimodal_cli

# 빌드된 바이너리 실행 (도움말 확인)
./bazel-bin/multimodal_cli --help
```

> [!IMPORTANT]
> 실행을 위해서는 `./models/multimodal_model.tflite` 경로에 유효한 LiteRT-LM 모델 파일이 있어야 합니다.

## 🚀 사용 방법

명령줄 인자를 통해 모델 경로, 포트 및 가속 방식을 지정할 수 있습니다. 실행하면 API 서버가 시작됩니다.

### ⚡ GPU 가속 (`--gpu`)
`--gpu` 옵션을 붙이면 GPU 백엔드 가속을 사용합니다.
```bash
./bazel-bin/multimodal_cli --gpu <모델경로>
```

### API 서버
기본 포트는 `11434`이며, `--port`로 변경할 수 있습니다.
```bash
./bazel-bin/multimodal_cli --gpu --port 11434 <모델경로>
```

### 🌐 지원 API 엔드포인트

이 서버는 **Ollama API**와 **OpenAI API**를 모두 완벽하게 지원합니다.

#### 1. OpenAI 호환 API (`/v1` 및 프리픽스 없는 경로 모두 지원)
- **`GET /v1/models` & `GET /models`**: 모델 목록 조회
- **`GET /v1/models/{model}` & `GET /models/{model}`**: 모델 단건 상세 조회
- **`POST /v1/chat/completions` & `POST /chat/completions`**: 채팅 완료 (Streaming 및 Non-streaming, Vision 이미지 분석, `stream_options.include_usage`, `developer` role, `stop` 시퀀스 등 완벽 대응)
- **`POST /v1/completions` & `POST /completions`**: 레거시 텍스트 완성 API
- **`POST /v1/embeddings`**: 미지원 시 규격에 맞는 안내 에러(400) 반환

##### Python `openai` SDK 사용 예시
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="none"  # 로컬 서버이므로 임의의 값
)

# 1. 모델 목록 확인
models = client.models.list()
print("사용 가능한 모델:", [m.id for m in models])

# 2. 스트리밍 채팅
response = client.chat.completions.create(
    model="litert-lm:latest",
    messages=[
        {"role": "system", "content": "친절한 AI 조수입니다."},
        {"role": "user", "content": "대한민국의 수도는 어디인가요?"}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

#### 2. Ollama 호환 API
- **`GET /`**: 상태 확인 (`Ollama is running`)
- **`GET /api/version`**: 버전 확인
- **`GET /api/tags`**: 모델 목록
- **`POST /api/show`**: 모델 세부 정보
- **`POST /api/chat`**: 채팅 완료 (멀티턴, 이미지 Base64, 스트리밍)
- **`POST /api/generate`**: 단일 프롬프트 생성 (스트리밍/비스트리밍)

## 📂 프로젝트 구조

```text
.
├── main.cpp            # 프로그램 진입점 및 인자 파싱
├── llm_engine.h/cpp    # LiteRT-LM 엔진 래퍼 및 코어 로직
├── server_mode.h/cpp   # HTTP API 서버 (Ollama 호환) 구현
├── config.h/cpp        # 설정 로드/저장 및 파라미터 관리
├── utils.h/cpp         # 문자열 처리, 경로 확장, 클립보드 등 유틸리티
├── BUILD               # Bazel 빌드 설정 파일
├── CMakeLists.txt      # CMake 빌드 설정 파일 (보조)
├── config.json         # LLM 파라미터 저장 파일 (생성됨)
└── system_prompt.txt   # 시스템 프롬프트 저장 파일 (생성됨)
```

## 📝 코드 설명

- **`MultimodalCliApp`**: 엔진의 생명주기를 관리하고, 대화 컨텍스트를 유지하며 텍스트/이미지 메시지를 엔진에 전달합니다.

## ⚙️ 설정 관련

- **`system_prompt.txt`**: AI의 페르소나를 정의합니다. 기본적으로 한국어 사용 및 친절한 답변이 설정되어 있습니다.
- **`config.json`**: `temperature`, `top_p`, `top_k`, `max_tokens` 등 생성 옵션을 저장합니다.

---
**Note**: 이 프로젝트는 Google DeepMind의 LiteRT-LM 라이브러리를 기반으로 합니다.