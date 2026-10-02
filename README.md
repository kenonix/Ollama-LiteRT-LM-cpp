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
./bazel-bin/multimodal_cli --port 11434 <모델경로>
```

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