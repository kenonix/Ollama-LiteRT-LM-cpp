# LiteRT-LM Multimodal CLI

LiteRT-LM 엔진을 사용한 멀티모달(텍스트 및 이미지) AI 대화 도구입니다. 이 프로젝트는 모듈화된 C++로 작성되었으며, 인터랙티브 CLI, 현대적인 TUI, 그리고 Ollama 호환 API 서버 모드를 지원합니다.

## ✨ 주요 기능

- **멀티모달 지원**: 텍스트 질문뿐만 아니라 이미지 파일을 함께 전달하여 분석할 수 있습니다.
- **TUI 모드**: FTXUI 라이브러리를 사용한 아름답고 직관적인 터미널 사용자 인터페이스를 제공합니다.
- **서버 모드**: Ollama 및 OpenAI 호환 API 서버를 실행하여 다른 앱에서 이 엔진을 호출할 수 있습니다.
- **수식 변환**: TUI에서 LaTeX 수학 기호를 유니코드로 자동 변환하여 깔끔하게 표시합니다.
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

명령줄 인자를 통해 실행 모드를 선택할 수 있습니다.

### 1. 인터랙티브 CLI 모드 (기본)
가장 간단한 텍스트 기반 대화 모드입니다.
```bash
./bazel-bin/multimodal_cli --model <모델경로>
```

### 2. TUI 모드 (`--ui`)
화면 전체를 사용하는 그래픽 터미널 인터페이스입니다.
```bash
./bazel-bin/multimodal_cli --ui
```
- **F1**: 채팅 탭으로 이동
- **F2**: 설정 탭으로 이동 (실시간 파라미터 조정)
- **Ctrl+C**: 마지막 AI 답변 클립보드 복사
- **명령어**: `/img <경로>` (이미지 첨부), `/clear` (대화 초기화)

### 3. 서버 모드 (`--server`)
Ollama 호환 API 서버를 실행합니다.
```bash
./bazel-bin/multimodal_cli --server --port 11434
```

## 📂 프로젝트 구조

```text
.
├── main.cpp            # 프로그램 진입점 및 인자 파싱
├── llm_engine.h/cpp    # LiteRT-LM 엔진 래퍼 및 코어 로직
├── tui_interface.h/cpp # FTXUI 기반 터미널 UI 구현
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
- **`StreamContext`**: 비동기 스트리밍 콜백과 메인 스레드 간의 동기화를 담당합니다.
- **`ChatMessage`**: TUI에서 사용되는 메시지 데이터 구조로 역할(User/AI)과 스트리밍 상태를 저장합니다.
- **`latex_to_unicode`**: 정규표현식을 사용하여 LaTeX 수식을 터미널에서 읽기 쉬운 유니코드 기호로 변환합니다.

## ⚙️ 설정 관련

- **`system_prompt.txt`**: AI의 페르소나를 정의합니다. 기본적으로 한국어 사용 및 친절한 답변이 설정되어 있습니다.
- **`config.json`**: `temperature`, `top_p`, `top_k`, `max_tokens` 등 생성 옵션을 저장합니다. TUI 설정 탭에서 저장 시 즉시 반영됩니다.

---
**Note**: 이 프로젝트는 Google DeepMind의 LiteRT-LM 라이브러리를 기반으로 합니다.