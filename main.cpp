#include "server_mode.h"
#include "llm_engine.h"
#include <iostream>

void print_help(const char *prog_name) {
  std::cout << "LiteRT-LM Multimodal CLI\n\n"
            << "사용법: " << prog_name << " [옵션] [모델경로]\n\n"
            << "옵션:\n"
            << "  --gpu                 GPU 가속 (Vulkan/WebGPU) 활성화\n"
            << "  --port <PORT>         서버 수신 대기 포트 (기본값: 11434)\n"
            << "  --model-name <NAME>   서버가 보고할 모델 이름 (기본값: litert-lm:latest)\n"
            << "  --max-tokens <N>      최대 컨텍스트/생성 토큰 수 (기본값: 2048, RAM 최적화)\n"
            << "  -h, --help            도움말 출력\n\n"
            << "예시:\n"
            << "  " << prog_name << " --gpu gemma-4-E2B-it.litertlm\n"
            << "  " << prog_name << " --gpu --port 11434 gemma-4-E2B-it.litertlm\n";
}

/**
 * @brief 프로젝트 메인 엔트리 포인트
 * 사용자의 CLI 인자를 분석하여 Ollama/OpenAI 호환 API 서버를 실행합니다.
 */
int main(int argc, char *argv[]) {
  // 기본 설정 옵션들
  bool use_gpu = false;
  int port = 11434;
  int max_tokens = 2048;
  std::string model_path = "./models/multimodal_model.tflite";
  std::string model_name = "litert-lm:latest";

  // 명령줄 인자 파싱 루프
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--gpu") {
      use_gpu = true;
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--model-name" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg[0] != '-') {
      model_path = arg;
    }
  }

  try {
    // LLM 엔진 앱 인스턴스 초기화
    MultimodalCliApp app(model_path, "", use_gpu, max_tokens);
    RunServer(app, port, model_name);
  } catch (const std::exception &e) {
    // 예외 발생 시 치명적 오류 메시지 출력
    std::cerr << "[치명적 예외] " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
