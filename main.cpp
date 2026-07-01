#include "tui_interface.h"
#include "server_mode.h"
#include "llm_engine.h"
#include <iostream>

/**
 * @brief 프로젝트 메인 엔트리 포인트
 * 사용자의 CLI 인자를 분석하여 TUI 모드, 서버 모드, 또는 인터랙티브 CLI 모드를 실행합니다.
 */
int main(int argc, char *argv[]) {
  // 기본 설정 옵션들
  bool is_server = false;
  bool is_ui = false;
  bool use_gpu = false;
  int port = 11434;
  std::string model_path = "./models/multimodal_model.tflite";
  std::string model_name = "litert-lm:latest";

  // 명령줄 인자 파싱 루프
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--server") is_server = true;
    else if (arg == "--ui") is_ui = true;
    else if (arg == "--gpu") use_gpu = true;
    else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (arg == "--model-name" && i + 1 < argc) model_name = argv[++i];
    else if (arg[0] != '-') model_path = arg;
  }

  // UI 모드 실행 조건
  if (is_ui) {
    RunTUI(model_path, use_gpu);
    return 0;
  }

  try {
    // LLM 엔진 앱 인스턴스 초기화
    MultimodalCliApp app(model_path, "", use_gpu);
    
    // 서버 모드 또는 일반 CLI 모드 실행 로직 분기
    if (is_server) RunServer(app, port, model_name);
    else app.RunInteractive();
  } catch (const std::exception &e) {
    // 예외 발생 시 치명적 오류 메시지 출력
    std::cerr << "[치명적 예외] " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
