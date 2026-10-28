#include "server_mode.h"
#include "llm_engine.h"
#include "embedding_engine.h"
#include <iostream>

void print_help(const char *prog_name) {
  std::cout << "LiteRT-LM Multimodal CLI\n\n"
            << "사용법: " << prog_name << " [옵션] [모델경로]\n\n"
            << "옵션:\n"
            << "  --gpu                     GPU 가속 (Vulkan/WebGPU) 활성화\n"
            << "  --mtp                     MTP (Multi-Token Prediction) 투기적 디코딩 가속 활성화\n"
            << "  --port <PORT>             서버 수신 대기 포트 (기본값: 11434)\n"
            << "  --model-name <NAME>       서버가 보고할 모델 이름 (기본값: litert-lm:latest)\n"
            << "  --max-tokens <N>          최대 컨텍스트/생성 토큰 수 (기본값: 2048, RAM 최적화)\n"
            << "  --embedding-model <PATH>  EmbeddingGemma 2 임베딩 모델 경로 (생략 시 자동 탐색/다운로드)\n"
            << "  --embedding-backend <B>   임베딩 백엔드 (기본값: cpu, 25ms 고속 및 VRAM 절약, 또는 gpu)\n"
            << "  --no-embedding            임베딩 기능 비활성화\n"
            << "  -h, --help                도움말 출력\n\n"
            << "예시:\n"
            << "  " << prog_name << " --gpu --mtp gemma-4-E2B-it.litertlm\n"
            << "  " << prog_name << " --gpu --mtp --port 11434 gemma-4-E2B-it.litertlm\n";
}

/**
 * @brief 프로젝트 메인 엔트리 포인트
 * 사용자의 CLI 인자를 분석하여 Ollama/OpenAI 호환 API 서버를 실행합니다.
 */
int main(int argc, char *argv[]) {
  // 기본 설정 옵션들
  bool use_gpu = false;
  bool enable_mtp = false;
  int port = 11434;
  int max_tokens = 2048;
  std::string model_path = "./models/multimodal_model.tflite";
  std::string model_name = "litert-lm:latest";

  bool enable_embedding = true;
  std::string embedding_model_path = "";
  std::string embedding_backend = "cpu";

  // 명령줄 인자 파싱 루프
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--server") {
      // 서버 모드 명시적 플래그 (기본값이 서버 모드이므로 유지)
    } else if (arg == "--gpu") {
      use_gpu = true;
    } else if (arg == "--mtp" || arg == "--speculative" ||
               arg == "--enable-speculative-decoding" ||
               arg == "--enable_speculative_decoding" ||
               arg == "--enable_speculative_decoding=true") {
      enable_mtp = true;
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--model-name" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--embedding-model" && i + 1 < argc) {
      embedding_model_path = argv[++i];
    } else if (arg == "--embedding-backend" && i + 1 < argc) {
      embedding_backend = argv[++i];
    } else if (arg == "--no-embedding") {
      enable_embedding = false;
    } else if (arg[0] != '-') {
      model_path = arg;
    }
  }

  try {
    // LLM 엔진 앱 인스턴스 초기화
    MultimodalCliApp app(model_path, "", use_gpu, max_tokens, enable_mtp);

    // EmbeddingGemma 2 임베딩 엔진 초기화
    EmbeddingEngineWrapper emb_engine;
    if (enable_embedding) {
      std::string resolved_emb_path = EmbeddingEngineWrapper::EnsureModelAvailable(embedding_model_path);
      if (!resolved_emb_path.empty()) {
        emb_engine.Initialize(resolved_emb_path, embedding_backend);
      } else {
        std::cerr << "[경고] 임베딩 모델을 찾지 못하여 임베딩 기능이 비활성화됩니다." << std::endl;
      }
    }

    RunServer(app, port, model_name, emb_engine.IsLoaded() ? &emb_engine : nullptr);
  } catch (const std::exception &e) {
    // 예외 발생 시 치명적 오류 메시지 출력
    std::cerr << "[치명적 예외] " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

