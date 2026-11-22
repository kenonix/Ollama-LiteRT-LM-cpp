#pragma once

#include "llm_engine.h"

#ifdef HAS_LLAMA_CPP

#include "llama.h"
#include <mutex>
#include <string>
#include <vector>

struct common_speculative;

/**
 * @brief llama.cpp 기반 GGUF 모델 실행 엔진 클래스 (ILlmEngine 인터페이스 구현)
 */
class LlamaCppEngine : public ILlmEngine {
private:
  llama_model *model_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_context *ctx_dft_ = nullptr;
  common_speculative *spec_ = nullptr;
  int max_tokens_ = 2048;
  bool enable_mtp_ = false;
  std::mutex mutex_;
  std::string model_path_;

  std::string FormatChatPrompt(const std::string &system_msg_str,
                               const std::string &history_json,
                               const std::string &current_msg);

public:
  LlamaCppEngine(const std::string &model_path, int max_tokens = 2048, bool use_gpu = true, int n_gpu_layers = 99, bool enable_mtp = false);
  ~LlamaCppEngine() override;

  int GetMaxTokens() const override { return max_tokens_; }
  bool SupportsSystemRole() const override { return true; }
  int CountTokens(const std::string &text) override;

  std::string GenerateForServer(const std::string &system_msg_str,
                                const std::string &history_json,
                                const std::string &current_msg,
                                const GenerationOptions &opts = {}) override;

  void StreamForServer(const std::string &system_msg_str,
                       const std::string &history_json,
                       const std::string &current_msg,
                       std::function<void(const std::string &chunk)> chunk_cb,
                       std::function<void()> done_cb,
                       std::function<void(const std::string &err)> error_cb,
                       std::function<bool()> is_cancelled = nullptr,
                       const GenerationOptions &opts = {}) override;
};

#endif // HAS_LLAMA_CPP
