#include "llm_engine.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <condition_variable>
#include <unistd.h>

// 엔진 응답 JSON에서 텍스트를 파싱하여 추출
std::string extract_text_from_chunk(const char *chunk) {
  if (!chunk) return "";
  try {
    auto j = json::parse(chunk);
    if (j.contains("content")) {
      if (j["content"].is_string()) {
        return j["content"].get<std::string>();
      } else if (j["content"].is_array()) {
        std::string result;
        for (auto &part : j["content"]) {
          if (part.contains("text") && part["text"].is_string()) {
            result += part["text"].get<std::string>();
          }
        }
        return result;
      }
    }
    if (j.is_object()) {
      return "";
    }
  } catch (...) {}
  return std::string(chunk);
}

static bool probe_system_role_support(LiteRtLmEngine *eng) {
  if (!eng) return false;
  LiteRtLmConversationConfig *cc = nullptr;
  if (litert_lm_conversation_config_create(&cc) != kLiteRtLmStatusOk || !cc) return false;
  const char *probe_marker = "__LITERT_SYS_ROLE_PROBE__";
  std::string probe_json = json(probe_marker).dump();
  litert_lm_conversation_config_set_system_message(cc, probe_json.c_str());

  LiteRtLmConversation *conv = nullptr;
  LiteRtLmStatusCode st = litert_lm_conversation_create(eng, cc, &conv);
  litert_lm_conversation_config_delete(cc);
  if (st != kLiteRtLmStatusOk || !conv) return false;

  const char *rendered = nullptr;
  bool supported = false;
  if (litert_lm_conversation_render_preface_to_string(conv, &rendered) == kLiteRtLmStatusOk && rendered) {
    supported = (std::string(rendered).find(probe_marker) != std::string::npos);
  }
  litert_lm_conversation_delete(conv);
  return supported;
}

// MultimodalCliApp 생성자: 엔진 설정 및 인스턴스 생성
MultimodalCliApp::MultimodalCliApp(const std::string &model_path, const std::string &system_prompt, bool use_gpu, int max_tokens, bool enable_mtp)
    : system_prompt_(system_prompt), max_tokens_(max_tokens) {
  if (system_prompt_.empty()) {
    std::ifstream pfile(PROMPT_FILE);
    if (pfile.is_open()) {
      system_prompt_ = std::string((std::istreambuf_iterator<char>(pfile)),
                                   std::istreambuf_iterator<char>());
    }
    if (system_prompt_.empty())
      system_prompt_ = DEFAULT_SYSTEM_PROMPT;
  }

  std::cout << "[시스템] 모델 로딩 중 (" << (use_gpu ? "GPU 가속" : "CPU")
            << ", Max Tokens: " << max_tokens_
            << (enable_mtp ? ", MTP 활성화" : ", MTP 비활성화") << ")..." << std::endl;

  std::string dispatch_dir = "./lib";
#if defined(__aarch64__)
  if (access("./lib/aarch64", F_OK) == 0) dispatch_dir = "./lib/aarch64";
#elif defined(__x86_64__)
  if (access("./lib/x86_64", F_OK) == 0) dispatch_dir = "./lib/x86_64";
#endif

  const char* backend = use_gpu ? "gpu" : "cpu";

  auto create_engine_with_audio = [&](const char* audio_backend) -> LiteRtLmEngine* {
    LiteRtLmEngineSettings *settings = nullptr;
    if (litert_lm_engine_settings_create(
            model_path.c_str(), backend, backend, audio_backend, &settings) != kLiteRtLmStatusOk || !settings) {
      return nullptr;
    }

    litert_lm_engine_settings_set_max_num_tokens(settings, max_tokens_);
    litert_lm_engine_settings_set_parallel_file_section_loading(settings, false);
    litert_lm_engine_settings_set_use_ringbuffers_local_attention(settings, true);

    if (enable_mtp) {
      LiteRtLmSpeculativeDecodingConfig *spec_cfg = nullptr;
      if (litert_lm_speculative_decoding_config_create(&spec_cfg) == kLiteRtLmStatusOk && spec_cfg) {
        litert_lm_speculative_decoding_config_set_mode(spec_cfg, kLiteRtLmSpeculativeDecodingModeOn);
        litert_lm_engine_settings_set_speculative_decoding_config(settings, spec_cfg);
        litert_lm_speculative_decoding_config_delete(spec_cfg);
      }
    }

    litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, dispatch_dir.c_str());

    LiteRtLmEngine *eng = nullptr;
    LiteRtLmStatusCode st = litert_lm_engine_create(settings, &eng);
    litert_lm_engine_settings_delete(settings);
    return (st == kLiteRtLmStatusOk) ? eng : nullptr;
  };

  auto run_warmup = [&](LiteRtLmEngine *eng) -> bool {
    if (!eng) return false;
    LiteRtLmSessionConfig *warmup_sc = nullptr;
    if (litert_lm_session_config_create(&warmup_sc) == kLiteRtLmStatusOk && warmup_sc) {
      litert_lm_session_config_set_max_output_tokens(warmup_sc, 1);
    }
    LiteRtLmConversationConfig *warmup_cc = nullptr;
    if (litert_lm_conversation_config_create(&warmup_cc) == kLiteRtLmStatusOk && warmup_cc) {
      if (warmup_sc) {
        litert_lm_conversation_config_set_session_config(warmup_cc, warmup_sc);
      }
      LiteRtLmThinkingConfig *t_cfg = nullptr;
      if (litert_lm_thinking_config_create(&t_cfg) == kLiteRtLmStatusOk && t_cfg) {
        litert_lm_thinking_config_set_enable_thinking(t_cfg, false);
        litert_lm_conversation_config_set_thinking_config(warmup_cc, t_cfg);
        litert_lm_thinking_config_delete(t_cfg);
      }
    }
    if (warmup_sc) litert_lm_session_config_delete(warmup_sc);

    LiteRtLmConversation *warmup_conv = nullptr;
    LiteRtLmStatusCode st = litert_lm_conversation_create(eng, warmup_cc, &warmup_conv);
    if (warmup_cc) litert_lm_conversation_config_delete(warmup_cc);
    if (st != kLiteRtLmStatusOk || !warmup_conv) return false;

    LiteRtLmConversationOptionalArgs *warmup_opt = nullptr;
    if (litert_lm_conversation_optional_args_create(&warmup_opt) == kLiteRtLmStatusOk && warmup_opt) {
      litert_lm_conversation_optional_args_set_max_output_tokens(warmup_opt, 1);
    }
    LiteRtLmJsonResponse *w_res = nullptr;
    litert_lm_conversation_send_message(
        warmup_conv, "{\"role\":\"user\",\"content\":\"hi\"}", nullptr, warmup_opt, &w_res);
    if (w_res) litert_lm_json_response_delete(w_res);
    if (warmup_opt) litert_lm_conversation_optional_args_delete(warmup_opt);
    litert_lm_conversation_delete(warmup_conv);
    return true;
  };

  if (enable_mtp) {
    std::cout << "[시스템] MTP(Multi-Token Prediction / Speculative Decoding) 활성화 설정 적용..." << std::endl;
  }

  bool model_has_audio_section = false;
  {
    std::ifstream mf(model_path, std::ios::binary);
    if (mf.is_open()) {
      std::string header_buf(65536, '\0');
      mf.read(&header_buf[0], static_cast<std::streamsize>(header_buf.size()));
      header_buf.resize(static_cast<size_t>(mf.gcount()));
      if (header_buf.find("tf_lite_audio_encoder") != std::string::npos) {
        model_has_audio_section = true;
      }
    }
  }

  if (model_has_audio_section) {
    std::cout << "[시스템] 오디오 인코더 감지 -> 오디오 모달리티(CPU 백엔드) 포함 엔진 초기화 중..." << std::endl;
    engine_ = create_engine_with_audio("cpu");
    if (engine_) {
      std::cout << "[시스템] 파이프라인 및 세션 워밍업 중..." << std::endl;
      if (run_warmup(engine_)) {
        audio_enabled_ = true;
        std::cout << "[시스템] 오디오 모달리티 활성화 완료!" << std::endl;
      } else {
        std::cout << "[시스템] 오디오 세션 워밍업 실패 -> 오디오 비활성화 모드로 재초기화합니다..." << std::endl;
        litert_lm_engine_delete(engine_);
        engine_ = nullptr;
      }
    } else {
      std::cout << "[시스템] 오디오 백엔드 포함 초기화 실패 -> 오디오 비활성화 모드로 재시도합니다..." << std::endl;
    }
  }

  if (!engine_) {
    engine_ = create_engine_with_audio(nullptr);
    if (!engine_)
      throw std::runtime_error("엔진 생성 실패");
    std::cout << "[시스템] GPU/CPU 파이프라인 및 세션 워밍업 중..." << std::endl;
    run_warmup(engine_);
    audio_enabled_ = false;
  }

  supports_system_role_ = probe_system_role_support(engine_);
  std::cout << "[시스템] 준비 완료! (오디오 입력: " << (audio_enabled_ ? "지원" : "미지원")
            << ", 네이티브 system 역할: " << (supports_system_role_ ? "지원" : "미지원") << ")" << std::endl;
}

MultimodalCliApp::~MultimodalCliApp() {
  std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
  std::lock_guard<std::mutex> tok_lock(tokenizer_mutex_);
  if (engine_) {
    litert_lm_engine_delete(engine_);
    engine_ = nullptr;
  }
}

int MultimodalCliApp::CountTokens(const std::string &text) {
  if (text.empty()) return 0;
  std::lock_guard<std::mutex> tok_lock(tokenizer_mutex_);
  if (!engine_) return std::max(1, static_cast<int>(text.size() / 3));

  LiteRtLmTokenizeResult *res = nullptr;
  if (litert_lm_engine_tokenize(engine_, text.c_str(), &res) == kLiteRtLmStatusOk && res) {
    size_t count = 0;
    LiteRtLmStatusCode st = litert_lm_tokenize_result_get_num_tokens(res, &count);
    litert_lm_tokenize_result_delete(res);
    if (st == kLiteRtLmStatusOk) {
      return static_cast<int>(count);
    }
  }
  return std::max(1, static_cast<int>(text.size() / 3));
}

std::string MultimodalCliApp::GenerateForServer(const std::string &system_msg_str,
                                                const std::string &history_json,
                                                const std::string &current_msg,
                                                const GenerationOptions &opts) {
  std::lock_guard<std::recursive_mutex> engine_lock(engine_mutex_);

  int max_tok = (opts.max_tokens > 0) ? opts.max_tokens : max_tokens_;

  LiteRtLmSessionConfig *session_config = nullptr;
  if (litert_lm_session_config_create(&session_config) == kLiteRtLmStatusOk && session_config) {
    litert_lm_session_config_set_max_output_tokens(session_config, max_tok);

    LiteRtLmSamplerType s_type = (opts.temperature <= 0.001f) ? kLiteRtLmSamplerTypeGreedy : kLiteRtLmSamplerTypeTopP;
    LiteRtLmSamplerParams *sampler_params = nullptr;
    if (litert_lm_sampler_params_create(s_type, &sampler_params) == kLiteRtLmStatusOk && sampler_params) {
      litert_lm_sampler_params_set_temperature(sampler_params, opts.temperature);
      litert_lm_sampler_params_set_top_p(sampler_params, opts.top_p);
      litert_lm_sampler_params_set_top_k(sampler_params, opts.top_k);
      if (opts.seed >= 0) {
        litert_lm_sampler_params_set_seed(sampler_params, opts.seed);
      }
      litert_lm_session_config_set_sampler_params(session_config, sampler_params);
      litert_lm_sampler_params_delete(sampler_params);
    }
  }

  LiteRtLmConversationConfig *conv_config = nullptr;
  if (litert_lm_conversation_config_create(&conv_config) == kLiteRtLmStatusOk && conv_config) {
    if (session_config) {
      litert_lm_conversation_config_set_session_config(conv_config, session_config);
    }
    if (supports_system_role_ && !system_msg_str.empty()) {
      std::string sys_content_json = json(system_msg_str).dump();
      litert_lm_conversation_config_set_system_message(conv_config, sys_content_json.c_str());
    }
    if (!history_json.empty()) {
      litert_lm_conversation_config_set_messages(conv_config, history_json.c_str());
    }
    litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(conv_config, true);
    LiteRtLmThinkingConfig *thinking_cfg = nullptr;
    if (litert_lm_thinking_config_create(&thinking_cfg) == kLiteRtLmStatusOk && thinking_cfg) {
      litert_lm_thinking_config_set_enable_thinking(thinking_cfg, false);
      litert_lm_conversation_config_set_thinking_config(conv_config, thinking_cfg);
      litert_lm_thinking_config_delete(thinking_cfg);
    }
  }
  if (session_config)
    litert_lm_session_config_delete(session_config);

  LiteRtLmConversation *conversation = nullptr;
  LiteRtLmStatusCode conv_st = litert_lm_conversation_create(engine_, conv_config, &conversation);
  if (conv_config)
    litert_lm_conversation_config_delete(conv_config);
  if (conv_st != kLiteRtLmStatusOk || !conversation)
    return "";

  LiteRtLmConversationOptionalArgs *opt_args = nullptr;
  if (litert_lm_conversation_optional_args_create(&opt_args) == kLiteRtLmStatusOk && opt_args) {
    litert_lm_conversation_optional_args_set_max_output_tokens(opt_args, max_tok);
    LiteRtLmThinkingConfig *thinking_cfg = nullptr;
    if (litert_lm_thinking_config_create(&thinking_cfg) == kLiteRtLmStatusOk && thinking_cfg) {
      litert_lm_thinking_config_set_enable_thinking(thinking_cfg, false);
      litert_lm_conversation_optional_args_set_thinking_config(opt_args, thinking_cfg);
      litert_lm_thinking_config_delete(thinking_cfg);
    }
    if (opts.frequency_penalty != 0.0f || opts.presence_penalty != 0.0f) {
      LiteRtLmRepetitionPenaltyConfig *rep_cfg = nullptr;
      if (litert_lm_repetition_penalty_config_create(&rep_cfg) == kLiteRtLmStatusOk && rep_cfg) {
        litert_lm_repetition_penalty_config_set_presence_penalty(rep_cfg, opts.presence_penalty);
        litert_lm_repetition_penalty_config_set_frequency_penalty(rep_cfg, opts.frequency_penalty);
        litert_lm_conversation_optional_args_set_repetition_penalty_config(opt_args, rep_cfg);
        litert_lm_repetition_penalty_config_delete(rep_cfg);
      }
    }
  }

  std::string out_text = "";
  LiteRtLmJsonResponse *response_obj = nullptr;
  LiteRtLmStatusCode send_st = litert_lm_conversation_send_message(
      conversation, current_msg.c_str(), nullptr, opt_args, &response_obj);
  if (opt_args) {
    litert_lm_conversation_optional_args_delete(opt_args);
  }

  if (send_st == kLiteRtLmStatusOk && response_obj) {
    const char *res_text = nullptr;
    if (litert_lm_json_response_get_string(response_obj, &res_text) == kLiteRtLmStatusOk && res_text) {
      out_text = extract_text_from_chunk(res_text);
    }
    litert_lm_json_response_delete(response_obj);
  }

  size_t kv_count = 0;
  litert_lm_conversation_get_token_count(conversation, &kv_count);
  int total_kv_tokens = static_cast<int>(kv_count);
  litert_lm_conversation_delete(conversation);

  bool stop_triggered = false;
  std::string matched_stop = "";
  if (!opts.stop.empty()) {
    size_t earliest = std::string::npos;
    for (const auto &sw : opts.stop) {
      if (sw.empty()) continue;
      size_t pos = out_text.find(sw);
      if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
        earliest = pos;
        matched_stop = sw;
      }
    }
    if (earliest != std::string::npos) {
      out_text = out_text.substr(0, earliest);
      stop_triggered = true;
    }
  }

  int out_tokens = CountTokens(out_text);
  if (total_kv_tokens >= max_tokens_) {
    std::cerr << "[경고] KV 캐시 최대 토큰 한도 도달! (KV 사용량: " << total_kv_tokens
              << " / " << max_tokens_ << " 토큰, 출력: " << out_tokens
              << " 토큰) -> 문장이 중간에 잘렸을 수 있습니다." << std::endl;
  } else if (opts.max_tokens > 0 && out_tokens >= opts.max_tokens) {
    std::cerr << "[경고] 요청된 max_tokens(" << opts.max_tokens
              << ") 한도 도달! (출력: " << out_tokens << " 토큰, KV 사용량: "
              << total_kv_tokens << " / " << max_tokens_ << " 토큰)" << std::endl;
  } else if (stop_triggered) {
    std::cout << "[생성 완료] Stop 시퀀스(\"" << matched_stop << "\") 도달 (출력: "
              << out_tokens << " 토큰, KV 사용량: " << total_kv_tokens
              << " / " << max_tokens_ << " 토큰)" << std::endl;
  } else {
    std::cout << "[생성 완료] 정상 종료 (출력: " << out_tokens
              << " 토큰, KV 사용량: " << total_kv_tokens << " / " << max_tokens_
              << " 토큰)" << std::endl;
  }

  return out_text;
}

void MultimodalCliApp::StreamForServer(const std::string &system_msg_str,
                                      const std::string &history_json,
                                      const std::string &current_msg,
                                      std::function<void(const std::string &chunk)> chunk_cb,
                                      std::function<void()> done_cb,
                                      std::function<void(const std::string &err)> error_cb,
                                      std::function<bool()> is_cancelled,
                                      const GenerationOptions &opts) {
  std::lock_guard<std::recursive_mutex> engine_lock(engine_mutex_);

  int max_tok = (opts.max_tokens > 0) ? opts.max_tokens : max_tokens_;

  LiteRtLmSessionConfig *session_config = nullptr;
  if (litert_lm_session_config_create(&session_config) == kLiteRtLmStatusOk && session_config) {
    litert_lm_session_config_set_max_output_tokens(session_config, max_tok);

    LiteRtLmSamplerType s_type = (opts.temperature <= 0.001f) ? kLiteRtLmSamplerTypeGreedy : kLiteRtLmSamplerTypeTopP;
    LiteRtLmSamplerParams *sampler_params = nullptr;
    if (litert_lm_sampler_params_create(s_type, &sampler_params) == kLiteRtLmStatusOk && sampler_params) {
      litert_lm_sampler_params_set_temperature(sampler_params, opts.temperature);
      litert_lm_sampler_params_set_top_p(sampler_params, opts.top_p);
      litert_lm_sampler_params_set_top_k(sampler_params, opts.top_k);
      if (opts.seed >= 0) {
        litert_lm_sampler_params_set_seed(sampler_params, opts.seed);
      }
      litert_lm_session_config_set_sampler_params(session_config, sampler_params);
      litert_lm_sampler_params_delete(sampler_params);
    }
  }

  LiteRtLmConversationConfig *conv_config = nullptr;
  if (litert_lm_conversation_config_create(&conv_config) == kLiteRtLmStatusOk && conv_config) {
    if (session_config) {
      litert_lm_conversation_config_set_session_config(conv_config, session_config);
    }
    if (supports_system_role_ && !system_msg_str.empty()) {
      std::string sys_content_json = json(system_msg_str).dump();
      litert_lm_conversation_config_set_system_message(conv_config, sys_content_json.c_str());
    }
    if (!history_json.empty()) {
      litert_lm_conversation_config_set_messages(conv_config, history_json.c_str());
    }
    litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(conv_config, true);
    LiteRtLmThinkingConfig *thinking_cfg = nullptr;
    if (litert_lm_thinking_config_create(&thinking_cfg) == kLiteRtLmStatusOk && thinking_cfg) {
      litert_lm_thinking_config_set_enable_thinking(thinking_cfg, false);
      litert_lm_conversation_config_set_thinking_config(conv_config, thinking_cfg);
      litert_lm_thinking_config_delete(thinking_cfg);
    }
  }
  if (session_config)
    litert_lm_session_config_delete(session_config);

  LiteRtLmConversation *conversation = nullptr;
  LiteRtLmStatusCode conv_st = litert_lm_conversation_create(engine_, conv_config, &conversation);
  if (conv_config)
    litert_lm_conversation_config_delete(conv_config);
  if (conv_st != kLiteRtLmStatusOk || !conversation) {
    error_cb("대화 세션 생성 실패");
    return;
  }

  LiteRtLmConversationOptionalArgs *opt_args = nullptr;
  if (litert_lm_conversation_optional_args_create(&opt_args) == kLiteRtLmStatusOk && opt_args) {
    litert_lm_conversation_optional_args_set_max_output_tokens(opt_args, max_tok);
    LiteRtLmThinkingConfig *thinking_cfg = nullptr;
    if (litert_lm_thinking_config_create(&thinking_cfg) == kLiteRtLmStatusOk && thinking_cfg) {
      litert_lm_thinking_config_set_enable_thinking(thinking_cfg, false);
      litert_lm_conversation_optional_args_set_thinking_config(opt_args, thinking_cfg);
      litert_lm_thinking_config_delete(thinking_cfg);
    }
    if (opts.frequency_penalty != 0.0f || opts.presence_penalty != 0.0f) {
      LiteRtLmRepetitionPenaltyConfig *rep_cfg = nullptr;
      if (litert_lm_repetition_penalty_config_create(&rep_cfg) == kLiteRtLmStatusOk && rep_cfg) {
        litert_lm_repetition_penalty_config_set_presence_penalty(rep_cfg, opts.presence_penalty);
        litert_lm_repetition_penalty_config_set_frequency_penalty(rep_cfg, opts.frequency_penalty);
        litert_lm_conversation_optional_args_set_repetition_penalty_config(opt_args, rep_cfg);
        litert_lm_repetition_penalty_config_delete(rep_cfg);
      }
    }
  }

  struct ServerStreamCtx {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::string> chunks;
    bool done = false;
    bool has_error = false;
    std::string error_msg;
  };
  auto ctx = std::make_shared<ServerStreamCtx>();

  auto callback = [](void *data, const LiteRtLmStreamChunk *chunk) {
    auto *c = static_cast<ServerStreamCtx *>(data);
    if (!chunk) return;
    std::lock_guard<std::mutex> lock(c->mtx);
    const char *error_msg = nullptr;
    if (litert_lm_stream_chunk_get_error(chunk, &error_msg) == kLiteRtLmStatusOk && error_msg) {
      c->has_error = true;
      c->error_msg = error_msg;
      c->done = true;
      c->cv.notify_one();
      return;
    }
    const char *raw_chunk = nullptr;
    if (litert_lm_stream_chunk_get_text(chunk, &raw_chunk) == kLiteRtLmStatusOk && raw_chunk) {
      std::string text = extract_text_from_chunk(raw_chunk);
      if (!text.empty()) {
        c->chunks.push(std::move(text));
      }
    }
    bool is_final = false;
    if (litert_lm_stream_chunk_is_final(chunk, &is_final) == kLiteRtLmStatusOk && is_final) {
      c->done = true;
    }
    c->cv.notify_one();
  };

  LiteRtLmStatusCode result = litert_lm_conversation_send_message_stream(
      conversation, current_msg.c_str(), nullptr, opt_args, callback, ctx.get());

  if (opt_args) {
    litert_lm_conversation_optional_args_delete(opt_args);
  }

  if (result != kLiteRtLmStatusOk) {
    error_cb("스트리밍 시작 실패 (코드: " + std::to_string(static_cast<int>(result)) + ")");
    litert_lm_conversation_delete(conversation);
    return;
  }

  std::string accumulated = "";
  bool stop_triggered = false;
  std::string matched_stop = "";
  bool client_cancelled = false;

  while (true) {
    if (is_cancelled && is_cancelled()) {
      client_cancelled = true;
      litert_lm_conversation_cancel_process(conversation);
      break;
    }

    std::unique_lock<std::mutex> lock(ctx->mtx);
    ctx->cv.wait_for(lock, std::chrono::milliseconds(50), [&ctx] {
      return !ctx->chunks.empty() || ctx->done;
    });

    while (!ctx->chunks.empty()) {
      std::string c = std::move(ctx->chunks.front());
      ctx->chunks.pop();
      lock.unlock();

      if (!stop_triggered) {
        if (!opts.stop.empty()) {
          std::string test_str = accumulated + c;
          size_t earliest = std::string::npos;
          for (const auto &sw : opts.stop) {
            if (sw.empty()) continue;
            size_t pos = test_str.find(sw);
            if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
              earliest = pos;
              matched_stop = sw;
            }
          }
          if (earliest != std::string::npos) {
            stop_triggered = true;
            litert_lm_conversation_cancel_process(conversation);
            if (earliest > accumulated.size()) {
              std::string part_to_send = test_str.substr(accumulated.size(), earliest - accumulated.size());
              if (!part_to_send.empty()) {
                chunk_cb(part_to_send);
              }
            }
            accumulated = test_str.substr(0, earliest);
            lock.lock();
            break;
          }
        }
        accumulated += c;
        chunk_cb(c);
      }
      lock.lock();
    }

    if (stop_triggered) {
      lock.unlock();
      done_cb();
      break;
    }

    if (ctx->done) {
      lock.unlock();
      if (ctx->has_error) error_cb(ctx->error_msg);
      else done_cb();
      break;
    }
  }

  size_t kv_count = 0;
  litert_lm_conversation_get_token_count(conversation, &kv_count);
  int total_kv_tokens = static_cast<int>(kv_count);
  litert_lm_conversation_delete(conversation);

  int out_tokens = CountTokens(accumulated);
  if (client_cancelled) {
    std::cerr << "[경고] 클라이언트 연결 종료로 스트리밍이 중단되었습니다. (출력: "
              << out_tokens << " 토큰, KV 사용량: " << total_kv_tokens
              << " / " << max_tokens_ << " 토큰)" << std::endl;
  } else if (total_kv_tokens >= max_tokens_ ||
             (ctx->has_error && ctx->error_msg.find("Max number of tokens") != std::string::npos)) {
    std::cerr << "[경고] KV 캐시 최대 토큰 한도 도달! (KV 사용량: " << total_kv_tokens
              << " / " << max_tokens_ << " 토큰, 출력: " << out_tokens
              << " 토큰) -> 문장이 중간에 잘렸습니다." << std::endl;
  } else if (ctx->has_error) {
    std::cerr << "[생성 오류] " << ctx->error_msg << " (출력: " << out_tokens
              << " 토큰, KV 사용량: " << total_kv_tokens << " / " << max_tokens_
              << " 토큰)" << std::endl;
  } else if (opts.max_tokens > 0 && out_tokens >= opts.max_tokens) {
    std::cerr << "[경고] 요청된 max_tokens(" << opts.max_tokens
              << ") 한도 도달! (출력: " << out_tokens << " 토큰, KV 사용량: "
              << total_kv_tokens << " / " << max_tokens_ << " 토큰)" << std::endl;
  } else if (stop_triggered) {
    std::cout << "[생성 완료] Stop 시퀀스(\"" << matched_stop << "\") 도달 (출력: "
              << out_tokens << " 토큰, KV 사용량: " << total_kv_tokens
              << " / " << max_tokens_ << " 토큰)" << std::endl;
  } else {
    std::cout << "[생성 완료] 정상 종료 (출력: " << out_tokens
              << " 토큰, KV 사용량: " << total_kv_tokens << " / " << max_tokens_
              << " 토큰)" << std::endl;
  }
}
