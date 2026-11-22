#include "llama_engine.h"

#ifdef HAS_LLAMA_CPP

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "json.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

using json = nlohmann::ordered_json;

static std::string extract_text_from_json(const json &content_j) {
  if (content_j.is_string()) return content_j.get<std::string>();
  if (content_j.is_array()) {
    std::string s;
    for (const auto &p : content_j) {
      if (p.is_object() && p.value("type", "") == "text" && p.contains("text")) {
        if (!s.empty()) s += "\n";
        s += p.value("text", "");
      }
    }
    return s;
  }
  return content_j.dump();
}

LlamaCppEngine::LlamaCppEngine(const std::string &model_path, int max_tokens, bool use_gpu, int n_gpu_layers, bool enable_mtp)
    : max_tokens_(max_tokens), enable_mtp_(enable_mtp), model_path_(model_path) {
  ggml_backend_load_all();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
  mparams.load_mtp = enable_mtp_;

  std::cout << "[llama.cpp] GGUF 모델 로딩 중: " << model_path
            << " (GPU 가속: " << (use_gpu ? "ON (layers=" + std::to_string(n_gpu_layers) + ")" : "OFF")
            << ", MTP: " << (enable_mtp_ ? "ON" : "OFF")
            << ", Max Tokens: " << max_tokens_ << ")..." << std::endl;

  model_ = llama_model_load_from_file(model_path.c_str(), mparams);
  if (!model_) {
    throw std::runtime_error("llama.cpp 모델 로드 실패: " + model_path);
  }
  vocab_ = llama_model_get_vocab(model_);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = max_tokens_;
  cparams.n_batch = 512;
  cparams.n_ubatch = 256;
  cparams.no_perf = true;
  if (enable_mtp_) {
    cparams.n_rs_seq = 2;
    auto out_limits = common_speculative_get_output_limits(cparams.n_batch, 1, 2);
    cparams.n_outputs_max = out_limits.total;
    cparams.n_outputs_max_per_seq = out_limits.per_seq;
  }

  ctx_ = llama_init_from_model(model_, cparams);
  if (!ctx_) {
    llama_model_free(model_);
    model_ = nullptr;
    throw std::runtime_error("llama.cpp 컨텍스트 생성 실패");
  }

  if (enable_mtp_) {
    if (llama_model_n_layer_nextn(model_) > 0) {
      llama_context_params cparams_dft = cparams;
      cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
      cparams_dft.n_rs_seq = 0;
      cparams_dft.n_outputs_max = 1;
      cparams_dft.n_outputs_max_per_seq = 1;
      cparams_dft.ctx_other = ctx_;
      ctx_dft_ = llama_init_from_model(model_, cparams_dft);
      if (ctx_dft_) {
        common_params_speculative spec_params;
        spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        spec_params.draft.ctx_tgt = ctx_;
        spec_params.draft.ctx_dft = ctx_dft_;
        spec_params.draft.n_max = 2;
        spec_params.draft.n_min = 0;
        spec_params.draft.p_min = 0.0f;
        spec_ = common_speculative_init(spec_params, 1);
        if (spec_) {
          std::cout << "[llama.cpp] MTP (Multi-Token Prediction) 드래프트 컨텍스트 초기화 완료! (nextn layers="
                    << llama_model_n_layer_nextn(model_) << ")" << std::endl;
        } else {
          std::cerr << "[경고] MTP 투기적 디코딩 엔진 초기화에 실패하여 일반 디코딩 모드로 동작합니다." << std::endl;
          llama_free(ctx_dft_);
          ctx_dft_ = nullptr;
          enable_mtp_ = false;
        }
      } else {
        std::cerr << "[경고] MTP 드래프트 컨텍스트 생성에 실패하여 일반 디코딩 모드로 동작합니다." << std::endl;
        enable_mtp_ = false;
      }
    } else {
      std::cout << "[경고] 모델에 내장된 MTP (nextn) 레이어가 없어 일반 디코딩 모드로 동작합니다." << std::endl;
      enable_mtp_ = false;
    }
  }

  std::cout << "[llama.cpp] GGUF 모델 준비 완료!" << std::endl;
}

LlamaCppEngine::~LlamaCppEngine() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (spec_) {
    common_speculative_free(spec_);
    spec_ = nullptr;
  }
  if (ctx_dft_) {
    llama_free(ctx_dft_);
    ctx_dft_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

int LlamaCppEngine::CountTokens(const std::string &text) {
  if (text.empty()) return 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!vocab_) return std::max(1, static_cast<int>(text.size() / 3));

  int n_tok = -llama_tokenize(vocab_, text.c_str(), text.size(), NULL, 0, false, true);
  if (n_tok <= 0) return std::max(1, static_cast<int>(text.size() / 3));
  return n_tok;
}

std::string LlamaCppEngine::FormatChatPrompt(const std::string &system_msg_str,
                                             const std::string &history_json,
                                             const std::string &current_msg) {
  std::vector<std::string> roles;
  std::vector<std::string> contents;

  if (!system_msg_str.empty()) {
    roles.push_back("system");
    contents.push_back(system_msg_str);
  }

    if (!history_json.empty()) {
    try {
      auto j_hist = json::parse(history_json);
      if (j_hist.is_array()) {
        for (const auto &m : j_hist) {
          if (!m.is_object()) continue;
          std::string r = m.value("role", "user");
          std::string c = "";
          if (m.contains("content")) {
            c = extract_text_from_json(m["content"]);
          }
          if (!c.empty() || r == "assistant") {
            roles.push_back(r);
            contents.push_back(c);
          }
        }
      }
    } catch (...) {}
  }

  if (!current_msg.empty()) {
    try {
      auto j_cur = json::parse(current_msg);
      if (j_cur.is_object()) {
        std::string r = j_cur.value("role", "user");
        std::string c = "";
        if (j_cur.contains("content")) {
          c = extract_text_from_json(j_cur["content"]);
        }
        roles.push_back(r);
        contents.push_back(c);
      } else {
        roles.push_back("user");
        contents.push_back(current_msg);
      }
    } catch (...) {
      roles.push_back("user");
      contents.push_back(current_msg);
    }
  }

  if (roles.empty()) return "";

  const char *tmpl = llama_model_chat_template(model_, nullptr);
  const bool uses_empty_think = (tmpl && std::strstr(tmpl, "<think></think>") != nullptr);
  if (uses_empty_think) {
    for (size_t i = 0; i < roles.size(); ++i) {
      if (roles[i] == "assistant" && contents[i].rfind("<think>", 0) != 0) {
        contents[i] = "<think></think>\n\n" + contents[i];
      }
    }
  }

  // llama_chat_message 리스트 구성
  std::vector<llama_chat_message> chat_msgs;
  chat_msgs.reserve(roles.size());
  for (size_t i = 0; i < roles.size(); ++i) {
    chat_msgs.push_back({roles[i].c_str(), contents[i].c_str()});
  }

  std::vector<char> buf(8192);
  int res = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, buf.data(), buf.size());
  if (res > static_cast<int>(buf.size())) {
    buf.resize(res + 256);
    res = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, buf.data(), buf.size());
  }

  if (res > 0) {
    std::string formatted(buf.data(), res);
    // raon-1-nano 등 <think></think> SFT 프로토콜을 사용하는 모델인 경우,
    // 불필요한 영어 혼잣말(thinking) 누출을 방지하고 한국어 본문 생성에 바로 집중하도록 빈 <think></think>\n\n 프리필 추가
    if (uses_empty_think) {
      const std::string ass_suffix = "<|im_start|>assistant\n";
      if (formatted.size() >= ass_suffix.size() &&
          formatted.compare(formatted.size() - ass_suffix.size(), ass_suffix.size(), ass_suffix) == 0) {
        formatted += "<think></think>\n\n";
      }
    }
    return formatted;
  }

  // 템플릿 미내장 시 Fallback 형식
  std::string fallback;
  for (size_t i = 0; i < roles.size(); ++i) {
    fallback += "<start_of_turn>" + roles[i] + "\n" + contents[i] + "<end_of_turn>\n";
  }
  fallback += "<start_of_turn>assistant\n";
  return fallback;
}

void LlamaCppEngine::StreamForServer(const std::string &system_msg_str,
                                     const std::string &history_json,
                                     const std::string &current_msg,
                                     std::function<void(const std::string &chunk)> chunk_cb,
                                     std::function<void()> done_cb,
                                     std::function<void(const std::string &err)> error_cb,
                                     std::function<bool()> is_cancelled,
                                     const GenerationOptions &opts) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!ctx_ || !model_ || !vocab_) {
    error_cb("llama.cpp 엔진이 준비되지 않았습니다.");
    return;
  }

  std::string prompt = FormatChatPrompt(system_msg_str, history_json, current_msg);
  if (prompt.empty()) {
    done_cb();
    return;
  }

  int n_prompt = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), NULL, 0, true, true);
  if (n_prompt <= 0) n_prompt = 1;
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
    error_cb("프롬프트 토큰화 실패");
    return;
  }

  if (static_cast<int>(prompt_tokens.size()) >= max_tokens_) {
    error_cb("프롬프트 토큰 수가 최대 컨텍스트(" + std::to_string(max_tokens_) + ")를 초과했습니다.");
    return;
  }

  // 이전 KV 캐시 초기화
  llama_memory_clear(llama_get_memory(ctx_), true);
  if (ctx_dft_) {
    llama_memory_clear(llama_get_memory(ctx_dft_), true);
  }

  std::vector<std::string> effective_stops = opts.stop;
  effective_stops.push_back("<end_of_turn>");
  effective_stops.push_back("<start_of_turn>");
  effective_stops.push_back("<|eot_id|>");
  effective_stops.push_back("<|im_end|>");
  effective_stops.push_back("<|endoftext|>");
  effective_stops.push_back("<eos>");
  effective_stops.push_back("</s>");

  int max_gen = (opts.max_tokens > 0) ? opts.max_tokens : (max_tokens_ - static_cast<int>(prompt_tokens.size()));
  if (max_gen <= 0) max_gen = 256;

  std::string pending_buf = "";
  auto emit_piece = [&](const std::string &piece) -> bool {
    pending_buf += piece;

    // 1. 전체 매칭 검사
    size_t earliest = std::string::npos;
    for (const auto &sw : effective_stops) {
      if (sw.empty()) continue;
      size_t pos = pending_buf.find(sw);
      if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
        earliest = pos;
      }
    }
    if (earliest != std::string::npos) {
      if (earliest > 0) {
        chunk_cb(pending_buf.substr(0, earliest));
      }
      pending_buf.clear();
      return true;
    }

    // 2. 부분 접두사(prefix) 매칭 검사: pending_buf의 끝이 stop 단어의 접두사인지 확인
    size_t longest_prefix_match = 0;
    for (const auto &sw : effective_stops) {
      if (sw.empty()) continue;
      for (size_t plen = std::min(pending_buf.size(), sw.size()); plen > 0; --plen) {
        if (pending_buf.compare(pending_buf.size() - plen, plen, sw, 0, plen) == 0) {
          if (plen > longest_prefix_match) {
            longest_prefix_match = plen;
          }
          break;
        }
      }
    }

    if (longest_prefix_match > 0) {
      size_t safe_len = pending_buf.size() - longest_prefix_match;
      if (safe_len > 0) {
        chunk_cb(pending_buf.substr(0, safe_len));
        pending_buf = pending_buf.substr(safe_len);
      }
    } else {
      chunk_cb(pending_buf);
      pending_buf.clear();
    }
    return false;
  };

  auto flush_pending = [&]() {
    if (!pending_buf.empty()) {
      bool is_stop_prefix = false;
      for (const auto &sw : effective_stops) {
        if (!sw.empty() && sw.find(pending_buf) == 0) {
          is_stop_prefix = true;
          break;
        }
      }
      if (!is_stop_prefix) {
        chunk_cb(pending_buf);
      }
      pending_buf.clear();
    }
  };

  if (enable_mtp_ && ctx_dft_ && spec_ && prompt_tokens.size() >= 1) {
    common_params_sampling sp;
    sp.temp = opts.temperature;
    sp.top_k = opts.top_k;
    sp.top_p = opts.top_p;
    sp.seed = (opts.seed >= 0) ? static_cast<uint32_t>(opts.seed) : LLAMA_DEFAULT_SEED;
    sp.penalty_freq = opts.frequency_penalty;
    sp.penalty_present = opts.presence_penalty;
    sp.no_perf = true;
    common_sampler_ptr smpl(common_sampler_init(model_, sp));

    llama_seq_id seq_id = 0;
    size_t n_prefill = prompt_tokens.size() - 1;
    int n_batch = 256;
    for (size_t i = 0; i < n_prefill; i += n_batch) {
      if (is_cancelled && is_cancelled()) {
        done_cb();
        return;
      }
      size_t cur_batch = std::min(static_cast<size_t>(n_batch), n_prefill - i);
      common_batch batch_prompt(ctx_);
      for (size_t j = 0; j < cur_batch; ++j) {
        batch_prompt.add(prompt_tokens[i + j], static_cast<llama_pos>(i + j), seq_id, false);
      }
      if (llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0) {
        error_cb("llama_process prefill 실패");
        return;
      }
      if (!common_speculative_process(spec_, batch_prompt)) {
        error_cb("common_speculative_process prefill 실패");
        return;
      }
    }

    llama_token id_last = prompt_tokens.back();
    llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.end() - 1);
    prompt_tgt.reserve(max_tokens_);
    int n_past = static_cast<int>(prompt_tokens.size()) - 1;

    common_speculative_begin(spec_, seq_id, prompt_tgt);
    common_batch batch_tgt(ctx_);
    llama_tokens draft;

    int n_predict = 0;
    bool has_eos = false;

    while (n_predict < max_gen && !has_eos) {
      if (is_cancelled && is_cancelled()) {
        break;
      }

      if (draft.empty()) {
        int n_draft_max = std::min(max_tokens_ - n_past - 2, max_gen - n_predict - 1);
        n_draft_max = std::max(n_draft_max, 0);

        if (n_draft_max > 0) {
          common_speculative_get_draft_params(spec_, seq_id) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_max,
            /* .pos0     = */ n_past,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt_tgt,
            /* .result   = */ &draft,
          };
          common_speculative_draft(spec_);
          llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq_id, n_past, -1);
        }
      }

      batch_tgt.clear();
      batch_tgt.add(id_last, n_past++, seq_id, true);
      for (size_t i = 0; i < draft.size(); ++i) {
        batch_tgt.add(draft[i], n_past + static_cast<int>(i), seq_id, true);
      }

      if (llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) {
        break;
      }

      if (!common_speculative_process(spec_, batch_tgt)) {
        break;
      }

      auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_, draft);
      if (ids.empty()) {
        break;
      }

      common_speculative_accept(spec_, seq_id, ids.size() - 1);

      n_past += static_cast<int>(ids.size()) - 1;
      n_predict += static_cast<int>(ids.size());

      for (size_t i = 0; i < ids.size(); ++i) {
        prompt_tgt.push_back(id_last);
        id_last = ids[i];

        if (llama_vocab_is_eog(vocab_, id_last)) {
          has_eos = true;
          break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab_, id_last, buf, sizeof(buf), 0, false);
        if (n < 0) {
          has_eos = true;
          break;
        }
        if (emit_piece(std::string(buf, n))) {
          has_eos = true;
          break;
        }
      }

      draft.clear();
      llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, n_past, -1);
      llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq_id, n_past, -1);
    }

    flush_pending();
    done_cb();
    return;
  }

  // 프롬프트 Prefill (청크 단위 배치 처리)
  int n_batch = 512;
  for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
    if (is_cancelled && is_cancelled()) {
      done_cb();
      return;
    }
    size_t cur_batch = std::min(static_cast<size_t>(n_batch), prompt_tokens.size() - i);
    llama_batch batch = llama_batch_get_one(&prompt_tokens[i], cur_batch);
    if (llama_decode(ctx_, batch)) {
      error_cb("llama_decode prefill 실패");
      return;
    }
  }

  // 샘플러 체인 초기화
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler *smpl = llama_sampler_chain_init(sparams);

  if (opts.temperature <= 0.001f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  } else {
    if (opts.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(opts.top_k));
    if (opts.top_p > 0.0f && opts.top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(opts.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(opts.temperature));
    uint32_t seed = (opts.seed >= 0) ? static_cast<uint32_t>(opts.seed) : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
  }
  if (opts.frequency_penalty != 0.0f || opts.presence_penalty != 0.0f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
      llama_vocab_n_tokens(vocab_),
      64,
      1.0f,
      opts.frequency_penalty,
      opts.presence_penalty
    ));
  }

  for (int step = 0; step < max_gen; ++step) {
    if (is_cancelled && is_cancelled()) {
      break;
    }

    llama_token token_id = llama_sampler_sample(smpl, ctx_, -1);
    if (llama_vocab_is_eog(vocab_, token_id)) {
      break;
    }

    char buf[256];
    int n = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), 0, false);
    if (n < 0) {
      break;
    }
    if (emit_piece(std::string(buf, n))) {
      break;
    }

    llama_batch batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(ctx_, batch)) {
      break;
    }
  }

  flush_pending();
  llama_sampler_free(smpl);
  done_cb();
}

std::string LlamaCppEngine::GenerateForServer(const std::string &system_msg_str,
                                              const std::string &history_json,
                                              const std::string &current_msg,
                                              const GenerationOptions &opts) {
  std::string result;
  StreamForServer(system_msg_str, history_json, current_msg,
    [&result](const std::string &chunk) {
      result += chunk;
    },
    []() {},
    [](const std::string &err) {
      std::cerr << "[llama.cpp Error] " << err << std::endl;
    },
    nullptr,
    opts
  );
  return result;
}

#endif // HAS_LLAMA_CPP
