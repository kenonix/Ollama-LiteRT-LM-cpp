#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

extern "C" {

struct LlamaShimEngine {
  llama_model *model = nullptr;
  const llama_vocab *vocab = nullptr;
  llama_context *ctx = nullptr;
  llama_context *ctx_dft = nullptr;
  common_speculative *spec = nullptr;
  int max_tokens = 2048;
  bool enable_mtp = false;
  std::mutex infer_mutex;
};

struct LlamaShimGenOpts {
  float temperature;
  float top_p;
  int top_k;
  int max_tokens;
  int seed;
  float frequency_penalty;
  float presence_penalty;
  const char *const *stops;
  size_t num_stops;
};

typedef void (*LlamaChunkCallback)(void *user_data, const char *data, size_t len);
typedef void (*LlamaDoneCallback)(void *user_data);
typedef void (*LlamaErrorCallback)(void *user_data, const char *err);
typedef bool (*LlamaIsCancelledCallback)(void *user_data);

static std::string extract_text_from_json(const json &content_j) {
  if (content_j.is_string()) return content_j.get<std::string>();
  if (content_j.is_array()) {
    std::string s;
    for (const auto &p : content_j) {
      if (p.is_object() && p.value("type", "") == "text" && p.contains("text") && p["text"].is_string()) {
        if (!s.empty()) s += "\n";
        s += p["text"].get<std::string>();
      }
    }
    return s;
  }
  return content_j.dump();
}

static std::string format_chat_prompt(llama_model *model,
                                      const char *system_msg_str,
                                      const char *history_json,
                                      const char *current_msg) {
  std::vector<std::string> roles;
  std::vector<std::string> contents;

  if (system_msg_str && system_msg_str[0] != '\0') {
    roles.push_back("system");
    contents.push_back(system_msg_str);
  }

  if (history_json && history_json[0] != '\0') {
    try {
      auto j_hist = json::parse(history_json);
      if (j_hist.is_array()) {
        for (const auto &m : j_hist) {
          if (!m.is_object()) continue;
          std::string r = m.value("role", "user");
          std::string c = m.contains("content") ? extract_text_from_json(m["content"]) : "";
          if (!c.empty() || r == "assistant") {
            roles.push_back(r);
            contents.push_back(c);
          }
        }
      }
    } catch (...) {}
  }

  if (current_msg && current_msg[0] != '\0') {
    try {
      auto j_cur = json::parse(current_msg);
      if (j_cur.is_object()) {
        std::string r = j_cur.value("role", "user");
        std::string c = j_cur.contains("content") ? extract_text_from_json(j_cur["content"]) : "";
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

  const char *tmpl = llama_model_chat_template(model, nullptr);
  const bool uses_empty_think = (tmpl && std::strstr(tmpl, "<think></think>") != nullptr);
  if (uses_empty_think) {
    for (size_t i = 0; i < roles.size(); ++i) {
      if (roles[i] == "assistant" && contents[i].rfind("<think>", 0) != 0) {
        contents[i] = "<think></think>\n\n" + contents[i];
      }
    }
  }

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
    if (uses_empty_think) {
      const std::string ass_suffix = "<|im_start|>assistant\n";
      if (formatted.size() >= ass_suffix.size() &&
          formatted.compare(formatted.size() - ass_suffix.size(), ass_suffix.size(), ass_suffix) == 0) {
        formatted += "<think></think>\n\n";
      }
    }
    return formatted;
  }

  std::string fallback;
  for (size_t i = 0; i < roles.size(); ++i) {
    fallback += "<start_of_turn>" + roles[i] + "\n" + contents[i] + "<end_of_turn>\n";
  }
  fallback += "<start_of_turn>assistant\n";
  return fallback;
}

LlamaShimEngine *llama_shim_create(const char *model_path,
                                   int max_tokens,
                                   bool use_gpu,
                                   int n_gpu_layers,
                                   bool enable_mtp) {
  if (!model_path) return nullptr;
  ggml_backend_load_all();

  auto *eng = new LlamaShimEngine();
  eng->max_tokens = max_tokens;
  eng->enable_mtp = enable_mtp;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
  mparams.load_mtp = enable_mtp;

  eng->model = llama_model_load_from_file(model_path, mparams);
  if (!eng->model) {
    delete eng;
    return nullptr;
  }
  eng->vocab = llama_model_get_vocab(eng->model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = max_tokens;
  cparams.n_batch = 512;
  cparams.n_ubatch = 256;
  cparams.no_perf = true;
  if (enable_mtp) {
    cparams.n_rs_seq = 2;
    auto out_limits = common_speculative_get_output_limits(cparams.n_batch, 1, 2);
    cparams.n_outputs_max = out_limits.total;
    cparams.n_outputs_max_per_seq = out_limits.per_seq;
  }

  eng->ctx = llama_init_from_model(eng->model, cparams);
  if (!eng->ctx) {
    llama_model_free(eng->model);
    delete eng;
    return nullptr;
  }

  if (enable_mtp && llama_model_n_layer_nextn(eng->model) > 0) {
    llama_context_params cparams_dft = cparams;
    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams_dft.n_rs_seq = 0;
    cparams_dft.n_outputs_max = 1;
    cparams_dft.n_outputs_max_per_seq = 1;
    cparams_dft.ctx_other = eng->ctx;
    eng->ctx_dft = llama_init_from_model(eng->model, cparams_dft);
    if (eng->ctx_dft) {
      common_params_speculative spec_params;
      spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
      spec_params.draft.ctx_tgt = eng->ctx;
      spec_params.draft.ctx_dft = eng->ctx_dft;
      spec_params.draft.n_max = 2;
      spec_params.draft.n_min = 0;
      spec_params.draft.p_min = 0.0f;
      eng->spec = common_speculative_init(spec_params, 1);
      if (!eng->spec) {
        llama_free(eng->ctx_dft);
        eng->ctx_dft = nullptr;
        eng->enable_mtp = false;
      }
    } else {
      eng->enable_mtp = false;
    }
  } else {
    eng->enable_mtp = false;
  }

  return eng;
}

void llama_shim_destroy(LlamaShimEngine *eng) {
  if (!eng) return;
  std::lock_guard<std::mutex> lock(eng->infer_mutex);
  if (eng->spec) common_speculative_free(eng->spec);
  if (eng->ctx_dft) llama_free(eng->ctx_dft);
  if (eng->ctx) llama_free(eng->ctx);
  if (eng->model) llama_model_free(eng->model);
  delete eng;
}

// vocab is immutable after model load -> lock-free & deadlock-free even from callbacks!
int llama_shim_count_tokens(LlamaShimEngine *eng, const char *text, size_t len) {
  if (!eng || !eng->vocab || !text || len == 0) return 0;
  int n_tok = -llama_tokenize(eng->vocab, text, static_cast<int32_t>(len), nullptr, 0, false, true);
  if (n_tok <= 0) return std::max(1, static_cast<int>(len / 3));
  return n_tok;
}

void llama_shim_stream(LlamaShimEngine *eng,
                       const char *system_msg,
                       const char *history_json,
                       const char *current_msg,
                       const LlamaShimGenOpts *opts,
                       LlamaChunkCallback chunk_cb,
                       LlamaDoneCallback done_cb,
                       LlamaErrorCallback error_cb,
                       LlamaIsCancelledCallback is_cancelled,
                       void *user_data) {
  if (!eng || !eng->ctx || !eng->model || !eng->vocab) {
    if (error_cb) error_cb(user_data, "llama.cpp 엔진이 준비되지 않았습니다.");
    return;
  }

  std::unique_lock<std::mutex> lock(eng->infer_mutex);

  std::string prompt = format_chat_prompt(eng->model, system_msg, history_json, current_msg);
  if (prompt.empty()) {
    lock.unlock();
    if (done_cb) done_cb(user_data);
    return;
  }

  int n_prompt = -llama_tokenize(eng->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
  if (n_prompt <= 0) n_prompt = 1;
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(eng->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                     prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()), true, true) < 0) {
    lock.unlock();
    if (error_cb) error_cb(user_data, "프롬프트 토큰화 실패");
    return;
  }

  if (static_cast<int>(prompt_tokens.size()) >= eng->max_tokens) {
    lock.unlock();
    if (error_cb) error_cb(user_data, "프롬프트 토큰 수가 최대 컨텍스트를 초과했습니다.");
    return;
  }

  llama_memory_clear(llama_get_memory(eng->ctx), true);
  if (eng->ctx_dft) {
    llama_memory_clear(llama_get_memory(eng->ctx_dft), true);
  }

  std::vector<std::string> effective_stops;
  if (opts && opts->stops) {
    for (size_t i = 0; i < opts->num_stops; ++i) {
      if (opts->stops[i] && opts->stops[i][0] != '\0') {
        effective_stops.emplace_back(opts->stops[i]);
      }
    }
  }
  effective_stops.push_back("<end_of_turn>");
  effective_stops.push_back("<start_of_turn>");
  effective_stops.push_back("<|eot_id|>");
  effective_stops.push_back("<|im_end|>");
  effective_stops.push_back("<|endoftext|>");
  effective_stops.push_back("<eos>");
  effective_stops.push_back("</s>");

  float temp = opts ? opts->temperature : 0.4f;
  float top_p = opts ? opts->top_p : 0.90f;
  int top_k = opts ? opts->top_k : 40;
  int seed_val = opts ? opts->seed : -1;
  float freq_pen = opts ? opts->frequency_penalty : 0.0f;
  float pres_pen = opts ? opts->presence_penalty : 0.0f;
  int max_gen = (opts && opts->max_tokens > 0)
                    ? opts->max_tokens
                    : (eng->max_tokens - static_cast<int>(prompt_tokens.size()));
  if (max_gen <= 0) max_gen = 256;

  std::string pending_buf;
  auto emit_piece = [&](const std::string &piece) -> bool {
    pending_buf += piece;
    size_t earliest = std::string::npos;
    for (const auto &sw : effective_stops) {
      if (sw.empty()) continue;
      size_t pos = pending_buf.find(sw);
      if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
        earliest = pos;
      }
    }
    if (earliest != std::string::npos) {
      if (earliest > 0 && chunk_cb) {
        chunk_cb(user_data, pending_buf.data(), earliest);
      }
      pending_buf.clear();
      return true;
    }

    size_t longest_prefix_match = 0;
    for (const auto &sw : effective_stops) {
      if (sw.empty()) continue;
      for (size_t plen = std::min(pending_buf.size(), sw.size()); plen > 0; --plen) {
        if (pending_buf.compare(pending_buf.size() - plen, plen, sw, 0, plen) == 0) {
          if (plen > longest_prefix_match) longest_prefix_match = plen;
          break;
        }
      }
    }

    if (longest_prefix_match > 0) {
      size_t safe_len = pending_buf.size() - longest_prefix_match;
      if (safe_len > 0) {
        if (chunk_cb) chunk_cb(user_data, pending_buf.data(), safe_len);
        pending_buf = pending_buf.substr(safe_len);
      }
    } else if (!pending_buf.empty()) {
      if (chunk_cb) chunk_cb(user_data, pending_buf.data(), pending_buf.size());
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
      if (!is_stop_prefix && chunk_cb) {
        chunk_cb(user_data, pending_buf.data(), pending_buf.size());
      }
      pending_buf.clear();
    }
  };

  if (eng->enable_mtp && eng->ctx_dft && eng->spec && !prompt_tokens.empty()) {
    common_params_sampling sp;
    sp.temp = temp;
    sp.top_k = top_k;
    sp.top_p = top_p;
    sp.seed = (seed_val >= 0) ? static_cast<uint32_t>(seed_val) : LLAMA_DEFAULT_SEED;
    sp.penalty_freq = freq_pen;
    sp.penalty_present = pres_pen;
    sp.no_perf = true;
    common_sampler_ptr smpl(common_sampler_init(eng->model, sp));

    llama_seq_id seq_id = 0;
    size_t n_prefill = prompt_tokens.size() - 1;
    int n_batch = 256;
    for (size_t i = 0; i < n_prefill; i += n_batch) {
      if (is_cancelled && is_cancelled(user_data)) {
        lock.unlock();
        if (done_cb) done_cb(user_data);
        return;
      }
      size_t cur_batch = std::min(static_cast<size_t>(n_batch), n_prefill - i);
      common_batch batch_prompt(eng->ctx);
      for (size_t j = 0; j < cur_batch; ++j) {
        batch_prompt.add(prompt_tokens[i + j], static_cast<llama_pos>(i + j), seq_id, false);
      }
      if (llama_process(eng->ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0 ||
          !common_speculative_process(eng->spec, batch_prompt)) {
        lock.unlock();
        if (error_cb) error_cb(user_data, "llama MTP prefill 실패");
        return;
      }
    }

    llama_token id_last = prompt_tokens.back();
    llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.end() - 1);
    prompt_tgt.reserve(eng->max_tokens);
    int n_past = static_cast<int>(prompt_tokens.size()) - 1;

    common_speculative_begin(eng->spec, seq_id, prompt_tgt);
    common_batch batch_tgt(eng->ctx);
    llama_tokens draft;
    int n_predict = 0;
    bool has_eos = false;

    while (n_predict < max_gen && !has_eos) {
      if (is_cancelled && is_cancelled(user_data)) break;

      if (draft.empty()) {
        int n_draft_max = std::max(0, std::min(eng->max_tokens - n_past - 2, max_gen - n_predict - 1));
        if (n_draft_max > 0) {
          common_speculative_get_draft_params(eng->spec, seq_id) = {
              true, n_draft_max, n_past, id_last, &prompt_tgt, &draft
          };
          common_speculative_draft(eng->spec);
          llama_memory_seq_rm(llama_get_memory(eng->ctx_dft), seq_id, n_past, -1);
        }
      }

      batch_tgt.clear();
      batch_tgt.add(id_last, n_past++, seq_id, true);
      for (size_t i = 0; i < draft.size(); ++i) {
        batch_tgt.add(draft[i], n_past + static_cast<int>(i), seq_id, true);
      }

      if (llama_process(eng->ctx, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0 ||
          !common_speculative_process(eng->spec, batch_tgt)) {
        break;
      }

      auto ids = common_sampler_sample_and_accept_n(smpl.get(), eng->ctx, draft);
      if (ids.empty()) break;

      common_speculative_accept(eng->spec, seq_id, ids.size() - 1);
      n_past += static_cast<int>(ids.size()) - 1;
      n_predict += static_cast<int>(ids.size());

      for (size_t i = 0; i < ids.size(); ++i) {
        prompt_tgt.push_back(id_last);
        id_last = ids[i];
        if (llama_vocab_is_eog(eng->vocab, id_last)) {
          has_eos = true;
          break;
        }
        char buf[256];
        int n = llama_token_to_piece(eng->vocab, id_last, buf, sizeof(buf), 0, false);
        if (n < 0 || emit_piece(std::string(buf, n))) {
          has_eos = true;
          break;
        }
      }

      draft.clear();
      llama_memory_seq_rm(llama_get_memory(eng->ctx), seq_id, n_past, -1);
      llama_memory_seq_rm(llama_get_memory(eng->ctx_dft), seq_id, n_past, -1);
    }

    flush_pending();
    lock.unlock();
    if (done_cb) done_cb(user_data);
    return;
  }

  // Standard prefill
  int n_batch = 512;
  for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
    if (is_cancelled && is_cancelled(user_data)) {
      lock.unlock();
      if (done_cb) done_cb(user_data);
      return;
    }
    size_t cur_batch = std::min(static_cast<size_t>(n_batch), prompt_tokens.size() - i);
    llama_batch batch = llama_batch_get_one(&prompt_tokens[i], static_cast<int32_t>(cur_batch));
    if (llama_decode(eng->ctx, batch)) {
      lock.unlock();
      if (error_cb) error_cb(user_data, "llama_decode prefill 실패");
      return;
    }
  }

  // Sampler chain: penalties MUST come BEFORE top_k / top_p / temp / dist / greedy!
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler *smpl = llama_sampler_chain_init(sparams);

  if (freq_pen != 0.0f || pres_pen != 0.0f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        llama_vocab_n_tokens(eng->vocab), 64, 1.0f, freq_pen, pres_pen));
  }
  if (temp <= 0.001f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  } else {
    if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    uint32_t seed = (seed_val >= 0) ? static_cast<uint32_t>(seed_val) : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
  }

  for (int step = 0; step < max_gen; ++step) {
    if (is_cancelled && is_cancelled(user_data)) break;
    llama_token token_id = llama_sampler_sample(smpl, eng->ctx, -1);
    if (llama_vocab_is_eog(eng->vocab, token_id)) break;

    char buf[256];
    int n = llama_token_to_piece(eng->vocab, token_id, buf, sizeof(buf), 0, false);
    if (n < 0 || emit_piece(std::string(buf, n))) break;

    llama_batch batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(eng->ctx, batch)) break;
  }

  flush_pending();
  llama_sampler_free(smpl);
  lock.unlock();
  if (done_cb) done_cb(user_data);
}

} // extern "C"
