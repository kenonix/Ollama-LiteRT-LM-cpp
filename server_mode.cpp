#include "server_mode.h"
#include "httplib.h"
#include "json.hpp"
#include "utils.h"
#include "config.h"
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

// 멀티모달 임시 이미지 파일 자동 정리 RAII 구조체
struct TempFilesCleanup {
  std::vector<std::string> paths;
  ~TempFilesCleanup() {
    for (const auto &p : paths) {
      if (!p.empty()) {
        std::remove(p.c_str());
      }
    }
  }
};

// 스트리밍 도중 UTF-8 멀티바이트 문자가 바이트 단위로 쪼개지는 것을 방지하는 리어셈블러
class Utf8StreamReassembler {
private:
  std::string buffer_;

public:
  // 입력 문자열을 받아 온전한 UTF-8 문자들만 반환하고, 덜 온 바이트는 버퍼에 보관
  std::string process(const std::string &incoming) {
    buffer_ += incoming;
    if (buffer_.empty()) return "";

    size_t valid_len = buffer_.size();
    size_t i = buffer_.size();

    while (i > 0 && i >= (buffer_.size() > 4 ? buffer_.size() - 4 : 0)) {
      --i;
      unsigned char c = static_cast<unsigned char>(buffer_[i]);
      if ((c & 0x80) == 0) {
        // ASCII 문자 (완전)
        break;
      } else if ((c & 0xC0) == 0xC0) {
        // 멀티바이트 선두 바이트 (0b11xxxxxx)
        size_t expected_len = 0;
        if ((c & 0xE0) == 0xC0) expected_len = 2;
        else if ((c & 0xF0) == 0xE0) expected_len = 3;
        else if ((c & 0xF8) == 0xF0) expected_len = 4;

        size_t actual_len = buffer_.size() - i;
        if (actual_len < expected_len) {
          valid_len = i;
        }
        break;
      }
    }

    if (valid_len == 0) return "";
    std::string ready = buffer_.substr(0, valid_len);
    buffer_.erase(0, valid_len);
    return ready;
  }

  // 스트림 완료 시 남은 버퍼 반환
  std::string flush() {
    std::string remaining = std::move(buffer_);
    buffer_.clear();
    return remaining;
  }
};

// 요청 메시지를 LiteRT-LM 엔진이 지원하는 내부 JSON 형식으로 변환 (Base64 / URL 이미지 처리 포함)
static json transform_message_for_litert(const json &msg, std::shared_ptr<TempFilesCleanup> temp_files) {
  json transformed = msg;
  std::string role = msg.value("role", "user");
  // OpenAI developer role을 system과 동일하게 취급 (LiteRT-LM 내부 처리용)
  if (role == "developer") {
    role = "system";
  }
  transformed["role"] = role;

  // 1. Ollama "images" 필드 (Base64 배열) 처리
  if (msg.contains("images") && msg["images"].is_array() && !msg["images"].empty()) {
    std::string text_content = "";
    if (msg.contains("content") && msg["content"].is_string()) {
      text_content = msg["content"].get<std::string>();
    }
    json content_arr = json::array();
    for (const auto &img_b64 : msg["images"]) {
      if (img_b64.is_string()) {
        std::string path = save_base64_to_temp_file(img_b64.get<std::string>());
        if (!path.empty()) {
          temp_files->paths.push_back(path);
          content_arr.push_back({{"type", "image"}, {"path", path}});
        }
      }
    }
    if (!text_content.empty()) {
      content_arr.push_back({{"type", "text"}, {"text", text_content}});
    }
    transformed["content"] = content_arr;
    transformed.erase("images");
    return transformed;
  }

  // 2. OpenAI 멀티모달 content 배열 (image_url: Base64 data URL, 일반 URL, 로컬 파일 등) 처리
  if (msg.contains("content") && msg["content"].is_array()) {
    json content_arr = json::array();
    for (const auto &part : msg["content"]) {
      if (!part.is_object()) continue;
      std::string type = part.value("type", "");
      if (type == "text") {
        content_arr.push_back(part);
      } else if (type == "image_url") {
        std::string img_src = "";
        if (part.contains("image_url")) {
          if (part["image_url"].is_string()) {
            img_src = part["image_url"].get<std::string>();
          } else if (part["image_url"].is_object() && part["image_url"].contains("url")) {
            img_src = part["image_url"]["url"].get<std::string>();
          }
        }
        if (!img_src.empty()) {
          bool is_temp = false;
          std::string path = save_image_source_to_temp_file(img_src, is_temp);
          if (!path.empty()) {
            if (is_temp) {
              temp_files->paths.push_back(path);
            }
            content_arr.push_back({{"type", "image"}, {"path", path}});
          }
        }
      }
    }
    transformed["content"] = content_arr;
    return transformed;
  }

  return transformed;
}

// 시스템 프롬프트를 사용자 메시지 앞단에 자연스럽게 병합 (Gemma/LiteRT-LM의 미지원 시스템 토큰 호환)
static void prepend_system_prompt_to_message(json &msg, const std::string &sys_prompt) {
  if (sys_prompt.empty()) return;
  if (msg.contains("content")) {
    if (msg["content"].is_string()) {
      std::string original = msg["content"].get<std::string>();
      if (original.empty()) {
        msg["content"] = sys_prompt;
      } else {
        msg["content"] = sys_prompt + "\n\n" + original;
      }
    } else if (msg["content"].is_array()) {
      bool prepended = false;
      for (auto &part : msg["content"]) {
        if (part.is_object() && part.value("type", "") == "text") {
          std::string original = part.value("text", "");
          if (original.empty()) {
            part["text"] = sys_prompt;
          } else {
            part["text"] = sys_prompt + "\n\n" + original;
          }
          prepended = true;
          break;
        }
      }
      if (!prepended) {
        msg["content"].insert(msg["content"].begin(), {{"type", "text"}, {"text", sys_prompt}});
      }
    }
  } else {
    msg["content"] = sys_prompt;
  }
}

// 요청 JSON으로부터 세부 생성 옵션(GenerationOptions)을 추출
static GenerationOptions parse_generation_options(const json &j_req) {
  GenerationOptions opts;

  // 1. OpenAI 표준 파라미터 파싱
  if (j_req.contains("temperature") && j_req["temperature"].is_number()) {
    opts.temperature = j_req["temperature"].get<float>();
  }
  if (j_req.contains("top_p") && j_req["top_p"].is_number()) {
    opts.top_p = j_req["top_p"].get<float>();
  }
  if (j_req.contains("top_k") && j_req["top_k"].is_number_integer()) {
    opts.top_k = j_req["top_k"].get<int>();
  }
  if (j_req.contains("max_tokens") && j_req["max_tokens"].is_number_integer()) {
    opts.max_tokens = j_req["max_tokens"].get<int>();
  } else if (j_req.contains("max_completion_tokens") && j_req["max_completion_tokens"].is_number_integer()) {
    opts.max_tokens = j_req["max_completion_tokens"].get<int>();
  }
  if (j_req.contains("seed") && j_req["seed"].is_number_integer()) {
    opts.seed = j_req["seed"].get<int>();
  }
  if (j_req.contains("frequency_penalty") && j_req["frequency_penalty"].is_number()) {
    opts.frequency_penalty = j_req["frequency_penalty"].get<float>();
  }
  if (j_req.contains("presence_penalty") && j_req["presence_penalty"].is_number()) {
    opts.presence_penalty = j_req["presence_penalty"].get<float>();
  }

  // 2. Ollama "options" 객체 파싱 (있는 경우 덮어쓰기)
  if (j_req.contains("options") && j_req["options"].is_object()) {
    const auto &o = j_req["options"];
    if (o.contains("temperature") && o["temperature"].is_number()) {
      opts.temperature = o["temperature"].get<float>();
    }
    if (o.contains("top_p") && o["top_p"].is_number()) {
      opts.top_p = o["top_p"].get<float>();
    }
    if (o.contains("top_k") && o["top_k"].is_number_integer()) {
      opts.top_k = o["top_k"].get<int>();
    }
    if (o.contains("num_predict") && o["num_predict"].is_number_integer()) {
      opts.max_tokens = o["num_predict"].get<int>();
    }
    if (o.contains("seed") && o["seed"].is_number_integer()) {
      opts.seed = o["seed"].get<int>();
    }
    if (o.contains("repeat_penalty") && o["repeat_penalty"].is_number()) {
      opts.frequency_penalty = o["repeat_penalty"].get<float>() - 1.0f;
    }
    if (o.contains("stop")) {
      if (o["stop"].is_string()) {
        opts.stop.push_back(o["stop"].get<std::string>());
      } else if (o["stop"].is_array()) {
        for (const auto &s : o["stop"]) {
          if (s.is_string()) opts.stop.push_back(s.get<std::string>());
        }
      }
    }
  }

  // 3. stop 단어 파싱
  if (j_req.contains("stop")) {
    if (j_req["stop"].is_string()) {
      opts.stop.push_back(j_req["stop"].get<std::string>());
    } else if (j_req["stop"].is_array()) {
      for (const auto &s : j_req["stop"]) {
        if (s.is_string()) opts.stop.push_back(s.get<std::string>());
      }
    }
  }

  return opts;
}

// OpenAI 규격 Model 정보 JSON 생성 헬퍼
static json make_openai_model_info(const std::string &model_name) {
  return {
    {"id", model_name},
    {"object", "model"},
    {"created", 1728000000},
    {"owned_by", "litert-lm"},
    {"permission", json::array({
      {
        {"id", "modelperm-" + model_name},
        {"object", "model_permission"},
        {"created", 1728000000},
        {"allow_create_engine", false},
        {"allow_sampling", true},
        {"allow_logprobs", true},
        {"allow_search_indices", false},
        {"allow_view", true},
        {"allow_fine_tuning", false},
        {"organization", "*"},
        {"group", nullptr},
        {"is_blocking", false}
      }
    })},
    {"root", model_name},
    {"parent", nullptr}
  };
}

// API 서버 실행부 구현
void RunServer(MultimodalCliApp &app, int port, const std::string &served_model_name) {
  httplib::Server svr;

  // CORS 및 프리플라이트 요청 처리 핸들러
  svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT");
    res.set_header("Access-Control-Allow-Headers", "*");
    if (req.method == "OPTIONS") {
      res.status = 204;
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  // 상태 확인용 기본 엔드포인트
  svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
    res.set_content("Ollama is running", "text/plain");
  });

  // Ollama 버전 확인 엔드포인트
  svr.Get("/api/version", [](const httplib::Request &, httplib::Response &res) {
    res.set_content(json({{"version", "0.3.14"}}).dump(), "application/json");
  });

  // 모델 목록 엔드포인트 (Ollama 호환)
  svr.Get("/api/tags", [&served_model_name](const httplib::Request &, httplib::Response &res) {
    json model_entry = {
      {"name", served_model_name}, 
      {"model", served_model_name}, 
      {"modified_at", get_iso8601_now()}, 
      {"size", 0}, 
      {"digest", "000000000000"}, 
      {"details", {{"format", "tflite"}, {"family", "litert"}, {"families", json::array({"litert"})}}}
    };
    json response = {{"models", json::array({model_entry})}};
    res.set_content(response.dump(), "application/json");
  });

  // 모델 상세 정보 엔드포인트 (Ollama 호환)
  svr.Post("/api/show", [&served_model_name](const httplib::Request &, httplib::Response &res) {
    json resp = {
      {"modelfile", "# Modelfile generated by Ollama-LiteRT-LM-cpp\nFROM " + served_model_name},
      {"parameters", "stop \"<end_of_turn>\""},
      {"template", "{{ .Prompt }}"},
      {"details", {
        {"parent_model", ""},
        {"format", "tflite"},
        {"family", "litert"},
        {"families", json::array({"litert"})},
        {"parameter_size", "2B"},
        {"quantization_level", "none"}
      }}
    };
    res.set_content(resp.dump(), "application/json");
  });

  // ==========================================
  // OpenAI 모델 목록 및 단건 조회 엔드포인트
  // ==========================================
  auto handle_models_list = [&served_model_name](const httplib::Request &, httplib::Response &res) {
    json response = {
      {"object", "list"},
      {"data", json::array({make_openai_model_info(served_model_name)})}
    };
    res.set_content(response.dump(), "application/json");
  };

  svr.Get("/v1/models", handle_models_list);
  svr.Get("/models", handle_models_list);

  auto handle_single_model = [&served_model_name](const httplib::Request &req, httplib::Response &res) {
    std::string requested = req.matches[1];
    res.set_content(make_openai_model_info(requested.empty() ? served_model_name : requested).dump(), "application/json");
  };

  svr.Get(R"(/v1/models/(.+))", handle_single_model);
  svr.Get(R"(/models/(.+))", handle_single_model);

  // ==========================================
  // OpenAI 임베딩 엔드포인트 (안내 에러 응답)
  // ==========================================
  auto handle_embeddings = [](const httplib::Request &, httplib::Response &res) {
    res.status = 400;
    json err = {
      {"error", {
        {"message", "This model is a text generative model and does not support embeddings."},
        {"type", "invalid_request_error"},
        {"param", nullptr},
        {"code", "model_not_supported"}
      }}
    };
    res.set_content(err.dump(), "application/json");
  };
  svr.Post("/v1/embeddings", handle_embeddings);
  svr.Post("/embeddings", handle_embeddings);

  // ==========================================
  // 채팅 완료 처리 핸들러 (OpenAI 및 Ollama 공용)
  // ==========================================
  auto handle_chat_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res, bool is_ollama) {
    try {
      auto j_req = json::parse(req.body);
      // Ollama 스펙상 stream 생략 시 기본값은 true, OpenAI는 false
      bool want_stream = j_req.contains("stream") ? j_req["stream"].get<bool>() : is_ollama;

      // OpenAI stream_options (include_usage) 확인
      bool include_usage = false;
      if (!is_ollama && j_req.contains("stream_options") && j_req["stream_options"].is_object()) {
        include_usage = j_req["stream_options"].value("include_usage", false);
      }

      GenerationOptions gen_opts = parse_generation_options(j_req);

      std::string sys_msg = "";
      json history_arr = json::array();
      json current_msg_j;
      std::string full_prompt_text = "";

      auto temp_files = std::make_shared<TempFilesCleanup>();

      // 메시지 파싱 및 대화 기록/시스템 메시지 분리
      if (j_req.contains("messages") && j_req["messages"].is_array()) {
        auto messages = j_req["messages"];
        for (const auto &m : messages) {
          if (!m.is_object()) continue;
          std::string role = m.value("role", "");
          if (role == "system" || role == "developer") {
            std::string content = "";
            if (m.contains("content") && m["content"].is_string()) {
              content = m["content"].get<std::string>();
            }
            if (!content.empty()) {
              if (!sys_msg.empty()) sys_msg += "\n\n";
              sys_msg += content;
            }
            full_prompt_text += content + "\n";
          } else {
            json tr = transform_message_for_litert(m, temp_files);
            history_arr.push_back(tr);
            if (m.contains("content") && m["content"].is_string()) {
              full_prompt_text += m["content"].get<std::string>() + "\n";
            }
          }
        }
      }

      if (sys_msg.empty()) {
        sys_msg = DEFAULT_SYSTEM_PROMPT;
      }

      if (!history_arr.empty()) {
        current_msg_j = history_arr.back();
        history_arr.erase(history_arr.end() - 1);
      } else {
        current_msg_j = {{"role", "user"}, {"content", ""}};
      }

      // LiteRT-LM / Gemma 엔진은 시스템 토큰이 없으므로 첫 번째 user 메시지 앞단에 시스템 프롬프트 병합
      if (!sys_msg.empty()) {
        bool injected = false;
        for (auto &m : history_arr) {
          if (m.value("role", "") == "user") {
            prepend_system_prompt_to_message(m, sys_msg);
            injected = true;
            break;
          }
        }
        if (!injected) {
          prepend_system_prompt_to_message(current_msg_j, sys_msg);
        }
      }

      std::string history_json_str = history_arr.empty() ? "" : history_arr.dump();
      std::string current_msg_str = current_msg_j.dump();

      // 프롬프트 토큰 수 계산 (정확도 높은 카운트)
      int prompt_tokens = std::max(1, app.CountTokens(full_prompt_text.empty() ? sys_msg : full_prompt_text));

      // OpenAI 규격용 ID 및 생성 타임스탬프
      std::string chat_id = generate_random_id("chatcmpl-");
      uint64_t created_ts = static_cast<uint64_t>(time(nullptr));

      // 스트리밍 응답 처리 루틴
      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);
        auto accumulated_text = std::make_shared<std::string>();
        auto is_first_chunk = std::make_shared<bool>(true);
        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();

        std::thread([&app, sys_msg, history_json_str, current_msg_str, served_model_name, is_ollama,
                     sink_ctx, cancelled, temp_files, gen_opts, chat_id, created_ts,
                     prompt_tokens, include_usage, accumulated_text, is_first_chunk, utf8_asm]() {
          app.StreamForServer(sys_msg, history_json_str, current_msg_str,
            // 청크 수신 콜백
            [served_model_name, is_ollama, sink_ctx, chat_id, created_ts, accumulated_text, is_first_chunk, utf8_asm](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              json chunk_j;
              if (is_ollama) {
                chunk_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"message", {{"role", "assistant"}, {"content", clean_chunk}}},
                  {"done", false}
                };
              } else {
                json delta_obj = json::object();
                if (*is_first_chunk) {
                  delta_obj["role"] = "assistant";
                  *is_first_chunk = false;
                }
                delta_obj["content"] = clean_chunk;

                chunk_j = {
                  {"id", chat_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"system_fingerprint", "fp_litert_lm"},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", delta_obj},
                      {"logprobs", nullptr},
                      {"finish_reason", nullptr}
                    }
                  })}
                };
                *accumulated_text += clean_chunk;
              }
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(is_ollama ? chunk_j.dump() + "\n" : "data: " + chunk_j.dump() + "\n\n");
              sink_ctx->cv.notify_one();
            },
            // 완료 콜백
            [&app, served_model_name, is_ollama, sink_ctx, chat_id, created_ts,
             prompt_tokens, include_usage, accumulated_text, utf8_asm, is_first_chunk]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              if (!leftover.empty()) {
                if (is_ollama) {
                  json chunk_j = {
                    {"model", served_model_name},
                    {"created_at", get_iso8601_now()},
                    {"message", {{"role", "assistant"}, {"content", leftover}}},
                    {"done", false}
                  };
                  sink_ctx->chunks.push(chunk_j.dump() + "\n");
                } else {
                  json delta_obj = json::object();
                  if (*is_first_chunk) {
                    delta_obj["role"] = "assistant";
                    *is_first_chunk = false;
                  }
                  delta_obj["content"] = leftover;
                  json chunk_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array({
                      {
                        {"index", 0},
                        {"delta", delta_obj},
                        {"logprobs", nullptr},
                        {"finish_reason", nullptr}
                      }
                    })}
                  };
                  sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
                  *accumulated_text += leftover;
                }
              }

              if (is_ollama) {
                json done_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"message", {{"role", "assistant"}, {"content", ""}}},
                  {"done_reason", "stop"},
                  {"done", true}
                };
                sink_ctx->chunks.push(done_j.dump() + "\n");
              } else {
                // OpenAI 스트리밍 종료 청크: finish_reason="stop"
                json stop_j = {
                  {"id", chat_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"system_fingerprint", "fp_litert_lm"},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", json::object()},
                      {"logprobs", nullptr},
                      {"finish_reason", "stop"}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + stop_j.dump() + "\n\n");

                // stream_options.include_usage가 설정된 경우 usage 청크 전송
                if (include_usage) {
                  int comp_tokens = app.CountTokens(*accumulated_text);
                  json usage_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array()},
                    {"usage", {
                      {"prompt_tokens", prompt_tokens},
                      {"completion_tokens", comp_tokens},
                      {"total_tokens", prompt_tokens + comp_tokens}
                    }}
                  };
                  sink_ctx->chunks.push("data: " + usage_j.dump() + "\n\n");
                }

                // 스트림 완료 신호
                sink_ctx->chunks.push("data: [DONE]\n\n");
              }
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            // 에러 콜백
            [sink_ctx](const std::string &) { 
              std::lock_guard<std::mutex> lock(sink_ctx->mtx); 
              sink_ctx->done = true; 
              sink_ctx->cv.notify_one(); 
            },
            // 취소 확인 콜백
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider(is_ollama ? "application/x-ndjson" : "text/event-stream", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } 
      // 일반 동기(비스트리밍) 응답 처리 루틴
      else {
        std::string output = app.GenerateForServer(sys_msg, history_json_str, current_msg_str, gen_opts);
        json api_res;
        if (is_ollama) {
          api_res = {
            {"model", served_model_name},
            {"created_at", get_iso8601_now()},
            {"message", {{"role", "assistant"}, {"content", output}}},
            {"done_reason", "stop"},
            {"done", true}
          };
        } else {
          int completion_tokens = app.CountTokens(output);
          api_res = {
            {"id", chat_id},
            {"object", "chat.completion"},
            {"created", created_ts},
            {"model", served_model_name},
            {"system_fingerprint", "fp_litert_lm"},
            {"choices", json::array({
              {
                {"index", 0},
                {"message", {
                  {"role", "assistant"},
                  {"content", output}
                }},
                {"logprobs", nullptr},
                {"finish_reason", "stop"}
              }
            })},
            {"usage", {
              {"prompt_tokens", prompt_tokens},
              {"completion_tokens", completion_tokens},
              {"total_tokens", prompt_tokens + completion_tokens}
            }}
          };
        }
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) { 
      res.status = 500; 
      res.set_content(e.what(), "text/plain"); 
    }
  };

  // ==========================================
  // OpenAI Legacy Completion (/v1/completions) 핸들러
  // ==========================================
  auto handle_legacy_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res) {
    try {
      auto j_req = json::parse(req.body);
      bool want_stream = j_req.value("stream", false);
      GenerationOptions gen_opts = parse_generation_options(j_req);

      std::string prompt = "";
      if (j_req.contains("prompt")) {
        if (j_req["prompt"].is_string()) {
          prompt = j_req["prompt"].get<std::string>();
        } else if (j_req["prompt"].is_array() && !j_req["prompt"].empty()) {
          prompt = j_req["prompt"][0].get<std::string>();
        }
      }

      int prompt_tokens = std::max(1, app.CountTokens(prompt));
      std::string cmpl_id = generate_random_id("cmpl-");
      uint64_t created_ts = static_cast<uint64_t>(time(nullptr));

      json current_msg_j = {{"role", "user"}, {"content", prompt}};
      std::string current_msg_str = current_msg_j.dump();

      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);

        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();

        std::thread([&app, current_msg_str, served_model_name, sink_ctx, cancelled, gen_opts, cmpl_id, created_ts, utf8_asm]() {
          app.StreamForServer(DEFAULT_SYSTEM_PROMPT, "", current_msg_str,
            [served_model_name, sink_ctx, cmpl_id, created_ts, utf8_asm](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              json chunk_j = {
                {"id", cmpl_id},
                {"object", "text_completion"},
                {"created", created_ts},
                {"model", served_model_name},
                {"choices", json::array({
                  {
                    {"text", clean_chunk},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", nullptr}
                  }
                })}
              };
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
              sink_ctx->cv.notify_one();
            },
            [served_model_name, sink_ctx, cmpl_id, created_ts, utf8_asm]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              if (!leftover.empty()) {
                json chunk_j = {
                  {"id", cmpl_id},
                  {"object", "text_completion"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"choices", json::array({
                    {
                      {"text", leftover},
                      {"index", 0},
                      {"logprobs", nullptr},
                      {"finish_reason", nullptr}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
              }

              json stop_j = {
                {"id", cmpl_id},
                {"object", "text_completion"},
                {"created", created_ts},
                {"model", served_model_name},
                {"choices", json::array({
                  {
                    {"text", ""},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", "stop"}
                  }
                })}
              };
              sink_ctx->chunks.push("data: " + stop_j.dump() + "\n\n");
              sink_ctx->chunks.push("data: [DONE]\n\n");
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [sink_ctx](const std::string &) {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider("text/event-stream", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } else {
        std::string output = app.GenerateForServer(DEFAULT_SYSTEM_PROMPT, "", current_msg_str, gen_opts);
        int completion_tokens = app.CountTokens(output);
        json api_res = {
          {"id", cmpl_id},
          {"object", "text_completion"},
          {"created", created_ts},
          {"model", served_model_name},
          {"choices", json::array({
            {
              {"text", output},
              {"index", 0},
              {"logprobs", nullptr},
              {"finish_reason", "stop"}
            }
          })},
          {"usage", {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}
          }}
        };
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) {
      res.status = 500;
      res.set_content(e.what(), "text/plain");
    }
  };

  // ==========================================
  // Ollama 단일 생성 (/api/generate) 핸들러
  // ==========================================
  auto handle_generate = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res) {
    try {
      auto j_req = json::parse(req.body);
      bool want_stream = j_req.contains("stream") ? j_req["stream"].get<bool>() : true;
      std::string prompt = j_req.value("prompt", "");
      std::string sys_msg = j_req.value("system", DEFAULT_SYSTEM_PROMPT);

      std::string effective_prompt = prompt;
      if (!sys_msg.empty()) {
        effective_prompt = sys_msg + "\n\n" + prompt;
      }

      GenerationOptions gen_opts = parse_generation_options(j_req);
      auto temp_files = std::make_shared<TempFilesCleanup>();
      json current_msg_j;

      // 이미지 처리 (Base64 배열)
      if (j_req.contains("images") && j_req["images"].is_array() && !j_req["images"].empty()) {
        json content_arr = json::array();
        for (const auto &img_b64 : j_req["images"]) {
          if (img_b64.is_string()) {
            std::string path = save_base64_to_temp_file(img_b64.get<std::string>());
            if (!path.empty()) {
              temp_files->paths.push_back(path);
              content_arr.push_back({{"type", "image"}, {"path", path}});
            }
          }
        }
        if (!effective_prompt.empty()) {
          content_arr.push_back({{"type", "text"}, {"text", effective_prompt}});
        }
        current_msg_j = {{"role", "user"}, {"content", content_arr}};
      } else {
        current_msg_j = {{"role", "user"}, {"content", effective_prompt}};
      }

      std::string current_msg_str = current_msg_j.dump();

      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);

        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();

        std::thread([&app, sys_msg, current_msg_str, served_model_name, sink_ctx, cancelled, temp_files, gen_opts, utf8_asm]() {
          app.StreamForServer(sys_msg, "", current_msg_str,
            [served_model_name, sink_ctx, utf8_asm](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              json chunk_j = {
                {"model", served_model_name},
                {"created_at", get_iso8601_now()},
                {"response", clean_chunk},
                {"done", false}
              };
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(chunk_j.dump() + "\n");
              sink_ctx->cv.notify_one();
            },
            [served_model_name, sink_ctx, utf8_asm]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              if (!leftover.empty()) {
                json chunk_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"response", leftover},
                  {"done", false}
                };
                sink_ctx->chunks.push(chunk_j.dump() + "\n");
              }

              json done_j = {
                {"model", served_model_name},
                {"created_at", get_iso8601_now()},
                {"response", ""},
                {"done_reason", "stop"},
                {"done", true}
              };
              sink_ctx->chunks.push(done_j.dump() + "\n");
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [sink_ctx](const std::string &) {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider("application/x-ndjson", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } else {
        std::string output = app.GenerateForServer(sys_msg, "", current_msg_str, gen_opts);
        json api_res = {
          {"model", served_model_name},
          {"created_at", get_iso8601_now()},
          {"response", output},
          {"done_reason", "stop"},
          {"done", true}
        };
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) {
      res.status = 500;
      res.set_content(e.what(), "text/plain");
    }
  };

  // OpenAI 호환 엔드포인트 등록 (경로 유연성 지원)
  svr.Post("/v1/chat/completions", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, false); });
  svr.Post("/chat/completions", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, false); });
  svr.Post("/v1/completions", handle_legacy_completion);
  svr.Post("/completions", handle_legacy_completion);

  // Ollama 호환 엔드포인트 등록
  svr.Post("/api/chat", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, true); });
  svr.Post("/api/generate", handle_generate);

  // 서버 바인딩 및 수신 대기
  std::cout << "[서버] 0.0.0.0:" << port << " 대기 중 (모델: " << served_model_name << ")" << std::endl;
  svr.listen("0.0.0.0", port);
}
