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

// 요청 메시지를 LiteRT-LM 엔진이 지원하는 내부 JSON 형식으로 변환 (Base64 이미지 디코딩 포함)
static json transform_message_for_litert(const json &msg, std::shared_ptr<TempFilesCleanup> temp_files) {
  json transformed = msg;
  std::string role = msg.value("role", "user");
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

  // 2. OpenAI 멀티모달 content 배열 (image_url Base64 등) 처리
  if (msg.contains("content") && msg["content"].is_array()) {
    json content_arr = json::array();
    for (const auto &part : msg["content"]) {
      if (!part.is_object()) continue;
      std::string type = part.value("type", "");
      if (type == "text") {
        content_arr.push_back(part);
      } else if (type == "image_url" && part.contains("image_url") && part["image_url"].contains("url")) {
        std::string url = part["image_url"]["url"].get<std::string>();
        size_t b64_idx = url.find("base64,");
        if (b64_idx != std::string::npos) {
          std::string b64 = url.substr(b64_idx + 7);
          std::string path = save_base64_to_temp_file(b64);
          if (!path.empty()) {
            temp_files->paths.push_back(path);
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

// API 서버 실행부 구현
void RunServer(MultimodalCliApp &app, int port, const std::string &served_model_name) {
  httplib::Server svr;

  // CORS 및 프리플라이트 요청 처리 핸들러
  svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT");
    res.set_header("Access-Control-Allow-Headers", "*");
    if (req.method == "OPTIONS") { res.status = 204; return httplib::Server::HandlerResponse::Handled; }
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

  // 채팅 완료 처리 핵심 핸들러 (OpenAI 및 Ollama 엔드포인트 공용)
  auto handle_chat_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res, bool is_ollama) {
    try {
      auto j_req = json::parse(req.body);
      // Ollama 스펙상 stream 생략 시 기본값은 true, OpenAI는 false
      bool want_stream = j_req.contains("stream") ? j_req["stream"].get<bool>() : is_ollama;
      std::string sys_msg = "";
      json history_arr = json::array();
      json current_msg_j;

      auto temp_files = std::make_shared<TempFilesCleanup>();

      // 메시지 파싱 및 대화 기록/시스템 메시지 분리
      if (j_req.contains("messages") && j_req["messages"].is_array()) {
        auto messages = j_req["messages"];
        for (const auto &m : messages) {
          if (!m.is_object()) continue;
          std::string role = m.value("role", "");
          if (role == "system") {
            std::string content = m.value("content", "");
            if (!content.empty()) {
              if (!sys_msg.empty()) sys_msg += "\n\n";
              sys_msg += content;
            }
          } else {
            history_arr.push_back(transform_message_for_litert(m, temp_files));
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

      // LiteRT-LM / Gemma 엔진은 내부 템플릿에 system 토큰이 없으므로,
      // 첫 번째 user 메시지 앞단에 시스템 프롬프트를 자동으로 병합하여 100% 지침 준수 보장
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

        // 별도 스레드에서 엔진 실행 및 데이터 수신
        std::thread([&app, sys_msg, history_json_str, current_msg_str, served_model_name, is_ollama, sink_ctx, cancelled, temp_files]() {
          app.StreamForServer(sys_msg, history_json_str, current_msg_str,
            // 청크 수신 콜백
            [served_model_name, is_ollama, sink_ctx](const std::string &chunk) {
              json chunk_j;
              if (is_ollama) {
                chunk_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"message", {{"role", "assistant"}, {"content", chunk}}},
                  {"done", false}
                };
              } else {
                chunk_j = {
                  {"id", "chatcmpl-litert"},
                  {"object", "chat.completion.chunk"},
                  {"created", time(nullptr)},
                  {"model", served_model_name},
                  {"choices", {{{{"delta", {{"content", chunk}}}}, {{"finish_reason", nullptr}}}}}
                };
              }
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(is_ollama ? chunk_j.dump() + "\n" : "data: " + chunk_j.dump() + "\n\n");
              sink_ctx->cv.notify_one();
            },
            // 완료 콜백
            [served_model_name, is_ollama, sink_ctx]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
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
            [cancelled]() { return cancelled->load(); }
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
      // 일반 동기 응답 처리 루틴
      else {
        std::string output = app.GenerateForServer(sys_msg, history_json_str, current_msg_str);
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
          api_res = {
            {"choices", {{{"message", {{"role", "assistant"}, {"content", output}}}, {"finish_reason", "stop"}}}}
          };
        }
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) { 
      res.status = 500; 
      res.set_content(e.what(), "text/plain"); 
    }
  };

  // Ollama 단일 생성 (/api/generate) 핸들러
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

        std::thread([&app, sys_msg, current_msg_str, served_model_name, sink_ctx, cancelled, temp_files]() {
          app.StreamForServer(sys_msg, "", current_msg_str,
            [served_model_name, sink_ctx](const std::string &chunk) {
              json chunk_j = {
                {"model", served_model_name},
                {"created_at", get_iso8601_now()},
                {"response", chunk},
                {"done", false}
              };
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(chunk_j.dump() + "\n");
              sink_ctx->cv.notify_one();
            },
            [served_model_name, sink_ctx]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
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
            [cancelled]() { return cancelled->load(); }
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
        std::string output = app.GenerateForServer(sys_msg, "", current_msg_str);
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

  // OpenAI 호환 엔드포인트 등록
  svr.Post("/v1/chat/completions", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, false); });
  
  // Ollama 호환 엔드포인트 등록
  svr.Post("/api/chat", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, true); });
  svr.Post("/api/generate", handle_generate);

  // 서버 바인딩 및 수신 대기
  std::cout << "[서버] 0.0.0.0:" << port << " 대기 중 (모델: " << served_model_name << ")" << std::endl;
  svr.listen("0.0.0.0", port);
}
