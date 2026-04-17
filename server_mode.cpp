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
  svr.Get("/", [](const httplib::Request &, httplib::Response &res) { res.set_content("Ollama is running", "text/plain"); });

  // 모델 목록 엔드포인트 (Ollama 호환)
  svr.Get("/api/tags", [&served_model_name](const httplib::Request &, httplib::Response &res) {
    json model_entry = {
      {"name", served_model_name}, 
      {"model", served_model_name}, 
      {"modified_at", get_iso8601_now()}, 
      {"size", 0}, 
      {"digest", "000000"}, 
      {"details", {{"format", "tflite"}, {"family", "litert"}}}
    };
    json response = {{"models", json::array({model_entry})}};
    res.set_content(response.dump(), "application/json");
  });

  // 채팅 완료 처리 핵심 핸들러 (OpenAI 및 Ollama 엔드포인트 공용)
  auto handle_chat_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res, bool is_ollama) {
    try {
      auto j_req = json::parse(req.body);
      bool want_stream = j_req.contains("stream") && j_req["stream"].get<bool>();
      std::string sys_msg = DEFAULT_SYSTEM_PROMPT;
      json history_arr = json::array();
      json current_msg_j;

      // 메시지 파싱 및 대화 기록/시스템 메시지 분리
      if (j_req.contains("messages") && j_req["messages"].is_array()) {
        auto messages = j_req["messages"];
        if (!messages.empty()) {
          current_msg_j = messages.back(); // 마지막 메시지를 현재 질문으로 설정
          for (size_t i = 0; i < messages.size() - 1; ++i) {
            if (messages[i]["role"] == "system") sys_msg = messages[i]["content"].get<std::string>();
            else history_arr.push_back(messages[i]);
          }
        }
      }

      std::string history_json_str = history_arr.empty() ? "" : history_arr.dump();
      std::string current_msg_str = current_msg_j.dump();

      // 스트리밍 응답 처리 루틴
      if (want_stream) {
        // 스트리밍 데이터를 모으기 위한 컨텍스트
        struct SinkCtx { std::mutex mtx; std::condition_variable cv; std::queue<std::string> chunks; bool done = false; };
        auto sink_ctx = std::make_shared<SinkCtx>();

        // 별도 스레드에서 엔진 실행 및 데이터 수신
        std::thread([&app, sys_msg, history_json_str, current_msg_str, served_model_name, is_ollama, sink_ctx]() {
          app.StreamForServer(sys_msg, history_json_str, current_msg_str,
            // 엔진으로부터 텍스트 청크 수신 시
            [&served_model_name, is_ollama, sink_ctx](const std::string &chunk) {
              json chunk_j;
              if (is_ollama) {
                // Ollama 형식 청크
                chunk_j = {{"model", served_model_name}, {"created_at", get_iso8601_now()}, {"message", {{"role", "assistant"}, {"content", chunk}}}, {"done", false}};
              } else {
                // OpenAI 형식 청크
                chunk_j = {{"id", "chatcmpl-litert"}, {"object", "chat.completion.chunk"}, {"created", time(0)}, {"model", served_model_name}, {"choices", {{{{"delta", {{"content", chunk}}}}, {{"finish_reason", nullptr}}}}}};
              }
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(is_ollama ? chunk_j.dump() + "\n" : "data: " + chunk_j.dump() + "\n\n");
              sink_ctx->cv.notify_one();
            },
            // 엔진 작업 완료 시
            [is_ollama, &served_model_name, sink_ctx]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(is_ollama ? json({{"model", served_model_name}, {"done", true}}).dump() + "\n" : "data: [DONE]\n\n");
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            // 오류 발생 시
            [sink_ctx](const std::string &) { std::lock_guard<std::mutex> lock(sink_ctx->mtx); sink_ctx->done = true; sink_ctx->cv.notify_one(); });
        }).detach();

        // HTTP Chunked 데이터 전송 핸들러 설정
        res.set_chunked_content_provider(is_ollama ? "application/x-ndjson" : "text/event-stream", [sink_ctx](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) return false; // 연결 끊김 처리
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
          api_res = {{"model", served_model_name}, {"message", {{"role", "assistant"}, {"content", output}}}, {"done", true}};
        } else {
          api_res = {{"choices", {{{"message", {{"role", "assistant"}, {"content", output}}}, {"finish_reason", "stop"}}}}};
        }
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

  // 서버 바인딩 및 수신 대기
  svr.listen("0.0.0.0", port);
}
