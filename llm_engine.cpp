#include "llm_engine.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <queue>

// 엔진 응답 JSON에서 텍스트를 파싱하여 추출
std::string extract_text_from_chunk(const char *chunk) {
  if (!chunk) return "";
  try {
    auto j = json::parse(chunk);
    if (j.contains("content")) {
      // content가 문자열인 경우 (단순 텍스트 응답)
      if (j["content"].is_string()) {
        return j["content"].get<std::string>();
      } 
      // content가 배열인 경우 (멀티모달 응답 등)
      else if (j["content"].is_array()) {
        std::string result;
        for (auto &part : j["content"]) {
          if (part.contains("text") && part["text"].is_string()) {
            result += part["text"].get<std::string>();
          }
        }
        return result;
      }
    }
  } catch (...) {}
  // 파싱 실패 시 원본 문자열 반환
  return std::string(chunk);
}

// 스트리밍 데이터를 가로채서 화면에 출력하고 완료 시 시그널링
void stream_callback(void *callback_data, const char *chunk, bool is_final, const char *error_msg) {
  auto *ctx = static_cast<StreamContext *>(callback_data);

  // 오류 발생 시 처리
  if (error_msg) {
    std::cerr << "\n[스트리밍 오류] " << error_msg << std::endl;
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->has_error = true;
    ctx->error_msg = error_msg;
    ctx->done = true;
    ctx->cv.notify_one();
    return;
  }

  // 텍스트 청크 출력
  if (chunk) {
    std::string text = extract_text_from_chunk(chunk);
    if (!text.empty()) {
      std::cout << text << std::flush;
    }
  }

  // 최종 응답 수신 시 완료 알림
  if (is_final) {
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->done = true;
    ctx->cv.notify_one();
  }
}

// MultimodalCliApp 생성자: 엔진 설정 및 인스턴스 생성
MultimodalCliApp::MultimodalCliApp(const std::string &model_path, const std::string &system_prompt)
    : system_prompt_(system_prompt) {
  // 시스템 프롬프트가 제공되지 않은 경우 파일에서 로드 시도
  if (system_prompt_.empty()) {
    std::ifstream pfile(PROMPT_FILE);
    if (pfile.is_open()) {
      system_prompt_ = std::string((std::istreambuf_iterator<char>(pfile)),
                                   std::istreambuf_iterator<char>());
    }
    // 여전히 비어있으면 기본값 적용
    if (system_prompt_.empty())
      system_prompt_ = DEFAULT_SYSTEM_PROMPT;
  }

  std::cout << "[시스템] 모델 로딩 중..." << std::endl;
  // 엔진 설정 생성 (CPU 사용 설정)
  LiteRtLmEngineSettings *settings = litert_lm_engine_settings_create(
      model_path.c_str(), "cpu", "cpu", nullptr);
  if (!settings)
    throw std::runtime_error("엔진 설정 생성 실패");

  // 엔진 인스턴스 생성
  engine_ = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine_)
    throw std::runtime_error("엔진 생성 실패");

  std::cout << "[시스템] 준비 완료!" << std::endl;
}

// 소멸자: LiteRT-LM 엔진 자원 해제
MultimodalCliApp::~MultimodalCliApp() {
  if (engine_)
    litert_lm_engine_delete(engine_);
}

// CLI 상호작용 루프 실행
void MultimodalCliApp::RunInteractive() {
  // 시스템 프롬프트를 포함한 대화 설정 구성
  std::string sys_json = json({{"role", "system"}, {"content", system_prompt_}}).dump();

  LiteRtLmSessionConfig *session_config = litert_lm_session_config_create();
  if (session_config) {
    // 최대 출력 토큰 설정
    litert_lm_session_config_set_max_output_tokens(session_config, 8192);
  }

  // 대화 세션 컨디그 생성
  LiteRtLmConversationConfig *conv_config =
      litert_lm_conversation_config_create(
          engine_, session_config, sys_json.c_str(), nullptr, nullptr, false);
  if (session_config)
    litert_lm_session_config_delete(session_config);

  // 대화 인스턴스 생성
  LiteRtLmConversation *conversation =
      litert_lm_conversation_create(engine_, conv_config);
  if (conv_config)
    litert_lm_conversation_config_delete(conv_config);

  if (!conversation)
    throw std::runtime_error("대화 세션 생성 실패");

  std::string user_input, image_input;
  while (true) {
    std::cout << "User (입력 없으면 빈칸 후 엔터) > ";
    if (!std::getline(std::cin, user_input) || user_input == "/exit")
      break;

    std::cout << "Image Path (없으면 엔터) > ";
    std::getline(std::cin, image_input);
    image_input = expand_path(trim(image_input));

    if (user_input.empty() && image_input.empty())
      continue;

    // 메시지 구성 (텍스트 전용 또는 멀티모달)
    std::string message_json;
    if (image_input.empty()) {
      message_json = json({{"role", "user"}, {"content", user_input}}).dump();
    } else {
      // 이미지가 포함된 경우 (리스트 형태의 content)
      if (user_input.empty() || user_input == " ") {
        message_json = json({{"role", "user"}, {"content", json::array({{{"type", "image"}, {"path", image_input}}})}}).dump();
      } else {
        message_json = json({{"role", "user"}, {"content", json::array({{{"type", "image"}, {"path", image_input}}, {{"type", "text"}, {"text", user_input}}})}}).dump();
      }
    }

    std::cout << "\n";
    StreamContext ctx;
    // 스트리밍 방식으로 메시지 전송
    int stream_result = litert_lm_conversation_send_message_stream(
        conversation, message_json.c_str(), nullptr, stream_callback, &ctx);

    if (stream_result != 0) {
      std::cerr << "[오류] 스트리밍 시작 실패 (코드: " << stream_result << ")" << std::endl;
    } else {
      // 스트리밍이 완료될 때까지 대기
      std::unique_lock<std::mutex> lock(ctx.mtx);
      ctx.cv.wait(lock, [&ctx] { return ctx.done; });
      if (ctx.has_error) {
        std::cerr << "\n[오류] 스트리밍 중 에러: " << ctx.error_msg << std::endl;
      }
    }
    std::cout << "\n" << std::endl;
  }
  litert_lm_conversation_delete(conversation);
}

// 서버 요청을 위한 비스트리밍(동기) 응답 생성
std::string MultimodalCliApp::GenerateForServer(const std::string &system_msg_str,
                                                const std::string &history_json,
                                                const std::string &current_msg) {
  std::string sys_json = json({{"role", "system"}, {"content", system_msg_str}}).dump();

  // 대화 기록을 포함하여 세션 구성
  LiteRtLmConversationConfig *conv_config =
      litert_lm_conversation_config_create(
          engine_, nullptr, sys_json.c_str(), nullptr,
          history_json.empty() ? nullptr : history_json.c_str(), false);

  LiteRtLmConversation *conversation =
      litert_lm_conversation_create(engine_, conv_config);
  if (conv_config)
    litert_lm_conversation_config_delete(conv_config);
  if (!conversation)
    return "";

  std::string out_text = "";
  // 동기식 메시지 전송
  LiteRtLmJsonResponse *response_obj = litert_lm_conversation_send_message(
      conversation, current_msg.c_str(), nullptr);
  if (response_obj) {
    const char *res_text = litert_lm_json_response_get_string(response_obj);
    if (res_text) {
      try {
        auto res_j = json::parse(res_text);
        // 응답 JSON에서 content 추출
        if (res_j.contains("content")) {
          if (res_j["content"].is_string()) {
            out_text = res_j["content"].get<std::string>();
          } else if (res_j["content"].is_array() && !res_j["content"].empty() && res_j["content"][0].contains("text")) {
            out_text = res_j["content"][0]["text"].get<std::string>();
          }
        } else {
          out_text = res_text;
        }
      } catch (...) {
        out_text = res_text;
      }
    }
    litert_lm_json_response_delete(response_obj);
  }
  litert_lm_conversation_delete(conversation);
  return out_text;
}

// 서버 요청을 위한 스트리밍 응답 생성
void MultimodalCliApp::StreamForServer(const std::string &system_msg_str,
                                      const std::string &history_json,
                                      const std::string &current_msg,
                                      std::function<void(const std::string &chunk)> chunk_cb,
                                      std::function<void()> done_cb,
                                      std::function<void(const std::string &err)> error_cb) {
  std::string sys_json = json({{"role", "system"}, {"content", system_msg_str}}).dump();

  LiteRtLmConversationConfig *conv_config =
      litert_lm_conversation_config_create(
          engine_, nullptr, sys_json.c_str(), nullptr,
          history_json.empty() ? nullptr : history_json.c_str(), false);

  LiteRtLmConversation *conversation =
      litert_lm_conversation_create(engine_, conv_config);
  if (conv_config)
    litert_lm_conversation_config_delete(conv_config);
  if (!conversation) {
    error_cb("대화 세션 생성 실패");
    return;
  }

  // 서버용 스트리밍 관리를 위한 내부 컨텍스트
  struct ServerStreamCtx {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::string> chunks; // 큐를 이용해 순차적 데이터 전달
    bool done = false;
    bool has_error = false;
    std::string error_msg;
  };
  auto ctx = std::make_shared<ServerStreamCtx>();

  // 내부 콜백: 수신된 데이터를 큐에 저장
  auto callback = [](void *data, const char *chunk, bool is_final, const char *error_msg) {
    auto *c = static_cast<ServerStreamCtx *>(data);
    std::lock_guard<std::mutex> lock(c->mtx);
    if (error_msg) {
      c->has_error = true;
      c->error_msg = error_msg;
      c->done = true;
      c->cv.notify_one();
      return;
    }
    if (chunk) {
      std::string text = extract_text_from_chunk(chunk);
      if (!text.empty()) {
        c->chunks.push(std::move(text));
      }
    }
    if (is_final) {
      c->done = true;
    }
    c->cv.notify_one();
  };

  // 비동기 스트리밍 시작
  int result = litert_lm_conversation_send_message_stream(
      conversation, current_msg.c_str(), nullptr, callback, ctx.get());

  if (result != 0) {
    error_cb("스트리밍 시작 실패 (코드: " + std::to_string(result) + ")");
    litert_lm_conversation_delete(conversation);
    return;
  }

  // 큐 소비 루프: 데이터를 수신하는 대로 외부 콜백(chunk_cb) 호출
  while (true) {
    std::unique_lock<std::mutex> lock(ctx->mtx);
    // 데이터 수신 또는 종료 시까지 대기
    ctx->cv.wait(lock, [&ctx] { return !ctx->chunks.empty() || ctx->done; });
    while (!ctx->chunks.empty()) {
      std::string c = std::move(ctx->chunks.front());
      ctx->chunks.pop();
      lock.unlock();
      chunk_cb(c); // 전달받은 텍스트 전달
      lock.lock();
    }
    if (ctx->done) {
      lock.unlock();
      if (ctx->has_error) error_cb(ctx->error_msg);
      else done_cb(); // 완료 통보
      break;
    }
  }
  litert_lm_conversation_delete(conversation);
}
