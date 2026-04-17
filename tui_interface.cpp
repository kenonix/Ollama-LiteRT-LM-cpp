#include "tui_interface.h"
#include "utils.h"
#include <fstream>
#include <thread>
#include <chrono>
#include <queue>
#include <atomic>

// 테마 색상 및 애니메이션 상수 정의
namespace Theme {
const Color Bg = Color::RGB(18, 18, 18);
const Color UserBubble = Color::RGB(30, 100, 220);
const Color AiBubble = Color::RGB(40, 42, 54);
const Color Accent = Color::RGB(255, 121, 198);
const Color Cyan = Color::RGB(139, 233, 253);
const Color Gray = Color::RGB(98, 114, 164);
const Color Text = Color::RGB(248, 248, 242);

// 생성 중 표시될 스피너 프레임들
const std::vector<std::string> SPINNER_FRAMES = {
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
};
}

// 설정 파일로부터 데이터 로드 및 UI 버퍼 업데이트
void load_settings(AppState &state) {
  // 시스템 프롬프트 로드
  std::ifstream pfile(PROMPT_FILE);
  if (pfile.is_open()) {
    std::string content((std::istreambuf_iterator<char>(pfile)),
                        std::istreambuf_iterator<char>());
    if (!content.empty()) state.system_prompt = content;
  } else {
    // 파일이 없으면 기본값으로 생성
    std::ofstream opfile(PROMPT_FILE);
    opfile << DEFAULT_SYSTEM_PROMPT;
    state.system_prompt = DEFAULT_SYSTEM_PROMPT;
  }

  // LLM 파라미터(온도 등) 로드
  std::ifstream cfile(CONFIG_FILE);
  if (cfile.is_open()) {
    try {
      json j;
      cfile >> j;
      state.config.from_json(j);
    } catch (...) {}
  } else {
    // 파일이 없으면 현재 설정으로 생성
    std::ofstream ocfile(CONFIG_FILE);
    ocfile << state.config.to_json().dump(2);
  }

  // UI 입력 필드 버퍼 동기화
  state.ui_prompt_buf = state.system_prompt;
  state.ui_temp_buf = std::to_string(state.config.temperature);
  state.ui_top_p_buf = std::to_string(state.config.top_p);
  state.ui_top_k_buf = std::to_string(state.config.top_k);
  state.ui_max_tokens_buf = std::to_string(state.config.max_tokens);
}

// 현재 설정을 파일에 저장
void save_settings(AppState &state) {
  std::ofstream pfile(PROMPT_FILE);
  pfile << state.system_prompt;
  std::ofstream cfile(CONFIG_FILE);
  cfile << state.config.to_json().dump(2);
}

// 대화 세션 초기화
void tui_init_conversation(AppState &state) {
  if (state.conversation) {
    litert_lm_conversation_delete(state.conversation);
  }
  // 시스템 메시지 포함하여 컨피그 생성
  std::string sys_json = json({{"role", "system"}, {"content", state.system_prompt}}).dump();
  LiteRtLmConversationConfig *conv_config = litert_lm_conversation_config_create(
      state.engine, nullptr, sys_json.c_str(), nullptr, nullptr, false);
  state.conversation = litert_lm_conversation_create(state.engine, conv_config);
  if (conv_config) litert_lm_conversation_config_delete(conv_config);
}

// 비동기 메시지 전송 및 스트리밍 처리 인터페이스용 함수
void send_message_async(AppState &state, const std::string &user_text,
                                const std::string &image_path,
                                ScreenInteractive &screen) {
  {
    // 메시지 목록에 사용자 메시지와 비어있는 AI 메시지 추가
    std::lock_guard<std::mutex> lock(state.mtx);
    state.messages.push_back({Role::User, user_text, image_path, false});
    state.messages.push_back({Role::AI, "", "", true});
    state.is_generating = true;
    state.status_text = "생성 중...";
  }
  screen.Post(Event::Custom); // UI 갱신 신호 발송

  // 별도 스레드에서 생성 작업 수행
  std::thread([&state, &screen, user_text, image_path]() {
    // 메시지 데이터 구성
    std::string message_json;
    if (image_path.empty()) {
      message_json = json({{"role", "user"}, {"content", user_text}}).dump();
    } else if (user_text.empty() || user_text == " ") {
      message_json = json({{"role", "user"}, {"content", json::array({{{"type", "image"}, {"path", image_path}}})}}).dump();
    } else {
      message_json = json({{"role", "user"}, {"content", json::array({{{"type", "image"}, {"path", image_path}}, {{"type", "text"}, {"text", user_text}}})}}).dump();
    }

    // 스트리밍 데이터 수신을 위한 내부 상태
    struct StreamCtx {
      std::mutex mtx;
      std::condition_variable cv;
      std::queue<std::string> chunks;
      bool done = false;
      bool has_error = false;
      std::string error_msg;
    };
    auto ctx = std::make_shared<StreamCtx>();

    // 스트리밍 콜백 람다
    auto callback = [](void *data, const char *chunk, bool is_final, const char *error_msg) {
      auto *c = static_cast<StreamCtx *>(data);
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
        if (!text.empty()) c->chunks.push(std::move(text));
      }
      if (is_final) c->done = true;
      c->cv.notify_one();
    };

    // 현재 설정 로드
    std::string config_json;
    {
      std::lock_guard<std::mutex> lock(state.mtx);
      config_json = state.config.to_json().dump();
    }

    // 엔진에 메시지 전송 (스트리밍 방식)
    int result = litert_lm_conversation_send_message_stream(
        state.conversation, message_json.c_str(), config_json.c_str(), callback, ctx.get());

    if (result != 0) {
      // 시작 실패 시 처리
      std::lock_guard<std::mutex> lock(state.mtx);
      if (!state.messages.empty()) {
        state.messages.back().content = "[오류] 스트리밍 실패 (코드: " + std::to_string(result) + ")";
        state.messages.back().is_streaming = false;
      }
      state.is_generating = false;
      state.status_text = "오류 발생";
      screen.Post(Event::Custom);
      return;
    }

    // 데이터 수신 대기 및 UI 업데이트 루프
    while (true) {
      std::unique_lock<std::mutex> lock(ctx->mtx);
      ctx->cv.wait(lock, [&ctx] { return !ctx->chunks.empty() || ctx->done; });
      
      std::vector<std::string> pending;
      while (!ctx->chunks.empty()) {
        pending.push_back(std::move(ctx->chunks.front()));
        ctx->chunks.pop();
      }
      
      bool is_done = ctx->done;
      bool has_error = ctx->has_error;
      std::string err_msg = ctx->error_msg;
      lock.unlock();

      // 수신된 텍스트 청크를 메시지에 추가
      if (!pending.empty()) {
        std::lock_guard<std::mutex> slock(state.mtx);
        if (!state.messages.empty()) {
          for (auto &c : pending) state.messages.back().content += c;
        }
        screen.Post(Event::Custom);
      }
      
      // 종료 조건 확인
      if (is_done) {
        std::lock_guard<std::mutex> slock(state.mtx);
        if (!state.messages.empty()) state.messages.back().is_streaming = false;
        state.status_text = has_error ? "오류: " + err_msg : "준비 완료";
        state.is_generating = false;
        screen.Post(Event::Custom);
        break;
      }
    }
  }).detach();
}

// 개별 메시지 렌더링 함수
Element render_message(const ChatMessage &msg, int spinner_frame) {
  if (msg.role == Role::User) {
    // 사용자 메시지: 오른쪽 정렬, 파란색 말풍선
    Elements user_parts;
    if (!msg.image_path.empty()) {
      user_parts.push_back(hbox({text(" 🖼 이미지: ") | color(Theme::Cyan) | dim, text(msg.image_path) | color(Theme::Cyan) | bold}) | borderEmpty | borderStyled(ROUNDED, Theme::Cyan));
    }
    user_parts.push_back(paragraph(msg.content) | borderEmpty);
    return hbox({filler(), vbox({hbox({filler(), text("USER") | bold | color(Theme::Cyan) | borderEmpty}), vbox(user_parts) | bgcolor(Theme::UserBubble) | color(Theme::Text) | borderStyled(ROUNDED, Theme::UserBubble)}) | size(WIDTH, LESS_THAN, 80)});
  } else if (msg.role == Role::AI) {
    // AI 메시지: 왼쪽 정렬, 어두운 회색 말풍선
    Elements content_parts;
    if (msg.content.empty() && msg.is_streaming) {
      // 텍스트 수신 전 로딩 스피너 표시
      content_parts.push_back(hbox({text(Theme::SPINNER_FRAMES[spinner_frame % Theme::SPINNER_FRAMES.size()]) | color(Theme::Accent) | bold, text(" 생각 중...") | color(Theme::Gray)}) | borderEmpty);
    } else {
      // LaTeX 수식 변환 후 텍스트 표시
      content_parts.push_back(paragraph(latex_to_unicode(msg.content)) | color(Theme::Text) | borderEmpty);
      if (msg.is_streaming) content_parts.push_back(text(" ▌") | color(Theme::Accent) | blink); // 커서 점멸 효과
    }
    return hbox({vbox({text("AI") | bold | color(Theme::Accent) | borderEmpty, vbox(content_parts) | bgcolor(Theme::AiBubble) | borderStyled(ROUNDED, Theme::AiBubble)}) | size(WIDTH, LESS_THAN, 80), filler()});
  }
  // 시스템 메시지: 중앙 정렬
  return hbox({filler(), text("── " + msg.content + " ──") | color(Theme::Gray) | center, filler()}) | borderEmpty;
}

// TUI 메인 실행 함수
void RunTUI(const std::string &model_path) {
  AppState state;
  // 엔진 설정 및 생성
  LiteRtLmEngineSettings *settings = litert_lm_engine_settings_create(model_path.c_str(), "cpu", "cpu", nullptr);
  if (!settings) return;
  state.engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!state.engine) return;

  // 설정 로드 및 세션 초기화
  load_settings(state);
  tui_init_conversation(state);
  if (!state.conversation) return;

  {
    std::lock_guard<std::mutex> lock(state.mtx);
    state.messages.push_back({Role::System, "💬 AI Chat — F1:대화 / F2:설정 · /clear 초기화", "", false});
  }

  auto screen = ScreenInteractive::Fullscreen();
  std::string input_text;
  auto input_option = InputOption();
  input_option.placeholder = "메시지를 입력하세요... (/img <경로>로 이미지 첨부)";
  
  // 엔터 입력 시 처리
  input_option.on_enter = [&] {
    if (input_text.empty()) {
      std::lock_guard<std::mutex> lock(state.mtx);
      if (state.pending_image.empty()) return;
    }

    // 명령어 처리 (/clear)
    if (input_text == "/clear") {
      std::lock_guard<std::mutex> lock(state.mtx);
      state.messages.clear();
      state.messages.push_back({Role::System, "💬 대화가 초기화되었습니다.", "", false});
      state.pending_image.clear();
      tui_init_conversation(state);
      input_text.clear();
      return;
    }
    
    // 이미지 첨부 명령어 처리 (/img)
    if (input_text.rfind("/img ", 0) == 0) {
      std::lock_guard<std::mutex> lock(state.mtx);
      state.pending_image = expand_path(trim(input_text.substr(5)));
      state.status_text = "🖼 이미지 첨부됨: " + state.pending_image;
      input_text.clear();
      return;
    }
    
    // 데이터 전송 준비
    std::string msg = input_text;
    std::string img;
    {
      std::lock_guard<std::mutex> lock(state.mtx);
      if (state.is_generating) return;
      img = state.pending_image;
      state.pending_image.clear();
    }
    input_text.clear();
    send_message_async(state, msg, img, screen);
  };

  auto input_component = Input(&input_text, input_option);
  
  // 스피너 애니메이션을 위한 별도 스레드
  std::atomic<bool> running{true};
  std::thread spinner_thread([&] {
    while (running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(80));
      {
        std::lock_guard<std::mutex> lock(state.mtx);
        if (state.is_generating) state.spinner_frame++;
      }
      screen.Post(Event::Custom);
    }
  });

  // 채팅 화면 렌더러
  auto chat_renderer = Renderer(input_component, [&]() -> Element {
    std::lock_guard<std::mutex> lock(state.mtx);
    auto title_bar = hbox({text("  ✨ ") | bold | color(Theme::Accent), text("Multimodal AI - CHAT") | bold | color(Theme::Text), filler(), text(state.status_text + "  ") | color(Theme::Gray)}) | bgcolor(Theme::AiBubble);
    
    Elements msg_elements;
    for (auto &msg : state.messages) {
      msg_elements.push_back(render_message(msg, state.spinner_frame));
      msg_elements.push_back(separatorEmpty());
    }
    
    auto chat_area = vbox(msg_elements) | focusPositionRelative(0, 1) | yframe | flex;
    auto input_area = hbox({text(" ❯ ") | color(Theme::Accent) | bold, input_component->Render() | flex | color(Theme::Text)}) | borderStyled(ROUNDED, state.is_generating ? Theme::Accent : Theme::Gray);
    
    return vbox({title_bar, chat_area | borderEmpty, input_area | borderEmpty, hbox({text(" F1:채팅  F2:설정  Ctrl+C:복사  ESC:종료") | dim}) | borderEmpty}) | bgcolor(Theme::Bg);
  });

  // 설정 화면 구성 요소
  auto prompt_input = Input(&state.ui_prompt_buf, "System Prompt...");
  auto temp_input = Input(&state.ui_temp_buf, "0.7");
  auto top_p_input = Input(&state.ui_top_p_buf, "0.95");
  auto top_k_input = Input(&state.ui_top_k_buf, "40");
  auto max_tokens_input = Input(&state.ui_max_tokens_buf, "2048");
  
  // 설정 저장 버튼
  auto save_button = Button("  💾 SAVE & APPLY  ", [&] {
    std::lock_guard<std::mutex> lock(state.mtx);
    state.system_prompt = state.ui_prompt_buf;
    try {
      state.config.temperature = std::stof(state.ui_temp_buf);
      state.config.top_p = std::stof(state.ui_top_p_buf);
      state.config.top_k = std::stoi(state.ui_top_k_buf);
      state.config.max_tokens = std::stoi(state.ui_max_tokens_buf);
    } catch (...) {}
    save_settings(state);
    tui_init_conversation(state);
    state.status_text = "설정 저장됨";
    state.selected_tab = 0; // 저장 후 채팅 탭으로 이동
  });

  // 설정 레이아웃 구성
  auto settings_container = Container::Vertical({prompt_input, temp_input, top_p_input, top_k_input, max_tokens_input, save_button});
  auto settings_renderer = Renderer(settings_container, [&]() -> Element {
    return vbox({hbox({text("  ⚙️  SETTINGS") | bold}) | bgcolor(Theme::AiBubble), vbox({text(" SYSTEM PROMPT") | bold | color(Theme::Cyan), prompt_input->Render() | borderStyled(ROUNDED, Theme::Gray), separatorEmpty(), hbox({vbox({text(" TEMP"), temp_input->Render() | border}) | flex, vbox({text(" TOP-P"), top_p_input->Render() | border}) | flex}), hbox({vbox({text(" TOP-K"), top_k_input->Render() | border}) | flex, vbox({text(" MAX_TOK"), max_tokens_input->Render() | border}) | flex}), separatorEmpty(), save_button->Render() | center | color(Theme::Accent)}) | borderEmpty | flex}) | bgcolor(Theme::Bg);
  });

  // 탭 구성 및 이벤트 처리
  auto main_container = Container::Tab({chat_renderer, settings_renderer}, &state.selected_tab);
  auto component = CatchEvent(main_container, [&](Event event) {
    if (event == Event::Escape) { screen.ExitLoopClosure()(); return true; }
    if (event == Event::F1) { state.selected_tab = 0; return true; }
    if (event == Event::F2) { state.selected_tab = 1; return true; }
    // Ctrl+C 입력 시 마지막 AI 답변 복사
    if (event == Event::Special({3})) { 
      std::lock_guard<std::mutex> lock(state.mtx);
      std::string last_ai_content;
      for (auto it = state.messages.rbegin(); it != state.messages.rend(); ++it) {
        if (it->role == Role::AI && !it->content.empty()) {
          last_ai_content = it->content;
          break;
        }
      }
      if (!last_ai_content.empty()) {
        copy_to_clipboard(last_ai_content);
        state.status_text = "📋 AI 답변 복사됨";
      }
      return true;
    }
    return false;
  });

  // 로그 레벨 조절 및 루프 시작
  litert_lm_set_min_log_level(1); // WARNING 레벨 이상만 표시
  screen.Loop(component);
  
  // 종료 처리
  running = false;
  if (spinner_thread.joinable()) spinner_thread.join();
  if (state.conversation) litert_lm_conversation_delete(state.conversation);
  if (state.engine) litert_lm_engine_delete(state.engine);
}
