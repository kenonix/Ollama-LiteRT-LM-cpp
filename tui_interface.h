#pragma once

#include "llm_engine.h"
#include "config.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <vector>
#include <string>
#include <mutex>

using namespace ftxui;

// 메시지 보낸 사람의 역할 구분
enum class Role { User, AI, System };

// 채팅 메시지 구조체
struct ChatMessage {
  Role role;               // 보낸 사람 역할
  std::string content;     // 메시지 내용
  std::string image_path;  // 첨부 이미지 경로 (이미지 있을 경우)
  bool is_streaming = false; // 현재 스트리밍 중 여부
};

// TUI 테마 색상 및 상수 정의
namespace Theme {
extern const Color Bg;         // 배경색
extern const Color UserBubble;  // 사용자 말풍선 배경색
extern const Color AiBubble;    // AI 말풍선 배경색
extern const Color Accent;      // 하이라이트 색상 (핑크)
extern const Color Cyan;        // 강조 색상 (시인성용)
extern const Color Gray;        // 보조 텍스트 색상
extern const Color Text;        // 기본 텍스트 색상
extern const std::vector<std::string> SPINNER_FRAMES; // 로딩 스피너 프레임
}

// TUI 애플리케이션의 전체 상태 관리 구조체
struct AppState {
  std::mutex mtx;                      // 멀티스레드 상태 보호용 뮤텍스
  std::vector<ChatMessage> messages;   // 대화 내역 목록
  bool is_generating = false;          // AI 응답 생성 중 여부
  int spinner_frame = 0;               // 현재 스피너 애니메이션 프레임 번호
  std::string status_text = "준비 완료"; // 하단 상태 바 텍스트
  std::string pending_image;           // 전송 대기 중인 이미지 경로

  LiteRtLmEngine *engine = nullptr;         // 엔진 포인터
  LiteRtLmConversation *conversation = nullptr; // 대화 세션 포인터

  std::string system_prompt = DEFAULT_SYSTEM_PROMPT; // 현재 시스템 프롬프트
  ChatConfig config;                                 // LLM 파라미터 설정
  int selected_tab = 0;                              // 현재 선택된 탭 (0: 채팅, 1: 설정)

  // 설정 UI용 임시 입력 버퍼들
  std::string ui_prompt_buf;
  std::string ui_temp_buf;
  std::string ui_top_p_buf;
  std::string ui_top_k_buf;
  std::string ui_max_tokens_buf;
};

/**
 * @brief 대화 세션을 초기화하거나 기존 세션을 삭제하고 새로 생성합니다.
 */
void tui_init_conversation(AppState &state);

/**
 * @brief 사용자 메시지를 비동기로 전송하고 AI 응답을 스트리밍으로 수신하여 화면에 업데이트합니다.
 */
void send_message_async(AppState &state, const std::string &user_text,
                        const std::string &image_path,
                        ftxui::ScreenInteractive &screen);

/**
 * @brief 단일 채팅 메시지를 FTXUI 렌더링 요소로 변환합니다.
 */
Element render_message(const ChatMessage &msg, int spinner_frame);

/**
 * @brief TUI 환경을 초기화하고 메인 루프를 실행합니다.
 */
void RunTUI(const std::string &model_path);
