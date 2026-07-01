#pragma once

#include "c/engine.h"
#include "json.hpp"
#include "config.h"
#include <mutex>
#include <condition_variable>
#include <string>
#include <functional>

using json = nlohmann::ordered_json;

// 스트리밍 콜백에서 상태와 데이터 동기화를 위한 컨텍스트 구조체
struct StreamContext {
  std::mutex mtx;             // 공유 데이터 보호용 뮤텍스
  std::condition_variable cv; // 데이터 수신 대기/알림용 조건 변수
  bool done = false;          // 스트리밍 완료 여부
  bool has_error = false;     // 오류 발생 여부
  std::string error_msg;      // 오류 메시지 내용
};

/**
 * @brief 엔진이 반환한 JSON 청크 데이터에서 텍스트 콘텐츠를 추출합니다.
 * 
 * @param chunk JSON 형식의 데이터 청크
 * @return std::string 추출된 텍스트 내용
 */
std::string extract_text_from_chunk(const char *chunk);

/**
 * @brief LiteRT-LM 엔진용 표준 스트리밍 콜백 함수입니다.
 */
void stream_callback(void *callback_data, const char *chunk, bool is_final, const char *error_msg);

/**
 * @brief LiteRT-LM 멀티모달 기능을 관리하는 고수준 래퍼 클래스입니다.
 */
class MultimodalCliApp {
private:
  LiteRtLmEngine *engine_ = nullptr; // 엔진 인스턴스 포인터
  std::string system_prompt_;        // 활성화된 시스템 프롬프트

public:
  // 생성자: 모델 경로와 시스템 프롬프트를 사용하여 엔진 초기화
  MultimodalCliApp(const std::string &model_path, const std::string &system_prompt = "", bool use_gpu = false);
  // 소멸자: 엔진 자원 해제
  ~MultimodalCliApp();

  // 인터랙티브 CLI 모드 실행
  void RunInteractive();

  // 서버용 동기 생성 함수 (OpenAI/Ollama API 대응)
  std::string GenerateForServer(const std::string &system_msg_str,
                                const std::string &history_json,
                                const std::string &current_msg);

  // 서버용 스트리밍 생성 함수
  void StreamForServer(const std::string &system_msg_str,
                       const std::string &history_json,
                       const std::string &current_msg,
                       std::function<void(const std::string &chunk)> chunk_cb,
                       std::function<void()> done_cb,
                       std::function<void(const std::string &err)> error_cb);
};
