#pragma once

#include "c/engine.h"
#include "c/conversation.h"
#include "json.hpp"
#include "config.h"
#include <mutex>
#include <string>
#include <functional>

using json = nlohmann::ordered_json;

/**
 * @brief 엔진이 반환한 JSON 청크 데이터에서 텍스트 콘텐츠를 추출합니다.
 * 
 * @param chunk JSON 형식의 데이터 청크
 * @return std::string 추출된 텍스트 내용
 */
std::string extract_text_from_chunk(const char *chunk);

/**
 * @brief LiteRT-LM 멀티모달 기능을 관리하는 고수준 래퍼 클래스입니다.
 */
class MultimodalCliApp {
private:
  LiteRtLmEngine *engine_ = nullptr; // 엔진 인스턴스 포인터
  std::string system_prompt_;        // 활성화된 시스템 프롬프트
  int max_tokens_ = 2048;            // 최대 토큰 수
  std::mutex engine_mutex_;          // 엔진 호출 직렬화 뮤텍스 (RAM 오버 및 레이스 방지)

public:
  // 생성자: 모델 경로와 시스템 프롬프트를 사용하여 엔진 초기화
  MultimodalCliApp(const std::string &model_path, const std::string &system_prompt = "", bool use_gpu = false, int max_tokens = 2048);
  // 소멸자: 엔진 자원 해제
  ~MultimodalCliApp();

  // 서버용 동기 생성 함수 (OpenAI/Ollama API 대응)
  std::string GenerateForServer(const std::string &system_msg_str,
                                const std::string &history_json,
                                const std::string &current_msg);

  // 서버용 스트리밍 생성 함수 (취소 콜백 지원)
  void StreamForServer(const std::string &system_msg_str,
                       const std::string &history_json,
                       const std::string &current_msg,
                       std::function<void(const std::string &chunk)> chunk_cb,
                       std::function<void()> done_cb,
                       std::function<void(const std::string &err)> error_cb,
                       std::function<bool()> is_cancelled = nullptr);
};
