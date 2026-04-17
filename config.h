#pragma once

#include "json.hpp"
#include <string>

using json = nlohmann::ordered_json;

// 설정 파일 및 기본값 상수 정의
extern const std::string PROMPT_FILE;          // 시스템 프롬프트 저장 파일명
extern const std::string CONFIG_FILE;          // LLM 파라미터 저장 파일명
extern const std::string DEFAULT_SYSTEM_PROMPT; // 기본 시스템 프롬프트 내용

// 채팅 관련 파라미터 설정 구조체
struct ChatConfig {
  float temperature = 0.7f; // 온도 (창의성 조절)
  float top_p = 0.95f;      // Top-P 샘플링
  int top_k = 40;           // Top-K 샘플링
  int max_tokens = 2048;    // 최대 생성 토큰 수

  // 설정을 JSON 객체로 변환
  json to_json() const;
  // JSON 객체로부터 설정 로드
  void from_json(const json &j);
};

// 앱 상태 구조체 전방 선언
struct AppState; 

// 설정 로드 및 저장 함수
void load_settings(AppState &state);
void save_settings(AppState &state);
