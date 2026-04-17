#include "config.h"
#include <fstream>

// 파일명 및 기본 설정값 초기화
const std::string PROMPT_FILE = "system_prompt.txt";
const std::string CONFIG_FILE = "config.json";

// 기본 시스템 프롬프트 (한국어 설정)
const std::string DEFAULT_SYSTEM_PROMPT = R"(당신의 이름은 AI입니다.
한국어만 사용하며, 친절하고 명확하게 답변합니다.)";

// 설정을 LiteRT-LM 엔진 호환 JSON 형식으로 변환
json ChatConfig::to_json() const {
  return {{"temperature", temperature},
          {"top_p", top_p},
          {"top_k", top_k},
          {"max_output_tokens", max_tokens}}; // 엔진 필드명은 max_output_tokens
}

// JSON 객체로부터 필드가 존재할 경우 값을 업데이트
void ChatConfig::from_json(const json &j) {
  if (j.contains("temperature"))
    temperature = j["temperature"];
  if (j.contains("top_p"))
    top_p = j["top_p"];
  if (j.contains("top_k"))
    top_k = j["top_k"];
  if (j.contains("max_tokens"))
    max_tokens = j["max_tokens"];
}
