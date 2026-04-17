#pragma once

#include <string>

/**
 * @brief 문자열의 앞뒤 공백(스페이스, 탭, 줄바꿈 등)을 제거합니다.
 * 
 * @param s 처리할 원본 문자열
 * @return std::string 공백이 제거된 문자열
 */
std::string trim(const std::string &s);

/**
 * @brief 경로에 포함된 틸드(~) 기호를 사용자의 홈 디렉토리 절대 경로로 확장합니다.
 * 
 * @param path 확장할 경로 문자열
 * @return std::string 확장된 절대 경로
 */
std::string expand_path(const std::string &path);

/**
 * @brief 텍스트를 시스템 클립보드에 복사합니다. (wl-copy 또는 xclip 사용)
 * 
 * @param text 복사할 텍스트
 */
void copy_to_clipboard(const std::string &text);

/**
 * @brief LaTeX 수학 기호를 TUI 표시를 위해 유니코드 기호로 변환합니다.
 * 
 * @param input LaTeX 기호가 포함된 문자열
 * @return std::string 유니코드로 변환된 문자열
 */
std::string latex_to_unicode(const std::string &input);

/**
 * @brief 현재 시간을 마이크로초 정밀도의 ISO8601 형식 문자열로 반환합니다.
 * 
 * @return std::string ISO8601 타임스탬프 (예: 2024-04-17T10:00:00.123456Z)
 */
std::string get_iso8601_now();
