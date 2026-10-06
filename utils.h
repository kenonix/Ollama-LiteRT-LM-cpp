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
 * @brief 현재 시간을 마이크로초 정밀도의 ISO8601 형식 문자열로 반환합니다.
 * 
 * @return std::string ISO8601 타임스탬프 (예: 2024-04-17T10:00:00.123456Z)
 */
std::string get_iso8601_now();

/**
 * @brief Base64 인코딩된 문자열을 바이너리 데이터(std::string)로 디코딩합니다.
 * 
 * @param in Base64 문자열
 * @return std::string 디코딩된 바이트 문자열
 */
std::string base64_decode(const std::string &in);

/**
 * @brief Base64 인코딩된 이미지 데이터를 임시 파일로 저장하고 경로를 반환합니다.
 * 
 * @param b64_data Base64 문자열
 * @return std::string 생성된 임시 파일 절대 경로 (실패 시 빈 문자열)
 */
std::string save_base64_to_temp_file(const std::string &b64_data);

/**
 * @brief 무작위 식별자(ID) 문자열을 생성합니다 (예: chatcmpl-xyz, cmpl-xyz).
 * 
 * @param prefix 접두사 (예: "chatcmpl-")
 * @return std::string 접두사와 무작위 영숫자가 결합된 ID
 */
std::string generate_random_id(const std::string &prefix);

/**
 * @brief 이미지 소스(URL, Base64, 파일 경로)를 판별하여 로컬 파일 경로로 변환합니다.
 * 
 * @param source URL (http/https), Base64, 또는 로컬 파일 경로
 * @param is_temp [out] 생성된 파일이 임시 파일인지 여부 (정리 대상 여부)
 * @return std::string 로컬 파일 절대 경로 (실패 시 빈 문자열)
 */
std::string save_image_source_to_temp_file(const std::string &source, bool &is_temp);
