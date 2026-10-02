#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <vector>
#include <unistd.h>

// 문자열의 앞뒤 공백 제거 구현
std::string trim(const std::string &s) {
  size_t first = s.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return ""; // 공백만 있는 경우 빈 문자열 반환
  size_t last = s.find_last_not_of(" \t\n\r");
  return s.substr(first, (last - first + 1));
}

// 틸드(~) 경로를 홈 디렉토리로 확장 구현
std::string expand_path(const std::string &path) {
  if (path.empty() || path[0] != '~') return path;
  const char *home = std::getenv("HOME");
  if (!home) return path; // HOME 환경변수가 없는 경우 원본 반환
  return std::string(home) + path.substr(1);
}

// 현재 시간 ISO8601 형식 문자열 생성 구현
std::string get_iso8601_now() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;
  char buf[64];
  // 초 단위까지 변환
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::gmtime(&time_t_now));
  char result[80];
  // 마이크로초와 Z(UTC) 문자 추가
  std::snprintf(result, sizeof(result), "%s.%06ldZ", buf, (long)us.count());
  return std::string(result);
}

// Base64 디코딩 구현
std::string base64_decode(const std::string &in) {
  std::string clean_in = in;
  // data:image/...;base64, 접두사 제거
  size_t comma_pos = clean_in.find(',');
  if (comma_pos != std::string::npos && clean_in.substr(0, comma_pos).find("base64") != std::string::npos) {
    clean_in = clean_in.substr(comma_pos + 1);
  }

  std::vector<int> T(256, -1);
  const std::string b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (int i = 0; i < 64; i++) T[(unsigned char)b64[i]] = i;

  std::string out;
  int val = 0, valb = -8;
  for (unsigned char c : clean_in) {
    if (c == '=') break;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (T[c] == -1) continue;
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

// Base64 이미지 데이터를 임시 파일로 저장
std::string save_base64_to_temp_file(const std::string &b64_data) {
  std::string decoded = base64_decode(b64_data);
  if (decoded.empty()) return "";

  // 임시 파일 생성
  char tmp_path[] = "/tmp/litert_img_XXXXXX.png";
  int fd = mkstemps(tmp_path, 4);
  if (fd == -1) {
    char tmp_path2[] = "/tmp/litert_img_XXXXXX";
    fd = mkstemp(tmp_path2);
    if (fd == -1) return "";
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); return ""; }
    fwrite(decoded.data(), 1, decoded.size(), f);
    fclose(f);
    return std::string(tmp_path2);
  }
  FILE *f = fdopen(fd, "wb");
  if (!f) { close(fd); return ""; }
  fwrite(decoded.data(), 1, decoded.size(), f);
  fclose(f);
  return std::string(tmp_path);
}
