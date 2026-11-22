#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <random>
#include <fstream>
#include <algorithm>
#include <cctype>

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

// 무작위 식별자(ID) 문자열 생성 구현
std::string generate_random_id(const std::string &prefix) {
  static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  static std::mt19937_64 rng(std::random_device{}());
  static std::uniform_int_distribution<size_t> dist(0, sizeof(charset) - 2);
  std::string id = prefix;
  for (int i = 0; i < 28; ++i) {
    id += charset[dist(rng)];
  }
  return id;
}

// 쉘을 거치지 않고 외부 프로세스를 안전하게 실행 (명령어 인젝션 방지)
static int run_process_no_shell(const std::vector<std::string> &args) {
  if (args.empty()) return -1;
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    std::vector<char *> c_args;
    c_args.reserve(args.size() + 1);
    for (const auto &a : args) {
      c_args.push_back(const_cast<char *>(a.c_str()));
    }
    c_args.push_back(nullptr);
    execvp(c_args[0], c_args.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return -1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

// 이미지 소스 판별 및 파일 경로 획득 구현
std::string save_image_source_to_temp_file(const std::string &source, bool &is_temp) {
  is_temp = false;
  if (source.empty()) return "";

  // 1. 이미 존재하는 로컬 파일인지 확인
  std::string expanded = expand_path(source);
  if (access(expanded.c_str(), R_OK) == 0) {
    is_temp = false;
    return expanded;
  }

  // 2. HTTP 또는 HTTPS URL인 경우 curl로 다운로드
  if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) {
    char tmp_path[] = "/tmp/litert_url_XXXXXX.png";
    int fd = mkstemps(tmp_path, 4);
    if (fd == -1) return "";
    close(fd);

    int ret = run_process_no_shell({"curl", "-s", "-L", "--max-time", "15", source, "-o", tmp_path});
    if (ret == 0 && access(tmp_path, R_OK) == 0) {
      is_temp = true;
      return std::string(tmp_path);
    } else {
      std::remove(tmp_path);
      return "";
    }
  }

  // 3. Base64 형식인 경우 디코딩하여 임시 파일로 저장
  std::string b64_path = save_base64_to_temp_file(source);
  if (!b64_path.empty()) {
    is_temp = true;
    return b64_path;
  }

  return "";
}

// 바이트 헤더(매직 바이트)로 miniaudio 기본 지원 포맷(WAV/MP3/FLAC)인지 판별
static std::string detect_native_audio_ext(const char *data, size_t len) {
  if (!data || len < 4) return "";
  if (len >= 12 &&
      data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
      data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
    return ".wav";
  }
  if (data[0] == 'f' && data[1] == 'L' && data[2] == 'a' && data[3] == 'C') {
    return ".flac";
  }
  if (data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
    return ".mp3";
  }
  unsigned char b0 = static_cast<unsigned char>(data[0]);
  unsigned char b1 = static_cast<unsigned char>(data[1]);
  if (b0 == 0xFF && (b1 & 0xE0) == 0xE0) {
    return ".mp3";
  }
  return "";
}

// 미지원 오디오 파일(webm/ogg/m4a 등)을 ffmpeg로 16kHz 모노 WAV로 변환
static std::string convert_audio_to_wav_temp(const std::string &input_path) {
  char out_path[] = "/tmp/litert_audio_conv_XXXXXX.wav";
  int fd = mkstemps(out_path, 4);
  if (fd == -1) return "";
  close(fd);

  int ret = run_process_no_shell({
      "ffmpeg", "-y", "-loglevel", "error",
      "-i", input_path,
      "-ac", "1", "-ar", "16000", "-f", "wav",
      out_path
  });
  if (ret == 0 && access(out_path, R_OK) == 0) {
    return std::string(out_path);
  }
  std::remove(out_path);
  return "";
}

static std::string write_bytes_to_temp_audio(const std::string &bytes, const std::string &ext) {
  if (bytes.empty()) return "";
  std::string tmpl = "/tmp/litert_audio_XXXXXX" + ext;
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  int fd = mkstemps(buf.data(), static_cast<int>(ext.size()));
  if (fd == -1) return "";
  FILE *f = fdopen(fd, "wb");
  if (!f) {
    close(fd);
    std::remove(buf.data());
    return "";
  }
  fwrite(bytes.data(), 1, bytes.size(), f);
  fclose(f);
  return std::string(buf.data());
}

// 오디오 소스 판별 및 LiteRT-LM 호환 로컬 파일 경로 획득 구현
std::string save_audio_source_to_temp_file(const std::string &source, const std::string &format_hint, bool &is_temp) {
  is_temp = false;
  if (source.empty()) return "";

  auto normalize_ext = [](std::string fmt) -> std::string {
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!fmt.empty() && fmt[0] == '.') fmt.erase(0, 1);
    if (fmt == "wav" || fmt == "wave" || fmt == "x-wav") return ".wav";
    if (fmt == "mp3" || fmt == "mpeg") return ".mp3";
    if (fmt == "flac") return ".flac";
    if (fmt == "webm" || fmt == "weba") return ".webm";
    if (fmt == "ogg" || fmt == "opus") return ".ogg";
    if (fmt == "m4a" || fmt == "mp4" || fmt == "aac") return ".m4a";
    return ".bin";
  };

  auto ensure_miniaudio_compatible = [&](const std::string &file_path, bool owns_file) -> std::string {
    std::ifstream ifs(file_path, std::ios::binary);
    char header[16] = {0};
    ifs.read(header, sizeof(header));
    size_t n = static_cast<size_t>(ifs.gcount());
    ifs.close();
    if (!detect_native_audio_ext(header, n).empty()) {
      is_temp = owns_file;
      return file_path;
    }
    std::string converted = convert_audio_to_wav_temp(file_path);
    if (!converted.empty()) {
      if (owns_file) std::remove(file_path.c_str());
      is_temp = true;
      return converted;
    }
    // ffmpeg가 없거나 변환 실패 시 원본 파일 반환 (miniaudio에 위임)
    is_temp = owns_file;
    return file_path;
  };

  // 1. 이미 존재하는 로컬 파일인지 확인 (data: URI나 긴 base64 문자열은 제외)
  if (source.size() < 4096 && source.rfind("data:", 0) != 0 &&
      source.rfind("http://", 0) != 0 && source.rfind("https://", 0) != 0) {
    std::string expanded = expand_path(source);
    if (access(expanded.c_str(), R_OK) == 0) {
      return ensure_miniaudio_compatible(expanded, false);
    }
  }

  // 2. HTTP 또는 HTTPS URL인 경우 curl로 다운로드
  if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) {
    std::string ext = format_hint.empty() ? ".bin" : normalize_ext(format_hint);
    std::string tmpl = "/tmp/litert_audio_url_XXXXXX" + ext;
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkstemps(buf.data(), static_cast<int>(ext.size()));
    if (fd == -1) return "";
    close(fd);
    std::string tmp_path(buf.data());

    int ret = run_process_no_shell({"curl", "-s", "-L", "--max-time", "15", source, "-o", tmp_path});
    if (ret == 0 && access(tmp_path.c_str(), R_OK) == 0) {
      return ensure_miniaudio_compatible(tmp_path, true);
    }
    std::remove(tmp_path.c_str());
    return "";
  }

  // 3. Base64 / Data URI 디코딩
  std::string decoded = base64_decode(source);
  if (decoded.empty()) return "";

  std::string native_ext = detect_native_audio_ext(decoded.data(), decoded.size());
  if (!native_ext.empty()) {
    std::string path = write_bytes_to_temp_audio(decoded, native_ext);
    if (!path.empty()) {
      is_temp = true;
      return path;
    }
    return "";
  }

  // miniaudio 미지원 포맷(webm/ogg/m4a 등)인 경우 임시 저장 후 ffmpeg로 WAV 변환 시도
  std::string hint_ext = format_hint.empty() ? ".bin" : normalize_ext(format_hint);
  std::string raw_path = write_bytes_to_temp_audio(decoded, hint_ext);
  if (raw_path.empty()) return "";
  return ensure_miniaudio_compatible(raw_path, true);
}
