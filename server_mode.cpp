#include "server_mode.h"
#include "httplib.h"
#include "json.hpp"
#include "utils.h"
#include "config.h"
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

// 멀티모달 임시 이미지 파일 자동 정리 RAII 구조체
struct TempFilesCleanup {
  std::vector<std::string> paths;
  ~TempFilesCleanup() {
    for (const auto &p : paths) {
      if (!p.empty()) {
        std::remove(p.c_str());
      }
    }
  }
};

// 스트리밍 도중 UTF-8 멀티바이트 문자가 바이트 단위로 쪼개지는 것을 방지하는 리어셈블러
class Utf8StreamReassembler {
private:
  std::string buffer_;

public:
  // 입력 문자열을 받아 온전한 UTF-8 문자들만 반환하고, 덜 온 바이트는 버퍼에 보관
  std::string process(const std::string &incoming) {
    buffer_ += incoming;
    if (buffer_.empty()) return "";

    size_t valid_len = buffer_.size();
    size_t i = buffer_.size();

    while (i > 0 && i >= (buffer_.size() > 4 ? buffer_.size() - 4 : 0)) {
      --i;
      unsigned char c = static_cast<unsigned char>(buffer_[i]);
      if ((c & 0x80) == 0) {
        // ASCII 문자 (완전)
        break;
      } else if ((c & 0xC0) == 0xC0) {
        // 멀티바이트 선두 바이트 (0b11xxxxxx)
        size_t expected_len = 0;
        if ((c & 0xE0) == 0xC0) expected_len = 2;
        else if ((c & 0xF0) == 0xE0) expected_len = 3;
        else if ((c & 0xF8) == 0xF0) expected_len = 4;

        size_t actual_len = buffer_.size() - i;
        if (actual_len < expected_len) {
          valid_len = i;
        }
        break;
      }
    }

    if (valid_len == 0) return "";
    std::string ready = buffer_.substr(0, valid_len);
    buffer_.erase(0, valid_len);
    return ready;
  }

  // 스트림 완료 시 남은 버퍼 반환
  std::string flush() {
    std::string remaining = std::move(buffer_);
    buffer_.clear();
    return remaining;
  }
};

// 파싱된 도구 호출 구조체
struct ParsedToolCall {
  size_t index = 0;
  std::string id;
  std::string name;
  json arguments;
};

// 긴 도구 결과나 텍스트 메시지를 안전한 크기로 요약/절삭
static void truncate_long_message(json &msg, int max_chars = 4000) {
  if (!msg.is_object() || !msg.contains("content")) return;
  if (msg["content"].is_string()) {
    std::string text = msg["content"].get<std::string>();
    if (text.size() > static_cast<size_t>(max_chars)) {
      size_t head_len = max_chars * 3 / 5;
      size_t tail_len = max_chars * 2 / 5;
      std::string truncated = text.substr(0, head_len) + 
                              "\n\n[... 중간 내용 " + std::to_string(text.size() - head_len - tail_len) + "자 생략됨 ...]\n\n" +
                              text.substr(text.size() - tail_len);
      msg["content"] = truncated;
    }
  }
}

// 도구 스키마(tools JSON 배열)를 Gemma 모델이 직관적으로 이해할 수 있는 압축 시스템 지침으로 변환
static std::string format_tools_system_prompt(const json &tools) {
  if (!tools.is_array() || tools.empty()) return "";

  std::string prompt = "\n\n# Available Tools\n"
                       "You have access to the following tools to interact with the environment, files, and system:\n<tools>\n";
  for (const auto &tool : tools) {
    if (!tool.is_object()) continue;
    json fn;
    if (tool.contains("function") && tool["function"].is_object()) {
      fn = tool["function"];
    } else {
      fn = tool;
    }

    // 도구 스키마 메타데이터를 핵심 필드(이름, 설명, 파라미터 타입)로 초경량 압축하여 수천 토큰 절약
    json compact_fn = json::object();
    compact_fn["name"] = fn.value("name", "");
    if (fn.contains("description") && fn["description"].is_string()) {
      std::string desc = fn["description"].get<std::string>();
      if (desc.size() > 140) desc = desc.substr(0, 137) + "...";
      compact_fn["description"] = desc;
    }
    if (fn.contains("parameters") && fn["parameters"].is_object()) {
      const auto &params = fn["parameters"];
      json compact_params = json::object();
      if (params.contains("properties") && params["properties"].is_object()) {
        for (auto it = params["properties"].begin(); it != params["properties"].end(); ++it) {
          std::string p_type = it.value().value("type", "string");
          std::string p_desc = it.value().value("description", "");
          if (p_desc.size() > 50) p_desc = p_desc.substr(0, 47) + "...";
          if (!p_desc.empty()) {
            compact_params[it.key()] = p_type + " (" + p_desc + ")";
          } else {
            compact_params[it.key()] = p_type;
          }
        }
      }
      compact_fn["parameters"] = compact_params;
      if (params.contains("required") && params["required"].is_array()) {
        compact_fn["required"] = params["required"];
      }
    }
    prompt += compact_fn.dump() + "\n";
  }
  prompt += "</tools>\n\n"
            "# Tool Calling Rules\n"
            "1. When the user asks you to read, write, edit, search files, execute commands, or manage tasks, you MUST execute the appropriate tool instead of just describing it or outputting code blocks in chat.\n"
            "2. To invoke a tool, output the tool call enclosed in <tool_call> tags using valid JSON formatted like this:\n"
            "<tool_call>\n"
            "{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"val1\"}}\n"
            "</tool_call>\n"
            "3. You may provide a brief explanation before or after the tool call, but the tool invocation itself MUST be wrapped in <tool_call> and </tool_call> tags.\n"
            "4. NEVER output raw markdown code blocks as a substitute for calling write or edit tools when file modification is requested.\n\n";
  return prompt;
}

// 스트리밍 및 논스트리밍 토큰에서 <tool_call> 태그를 실시간 추출/분리하는 파서
class ToolCallStreamParser {
private:
  std::string buffer_;
  std::vector<ParsedToolCall> tool_calls_;
  size_t call_index_ = 0;

  static bool extract_tag(const std::string &buf,
                          size_t &open_pos, size_t &open_len,
                          size_t &close_pos, size_t &close_len) {
    // 1) <tool_call> ... </tool_call>
    open_pos = buf.find("<tool_call>");
    if (open_pos != std::string::npos) {
      open_len = 11;
      close_pos = buf.find("</tool_call>", open_pos + open_len);
      if (close_pos != std::string::npos) {
        close_len = 12;
        return true;
      }
      return false;
    }

    // 2) ```tool_call ... ```
    open_pos = buf.find("```tool_call");
    if (open_pos != std::string::npos) {
      open_len = 12;
      close_pos = buf.find("```", open_pos + open_len);
      if (close_pos != std::string::npos) {
        close_len = 3;
        return true;
      }
      return false;
    }

    return false;
  }

public:
  struct ProcessResult {
    std::string text_to_emit;
    std::vector<ParsedToolCall> completed_calls;
  };

  ProcessResult process(const std::string &chunk) {
    buffer_ += chunk;
    ProcessResult res;

    while (!buffer_.empty()) {
      size_t open_pos = std::string::npos, open_len = 0;
      size_t close_pos = std::string::npos, close_len = 0;

      if (extract_tag(buffer_, open_pos, open_len, close_pos, close_len)) {
        if (open_pos > 0) {
          res.text_to_emit += buffer_.substr(0, open_pos);
        }

        std::string json_str = buffer_.substr(open_pos + open_len, close_pos - (open_pos + open_len));
        buffer_.erase(0, close_pos + close_len);

        parse_and_add_tool_call(json_str, res);
      } else if (open_pos != std::string::npos) {
        if (open_pos > 0) {
          res.text_to_emit += buffer_.substr(0, open_pos);
          buffer_.erase(0, open_pos);
        }
        break;
      } else {
        static const std::vector<std::string> potential_open_tags = {"<tool_call>", "```tool_call"};
        size_t safe_len = buffer_.size();

        for (const auto &tag : potential_open_tags) {
          for (size_t prefix_len = 1; prefix_len < tag.size() && prefix_len <= buffer_.size(); ++prefix_len) {
            std::string sub = buffer_.substr(buffer_.size() - prefix_len);
            if (tag.rfind(sub, 0) == 0) {
              if (buffer_.size() - prefix_len < safe_len) {
                safe_len = buffer_.size() - prefix_len;
              }
            }
          }
        }

        if (safe_len > 0) {
          res.text_to_emit += buffer_.substr(0, safe_len);
          buffer_.erase(0, safe_len);
        }
        break;
      }
    }

    return res;
  }

  ProcessResult flush() {
    ProcessResult res;
    if (buffer_.empty()) return res;

    size_t open_pos = std::string::npos, open_len = 0;
    size_t close_pos = std::string::npos, close_len = 0;
    if (extract_tag(buffer_, open_pos, open_len, close_pos, close_len)) {
      if (open_pos > 0) {
        res.text_to_emit += buffer_.substr(0, open_pos);
      }
      std::string json_str = buffer_.substr(open_pos + open_len, close_pos - (open_pos + open_len));
      parse_and_add_tool_call(json_str, res);
    } else if (open_pos != std::string::npos) {
      if (open_pos > 0) {
        res.text_to_emit += buffer_.substr(0, open_pos);
      }
      std::string json_str = buffer_.substr(open_pos + open_len);
      if (!parse_and_add_tool_call(json_str, res)) {
        res.text_to_emit += buffer_.substr(open_pos);
      }
    } else {
      res.text_to_emit += buffer_;
    }

    buffer_.clear();
    return res;
  }

  bool has_tool_calls() const {
    return !tool_calls_.empty();
  }

  const std::vector<ParsedToolCall>& get_all_tool_calls() const {
    return tool_calls_;
  }

private:
  bool parse_and_add_tool_call(std::string str, ProcessResult &res) {
    auto trim_str = [](std::string &s) {
      auto start = s.find_first_not_of(" \t\n\r");
      auto end = s.find_last_not_of(" \t\n\r");
      if (start != std::string::npos && end != std::string::npos) {
        s = s.substr(start, end - start + 1);
      }
    };
    trim_str(str);
    if (str.empty()) return false;

    try {
      json call_j = json::parse(str);
      std::string fn_name = "";
      json args = json::object();

      if (call_j.contains("name") && call_j["name"].is_string()) {
        fn_name = call_j["name"].get<std::string>();
      } else if (call_j.contains("function") && call_j["function"].is_object() && call_j["function"].contains("name")) {
        fn_name = call_j["function"]["name"].get<std::string>();
      }

      if (call_j.contains("arguments")) {
        if (call_j["arguments"].is_object()) {
          args = call_j["arguments"];
        } else if (call_j["arguments"].is_string()) {
          try {
            args = json::parse(call_j["arguments"].get<std::string>());
          } catch (...) {
            args = {{"raw", call_j["arguments"].get<std::string>()}};
          }
        }
      } else if (call_j.contains("parameters") && call_j["parameters"].is_object()) {
        args = call_j["parameters"];
      }

      if (!fn_name.empty()) {
        ParsedToolCall tc;
        tc.index = call_index_++;
        tc.id = generate_random_id("call_") + std::to_string(tc.index);
        tc.name = fn_name;
        tc.arguments = args;
        res.completed_calls.push_back(tc);
        tool_calls_.push_back(tc);
        return true;
      }
    } catch (...) {
      // JSON 파싱 실패
    }
    return false;
  }
};

// 요청 메시지를 LiteRT-LM 엔진이 지원하는 내부 JSON 형식으로 변환 (Base64 / URL 이미지 처리 포함)
static json transform_message_for_litert(const json &msg, std::shared_ptr<TempFilesCleanup> temp_files) {
  json transformed = msg;
  std::string role = msg.value("role", "user");
  // OpenAI developer role을 system과 동일하게 취급 (LiteRT-LM 내부 처리용)
  if (role == "developer") {
    role = "system";
  } else if (role == "tool") {
    // LiteRT-LM / Gemma는 tool role을 지원하지 않으므로 user 역할의 [Tool Result] 메시지로 변환
    role = "user";
    std::string tool_name = msg.value("name", "");
    std::string content_str = "";
    if (msg.contains("content")) {
      if (msg["content"].is_string()) {
        content_str = msg["content"].get<std::string>();
      } else {
        content_str = msg["content"].dump();
      }
    }
    std::string header = "[Tool Result" + (tool_name.empty() ? "" : " for " + tool_name) + "]:\n";
    transformed["role"] = "user";
    transformed["content"] = header + content_str;
    return transformed;
  }
  transformed["role"] = role;

  // assistant 역할에서 content가 비어있고 tool_calls가 있는 경우 <tool_call> 태그 텍스트로 복원
  if (role == "assistant") {
    bool has_text = msg.contains("content") && msg["content"].is_string() && !msg["content"].get<std::string>().empty();
    if (!has_text && msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty()) {
      std::string calls_text = "";
      for (const auto &tc : msg["tool_calls"]) {
        if (tc.contains("function") && tc["function"].is_object()) {
          calls_text += "<tool_call>\n" + tc["function"].dump() + "\n</tool_call>\n";
        }
      }
      transformed["content"] = calls_text;
      return transformed;
    }
  }

  // 1. Ollama "images" 필드 (Base64 배열) 처리
  if (msg.contains("images") && msg["images"].is_array() && !msg["images"].empty()) {
    std::string text_content = "";
    if (msg.contains("content") && msg["content"].is_string()) {
      text_content = msg["content"].get<std::string>();
    }
    json content_arr = json::array();
    for (const auto &img_b64 : msg["images"]) {
      if (img_b64.is_string()) {
        std::string path = save_base64_to_temp_file(img_b64.get<std::string>());
        if (!path.empty()) {
          temp_files->paths.push_back(path);
          content_arr.push_back({{"type", "image"}, {"path", path}});
        }
      }
    }
    if (!text_content.empty()) {
      content_arr.push_back({{"type", "text"}, {"text", text_content}});
    }
    transformed["content"] = content_arr;
    transformed.erase("images");
    return transformed;
  }

  // 2. OpenAI 멀티모달 content 배열 (image_url: Base64 data URL, 일반 URL, 로컬 파일 등) 처리
  if (msg.contains("content") && msg["content"].is_array()) {
    json content_arr = json::array();
    for (const auto &part : msg["content"]) {
      if (!part.is_object()) continue;
      std::string type = part.value("type", "");
      if (type == "text") {
        content_arr.push_back(part);
      } else if (type == "image_url") {
        std::string img_src = "";
        if (part.contains("image_url")) {
          if (part["image_url"].is_string()) {
            img_src = part["image_url"].get<std::string>();
          } else if (part["image_url"].is_object() && part["image_url"].contains("url")) {
            img_src = part["image_url"]["url"].get<std::string>();
          }
        }
        if (!img_src.empty()) {
          bool is_temp = false;
          std::string path = save_image_source_to_temp_file(img_src, is_temp);
          if (!path.empty()) {
            if (is_temp) {
              temp_files->paths.push_back(path);
            }
            content_arr.push_back({{"type", "image"}, {"path", path}});
          }
        }
      }
    }
    transformed["content"] = content_arr;
    return transformed;
  }

  // content가 null이거나 누락된 경우 빈 문자열로 안전 보장
  if (!transformed.contains("content") || transformed["content"].is_null()) {
    transformed["content"] = "";
  }

  return transformed;
}

// 시스템 프롬프트를 사용자 메시지 앞단에 자연스럽게 병합 (Gemma/LiteRT-LM의 미지원 시스템 토큰 호환)
static void prepend_system_prompt_to_message(json &msg, const std::string &sys_prompt) {
  if (sys_prompt.empty()) return;
  if (msg.contains("content")) {
    if (msg["content"].is_string()) {
      std::string original = msg["content"].get<std::string>();
      if (original.empty()) {
        msg["content"] = sys_prompt;
      } else {
        msg["content"] = sys_prompt + "\n\n" + original;
      }
    } else if (msg["content"].is_array()) {
      bool prepended = false;
      for (auto &part : msg["content"]) {
        if (part.is_object() && part.value("type", "") == "text") {
          std::string original = part.value("text", "");
          if (original.empty()) {
            part["text"] = sys_prompt;
          } else {
            part["text"] = sys_prompt + "\n\n" + original;
          }
          prepended = true;
          break;
        }
      }
      if (!prepended) {
        msg["content"].insert(msg["content"].begin(), {{"type", "text"}, {"text", sys_prompt}});
      }
    }
  } else {
    msg["content"] = sys_prompt;
  }
}

// 요청 JSON으로부터 세부 생성 옵션(GenerationOptions)을 추출
static GenerationOptions parse_generation_options(const json &j_req) {
  GenerationOptions opts;

  // 1. OpenAI 표준 파라미터 파싱
  if (j_req.contains("temperature") && j_req["temperature"].is_number()) {
    opts.temperature = j_req["temperature"].get<float>();
  }
  if (j_req.contains("top_p") && j_req["top_p"].is_number()) {
    opts.top_p = j_req["top_p"].get<float>();
  }
  if (j_req.contains("top_k") && j_req["top_k"].is_number_integer()) {
    opts.top_k = j_req["top_k"].get<int>();
  }
  if (j_req.contains("max_tokens") && j_req["max_tokens"].is_number_integer()) {
    opts.max_tokens = j_req["max_tokens"].get<int>();
  } else if (j_req.contains("max_completion_tokens") && j_req["max_completion_tokens"].is_number_integer()) {
    opts.max_tokens = j_req["max_completion_tokens"].get<int>();
  }
  if (j_req.contains("seed") && j_req["seed"].is_number_integer()) {
    opts.seed = j_req["seed"].get<int>();
  }
  if (j_req.contains("frequency_penalty") && j_req["frequency_penalty"].is_number()) {
    opts.frequency_penalty = j_req["frequency_penalty"].get<float>();
  }
  if (j_req.contains("presence_penalty") && j_req["presence_penalty"].is_number()) {
    opts.presence_penalty = j_req["presence_penalty"].get<float>();
  }

  // 2. Ollama "options" 객체 파싱 (있는 경우 덮어쓰기)
  if (j_req.contains("options") && j_req["options"].is_object()) {
    const auto &o = j_req["options"];
    if (o.contains("temperature") && o["temperature"].is_number()) {
      opts.temperature = o["temperature"].get<float>();
    }
    if (o.contains("top_p") && o["top_p"].is_number()) {
      opts.top_p = o["top_p"].get<float>();
    }
    if (o.contains("top_k") && o["top_k"].is_number_integer()) {
      opts.top_k = o["top_k"].get<int>();
    }
    if (o.contains("num_predict") && o["num_predict"].is_number_integer()) {
      opts.max_tokens = o["num_predict"].get<int>();
    }
    if (o.contains("seed") && o["seed"].is_number_integer()) {
      opts.seed = o["seed"].get<int>();
    }
    if (o.contains("repeat_penalty") && o["repeat_penalty"].is_number()) {
      opts.frequency_penalty = o["repeat_penalty"].get<float>() - 1.0f;
    }
    if (o.contains("stop")) {
      if (o["stop"].is_string()) {
        opts.stop.push_back(o["stop"].get<std::string>());
      } else if (o["stop"].is_array()) {
        for (const auto &s : o["stop"]) {
          if (s.is_string()) opts.stop.push_back(s.get<std::string>());
        }
      }
    }
  }

  // 3. stop 단어 파싱
  if (j_req.contains("stop")) {
    if (j_req["stop"].is_string()) {
      opts.stop.push_back(j_req["stop"].get<std::string>());
    } else if (j_req["stop"].is_array()) {
      for (const auto &s : j_req["stop"]) {
        if (s.is_string()) opts.stop.push_back(s.get<std::string>());
      }
    }
  }

  return opts;
}

// OpenAI 규격 Model 정보 JSON 생성 헬퍼
static json make_openai_model_info(const std::string &model_name) {
  return {
    {"id", model_name},
    {"object", "model"},
    {"created", 1728000000},
    {"owned_by", "litert-lm"},
    {"permission", json::array({
      {
        {"id", "modelperm-" + model_name},
        {"object", "model_permission"},
        {"created", 1728000000},
        {"allow_create_engine", false},
        {"allow_sampling", true},
        {"allow_logprobs", true},
        {"allow_search_indices", false},
        {"allow_view", true},
        {"allow_fine_tuning", false},
        {"organization", "*"},
        {"group", nullptr},
        {"is_blocking", false}
      }
    })},
    {"root", model_name},
    {"parent", nullptr}
  };
}

// API 서버 실행부 구현
void RunServer(MultimodalCliApp &app, int port, const std::string &served_model_name, EmbeddingEngineWrapper *emb_engine) {
  httplib::Server svr;

  // CORS 및 프리플라이트 요청 처리 핸들러
  svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT");
    res.set_header("Access-Control-Allow-Headers", "*");
    if (req.method == "OPTIONS") {
      res.status = 204;
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  // 상태 확인용 기본 엔드포인트
  svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
    res.set_content("Ollama is running", "text/plain");
  });

  // Ollama 버전 확인 엔드포인트
  svr.Get("/api/version", [](const httplib::Request &, httplib::Response &res) {
    res.set_content(json({{"version", "0.3.14"}}).dump(), "application/json");
  });

  // 모델 목록 엔드포인트 (Ollama 호환)
  svr.Get("/api/tags", [&served_model_name, emb_engine](const httplib::Request &, httplib::Response &res) {
    json models = json::array({
      {
        {"name", served_model_name}, 
        {"model", served_model_name}, 
        {"modified_at", get_iso8601_now()}, 
        {"size", 0}, 
        {"digest", "000000000000"}, 
        {"details", {{"format", "tflite"}, {"family", "litert"}, {"families", json::array({"litert"})}}}
      }
    });
    if (emb_engine && emb_engine->IsLoaded()) {
      models.push_back({
        {"name", "embeddinggemma-2:latest"},
        {"model", "embeddinggemma-2:latest"},
        {"modified_at", get_iso8601_now()},
        {"size", 164626432},
        {"digest", "embeddinggemma2"},
        {"details", {{"format", "litertlm"}, {"family", "embeddinggemma2"}, {"families", json::array({"embeddinggemma2", "embedding"})}}}
      });
      models.push_back({
        {"name", "embeddinggemma2"},
        {"model", "embeddinggemma2"},
        {"modified_at", get_iso8601_now()},
        {"size", 164626432},
        {"digest", "embeddinggemma2"},
        {"details", {{"format", "litertlm"}, {"family", "embeddinggemma2"}, {"families", json::array({"embeddinggemma2", "embedding"})}}}
      });
    }
    json response = {{"models", models}};
    res.set_content(response.dump(), "application/json");
  });

  // 모델 상세 정보 엔드포인트 (Ollama 호환)
  svr.Post("/api/show", [&served_model_name](const httplib::Request &, httplib::Response &res) {
    json resp = {
      {"modelfile", "# Modelfile generated by Ollama-LiteRT-LM-cpp\nFROM " + served_model_name},
      {"parameters", "stop \"<end_of_turn>\""},
      {"template", "{{ .Prompt }}"},
      {"details", {
        {"parent_model", ""},
        {"format", "tflite"},
        {"family", "litert"},
        {"families", json::array({"litert"})},
        {"parameter_size", "2B"},
        {"quantization_level", "none"}
      }}
    };
    res.set_content(resp.dump(), "application/json");
  });

  // ==========================================
  // OpenAI 모델 목록 및 단건 조회 엔드포인트
  // ==========================================
  auto handle_models_list = [&served_model_name, emb_engine](const httplib::Request &, httplib::Response &res) {
    json data = json::array({make_openai_model_info(served_model_name)});
    if (emb_engine && emb_engine->IsLoaded()) {
      data.push_back(make_openai_model_info("embeddinggemma-2:latest"));
      data.push_back(make_openai_model_info("embeddinggemma2"));
    }
    json response = {
      {"object", "list"},
      {"data", data}
    };
    res.set_content(response.dump(), "application/json");
  };

  svr.Get("/v1/models", handle_models_list);
  svr.Get("/models", handle_models_list);

  auto handle_single_model = [&served_model_name](const httplib::Request &req, httplib::Response &res) {
    std::string requested = req.matches[1];
    res.set_content(make_openai_model_info(requested.empty() ? served_model_name : requested).dump(), "application/json");
  };

  svr.Get(R"(/v1/models/(.+))", handle_single_model);
  svr.Get(R"(/models/(.+))", handle_single_model);

  // ==========================================
  // 임베딩 엔드포인트 (OpenAI / Ollama / llama.cpp 호환)
  // ==========================================
  // 1. OpenAI 호환 (/v1/embeddings, /embeddings)
  auto handle_openai_embeddings = [emb_engine](const httplib::Request &req, httplib::Response &res) {
    if (!emb_engine || !emb_engine->IsLoaded()) {
      res.status = 503;
      json err = {
        {"error", {
          {"message", "Embedding engine is not loaded or available."},
          {"type", "service_unavailable"},
          {"param", nullptr},
          {"code", "embedding_disabled"}
        }}
      };
      res.set_content(err.dump(), "application/json");
      return;
    }

    try {
      auto j = json::parse(req.body);
      std::string model = j.value("model", "embeddinggemma-2");
      std::vector<std::string> inputs;
      if (j.contains("input")) {
        if (j["input"].is_string()) {
          inputs.push_back(j["input"].get<std::string>());
        } else if (j["input"].is_array()) {
          for (const auto &item : j["input"]) {
            if (item.is_string()) {
              inputs.push_back(item.get<std::string>());
            }
          }
        }
      }

      if (inputs.empty()) {
        res.status = 400;
        json err = {
          {"error", {
            {"message", "Missing or invalid 'input' parameter in embedding request."},
            {"type", "invalid_request_error"},
            {"param", "input"},
            {"code", "invalid_input"}
          }}
        };
        res.set_content(err.dump(), "application/json");
        return;
      }

      json data_arr = json::array();
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto vec = emb_engine->ComputeEmbedding(inputs[i], true, 0);
        data_arr.push_back({
          {"object", "embedding"},
          {"index", i},
          {"embedding", vec}
        });
      }

      json response = {
        {"object", "list"},
        {"data", data_arr},
        {"model", model},
        {"usage", {
          {"prompt_tokens", 0},
          {"total_tokens", 0}
        }}
      };
      res.set_content(response.dump(), "application/json");
    } catch (const std::exception &e) {
      res.status = 400;
      json err = {
        {"error", {
          {"message", std::string("Malformed JSON body: ") + e.what()},
          {"type", "invalid_request_error"},
          {"param", nullptr},
          {"code", "invalid_json"}
        }}
      };
      res.set_content(err.dump(), "application/json");
    }
  };
  svr.Post("/v1/embeddings", handle_openai_embeddings);
  svr.Post("/embeddings", handle_openai_embeddings);

  // 2. Ollama 호환 단일 임베딩 (/api/embeddings)
  auto handle_ollama_embeddings = [emb_engine](const httplib::Request &req, httplib::Response &res) {
    if (!emb_engine || !emb_engine->IsLoaded()) {
      res.status = 503;
      res.set_content(json({{"error", "Embedding engine not available"}}).dump(), "application/json");
      return;
    }

    try {
      auto j = json::parse(req.body);
      std::string prompt = "";
      if (j.contains("prompt") && j["prompt"].is_string()) {
        prompt = j["prompt"].get<std::string>();
      } else if (j.contains("input") && j["input"].is_string()) {
        prompt = j["input"].get<std::string>();
      }

      auto vec = emb_engine->ComputeEmbedding(prompt, true, 0);
      json response = {
        {"embedding", vec}
      };
      res.set_content(response.dump(), "application/json");
    } catch (const std::exception &e) {
      res.status = 400;
      res.set_content(json({{"error", std::string("Invalid request: ") + e.what()}}).dump(), "application/json");
    }
  };
  svr.Post("/api/embeddings", handle_ollama_embeddings);

  // 3. Ollama v2 호환 배치 임베딩 (/api/embed)
  auto handle_ollama_embed = [emb_engine](const httplib::Request &req, httplib::Response &res) {
    if (!emb_engine || !emb_engine->IsLoaded()) {
      res.status = 503;
      res.set_content(json({{"error", "Embedding engine not available"}}).dump(), "application/json");
      return;
    }

    try {
      auto j = json::parse(req.body);
      std::string model = j.value("model", "embeddinggemma-2");
      std::vector<std::string> inputs;
      if (j.contains("input")) {
        if (j["input"].is_string()) {
          inputs.push_back(j["input"].get<std::string>());
        } else if (j["input"].is_array()) {
          for (const auto &item : j["input"]) {
            if (item.is_string()) {
              inputs.push_back(item.get<std::string>());
            }
          }
        }
      }

      json embeddings_arr = json::array();
      for (const auto &inp : inputs) {
        embeddings_arr.push_back(emb_engine->ComputeEmbedding(inp, true, 0));
      }

      json response = {
        {"model", model},
        {"embeddings", embeddings_arr}
      };
      res.set_content(response.dump(), "application/json");
    } catch (const std::exception &e) {
      res.status = 400;
      res.set_content(json({{"error", std::string("Invalid request: ") + e.what()}}).dump(), "application/json");
    }
  };
  svr.Post("/api/embed", handle_ollama_embed);

  // 4. llama.cpp 네이티브 호환 (/embedding)
  auto handle_llamacpp_embedding = [emb_engine](const httplib::Request &req, httplib::Response &res) {
    if (!emb_engine || !emb_engine->IsLoaded()) {
      res.status = 503;
      res.set_content(json({{"error", "Embedding engine not available"}}).dump(), "application/json");
      return;
    }

    try {
      auto j = json::parse(req.body);
      std::string content = "";
      if (j.contains("content") && j["content"].is_string()) {
        content = j["content"].get<std::string>();
      } else if (j.contains("prompt") && j["prompt"].is_string()) {
        content = j["prompt"].get<std::string>();
      }

      auto vec = emb_engine->ComputeEmbedding(content, true, 0);
      json response = {
        {"embedding", vec}
      };
      res.set_content(response.dump(), "application/json");
    } catch (const std::exception &e) {
      res.status = 400;
      res.set_content(json({{"error", std::string("Invalid request: ") + e.what()}}).dump(), "application/json");
    }
  };
  svr.Post("/embedding", handle_llamacpp_embedding);


  // ==========================================
  // 채팅 완료 처리 핸들러 (OpenAI 및 Ollama 공용)
  // ==========================================
  auto handle_chat_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res, bool is_ollama) {
    try {
      auto j_req = json::parse(req.body);
      std::cout << "[CHAT REQUEST] model=" << j_req.value("model", "")
                << " stream=" << (j_req.contains("stream") ? (j_req["stream"].get<bool>() ? "true" : "false") : (is_ollama ? "true" : "false"))
                << " msg_count=" << (j_req.contains("messages") && j_req["messages"].is_array() ? j_req["messages"].size() : 0)
                << " has_tools=" << (j_req.contains("tools") && j_req["tools"].is_array() && !j_req["tools"].empty() ? "true" : "false")
                << std::endl;
      // Ollama 스펙상 stream 생략 시 기본값은 true, OpenAI는 false
      bool want_stream = j_req.contains("stream") ? j_req["stream"].get<bool>() : is_ollama;

      // OpenAI stream_options (include_usage) 확인
      bool include_usage = false;
      if (!is_ollama && j_req.contains("stream_options") && j_req["stream_options"].is_object()) {
        include_usage = j_req["stream_options"].value("include_usage", false);
      }

      GenerationOptions gen_opts = parse_generation_options(j_req);

      std::string sys_msg = "";
      json history_arr = json::array();
      json current_msg_j;
      std::string full_prompt_text = "";

      auto temp_files = std::make_shared<TempFilesCleanup>();

      // 메시지 파싱 및 대화 기록/시스템 메시지 분리
      if (j_req.contains("messages") && j_req["messages"].is_array()) {
        auto messages = j_req["messages"];
        for (const auto &m : messages) {
          if (!m.is_object()) continue;
          std::string role = m.value("role", "");
          if (role == "system" || role == "developer") {
            std::string content = "";
            if (m.contains("content") && m["content"].is_string()) {
              content = m["content"].get<std::string>();
            }
            if (!content.empty()) {
              if (!sys_msg.empty()) sys_msg += "\n\n";
              sys_msg += content;
            }
            full_prompt_text += content + "\n";
          } else {
            json tr = transform_message_for_litert(m, temp_files);
            history_arr.push_back(tr);
            if (m.contains("content") && m["content"].is_string()) {
              full_prompt_text += m["content"].get<std::string>() + "\n";
            }
          }
        }
      }

      if (sys_msg.empty()) {
        sys_msg = DEFAULT_SYSTEM_PROMPT;
      }

      // 도구 정의(tools)가 요청에 포함된 경우 Gemma용 툴 시스템 지침 자동 주입
      if (j_req.contains("tools") && j_req["tools"].is_array() && !j_req["tools"].empty()) {
        sys_msg += format_tools_system_prompt(j_req["tools"]);
      }

      if (!history_arr.empty()) {
        current_msg_j = history_arr.back();
        history_arr.erase(history_arr.end() - 1);
      } else {
        current_msg_j = {{"role", "user"}, {"content", ""}};
      }

      // 1. 현재 메시지(도구 결과 등)가 비대할 경우 안전하게 중간 생략 (최대 4000자)
      truncate_long_message(current_msg_j, 4000);

      // 컨텍스트 슬라이딩 윈도우 (엔진 max_tokens 한도 초과 방지)
      int max_engine_tokens = app.GetMaxTokens();
      // 모델 응답 생성 버퍼(최소 1536 토큰) 및 오버헤드(300 토큰)를 제외한 안전 한도 설정
      int target_limit = std::max(2048, max_engine_tokens - 1836);

      std::string curr_text = "";
      if (current_msg_j.contains("content")) {
        if (current_msg_j["content"].is_string()) curr_text = current_msg_j["content"].get<std::string>();
        else curr_text = current_msg_j["content"].dump();
      }
      int sys_tokens = app.CountTokens(sys_msg);
      int curr_tokens = app.CountTokens(curr_text);
      int base_tokens = sys_tokens + curr_tokens + 50;

      int history_budget = target_limit - base_tokens;
      if (history_budget < 0) {
        history_arr.clear();

        // 비상 방어선 (Hard Clamping): 히스토리를 전부 비워도 시스템 지침 + 현재 메시지가 한도를 넘는 경우
        if (base_tokens > target_limit) {
          std::cout << "[컨텍스트 하드 클램핑] 시스템 지침(" << sys_tokens << ")과 현재 메시지(" 
                    << curr_tokens << ")가 안전 한도(" << target_limit << ")를 초과하여 강제 절삭합니다." << std::endl;
          // 현재 메시지를 더 타이트하게 2000자로 절삭
          truncate_long_message(current_msg_j, 2000);
          if (current_msg_j["content"].is_string()) curr_text = current_msg_j["content"].get<std::string>();
          else curr_text = current_msg_j["content"].dump();
          curr_tokens = app.CountTokens(curr_text);

          // 그래도 초과 시 시스템 지침의 후반부를 안전하게 절삭
          int allowed_sys_tokens = target_limit - curr_tokens - 100;
          if (allowed_sys_tokens < 1000) allowed_sys_tokens = 1000;
          if (sys_tokens > allowed_sys_tokens) {
            size_t cut_len = (sys_msg.size() * allowed_sys_tokens) / sys_tokens;
            if (cut_len < sys_msg.size()) {
              sys_msg = sys_msg.substr(0, cut_len) + "\n\n[... 프롬프트 일부 절삭됨 ...]";
              sys_tokens = app.CountTokens(sys_msg);
            }
          }
          base_tokens = sys_tokens + curr_tokens + 50;
        }
      } else if (!history_arr.empty()) {
        json trimmed_history = json::array();
        int accumulated_tokens = 0;
        int dropped_count = 0;

        // 최신 메시지(뒤)부터 역순으로 탐색하여 예산 내에 들어오는 메시지만 보존
        for (int idx = static_cast<int>(history_arr.size()) - 1; idx >= 0; --idx) {
          const auto &item = history_arr[idx];
          std::string item_text = "";
          if (item.contains("content")) {
            if (item["content"].is_string()) item_text = item["content"].get<std::string>();
            else item_text = item["content"].dump();
          }
          int item_tokens = app.CountTokens(item_text) + 20;

          if (accumulated_tokens + item_tokens <= history_budget) {
            trimmed_history.insert(trimmed_history.begin(), item);
            accumulated_tokens += item_tokens;
          } else {
            dropped_count = idx + 1;
            break;
          }
        }

        if (dropped_count > 0) {
          std::cout << "[컨텍스트 슬라이딩 윈도우] 오래된 메시지 " << dropped_count
                    << "개를 트리밍하여 토큰 초과를 방지했습니다. (예상 입력: "
                    << (base_tokens + accumulated_tokens) << " / " << max_engine_tokens << " 토큰)"
                    << std::endl;
        }

        // 트리밍 후 만약 첫 메시지가 'assistant'이면 대화 턴 일관성을 위해 첫 'user' 메시지 이전 고아 메시지 정리
        while (!trimmed_history.empty() && trimmed_history.front().value("role", "") != "user") {
          trimmed_history.erase(trimmed_history.begin());
        }

        history_arr = std::move(trimmed_history);
      }

      // LiteRT-LM / Gemma 엔진은 시스템 토큰이 없으므로 첫 번째 user 메시지 앞단에 시스템 프롬프트 병합
      if (!sys_msg.empty()) {
        bool injected = false;
        for (auto &m : history_arr) {
          if (m.value("role", "") == "user") {
            prepend_system_prompt_to_message(m, sys_msg);
            injected = true;
            break;
          }
        }
        if (!injected) {
          prepend_system_prompt_to_message(current_msg_j, sys_msg);
        }
      }

      std::string history_json_str = history_arr.empty() ? "" : history_arr.dump();
      std::string current_msg_str = current_msg_j.dump();

      // 프롬프트 토큰 수 계산 (정확도 높은 카운트)
      int prompt_tokens = std::max(1, app.CountTokens(full_prompt_text.empty() ? sys_msg : full_prompt_text));

      // OpenAI 규격용 ID 및 생성 타임스탬프
      std::string chat_id = generate_random_id("chatcmpl-");
      uint64_t created_ts = static_cast<uint64_t>(time(nullptr));

      // 스트리밍 응답 처리 루틴
      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);
        auto accumulated_text = std::make_shared<std::string>();
        auto is_first_chunk = std::make_shared<bool>(true);
        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();
        auto tool_parser = std::make_shared<ToolCallStreamParser>();

        std::thread([&app, sys_msg, history_json_str, current_msg_str, served_model_name, is_ollama,
                     sink_ctx, cancelled, temp_files, gen_opts, chat_id, created_ts,
                     prompt_tokens, include_usage, accumulated_text, is_first_chunk, utf8_asm, tool_parser]() {
          app.StreamForServer(sys_msg, history_json_str, current_msg_str,
            // 청크 수신 콜백
            [served_model_name, is_ollama, sink_ctx, chat_id, created_ts, accumulated_text, is_first_chunk, utf8_asm, tool_parser](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              auto p_res = tool_parser->process(clean_chunk);

              // 1. 일반 텍스트 청크 방출
              if (!p_res.text_to_emit.empty()) {
                std::string text_chunk = p_res.text_to_emit;
                *accumulated_text += text_chunk;

                if (is_ollama) {
                  json chunk_j = {
                    {"model", served_model_name},
                    {"created_at", get_iso8601_now()},
                    {"message", {{"role", "assistant"}, {"content", text_chunk}}},
                    {"done", false}
                  };
                  std::lock_guard<std::mutex> lock(sink_ctx->mtx);
                  sink_ctx->chunks.push(chunk_j.dump() + "\n");
                  sink_ctx->cv.notify_one();
                } else {
                  json delta_obj = json::object();
                  if (*is_first_chunk) {
                    delta_obj["role"] = "assistant";
                    *is_first_chunk = false;
                  }
                  delta_obj["content"] = text_chunk;

                  json chunk_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array({
                      {
                        {"index", 0},
                        {"delta", delta_obj},
                        {"logprobs", nullptr},
                        {"finish_reason", nullptr}
                      }
                    })}
                  };
                  std::lock_guard<std::mutex> lock(sink_ctx->mtx);
                  sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
                  sink_ctx->cv.notify_one();
                }
              }

              // 2. 완성된 Tool Call 청크 방출
              for (const auto &tc : p_res.completed_calls) {
                if (is_ollama) {
                  json chunk_j = {
                    {"model", served_model_name},
                    {"created_at", get_iso8601_now()},
                    {"message", {
                      {"role", "assistant"},
                      {"content", ""},
                      {"tool_calls", json::array({
                        {{"function", {{"name", tc.name}, {"arguments", tc.arguments}}}}
                      })}
                    }},
                    {"done", false}
                  };
                  std::lock_guard<std::mutex> lock(sink_ctx->mtx);
                  sink_ctx->chunks.push(chunk_j.dump() + "\n");
                  sink_ctx->cv.notify_one();
                } else {
                  json delta_obj = json::object();
                  if (*is_first_chunk) {
                    delta_obj["role"] = "assistant";
                    *is_first_chunk = false;
                  }
                  delta_obj["tool_calls"] = json::array({
                    {
                      {"index", tc.index},
                      {"id", tc.id},
                      {"type", "function"},
                      {"function", {
                        {"name", tc.name},
                        {"arguments", tc.arguments.dump()}
                      }}
                    }
                  });

                  json chunk_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array({
                      {
                        {"index", 0},
                        {"delta", delta_obj},
                        {"logprobs", nullptr},
                        {"finish_reason", nullptr}
                      }
                    })}
                  };
                  std::lock_guard<std::mutex> lock(sink_ctx->mtx);
                  sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
                  sink_ctx->cv.notify_one();
                }
              }
            },
            // 완료 콜백
            [&app, served_model_name, is_ollama, sink_ctx, chat_id, created_ts,
             prompt_tokens, include_usage, accumulated_text, utf8_asm, is_first_chunk, tool_parser]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              auto p_res = tool_parser->process(leftover);
              auto p_flush = tool_parser->flush();
              std::string final_text = p_res.text_to_emit + p_flush.text_to_emit;

              if (!final_text.empty()) {
                *accumulated_text += final_text;
                if (is_ollama) {
                  json chunk_j = {
                    {"model", served_model_name},
                    {"created_at", get_iso8601_now()},
                    {"message", {{"role", "assistant"}, {"content", final_text}}},
                    {"done", false}
                  };
                  sink_ctx->chunks.push(chunk_j.dump() + "\n");
                } else {
                  json delta_obj = json::object();
                  if (*is_first_chunk) {
                    delta_obj["role"] = "assistant";
                    *is_first_chunk = false;
                  }
                  delta_obj["content"] = final_text;
                  json chunk_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array({
                      {
                        {"index", 0},
                        {"delta", delta_obj},
                        {"logprobs", nullptr},
                        {"finish_reason", nullptr}
                      }
                    })}
                  };
                  sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
                }
              }

              // 잔여 버퍼에서 검출된 Tool Call 방출
              std::vector<ParsedToolCall> final_calls = p_res.completed_calls;
              final_calls.insert(final_calls.end(), p_flush.completed_calls.begin(), p_flush.completed_calls.end());
              for (const auto &tc : final_calls) {
                if (is_ollama) {
                  json chunk_j = {
                    {"model", served_model_name},
                    {"created_at", get_iso8601_now()},
                    {"message", {
                      {"role", "assistant"},
                      {"content", ""},
                      {"tool_calls", json::array({
                        {{"function", {{"name", tc.name}, {"arguments", tc.arguments}}}}
                      })}
                    }},
                    {"done", false}
                  };
                  sink_ctx->chunks.push(chunk_j.dump() + "\n");
                } else {
                  json delta_obj = json::object();
                  if (*is_first_chunk) {
                    delta_obj["role"] = "assistant";
                    *is_first_chunk = false;
                  }
                  delta_obj["tool_calls"] = json::array({
                    {
                      {"index", tc.index},
                      {"id", tc.id},
                      {"type", "function"},
                      {"function", {
                        {"name", tc.name},
                        {"arguments", tc.arguments.dump()}
                      }}
                    }
                  });
                  json chunk_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array({
                      {
                        {"index", 0},
                        {"delta", delta_obj},
                        {"logprobs", nullptr},
                        {"finish_reason", nullptr}
                      }
                    })}
                  };
                  sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
                }
              }

              bool has_tools_called = tool_parser->has_tool_calls();
              std::string finish_reason = has_tools_called ? "tool_calls" : "stop";

              // OpenAI 모드에서 첫 청크가 아직 방출되지 않은 경우(내용이 빈 채로 종료된 경우), 빈 content 델타를 명시적으로 방출
              if (!is_ollama && *is_first_chunk && !has_tools_called) {
                json init_j = {
                  {"id", chat_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"system_fingerprint", "fp_litert_lm"},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", {{"role", "assistant"}, {"content", ""}}},
                      {"logprobs", nullptr},
                      {"finish_reason", nullptr}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + init_j.dump() + "\n\n");
                *is_first_chunk = false;
              }

              if (is_ollama) {
                json done_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"message", {{"role", "assistant"}, {"content", ""}}},
                  {"done_reason", finish_reason},
                  {"done", true}
                };
                sink_ctx->chunks.push(done_j.dump() + "\n");
              } else {
                json stop_j = {
                  {"id", chat_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"system_fingerprint", "fp_litert_lm"},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", json::object()},
                      {"logprobs", nullptr},
                      {"finish_reason", finish_reason}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + stop_j.dump() + "\n\n");

                if (include_usage) {
                  int comp_tokens = app.CountTokens(*accumulated_text);
                  json usage_j = {
                    {"id", chat_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created_ts},
                    {"model", served_model_name},
                    {"system_fingerprint", "fp_litert_lm"},
                    {"choices", json::array()},
                    {"usage", {
                      {"prompt_tokens", prompt_tokens},
                      {"completion_tokens", comp_tokens},
                      {"total_tokens", prompt_tokens + comp_tokens}
                    }}
                  };
                  sink_ctx->chunks.push("data: " + usage_j.dump() + "\n\n");
                }

                sink_ctx->chunks.push("data: [DONE]\n\n");
              }
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            // 에러 콜백
            [sink_ctx, is_ollama, chat_id, created_ts, served_model_name](const std::string &err_msg) { 
              std::cerr << "[SERVER STREAM ERROR] " << err_msg << std::endl;
              std::lock_guard<std::mutex> lock(sink_ctx->mtx); 
              if (is_ollama) {
                json err_j = {
                  {"model", served_model_name},
                  {"error", err_msg},
                  {"done", true}
                };
                sink_ctx->chunks.push(err_j.dump() + "\n");
              } else {
                json err_j = {
                  {"id", chat_id},
                  {"object", "chat.completion.chunk"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"system_fingerprint", "fp_litert_lm"},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", {{"role", "assistant"}, {"content", "[오류 발생: " + err_msg + "]"}}},
                      {"logprobs", nullptr},
                      {"finish_reason", "stop"}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + err_j.dump() + "\n\n");
                sink_ctx->chunks.push("data: [DONE]\n\n");
              }
              sink_ctx->done = true; 
              sink_ctx->cv.notify_one(); 
            },
            // 취소 확인 콜백
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider(is_ollama ? "application/x-ndjson" : "text/event-stream", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } 
      // 일반 동기(비스트리밍) 응답 처리 루틴
      else {
        std::string raw_output = app.GenerateForServer(sys_msg, history_json_str, current_msg_str, gen_opts);
        ToolCallStreamParser parser;
        auto p_res = parser.process(raw_output);
        auto p_flush = parser.flush();
        std::string clean_text = p_res.text_to_emit + p_flush.text_to_emit;
        bool has_tools_called = parser.has_tool_calls();
        std::string finish_reason = has_tools_called ? "tool_calls" : "stop";

        json api_res;
        if (is_ollama) {
          json msg_obj = {{"role", "assistant"}, {"content", clean_text}};
          if (has_tools_called) {
            json tc_arr = json::array();
            for (const auto &tc : parser.get_all_tool_calls()) {
              tc_arr.push_back({{"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
            }
            msg_obj["tool_calls"] = tc_arr;
          }

          api_res = {
            {"model", served_model_name},
            {"created_at", get_iso8601_now()},
            {"message", msg_obj},
            {"done_reason", finish_reason},
            {"done", true}
          };
        } else {
          int completion_tokens = app.CountTokens(clean_text);
          json msg_obj = {{"role", "assistant"}, {"content", clean_text}};
          if (has_tools_called) {
            json tc_arr = json::array();
            for (const auto &tc : parser.get_all_tool_calls()) {
              tc_arr.push_back({
                {"id", tc.id},
                {"type", "function"},
                {"function", {{"name", tc.name}, {"arguments", tc.arguments.dump()}}}
              });
            }
            msg_obj["tool_calls"] = tc_arr;
          }

          api_res = {
            {"id", chat_id},
            {"object", "chat.completion"},
            {"created", created_ts},
            {"model", served_model_name},
            {"system_fingerprint", "fp_litert_lm"},
            {"choices", json::array({
              {
                {"index", 0},
                {"message", msg_obj},
                {"logprobs", nullptr},
                {"finish_reason", finish_reason}
              }
            })},
            {"usage", {
              {"prompt_tokens", prompt_tokens},
              {"completion_tokens", completion_tokens},
              {"total_tokens", prompt_tokens + completion_tokens}
            }}
          };
        }
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) { 
      res.status = 500; 
      res.set_content(e.what(), "text/plain"); 
    }
  };

  // ==========================================
  // OpenAI Legacy Completion (/v1/completions) 핸들러
  // ==========================================
  auto handle_legacy_completion = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res) {
    try {
      auto j_req = json::parse(req.body);
      bool want_stream = j_req.value("stream", false);
      GenerationOptions gen_opts = parse_generation_options(j_req);

      std::string prompt = "";
      if (j_req.contains("prompt")) {
        if (j_req["prompt"].is_string()) {
          prompt = j_req["prompt"].get<std::string>();
        } else if (j_req["prompt"].is_array() && !j_req["prompt"].empty()) {
          prompt = j_req["prompt"][0].get<std::string>();
        }
      }

      int prompt_tokens = std::max(1, app.CountTokens(prompt));
      std::string cmpl_id = generate_random_id("cmpl-");
      uint64_t created_ts = static_cast<uint64_t>(time(nullptr));

      json current_msg_j = {{"role", "user"}, {"content", prompt}};
      std::string current_msg_str = current_msg_j.dump();

      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);

        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();

        std::thread([&app, current_msg_str, served_model_name, sink_ctx, cancelled, gen_opts, cmpl_id, created_ts, utf8_asm]() {
          app.StreamForServer(DEFAULT_SYSTEM_PROMPT, "", current_msg_str,
            [served_model_name, sink_ctx, cmpl_id, created_ts, utf8_asm](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              json chunk_j = {
                {"id", cmpl_id},
                {"object", "text_completion"},
                {"created", created_ts},
                {"model", served_model_name},
                {"choices", json::array({
                  {
                    {"text", clean_chunk},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", nullptr}
                  }
                })}
              };
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
              sink_ctx->cv.notify_one();
            },
            [served_model_name, sink_ctx, cmpl_id, created_ts, utf8_asm]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              if (!leftover.empty()) {
                json chunk_j = {
                  {"id", cmpl_id},
                  {"object", "text_completion"},
                  {"created", created_ts},
                  {"model", served_model_name},
                  {"choices", json::array({
                    {
                      {"text", leftover},
                      {"index", 0},
                      {"logprobs", nullptr},
                      {"finish_reason", nullptr}
                    }
                  })}
                };
                sink_ctx->chunks.push("data: " + chunk_j.dump() + "\n\n");
              }

              json stop_j = {
                {"id", cmpl_id},
                {"object", "text_completion"},
                {"created", created_ts},
                {"model", served_model_name},
                {"choices", json::array({
                  {
                    {"text", ""},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", "stop"}
                  }
                })}
              };
              sink_ctx->chunks.push("data: " + stop_j.dump() + "\n\n");
              sink_ctx->chunks.push("data: [DONE]\n\n");
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [sink_ctx](const std::string &) {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider("text/event-stream", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } else {
        std::string output = app.GenerateForServer(DEFAULT_SYSTEM_PROMPT, "", current_msg_str, gen_opts);
        int completion_tokens = app.CountTokens(output);
        json api_res = {
          {"id", cmpl_id},
          {"object", "text_completion"},
          {"created", created_ts},
          {"model", served_model_name},
          {"choices", json::array({
            {
              {"text", output},
              {"index", 0},
              {"logprobs", nullptr},
              {"finish_reason", "stop"}
            }
          })},
          {"usage", {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}
          }}
        };
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) {
      res.status = 500;
      res.set_content(e.what(), "text/plain");
    }
  };

  // ==========================================
  // Ollama 단일 생성 (/api/generate) 핸들러
  // ==========================================
  auto handle_generate = [&app, &served_model_name](const httplib::Request &req, httplib::Response &res) {
    try {
      auto j_req = json::parse(req.body);
      bool want_stream = j_req.contains("stream") ? j_req["stream"].get<bool>() : true;
      std::string prompt = j_req.value("prompt", "");
      std::string sys_msg = j_req.value("system", DEFAULT_SYSTEM_PROMPT);

      std::string effective_prompt = prompt;
      if (!sys_msg.empty()) {
        effective_prompt = sys_msg + "\n\n" + prompt;
      }

      GenerationOptions gen_opts = parse_generation_options(j_req);
      auto temp_files = std::make_shared<TempFilesCleanup>();
      json current_msg_j;

      // 이미지 처리 (Base64 배열)
      if (j_req.contains("images") && j_req["images"].is_array() && !j_req["images"].empty()) {
        json content_arr = json::array();
        for (const auto &img_b64 : j_req["images"]) {
          if (img_b64.is_string()) {
            std::string path = save_base64_to_temp_file(img_b64.get<std::string>());
            if (!path.empty()) {
              temp_files->paths.push_back(path);
              content_arr.push_back({{"type", "image"}, {"path", path}});
            }
          }
        }
        if (!effective_prompt.empty()) {
          content_arr.push_back({{"type", "text"}, {"text", effective_prompt}});
        }
        current_msg_j = {{"role", "user"}, {"content", content_arr}};
      } else {
        current_msg_j = {{"role", "user"}, {"content", effective_prompt}};
      }

      std::string current_msg_str = current_msg_j.dump();

      if (want_stream) {
        struct SinkCtx { 
          std::mutex mtx; 
          std::condition_variable cv; 
          std::queue<std::string> chunks; 
          bool done = false; 
        };
        auto sink_ctx = std::make_shared<SinkCtx>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);

        auto utf8_asm = std::make_shared<Utf8StreamReassembler>();

        std::thread([&app, sys_msg, current_msg_str, served_model_name, sink_ctx, cancelled, temp_files, gen_opts, utf8_asm]() {
          app.StreamForServer(sys_msg, "", current_msg_str,
            [served_model_name, sink_ctx, utf8_asm](const std::string &chunk) {
              std::string clean_chunk = utf8_asm->process(chunk);
              if (clean_chunk.empty()) return;

              json chunk_j = {
                {"model", served_model_name},
                {"created_at", get_iso8601_now()},
                {"response", clean_chunk},
                {"done", false}
              };
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->chunks.push(chunk_j.dump() + "\n");
              sink_ctx->cv.notify_one();
            },
            [served_model_name, sink_ctx, utf8_asm]() {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              std::string leftover = utf8_asm->flush();
              if (!leftover.empty()) {
                json chunk_j = {
                  {"model", served_model_name},
                  {"created_at", get_iso8601_now()},
                  {"response", leftover},
                  {"done", false}
                };
                sink_ctx->chunks.push(chunk_j.dump() + "\n");
              }

              json done_j = {
                {"model", served_model_name},
                {"created_at", get_iso8601_now()},
                {"response", ""},
                {"done_reason", "stop"},
                {"done", true}
              };
              sink_ctx->chunks.push(done_j.dump() + "\n");
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [sink_ctx](const std::string &) {
              std::lock_guard<std::mutex> lock(sink_ctx->mtx);
              sink_ctx->done = true;
              sink_ctx->cv.notify_one();
            },
            [cancelled]() { return cancelled->load(); },
            gen_opts
          );
        }).detach();

        res.set_chunked_content_provider("application/x-ndjson", [sink_ctx, cancelled](size_t, httplib::DataSink &sink) {
          while (true) {
            std::unique_lock<std::mutex> lock(sink_ctx->mtx);
            sink_ctx->cv.wait(lock, [&sink_ctx]{ return !sink_ctx->chunks.empty() || sink_ctx->done; });
            while (!sink_ctx->chunks.empty()) {
              std::string data = std::move(sink_ctx->chunks.front());
              sink_ctx->chunks.pop();
              lock.unlock();
              if (!sink.write(data.c_str(), data.size())) {
                cancelled->store(true);
                return false;
              }
              lock.lock();
            }
            if (sink_ctx->done) { sink.done(); return true; }
          }
        });
      } else {
        std::string output = app.GenerateForServer(sys_msg, "", current_msg_str, gen_opts);
        json api_res = {
          {"model", served_model_name},
          {"created_at", get_iso8601_now()},
          {"response", output},
          {"done_reason", "stop"},
          {"done", true}
        };
        res.set_content(api_res.dump(), "application/json");
      }
    } catch (const std::exception &e) {
      res.status = 500;
      res.set_content(e.what(), "text/plain");
    }
  };

  // OpenAI 호환 엔드포인트 등록 (경로 유연성 지원)
  svr.Post("/v1/chat/completions", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, false); });
  svr.Post("/chat/completions", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, false); });
  svr.Post("/v1/completions", handle_legacy_completion);
  svr.Post("/completions", handle_legacy_completion);

  // Ollama 호환 엔드포인트 등록
  svr.Post("/api/chat", [&handle_chat_completion](const httplib::Request &req, httplib::Response &res) { handle_chat_completion(req, res, true); });
  svr.Post("/api/generate", handle_generate);

  // 서버 바인딩 및 수신 대기
  std::cout << "[서버] 0.0.0.0:" << port << " 대기 중 (모델: " << served_model_name << ")" << std::endl;
  if (!svr.listen("0.0.0.0", port)) {
    std::cerr << "\n[오류] 0.0.0.0:" << port << " 포트 바인딩에 실패했습니다!\n"
              << "원인: 해당 포트를 이미 다른 프로세스(예: 백그라운드에서 실행 중인 기존 Ollama 서비스)가 점유하고 있습니다.\n"
              << "해결 방법:\n"
              << "  1) 기존 Ollama 서비스 중지: sudo systemctl stop ollama\n"
              << "  2) 또는 다른 포트로 실행: PORT=11435 ./run.sh (또는 --port 11435)\n" << std::endl;
    throw std::runtime_error("Port " + std::to_string(port) + " is already in use.");
  }
}
