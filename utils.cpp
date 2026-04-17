#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <regex>
#include <vector>
#include <map>

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

// 시스템 클립보드 복사 구현
void copy_to_clipboard(const std::string &text) {
  // wl-copy (Wayland) 우선 시도 후 xclip (X11) 시도
  FILE* pipe = popen("wl-copy 2>/dev/null || xclip -selection clipboard 2>/dev/null", "w");
  if (pipe) {
    fwrite(text.c_str(), 1, text.size(), pipe);
    pclose(pipe);
  }
}

// LaTeX 기호를 유니코드로 변환하는 로직
std::string latex_to_unicode(const std::string &input) {
  // 변환할 주요 수학 기호 매핑 테이블
  static const std::vector<std::pair<std::string, std::string>> SYMBOLS = {
      {"\\alpha", "α"}, {"\\beta", "β"}, {"\\gamma", "γ"}, {"\\delta", "δ"},
      {"\\epsilon", "ε"}, {"\\zeta", "ζ"}, {"\\eta", "η"}, {"\\theta", "θ"},
      {"\\iota", "ι"}, {"\\kappa", "κ"}, {"\\lambda", "λ"}, {"\\mu", "μ"},
      {"\\nu", "ν"}, {"\\xi", "ξ"}, {"\\pi", "π"}, {"\\rho", "ρ"},
      {"\\sigma", "σ"}, {"\\tau", "τ"}, {"\\upsilon", "υ"}, {"\\phi", "φ"},
      {"\\chi", "χ"}, {"\\psi", "ψ"}, {"\\omega", "ω"}, {"\\Gamma", "Γ"},
      {"\\Delta", "Δ"}, {"\\Theta", "Θ"}, {"\\Lambda", "Λ"}, {"\\Xi", "Ξ"},
      {"\\Pi", "Π"}, {"\\Sigma", "Σ"}, {"\\Phi", "Φ"}, {"\\Psi", "Ψ"},
      {"\\Omega", "Ω"}, {"\\times", "×"}, {"\\div", "÷"}, {"\\cdot", "·"},
      {"\\pm", "±"}, {"\\mp", "∓"}, {"\\leq", "≤"}, {"\\geq", "≥"},
      {"\\neq", "≠"}, {"\\approx", "≈"}, {"\\equiv", "≡"}, {"\\sim", "∼"},
      {"\\propto", "∝"}, {"\\ll", "≪"}, {"\\gg", "≫"}, {"\\in", "∈"},
      {"\\notin", "∉"}, {"\\subset", "⊂"}, {"\\supset", "⊃"}, {"\\subseteq", "⊆"},
      {"\\supseteq", "⊇"}, {"\\cup", "∪"}, {"\\cap", "∩"}, {"\\emptyset", "∅"},
      {"\\forall", "∀"}, {"\\exists", "∃"}, {"\\neg", "¬"}, {"\\land", "∧"},
      {"\\lor", "∨"}, {"\\rightarrow", "→"}, {"\\leftarrow", "←"},
      {"\\Rightarrow", "⇒"}, {"\\Leftarrow", "⇐"}, {"\\leftrightarrow", "↔"},
      {"\\Leftrightarrow", "⇔"}, {"\\to", "→"}, {"\\mapsto", "↦"},
      {"\\infty", "∞"}, {"\\partial", "∂"}, {"\\nabla", "∇"}, {"\\sum", "Σ"},
      {"\\prod", "Π"}, {"\\int", "∫"}, {"\\sqrt", "√"}, {"\\angle", "∠"},
      {"\\degree", "°"}, {"\\circ", "∘"}, {"\\star", "★"}, {"\\bullet", "•"},
      {"\\dots", "…"}, {"\\cdots", "⋯"}, {"\\ldots", "…"}, {"\\therefore", "∴"},
      {"\\because", "∵"}, {"\\left(", "("}, {"\\right)", ")"}, {"\\left[", "["},
      {"\\right]", "]"}, {"\\left\\{", "{"}, {"\\right\\}", "}"}, {"\\left|", "|"},
      {"\\right|", "|"}, {"\\lfloor", "⌊"}, {"\\rfloor", "⌋"}, {"\\lceil", "⌈"},
      {"\\rceil", "⌉"},
  };

  // 위첨자 숫자 및 기호 매핑
  static const std::map<char, std::string> SUPERSCRIPTS = {
      {'0', "⁰"}, {'1', "¹"}, {'2', "²"}, {'3', "³"}, {'4', "⁴"}, {'5', "⁵"},
      {'6', "⁶"}, {'7', "⁷"}, {'8', "⁸"}, {'9', "⁹"}, {'n', "ⁿ"}, {'i', "ⁱ"},
      {'+', "⁺"}, {'-', "⁻"}, {'(', "⁽"}, {')', "⁾"},
  };
  // 아래첨자 숫자 및 기호 매핑
  static const std::map<char, std::string> SUBSCRIPTS = {
      {'0', "₀"}, {'1', "₁"}, {'2', "₂"}, {'3', "₃"}, {'4', "₄"}, {'5', "₅"},
      {'6', "₆"}, {'7', "₇"}, {'8', "₈"}, {'9', "₉"}, {'i', "ᵢ"}, {'j', "ⱼ"},
      {'n', "ₙ"}, {'m', "ₘ"}, {'+', "₊"}, {'-', "₋"}, {'(', "₍"}, {')', "₎"},
  };

  std::string result = input;
  // 기본적인 LaTeX 구분자 제거 ($$, $)
  result = std::regex_replace(result, std::regex(R"(\$\$)"), "");
  result = std::regex_replace(result, std::regex(R"(\$)"), "");
  // 분수 및 제곱근 등 복합 서식 간소화
  result = std::regex_replace(result, std::regex(R"(\\frac\{([^}]*)\}\{([^}]*)\})"), "($1/$2)");
  result = std::regex_replace(result, std::regex(R"(\\sqrt\{([^}]*)\})"), "√($1)");
  result = std::regex_replace(result, std::regex(R"(\\text\{([^}]*)\})"), "$1");
  result = std::regex_replace(result, std::regex(R"(\\mathrm\{([^}]*)\})"), "$1");
  result = std::regex_replace(result, std::regex(R"(\\mathbf\{([^}]*)\})"), "$1");

  // 위첨자 처리 (중괄호 형태 ^{...})
  std::regex sup_brace(R"(\^\{([^}]*)\})");
  std::smatch m;
  while (std::regex_search(result, m, sup_brace)) {
    std::string inner = m[1].str();
    std::string converted;
    for (char c : inner) {
      auto it = SUPERSCRIPTS.find(c);
      converted += (it != SUPERSCRIPTS.end()) ? it->second : std::string(1, c);
    }
    result = m.prefix().str() + converted + m.suffix().str();
  }
  // 위첨자 처리 (단일 문자 형태 ^x)
  std::regex sup_single(R"(\^([0-9ni]))");
  while (std::regex_search(result, m, sup_single)) {
    char c = m[1].str()[0];
    auto it = SUPERSCRIPTS.find(c);
    std::string rep = (it != SUPERSCRIPTS.end()) ? it->second : m[1].str();
    result = m.prefix().str() + rep + m.suffix().str();
  }

  // 아래첨자 처리 (중괄호 형태 _{...})
  std::regex sub_brace(R"(_\{([^}]*)\})");
  while (std::regex_search(result, m, sub_brace)) {
    std::string inner = m[1].str();
    std::string converted;
    for (char c : inner) {
      auto it = SUBSCRIPTS.find(c);
      converted += (it != SUBSCRIPTS.end()) ? it->second : std::string(1, c);
    }
    result = m.prefix().str() + converted + m.suffix().str();
  }
  // 아래첨자 처리 (단일 문자 형태 _x)
  std::regex sub_single(R"(_([0-9ijnm]))");
  while (std::regex_search(result, m, sub_single)) {
    char c = m[1].str()[0];
    auto it = SUBSCRIPTS.find(c);
    std::string rep = (it != SUBSCRIPTS.end()) ? it->second : m[1].str();
    result = m.prefix().str() + rep + m.suffix().str();
  }

  // 매핑된 기호들 일괄 치환
  for (auto &[latex, unicode] : SYMBOLS) {
    size_t pos = 0;
    while ((pos = result.find(latex, pos)) != std::string::npos) {
      result.replace(pos, latex.length(), unicode);
      pos += unicode.length();
    }
  }
  // 남은 중괄호 정리
  result = std::regex_replace(result, std::regex(R"(\{([^}]*)\})"), "$1");
  return result;
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
