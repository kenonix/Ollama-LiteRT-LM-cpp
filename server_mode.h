#pragma once

#include "llm_engine.h"
#include <string>

/**
 * @brief Ollama 호환 HTTP API 서버를 시작합니다.
 * 
 * @param app MultimodalCliApp 인스턴스 참조
 * @param port 서버가 대기할 포트 번호
 * @param served_model_name API 응답에서 보고할 모델명
 */
void RunServer(MultimodalCliApp &app, int port, const std::string &served_model_name);
