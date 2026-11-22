#pragma once

#include "llm_engine.h"
#include "embedding_engine.h"
#include <string>

/**
 * @brief Ollama 호환 HTTP API 서버를 시작합니다.
 * 
 * @param app ILlmEngine 인터페이스 참조 (LiteRT-LM 또는 llama.cpp GGUF)
 * @param port 서버가 대기할 포트 번호
 * @param served_model_name API 응답에서 보고할 모델명
 * @param emb_engine EmbeddingEngineWrapper 포인터 (선택)
 */
void RunServer(ILlmEngine &app, int port, const std::string &served_model_name, EmbeddingEngineWrapper *emb_engine = nullptr);
