#include "embedding_engine.h"
#include "c/engine.h"
#include <iostream>
#include <fstream>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

// LiteRT-LM C API 선언 (헤더 미포함 심볼 정의)
extern "C" {
typedef struct LiteRtLmEmbeddingEngine LiteRtLmEmbeddingEngine;
typedef struct LiteRtLmEmbeddingEngineSettings LiteRtLmEmbeddingEngineSettings;
typedef struct LiteRtLmEmbeddingOptions LiteRtLmEmbeddingOptions;
typedef struct LiteRtLmEmbeddingResponse LiteRtLmEmbeddingResponse;

LiteRtLmEmbeddingEngineSettings* litert_lm_embedding_engine_settings_create(
    const char* model_path,
    const char* backend,
    const char* vision_backend,
    const char* audio_backend
);
void litert_lm_embedding_engine_settings_delete(LiteRtLmEmbeddingEngineSettings* settings);
void litert_lm_embedding_engine_settings_set_cache_dir(LiteRtLmEmbeddingEngineSettings* settings, const char* cache_dir);
void litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(LiteRtLmEmbeddingEngineSettings* settings, const char* dir);

LiteRtLmEmbeddingEngine* litert_lm_embedding_engine_create(LiteRtLmEmbeddingEngineSettings* settings);
void litert_lm_embedding_engine_delete(LiteRtLmEmbeddingEngine* engine);

LiteRtLmEmbeddingOptions* litert_lm_embedding_options_create();
void litert_lm_embedding_options_delete(LiteRtLmEmbeddingOptions* options);
void litert_lm_embedding_options_set_normalize(LiteRtLmEmbeddingOptions* options, bool normalize);
void litert_lm_embedding_options_set_output_size(LiteRtLmEmbeddingOptions* options, int size);

LiteRtLmEmbeddingResponse* litert_lm_embedding_engine_compute_embedding(
    LiteRtLmEmbeddingEngine* engine,
    LiteRtLmInputData** input_data_array,
    size_t num_input_data,
    LiteRtLmEmbeddingOptions* options
);
void litert_lm_embedding_response_delete(LiteRtLmEmbeddingResponse* response);
size_t litert_lm_embedding_response_get_size(LiteRtLmEmbeddingResponse* response);
const float* litert_lm_embedding_response_get_values(LiteRtLmEmbeddingResponse* response);
}

static bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 1000);
}

struct EmbeddingEngineWrapper::Impl {
    LiteRtLmEmbeddingEngine* engine = nullptr;
    std::mutex mutex;
    std::string model_path;

    ~Impl() {
        std::lock_guard<std::mutex> lock(mutex);
        if (engine) {
            litert_lm_embedding_engine_delete(engine);
            engine = nullptr;
        }
    }
};

EmbeddingEngineWrapper::EmbeddingEngineWrapper()
    : impl_(std::make_unique<Impl>()) {}

EmbeddingEngineWrapper::~EmbeddingEngineWrapper() = default;

std::string EmbeddingEngineWrapper::EnsureModelAvailable(const std::string &preferred_path) {
    if (file_exists(preferred_path)) {
        return preferred_path;
    }

    std::vector<std::string> search_candidates = {
        "./embeddinggemma-2-text-270m.litertlm",
        "../embeddinggemma-2-text-270m.litertlm",
        "./models/embeddinggemma-2-text-270m.litertlm",
        "/home/kenonix/gits/Ollama-LiteRT-LM-cpp/embeddinggemma-2-text-270m.litertlm"
    };

    for (const auto &cand : search_candidates) {
        if (file_exists(cand)) {
            std::cout << "[임베딩] 기존 모델 파일 발견: " << cand << std::endl;
            return cand;
        }
    }

    // 모델 파일이 없는 경우 자동 다운로드 시도
    std::string target_path = "./embeddinggemma-2-text-270m.litertlm";
    std::string download_url = "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm";

    std::cout << "[임베딩] 모델 파일이 없습니다. HuggingFace에서 자동 다운로드 시작: " << download_url << std::endl;
    std::string curl_cmd = "curl -L -f --progress-bar -o \"" + target_path + ".tmp\" \"" + download_url + "\" && mv \"" + target_path + ".tmp\" \"" + target_path + "\"";
    
    int ret = std::system(curl_cmd.c_str());
    if (ret == 0 && file_exists(target_path)) {
        std::cout << "[임베딩] EmbeddingGemma 2 모델 자동 다운로드 완료: " << target_path << std::endl;
        return target_path;
    }

    std::cerr << "[임베딩] 자동 다운로드 실패. 수동으로 모델을 다운로드해주세요." << std::endl;
    return "";
}

bool EmbeddingEngineWrapper::Initialize(const std::string &model_path, const std::string &backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->engine) {
        litert_lm_embedding_engine_delete(impl_->engine);
        impl_->engine = nullptr;
    }

    if (!file_exists(model_path)) {
        std::cerr << "[임베딩] 모델 파일을 찾을 수 없습니다: " << model_path << std::endl;
        return false;
    }

    std::cout << "[임베딩] 엔진 초기화 중 (백엔드: " << backend << ", 모델: " << model_path << ")..." << std::endl;

    auto* settings = litert_lm_embedding_engine_settings_create(
        model_path.c_str(),
        backend.c_str(),
        nullptr,
        nullptr
    );
    if (!settings) {
        std::cerr << "[임베딩] 설정 생성 실패" << std::endl;
        return false;
    }

    litert_lm_embedding_engine_settings_set_cache_dir(settings, ".");

    impl_->engine = litert_lm_embedding_engine_create(settings);
    litert_lm_embedding_engine_settings_delete(settings);

    if (!impl_->engine) {
        std::cerr << "[임베딩] LiteRT-LM EmbeddingEngine 생성 실패" << std::endl;
        return false;
    }

    impl_->model_path = model_path;
    std::cout << "[임베딩] EmbeddingGemma 2 엔진 준비 완료! (768 차원 벡터 지원)" << std::endl;
    return true;
}

bool EmbeddingEngineWrapper::IsLoaded() const {
    return impl_ && impl_->engine != nullptr;
}

std::string EmbeddingEngineWrapper::GetModelPath() const {
    return impl_ ? impl_->model_path : "";
}

std::vector<float> EmbeddingEngineWrapper::ComputeEmbedding(const std::string &text, bool normalize, int output_size) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->engine) {
        return {};
    }

    std::string input_text = text;
    if (input_text.empty()) {
        input_text = " ";
    }

    LiteRtLmInputData* input_data = litert_lm_input_data_create(
        kLiteRtLmInputDataTypeText,
        input_text.data(),
        input_text.size()
    );
    if (!input_data) {
        return {};
    }

    auto* options = litert_lm_embedding_options_create();
    if (!options) {
        litert_lm_input_data_delete(input_data);
        return {};
    }

    litert_lm_embedding_options_set_normalize(options, normalize);
    if (output_size > 0) {
        litert_lm_embedding_options_set_output_size(options, output_size);
    }

    LiteRtLmInputData* inputs[] = { input_data };
    auto* resp = litert_lm_embedding_engine_compute_embedding(impl_->engine, inputs, 1, options);

    std::vector<float> result;
    if (resp) {
        size_t dim = litert_lm_embedding_response_get_size(resp);
        const float* vals = litert_lm_embedding_response_get_values(resp);
        if (dim > 0 && vals != nullptr) {
            result.assign(vals, vals + dim);
        }
        litert_lm_embedding_response_delete(resp);
    }

    litert_lm_embedding_options_delete(options);
    litert_lm_input_data_delete(input_data);

    return result;
}

std::vector<std::vector<float>> EmbeddingEngineWrapper::ComputeEmbeddingBatch(const std::vector<std::string> &texts, bool normalize, int output_size) {
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    for (const auto &text : texts) {
        results.push_back(ComputeEmbedding(text, normalize, output_size));
    }
    return results;
}
