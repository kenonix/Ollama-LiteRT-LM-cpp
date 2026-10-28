#include "embedding_engine.h"
#include "c/engine.h"
#include <iostream>
#include <fstream>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <dlfcn.h>

// LiteRT-LM C API 타입 및 동적 함수 포인터 정의 (다양한 플랫폼 및 구버전 라이브러리 링크 호환성 보장)
extern "C" {
typedef struct LiteRtLmEmbeddingEngine LiteRtLmEmbeddingEngine;
typedef struct LiteRtLmEmbeddingEngineSettings LiteRtLmEmbeddingEngineSettings;
typedef struct LiteRtLmEmbeddingOptions LiteRtLmEmbeddingOptions;
typedef struct LiteRtLmEmbeddingResponse LiteRtLmEmbeddingResponse;

typedef LiteRtLmEmbeddingEngineSettings* (*Fn_settings_create)(const char*, const char*, const char*, const char*);
typedef void (*Fn_settings_delete)(LiteRtLmEmbeddingEngineSettings*);
typedef void (*Fn_settings_set_cache_dir)(LiteRtLmEmbeddingEngineSettings*, const char*);
typedef void (*Fn_settings_set_litert_dispatch_lib_dir)(LiteRtLmEmbeddingEngineSettings*, const char*);

typedef LiteRtLmEmbeddingEngine* (*Fn_engine_create)(LiteRtLmEmbeddingEngineSettings*);
typedef void (*Fn_engine_delete)(LiteRtLmEmbeddingEngine*);

typedef LiteRtLmEmbeddingOptions* (*Fn_options_create)();
typedef void (*Fn_options_delete)(LiteRtLmEmbeddingOptions*);
typedef void (*Fn_options_set_normalize)(LiteRtLmEmbeddingOptions*, bool);
typedef void (*Fn_options_set_output_size)(LiteRtLmEmbeddingOptions*, int);

typedef LiteRtLmEmbeddingResponse* (*Fn_compute_embedding)(
    LiteRtLmEmbeddingEngine*,
    LiteRtLmInputData**,
    size_t,
    LiteRtLmEmbeddingOptions*
);
typedef void (*Fn_response_delete)(LiteRtLmEmbeddingResponse*);
typedef size_t (*Fn_response_get_size)(LiteRtLmEmbeddingResponse*);
typedef const float* (*Fn_response_get_values)(LiteRtLmEmbeddingResponse*);
}

struct EmbeddingApiSymbols {
    Fn_settings_create settings_create = nullptr;
    Fn_settings_delete settings_delete = nullptr;
    Fn_settings_set_cache_dir settings_set_cache_dir = nullptr;
    Fn_settings_set_litert_dispatch_lib_dir settings_set_litert_dispatch_lib_dir = nullptr;
    Fn_engine_create engine_create = nullptr;
    Fn_engine_delete engine_delete = nullptr;
    Fn_options_create options_create = nullptr;
    Fn_options_delete options_delete = nullptr;
    Fn_options_set_normalize options_set_normalize = nullptr;
    Fn_options_set_output_size options_set_output_size = nullptr;
    Fn_compute_embedding compute_embedding = nullptr;
    Fn_response_delete response_delete = nullptr;
    Fn_response_get_size response_get_size = nullptr;
    Fn_response_get_values response_get_values = nullptr;

    bool is_available = false;

    static EmbeddingApiSymbols& Get() {
        static EmbeddingApiSymbols instance;
        return instance;
    }

private:
    EmbeddingApiSymbols() {
        void* handle = RTLD_DEFAULT;
        settings_create = (Fn_settings_create)dlsym(handle, "litert_lm_embedding_engine_settings_create");
        settings_delete = (Fn_settings_delete)dlsym(handle, "litert_lm_embedding_engine_settings_delete");
        settings_set_cache_dir = (Fn_settings_set_cache_dir)dlsym(handle, "litert_lm_embedding_engine_settings_set_cache_dir");
        settings_set_litert_dispatch_lib_dir = (Fn_settings_set_litert_dispatch_lib_dir)dlsym(handle, "litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir");
        engine_create = (Fn_engine_create)dlsym(handle, "litert_lm_embedding_engine_create");
        engine_delete = (Fn_engine_delete)dlsym(handle, "litert_lm_embedding_engine_delete");
        options_create = (Fn_options_create)dlsym(handle, "litert_lm_embedding_options_create");
        options_delete = (Fn_options_delete)dlsym(handle, "litert_lm_embedding_options_delete");
        options_set_normalize = (Fn_options_set_normalize)dlsym(handle, "litert_lm_embedding_options_set_normalize");
        options_set_output_size = (Fn_options_set_output_size)dlsym(handle, "litert_lm_embedding_options_set_output_size");
        compute_embedding = (Fn_compute_embedding)dlsym(handle, "litert_lm_embedding_engine_compute_embedding");
        response_delete = (Fn_response_delete)dlsym(handle, "litert_lm_embedding_response_delete");
        response_get_size = (Fn_response_get_size)dlsym(handle, "litert_lm_embedding_response_get_size");
        response_get_values = (Fn_response_get_values)dlsym(handle, "litert_lm_embedding_response_get_values");

        is_available = (settings_create && settings_delete && engine_create && engine_delete &&
                        options_create && options_delete && compute_embedding &&
                        response_delete && response_get_size && response_get_values);
    }
};

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
            auto &api = EmbeddingApiSymbols::Get();
            if (api.engine_delete) {
                api.engine_delete(engine);
            }
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
    auto &api = EmbeddingApiSymbols::Get();

    if (!api.is_available) {
        std::cerr << "[임베딩] 현재 라이브러리에서 LiteRT-LM Embedding C API를 지원하지 않습니다." << std::endl;
        return false;
    }

    if (impl_->engine) {
        api.engine_delete(impl_->engine);
        impl_->engine = nullptr;
    }

    if (!file_exists(model_path)) {
        std::cerr << "[임베딩] 모델 파일을 찾을 수 없습니다: " << model_path << std::endl;
        return false;
    }

    std::cout << "[임베딩] 엔진 초기화 중 (백엔드: " << backend << ", 모델: " << model_path << ")..." << std::endl;

    auto* settings = api.settings_create(
        model_path.c_str(),
        backend.c_str(),
        nullptr,
        nullptr
    );
    if (!settings) {
        std::cerr << "[임베딩] 설정 생성 실패" << std::endl;
        return false;
    }

    if (api.settings_set_cache_dir) {
        api.settings_set_cache_dir(settings, ".");
    }

    impl_->engine = api.engine_create(settings);
    api.settings_delete(settings);

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

    auto &api = EmbeddingApiSymbols::Get();
    if (!api.is_available) {
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

    auto* options = api.options_create();
    if (!options) {
        litert_lm_input_data_delete(input_data);
        return {};
    }

    api.options_set_normalize(options, normalize);
    if (output_size > 0 && api.options_set_output_size) {
        api.options_set_output_size(options, output_size);
    }

    LiteRtLmInputData* inputs[] = { input_data };
    auto* resp = api.compute_embedding(impl_->engine, inputs, 1, options);

    std::vector<float> result;
    if (resp) {
        size_t dim = api.response_get_size(resp);
        const float* vals = api.response_get_values(resp);
        if (dim > 0 && vals != nullptr) {
            result.assign(vals, vals + dim);
        }
        api.response_delete(resp);
    }

    api.options_delete(options);
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
