#pragma once

#include <string>
#include <vector>
#include <memory>

/**
 * @brief LiteRT-LM Embedding Engine C++ 래퍼 클래스
 * Google EmbeddingGemma 2 모델 (768 차원 L2 정규화 임베딩)을 지원합니다.
 */
class EmbeddingEngineWrapper {
public:
    EmbeddingEngineWrapper();
    ~EmbeddingEngineWrapper();

    /**
     * @brief 임베딩 모델 파일이 존재하지 않는 경우 HuggingFace에서 자동 다운로드
     * @param preferred_path 우선 확인할 경로 (비어있으면 기본 경로)
     * @return 유효한 모델 파일 경로 (실패 시 빈 문자열)
     */
    static std::string EnsureModelAvailable(const std::string &preferred_path = "");

    /**
     * @brief 엔진 초기화
     * @param model_path .litertlm 임베딩 모델 파일 경로
     * @param backend "cpu" (권장, LLM GPU VRAM 보존 및 25ms 고속) 또는 "gpu"
     * @return 성공 여부
     */
    bool Initialize(const std::string &model_path, const std::string &backend = "cpu");

    /**
     * @brief 모델이 정상 로드되었는지 확인
     */
    bool IsLoaded() const;

    /**
     * @brief 로드된 모델 경로 반환
     */
    std::string GetModelPath() const;

    /**
     * @brief 단일 텍스트 임베딩 계산
     * @param text 입력 텍스트
     * @param normalize L2 정규화 여부 (기본 true)
     * @param output_size 출력 차원 (기본 0: 768)
     * @return 768차원 부동소수점 벡터
     */
    std::vector<float> ComputeEmbedding(const std::string &text, bool normalize = true, int output_size = 0);

    /**
     * @brief 배치 텍스트 임베딩 계산
     * @param texts 입력 텍스트 벡터
     * @param normalize L2 정규화 여부 (기본 true)
     * @param output_size 출력 차원 (기본 0: 768)
     * @return 각 텍스트에 대한 임베딩 벡터 목록
     */
    std::vector<std::vector<float>> ComputeEmbeddingBatch(const std::vector<std::string> &texts, bool normalize = true, int output_size = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
