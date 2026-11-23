pub const PROMPT_FILE: &str = "system_prompt.txt";

pub const DEFAULT_SYSTEM_PROMPT: &str = "당신은 한국어로 소통하는 유능하고 친절한 AI 어시스턴트입니다.\n\
반드시 자연스럽고 정확한 한국어 표준어로만 답변하며, 사용자의 질문 의도에 맞게 핵심을 명확하게 설명합니다.";

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_tokens: i32,
    pub seed: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub stop: Vec<String>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: 0.4,
            top_p: 0.90,
            top_k: 40,
            max_tokens: -1,
            seed: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            stop: Vec::new(),
        }
    }
}
