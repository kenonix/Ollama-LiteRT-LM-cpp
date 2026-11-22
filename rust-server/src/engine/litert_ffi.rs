#![allow(non_camel_case_types, dead_code)]

use libc::{c_char, c_float, c_int, c_void, size_t};

#[repr(C)]
pub struct LiteRtLmEngine {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmEngineSettings {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmSessionConfig {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmSamplerParams {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmRepetitionPenaltyConfig {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmTokenizeResult {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmInputData {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmStreamChunk {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmConversation {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmConversationConfig {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmConversationOptionalArgs {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmJsonResponse {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LiteRtLmThinkingConfig {
    _private: [u8; 0],
}

pub const SAMPLER_TYPE_TOP_P: c_int = 2;
pub const SAMPLER_TYPE_GREEDY: c_int = 3;
pub const INPUT_DATA_TYPE_TEXT: c_int = 0;

pub type LiteRtLmStreamCallback =
    extern "C" fn(callback_data: *mut c_void, chunk: *const LiteRtLmStreamChunk);

extern "C" {
    pub fn litert_lm_engine_settings_create(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings;
    pub fn litert_lm_engine_settings_delete(settings: *mut LiteRtLmEngineSettings);
    pub fn litert_lm_engine_settings_set_max_num_tokens(
        settings: *mut LiteRtLmEngineSettings,
        max_num_tokens: c_int,
    );
    pub fn litert_lm_engine_settings_set_parallel_file_section_loading(
        settings: *mut LiteRtLmEngineSettings,
        parallel: bool,
    );
    pub fn litert_lm_engine_settings_set_use_ringbuffers_local_attention(
        settings: *mut LiteRtLmEngineSettings,
        use_ringbuffers: bool,
    );
    pub fn litert_lm_engine_settings_set_enable_speculative_decoding(
        settings: *mut LiteRtLmEngineSettings,
        enable: bool,
    );
    pub fn litert_lm_engine_settings_set_litert_dispatch_lib_dir(
        settings: *mut LiteRtLmEngineSettings,
        lib_dir: *const c_char,
    );

    pub fn litert_lm_engine_create(settings: *const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine;
    pub fn litert_lm_engine_delete(engine: *mut LiteRtLmEngine);

    pub fn litert_lm_engine_tokenize(
        engine: *mut LiteRtLmEngine,
        text: *const c_char,
    ) -> *mut LiteRtLmTokenizeResult;
    pub fn litert_lm_tokenize_result_delete(result: *mut LiteRtLmTokenizeResult);
    pub fn litert_lm_tokenize_result_get_num_tokens(
        result: *const LiteRtLmTokenizeResult,
    ) -> size_t;

    pub fn litert_lm_input_data_create(
        data_type: c_int,
        data: *const c_void,
        size: size_t,
    ) -> *mut LiteRtLmInputData;
    pub fn litert_lm_input_data_delete(input_data: *mut LiteRtLmInputData);

    pub fn litert_lm_sampler_params_create(sampler_type: c_int) -> *mut LiteRtLmSamplerParams;
    pub fn litert_lm_sampler_params_delete(params: *mut LiteRtLmSamplerParams);
    pub fn litert_lm_sampler_params_set_top_k(params: *mut LiteRtLmSamplerParams, top_k: i32);
    pub fn litert_lm_sampler_params_set_top_p(params: *mut LiteRtLmSamplerParams, top_p: c_float);
    pub fn litert_lm_sampler_params_set_temperature(
        params: *mut LiteRtLmSamplerParams,
        temperature: c_float,
    );
    pub fn litert_lm_sampler_params_set_seed(params: *mut LiteRtLmSamplerParams, seed: i32);

    pub fn litert_lm_session_config_create() -> *mut LiteRtLmSessionConfig;
    pub fn litert_lm_session_config_delete(config: *mut LiteRtLmSessionConfig);
    pub fn litert_lm_session_config_set_max_output_tokens(
        config: *mut LiteRtLmSessionConfig,
        max_output_tokens: c_int,
    );
    pub fn litert_lm_session_config_set_sampler_params(
        config: *mut LiteRtLmSessionConfig,
        sampler_params: *const LiteRtLmSamplerParams,
    );

    pub fn litert_lm_repetition_penalty_config_create() -> *mut LiteRtLmRepetitionPenaltyConfig;
    pub fn litert_lm_repetition_penalty_config_delete(config: *mut LiteRtLmRepetitionPenaltyConfig);
    pub fn litert_lm_repetition_penalty_config_set_presence_penalty(
        config: *mut LiteRtLmRepetitionPenaltyConfig,
        presence_penalty: c_float,
    );
    pub fn litert_lm_repetition_penalty_config_set_frequency_penalty(
        config: *mut LiteRtLmRepetitionPenaltyConfig,
        frequency_penalty: c_float,
    );

    pub fn litert_lm_thinking_config_create() -> *mut LiteRtLmThinkingConfig;
    pub fn litert_lm_thinking_config_delete(config: *mut LiteRtLmThinkingConfig);
    pub fn litert_lm_thinking_config_set_enable_thinking(
        config: *mut LiteRtLmThinkingConfig,
        enable_thinking: bool,
    );

    pub fn litert_lm_conversation_config_create() -> *mut LiteRtLmConversationConfig;
    pub fn litert_lm_conversation_config_delete(config: *mut LiteRtLmConversationConfig);
    pub fn litert_lm_conversation_config_set_session_config(
        config: *mut LiteRtLmConversationConfig,
        session_config: *const LiteRtLmSessionConfig,
    );
    pub fn litert_lm_conversation_config_set_system_message(
        config: *mut LiteRtLmConversationConfig,
        system_message_json: *const c_char,
    );
    pub fn litert_lm_conversation_config_set_messages(
        config: *mut LiteRtLmConversationConfig,
        messages_json: *const c_char,
    );
    pub fn litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
        config: *mut LiteRtLmConversationConfig,
        filter: bool,
    );
    pub fn litert_lm_conversation_config_set_thinking_config(
        config: *mut LiteRtLmConversationConfig,
        thinking_config: *const LiteRtLmThinkingConfig,
    );

    pub fn litert_lm_conversation_optional_args_create() -> *mut LiteRtLmConversationOptionalArgs;
    pub fn litert_lm_conversation_optional_args_delete(args: *mut LiteRtLmConversationOptionalArgs);
    pub fn litert_lm_conversation_optional_args_set_max_output_tokens(
        args: *mut LiteRtLmConversationOptionalArgs,
        max_output_tokens: c_int,
    );
    pub fn litert_lm_conversation_optional_args_set_thinking_config(
        args: *mut LiteRtLmConversationOptionalArgs,
        thinking_config: *const LiteRtLmThinkingConfig,
    );
    pub fn litert_lm_conversation_optional_args_set_repetition_penalty_config(
        args: *mut LiteRtLmConversationOptionalArgs,
        rep_config: *const LiteRtLmRepetitionPenaltyConfig,
    );

    pub fn litert_lm_conversation_create(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmConversationConfig,
    ) -> *mut LiteRtLmConversation;
    pub fn litert_lm_conversation_delete(conversation: *mut LiteRtLmConversation);
    pub fn litert_lm_conversation_send_message(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        optional_args: *const LiteRtLmConversationOptionalArgs,
    ) -> *mut LiteRtLmJsonResponse;
    pub fn litert_lm_json_response_delete(response: *mut LiteRtLmJsonResponse);
    pub fn litert_lm_json_response_get_string(
        response: *const LiteRtLmJsonResponse,
    ) -> *const c_char;
    pub fn litert_lm_conversation_send_message_stream(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        optional_args: *const LiteRtLmConversationOptionalArgs,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> c_int;
    pub fn litert_lm_conversation_cancel_process(conversation: *mut LiteRtLmConversation);
    pub fn litert_lm_conversation_get_token_count(conversation: *mut LiteRtLmConversation)
        -> c_int;

    pub fn litert_lm_stream_chunk_get_text(chunk: *const LiteRtLmStreamChunk) -> *const c_char;
    pub fn litert_lm_stream_chunk_is_final(chunk: *const LiteRtLmStreamChunk) -> bool;
    pub fn litert_lm_stream_chunk_get_error(chunk: *const LiteRtLmStreamChunk) -> *const c_char;
}
