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
pub struct LiteRtLmSpeculativeDecodingConfig {
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

pub const STATUS_OK: c_int = 0;

pub const SAMPLER_TYPE_TOP_P: c_int = 2;
pub const SAMPLER_TYPE_GREEDY: c_int = 3;
pub const SPECULATIVE_DECODING_MODE_ON: c_int = 1;
pub const INPUT_DATA_TYPE_TEXT: c_int = 0;

pub type LiteRtLmStreamCallback =
    extern "C" fn(callback_data: *mut c_void, chunk: *const LiteRtLmStreamChunk);

extern "C" {
    pub fn litert_lm_get_last_error_message() -> *const c_char;

    pub fn litert_lm_engine_settings_create(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
        out_settings: *mut *mut LiteRtLmEngineSettings,
    ) -> c_int;
    pub fn litert_lm_engine_settings_delete(settings: *mut LiteRtLmEngineSettings);
    pub fn litert_lm_engine_settings_set_max_num_tokens(
        settings: *mut LiteRtLmEngineSettings,
        max_num_tokens: c_int,
    ) -> c_int;
    pub fn litert_lm_engine_settings_set_parallel_file_section_loading(
        settings: *mut LiteRtLmEngineSettings,
        parallel: bool,
    ) -> c_int;
    pub fn litert_lm_engine_settings_set_use_ringbuffers_local_attention(
        settings: *mut LiteRtLmEngineSettings,
        use_ringbuffers: bool,
    ) -> c_int;
    pub fn litert_lm_engine_settings_set_speculative_decoding_config(
        settings: *mut LiteRtLmEngineSettings,
        speculative_decoding_config: *const LiteRtLmSpeculativeDecodingConfig,
    ) -> c_int;
    pub fn litert_lm_engine_settings_set_litert_dispatch_lib_dir(
        settings: *mut LiteRtLmEngineSettings,
        lib_dir: *const c_char,
    ) -> c_int;

    pub fn litert_lm_speculative_decoding_config_create(
        out_config: *mut *mut LiteRtLmSpeculativeDecodingConfig,
    ) -> c_int;
    pub fn litert_lm_speculative_decoding_config_delete(
        config: *mut LiteRtLmSpeculativeDecodingConfig,
    );
    pub fn litert_lm_speculative_decoding_config_set_mode(
        config: *mut LiteRtLmSpeculativeDecodingConfig,
        mode: c_int,
    ) -> c_int;

    pub fn litert_lm_engine_create(
        settings: *const LiteRtLmEngineSettings,
        out_engine: *mut *mut LiteRtLmEngine,
    ) -> c_int;
    pub fn litert_lm_engine_delete(engine: *mut LiteRtLmEngine);

    pub fn litert_lm_engine_tokenize(
        engine: *mut LiteRtLmEngine,
        text: *const c_char,
        out_result: *mut *mut LiteRtLmTokenizeResult,
    ) -> c_int;
    pub fn litert_lm_tokenize_result_delete(result: *mut LiteRtLmTokenizeResult);
    pub fn litert_lm_tokenize_result_get_num_tokens(
        result: *const LiteRtLmTokenizeResult,
        out_num_tokens: *mut size_t,
    ) -> c_int;

    pub fn litert_lm_input_data_create(
        data_type: c_int,
        data: *const c_void,
        size: size_t,
        out_input_data: *mut *mut LiteRtLmInputData,
    ) -> c_int;
    pub fn litert_lm_input_data_delete(input_data: *mut LiteRtLmInputData);

    pub fn litert_lm_sampler_params_create(
        sampler_type: c_int,
        out_params: *mut *mut LiteRtLmSamplerParams,
    ) -> c_int;
    pub fn litert_lm_sampler_params_delete(params: *mut LiteRtLmSamplerParams);
    pub fn litert_lm_sampler_params_set_top_k(
        params: *mut LiteRtLmSamplerParams,
        top_k: i32,
    ) -> c_int;
    pub fn litert_lm_sampler_params_set_top_p(
        params: *mut LiteRtLmSamplerParams,
        top_p: c_float,
    ) -> c_int;
    pub fn litert_lm_sampler_params_set_temperature(
        params: *mut LiteRtLmSamplerParams,
        temperature: c_float,
    ) -> c_int;
    pub fn litert_lm_sampler_params_set_seed(
        params: *mut LiteRtLmSamplerParams,
        seed: i32,
    ) -> c_int;

    pub fn litert_lm_session_config_create(
        out_config: *mut *mut LiteRtLmSessionConfig,
    ) -> c_int;
    pub fn litert_lm_session_config_delete(config: *mut LiteRtLmSessionConfig);
    pub fn litert_lm_session_config_set_max_output_tokens(
        config: *mut LiteRtLmSessionConfig,
        max_output_tokens: c_int,
    ) -> c_int;
    pub fn litert_lm_session_config_set_sampler_params(
        config: *mut LiteRtLmSessionConfig,
        sampler_params: *const LiteRtLmSamplerParams,
    ) -> c_int;

    pub fn litert_lm_repetition_penalty_config_create(
        out_config: *mut *mut LiteRtLmRepetitionPenaltyConfig,
    ) -> c_int;
    pub fn litert_lm_repetition_penalty_config_delete(config: *mut LiteRtLmRepetitionPenaltyConfig);
    pub fn litert_lm_repetition_penalty_config_set_repetition_penalty(
        config: *mut LiteRtLmRepetitionPenaltyConfig,
        repetition_penalty: c_float,
    ) -> c_int;
    pub fn litert_lm_repetition_penalty_config_set_presence_penalty(
        config: *mut LiteRtLmRepetitionPenaltyConfig,
        presence_penalty: c_float,
    ) -> c_int;
    pub fn litert_lm_repetition_penalty_config_set_frequency_penalty(
        config: *mut LiteRtLmRepetitionPenaltyConfig,
        frequency_penalty: c_float,
    ) -> c_int;

    pub fn litert_lm_thinking_config_create(
        out_config: *mut *mut LiteRtLmThinkingConfig,
    ) -> c_int;
    pub fn litert_lm_thinking_config_delete(config: *mut LiteRtLmThinkingConfig);
    pub fn litert_lm_thinking_config_set_enable_thinking(
        config: *mut LiteRtLmThinkingConfig,
        enable_thinking: bool,
    ) -> c_int;

    pub fn litert_lm_conversation_config_create(
        out_config: *mut *mut LiteRtLmConversationConfig,
    ) -> c_int;
    pub fn litert_lm_conversation_config_delete(config: *mut LiteRtLmConversationConfig);
    pub fn litert_lm_conversation_config_set_session_config(
        config: *mut LiteRtLmConversationConfig,
        session_config: *const LiteRtLmSessionConfig,
    ) -> c_int;
    pub fn litert_lm_conversation_config_set_system_message(
        config: *mut LiteRtLmConversationConfig,
        system_message_json: *const c_char,
    ) -> c_int;
    pub fn litert_lm_conversation_config_set_messages(
        config: *mut LiteRtLmConversationConfig,
        messages_json: *const c_char,
    ) -> c_int;
    pub fn litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
        config: *mut LiteRtLmConversationConfig,
        filter: bool,
    ) -> c_int;
    pub fn litert_lm_conversation_config_set_thinking_config(
        config: *mut LiteRtLmConversationConfig,
        thinking_config: *const LiteRtLmThinkingConfig,
    ) -> c_int;

    pub fn litert_lm_conversation_optional_args_create(
        out_optional_args: *mut *mut LiteRtLmConversationOptionalArgs,
    ) -> c_int;
    pub fn litert_lm_conversation_optional_args_delete(args: *mut LiteRtLmConversationOptionalArgs);
    pub fn litert_lm_conversation_optional_args_set_max_output_tokens(
        args: *mut LiteRtLmConversationOptionalArgs,
        max_output_tokens: c_int,
    ) -> c_int;
    pub fn litert_lm_conversation_optional_args_set_thinking_config(
        args: *mut LiteRtLmConversationOptionalArgs,
        thinking_config: *const LiteRtLmThinkingConfig,
    ) -> c_int;
    pub fn litert_lm_conversation_optional_args_set_repetition_penalty_config(
        args: *mut LiteRtLmConversationOptionalArgs,
        rep_config: *const LiteRtLmRepetitionPenaltyConfig,
    ) -> c_int;

    pub fn litert_lm_conversation_create(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmConversationConfig,
        out_conversation: *mut *mut LiteRtLmConversation,
    ) -> c_int;
    pub fn litert_lm_conversation_delete(conversation: *mut LiteRtLmConversation);
    pub fn litert_lm_conversation_send_message(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        optional_args: *const LiteRtLmConversationOptionalArgs,
        out_response: *mut *mut LiteRtLmJsonResponse,
    ) -> c_int;
    pub fn litert_lm_json_response_delete(response: *mut LiteRtLmJsonResponse);
    pub fn litert_lm_json_response_get_string(
        response: *const LiteRtLmJsonResponse,
        out_json: *mut *const c_char,
    ) -> c_int;
    pub fn litert_lm_conversation_send_message_stream(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        optional_args: *const LiteRtLmConversationOptionalArgs,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> c_int;
    pub fn litert_lm_conversation_render_preface_to_string(
        conversation: *mut LiteRtLmConversation,
        out_rendered: *mut *const c_char,
    ) -> c_int;
    pub fn litert_lm_conversation_cancel_process(conversation: *mut LiteRtLmConversation) -> c_int;
    pub fn litert_lm_conversation_get_token_count(
        conversation: *mut LiteRtLmConversation,
        out_count: *mut size_t,
    ) -> c_int;

    pub fn litert_lm_stream_chunk_get_text(
        chunk: *const LiteRtLmStreamChunk,
        out_text: *mut *const c_char,
    ) -> c_int;
    pub fn litert_lm_stream_chunk_is_final(
        chunk: *const LiteRtLmStreamChunk,
        out_is_final: *mut bool,
    ) -> c_int;
    pub fn litert_lm_stream_chunk_get_error(
        chunk: *const LiteRtLmStreamChunk,
        out_error: *mut *const c_char,
    ) -> c_int;
}
