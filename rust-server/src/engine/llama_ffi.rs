#![allow(non_camel_case_types)]

use libc::{c_char, c_float, c_int, c_void, size_t};

#[repr(C)]
pub struct LlamaShimEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaShimGenOpts {
    pub temperature: c_float,
    pub top_p: c_float,
    pub top_k: c_int,
    pub max_tokens: c_int,
    pub seed: c_int,
    pub frequency_penalty: c_float,
    pub presence_penalty: c_float,
    pub stops: *const *const c_char,
    pub num_stops: size_t,
}

pub type LlamaChunkCallback = extern "C" fn(*mut c_void, *const c_char, size_t);
pub type LlamaDoneCallback = extern "C" fn(*mut c_void);
pub type LlamaErrorCallback = extern "C" fn(*mut c_void, *const c_char);
pub type LlamaIsCancelledCallback = extern "C" fn(*mut c_void) -> bool;

extern "C" {
    pub fn llama_shim_create(
        model_path: *const c_char,
        max_tokens: c_int,
        use_gpu: bool,
        n_gpu_layers: c_int,
        enable_mtp: bool,
    ) -> *mut LlamaShimEngine;

    pub fn llama_shim_destroy(eng: *mut LlamaShimEngine);

    pub fn llama_shim_count_tokens(
        eng: *mut LlamaShimEngine,
        text: *const c_char,
        len: size_t,
    ) -> c_int;

    pub fn llama_shim_stream(
        eng: *mut LlamaShimEngine,
        system_msg: *const c_char,
        history_json: *const c_char,
        current_msg: *const c_char,
        opts: *const LlamaShimGenOpts,
        chunk_cb: LlamaChunkCallback,
        done_cb: LlamaDoneCallback,
        error_cb: LlamaErrorCallback,
        is_cancelled: LlamaIsCancelledCallback,
        user_data: *mut c_void,
    );
}
