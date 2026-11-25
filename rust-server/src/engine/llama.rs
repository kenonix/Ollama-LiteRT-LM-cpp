use super::llama_ffi::*;
use super::{LlmEngine, StreamEvent};
use crate::config::GenerationOptions;
use anyhow::{anyhow, Result};
use libc::{c_char, c_void, size_t};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

struct SendLlamaPtr(*mut LlamaShimEngine);
unsafe impl Send for SendLlamaPtr {}
unsafe impl Sync for SendLlamaPtr {}

pub struct LlamaEngine {
    eng: SendLlamaPtr,
    max_tokens: usize,
}

impl LlamaEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_path: &str,
        draft_model_path: Option<&str>,
        device: Option<&str>,
        max_tokens: usize,
        use_gpu: bool,
        n_gpu_layers: i32,
        enable_mtp: bool,
        cpu_moe: bool,
        op_offload: bool,
        n_threads: i32,
    ) -> Result<Self> {
        println!(
            "[llama.cpp] GGUF 모델 로딩 중: {} (GPU: {}, Device: {}, Draft: {}, MTP: {}, CPU-MoE: {}, Max Tokens: {})...",
            model_path,
            if use_gpu { "ON" } else { "OFF" },
            device.unwrap_or("auto"),
            draft_model_path.unwrap_or("none"),
            if enable_mtp { "ON" } else { "OFF" },
            if cpu_moe { "ON" } else { "OFF" },
            max_tokens
        );
        let path_c = CString::new(model_path)?;
        let draft_c = CString::new(draft_model_path.unwrap_or(""))?;
        let dev_c = CString::new(device.unwrap_or(""))?;
        let ptr = unsafe {
            llama_shim_create(
                path_c.as_ptr(),
                draft_c.as_ptr(),
                dev_c.as_ptr(),
                max_tokens as i32,
                use_gpu,
                n_gpu_layers,
                enable_mtp,
                cpu_moe,
                op_offload,
                n_threads,
            )
        };
        if ptr.is_null() {
            return Err(anyhow!("llama.cpp 모델 로드 실패: {}", model_path));
        }
        println!("[llama.cpp] GGUF 모델 준비 완료!");
        Ok(Self {
            eng: SendLlamaPtr(ptr),
            max_tokens,
        })
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        if !self.eng.0.is_null() {
            unsafe { llama_shim_destroy(self.eng.0) };
            self.eng.0 = ptr::null_mut();
        }
    }
}

struct LlamaStreamUserData {
    tx: mpsc::UnboundedSender<StreamEvent>,
    cancelled: Arc<AtomicBool>,
}

extern "C" fn on_chunk(ud: *mut c_void, data: *const c_char, len: size_t) {
    if ud.is_null() || data.is_null() || len == 0 {
        return;
    }
    let ctx = unsafe { &*(ud as *const LlamaStreamUserData) };
    let slice = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
    let _ = ctx.tx.send(StreamEvent::Chunk(slice.to_vec()));
}

extern "C" fn on_done(ud: *mut c_void) {
    if ud.is_null() {
        return;
    }
    let ctx = unsafe { &*(ud as *const LlamaStreamUserData) };
    let _ = ctx.tx.send(StreamEvent::Done);
}

extern "C" fn on_error(ud: *mut c_void, err: *const c_char) {
    if ud.is_null() {
        return;
    }
    let ctx = unsafe { &*(ud as *const LlamaStreamUserData) };
    let msg = if err.is_null() {
        "llama.cpp 추론 오류".to_string()
    } else {
        unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() }
    };
    let _ = ctx.tx.send(StreamEvent::Error(msg));
}

extern "C" fn on_is_cancelled(ud: *mut c_void) -> bool {
    if ud.is_null() {
        return false;
    }
    let ctx = unsafe { &*(ud as *const LlamaStreamUserData) };
    ctx.cancelled.load(Ordering::Relaxed)
}

impl LlmEngine for LlamaEngine {
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn supports_system_role(&self) -> bool {
        true
    }

    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() || self.eng.0.is_null() {
            return 0;
        }
        let n = unsafe {
            llama_shim_count_tokens(self.eng.0, text.as_ptr() as *const c_char, text.len())
        };
        n.max(0) as usize
    }

    fn generate(
        &self,
        system_msg: &str,
        history_json: &str,
        current_msg: &str,
        opts: &GenerationOptions,
    ) -> String {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        self.stream(system_msg, history_json, current_msg, opts, tx, cancelled);
        let mut out_bytes = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Chunk(b) = ev {
                out_bytes.extend_from_slice(&b);
            }
        }
        String::from_utf8_lossy(&out_bytes).into_owned()
    }

    fn stream(
        &self,
        system_msg: &str,
        history_json: &str,
        current_msg: &str,
        opts: &GenerationOptions,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancelled: Arc<AtomicBool>,
    ) {
        let sys_c = CString::new(system_msg).unwrap_or_default();
        let hist_c = CString::new(history_json).unwrap_or_default();
        let curr_c = CString::new(current_msg).unwrap_or_default();

        let stop_cstrings: Vec<CString> = opts
            .stop
            .iter()
            .filter_map(|s| CString::new(s.as_str()).ok())
            .collect();
        let stop_ptrs: Vec<*const c_char> = stop_cstrings.iter().map(|c| c.as_ptr()).collect();

        let c_opts = LlamaShimGenOpts {
            temperature: opts.temperature,
            top_p: opts.top_p,
            top_k: opts.top_k,
            max_tokens: opts.max_tokens,
            seed: opts.seed,
            frequency_penalty: opts.frequency_penalty,
            presence_penalty: opts.presence_penalty,
            stops: if stop_ptrs.is_empty() {
                ptr::null()
            } else {
                stop_ptrs.as_ptr()
            },
            num_stops: stop_ptrs.len(),
        };

        let mut ud = LlamaStreamUserData { tx, cancelled };
        unsafe {
            llama_shim_stream(
                self.eng.0,
                sys_c.as_ptr(),
                hist_c.as_ptr(),
                curr_c.as_ptr(),
                &c_opts,
                on_chunk,
                on_done,
                on_error,
                on_is_cancelled,
                &mut ud as *mut LlamaStreamUserData as *mut c_void,
            );
        }
    }
}
