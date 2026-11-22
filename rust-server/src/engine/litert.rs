use super::litert_ffi::*;
use super::{LlmEngine, StreamEvent};
use crate::config::GenerationOptions;
use anyhow::{anyhow, Result};
use libc::c_void;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

pub fn extract_text_from_chunk(chunk: &str) -> String {
    if chunk.is_empty() {
        return String::new();
    }
    if let Ok(j) = serde_json::from_str::<Value>(chunk) {
        if let Some(content) = j.get("content") {
            if let Some(s) = content.as_str() {
                return s.to_string();
            }
            if let Some(arr) = content.as_array() {
                let mut result = String::new();
                for part in arr {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        result.push_str(t);
                    }
                }
                return result;
            }
        }
        if j.is_object() {
            return String::new();
        }
    }
    chunk.to_string()
}

fn find_dispatch_dir() -> String {
    let candidates = [
        #[cfg(target_arch = "aarch64")]
        "./lib/aarch64",
        #[cfg(target_arch = "aarch64")]
        "../lib/aarch64",
        #[cfg(target_arch = "x86_64")]
        "./lib/x86_64",
        #[cfg(target_arch = "x86_64")]
        "../lib/x86_64",
        "./lib",
        "../lib",
    ];
    for c in candidates {
        if Path::new(c).exists() {
            return c.to_string();
        }
    }
    "./lib".to_string()
}

fn litertlm_has_audio_section(model_path: &str) -> bool {
    let Ok(mut f) = File::open(model_path) else {
        return false;
    };
    let mut buf = vec![0u8; 65536];
    let n = f.read(&mut buf).unwrap_or(0);
    buf.truncate(n);
    let needle = b"tf_lite_audio_encoder";
    buf.windows(needle.len()).any(|w| w == needle)
}

struct SendEnginePtr(*mut LiteRtLmEngine);
unsafe impl Send for SendEnginePtr {}
unsafe impl Sync for SendEnginePtr {}

pub struct LiteRtEngine {
    engine: SendEnginePtr,
    max_tokens: usize,
    audio_enabled: bool,
    infer_mutex: Mutex<()>,
    tokenizer_mutex: Mutex<()>,
}

impl LiteRtEngine {
    pub fn new(
        model_path: &str,
        use_gpu: bool,
        max_tokens: usize,
        enable_mtp: bool,
    ) -> Result<Self> {
        println!(
            "[시스템] LiteRT-LM 모델 로딩 중 ({}, Max Tokens: {}, MTP: {})...",
            if use_gpu { "GPU 가속" } else { "CPU" },
            max_tokens,
            if enable_mtp { "ON" } else { "OFF" }
        );

        let dispatch_dir = find_dispatch_dir();
        let backend_c = CString::new(if use_gpu { "gpu" } else { "cpu" })?;
        let model_c = CString::new(model_path)?;
        let dispatch_c = CString::new(dispatch_dir.as_str())?;

        let create_engine = |audio_backend: Option<&str>| -> *mut LiteRtLmEngine {
            let audio_c = audio_backend.and_then(|s| CString::new(s).ok());
            let audio_ptr = audio_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
            unsafe {
                let settings = litert_lm_engine_settings_create(
                    model_c.as_ptr(),
                    backend_c.as_ptr(),
                    backend_c.as_ptr(),
                    audio_ptr,
                );
                if settings.is_null() {
                    return ptr::null_mut();
                }
                litert_lm_engine_settings_set_max_num_tokens(settings, max_tokens as i32);
                litert_lm_engine_settings_set_parallel_file_section_loading(settings, false);
                litert_lm_engine_settings_set_use_ringbuffers_local_attention(settings, true);
                if enable_mtp {
                    litert_lm_engine_settings_set_enable_speculative_decoding(settings, true);
                }
                litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, dispatch_c.as_ptr());
                let eng = litert_lm_engine_create(settings);
                litert_lm_engine_settings_delete(settings);
                eng
            }
        };

        let run_warmup = |eng: *mut LiteRtLmEngine| -> bool {
            if eng.is_null() {
                return false;
            }
            unsafe {
                let warmup_sc = litert_lm_session_config_create();
                if !warmup_sc.is_null() {
                    litert_lm_session_config_set_max_output_tokens(warmup_sc, 1);
                }
                let warmup_cc = litert_lm_conversation_config_create();
                if !warmup_cc.is_null() {
                    if !warmup_sc.is_null() {
                        litert_lm_conversation_config_set_session_config(warmup_cc, warmup_sc);
                    }
                    let t_cfg = litert_lm_thinking_config_create();
                    if !t_cfg.is_null() {
                        litert_lm_thinking_config_set_enable_thinking(t_cfg, false);
                        litert_lm_conversation_config_set_thinking_config(warmup_cc, t_cfg);
                        litert_lm_thinking_config_delete(t_cfg);
                    }
                }
                if !warmup_sc.is_null() {
                    litert_lm_session_config_delete(warmup_sc);
                }
                let conv = litert_lm_conversation_create(eng, warmup_cc);
                if !warmup_cc.is_null() {
                    litert_lm_conversation_config_delete(warmup_cc);
                }
                if conv.is_null() {
                    return false;
                }
                let opt = litert_lm_conversation_optional_args_create();
                if !opt.is_null() {
                    litert_lm_conversation_optional_args_set_max_output_tokens(opt, 1);
                }
                let hi = CString::new(r#"{"role":"user","content":"hi"}"#).unwrap();
                let res = litert_lm_conversation_send_message(conv, hi.as_ptr(), ptr::null(), opt);
                if !res.is_null() {
                    litert_lm_json_response_delete(res);
                }
                if !opt.is_null() {
                    litert_lm_conversation_optional_args_delete(opt);
                }
                litert_lm_conversation_delete(conv);
                true
            }
        };

        let mut engine_ptr = ptr::null_mut();
        let mut audio_enabled = false;

        if litertlm_has_audio_section(model_path) {
            println!("[시스템] 오디오 인코더 감지 -> 오디오 모달리티(CPU 백엔드) 포함 엔진 초기화 중...");
            engine_ptr = create_engine(Some("cpu"));
            if !engine_ptr.is_null() {
                println!("[시스템] 파이프라인 및 세션 워밍업 중...");
                if run_warmup(engine_ptr) {
                    audio_enabled = true;
                    println!("[시스템] 오디오 모달리티 활성화 완료!");
                } else {
                    println!("[시스템] 오디오 세션 워밍업 실패 -> 오디오 비활성화 모드로 재초기화합니다...");
                    unsafe { litert_lm_engine_delete(engine_ptr) };
                    engine_ptr = ptr::null_mut();
                }
            } else {
                println!("[시스템] 오디오 백엔드 포함 초기화 실패 -> 오디오 비활성화 모드로 재시도합니다...");
            }
        }

        if engine_ptr.is_null() {
            engine_ptr = create_engine(None);
            if engine_ptr.is_null() {
                return Err(anyhow!("LiteRT-LM 엔진 생성 실패"));
            }
            println!("[시스템] 파이프라인 및 세션 워밍업 중...");
            run_warmup(engine_ptr);
        }

        println!(
            "[시스템] 준비 완료! (오디오 입력: {})",
            if audio_enabled { "지원" } else { "미지원" }
        );

        Ok(Self {
            engine: SendEnginePtr(engine_ptr),
            max_tokens,
            audio_enabled,
            infer_mutex: Mutex::new(()),
            tokenizer_mutex: Mutex::new(()),
        })
    }

    pub fn audio_enabled(&self) -> bool {
        self.audio_enabled
    }

    unsafe fn create_conversation(
        &self,
        system_msg: &str,
        history_json: &str,
        opts: &GenerationOptions,
    ) -> *mut LiteRtLmConversation {
        let max_tok = if opts.max_tokens > 0 {
            opts.max_tokens
        } else {
            self.max_tokens as i32
        };

        let sc = litert_lm_session_config_create();
        if !sc.is_null() {
            litert_lm_session_config_set_max_output_tokens(sc, max_tok);
            let s_type = if opts.temperature <= 0.001 {
                SAMPLER_TYPE_GREEDY
            } else {
                SAMPLER_TYPE_TOP_P
            };
            let sp = litert_lm_sampler_params_create(s_type);
            if !sp.is_null() {
                litert_lm_sampler_params_set_temperature(sp, opts.temperature);
                litert_lm_sampler_params_set_top_p(sp, opts.top_p);
                litert_lm_sampler_params_set_top_k(sp, opts.top_k);
                if opts.seed >= 0 {
                    litert_lm_sampler_params_set_seed(sp, opts.seed);
                }
                litert_lm_session_config_set_sampler_params(sc, sp);
                litert_lm_sampler_params_delete(sp);
            }
        }

        let cc = litert_lm_conversation_config_create();
        if !cc.is_null() {
            if !sc.is_null() {
                litert_lm_conversation_config_set_session_config(cc, sc);
            }
            if !system_msg.is_empty() {
                let sys_json = json!({ "role": "system", "content": system_msg }).to_string();
                if let Ok(sys_c) = CString::new(sys_json) {
                    litert_lm_conversation_config_set_system_message(cc, sys_c.as_ptr());
                }
            }
            if !history_json.is_empty() {
                if let Ok(hist_c) = CString::new(history_json) {
                    litert_lm_conversation_config_set_messages(cc, hist_c.as_ptr());
                }
            }
            litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(cc, true);
            let t_cfg = litert_lm_thinking_config_create();
            if !t_cfg.is_null() {
                litert_lm_thinking_config_set_enable_thinking(t_cfg, false);
                litert_lm_conversation_config_set_thinking_config(cc, t_cfg);
                litert_lm_thinking_config_delete(t_cfg);
            }
        }
        if !sc.is_null() {
            litert_lm_session_config_delete(sc);
        }

        let conv = litert_lm_conversation_create(self.engine.0, cc);
        if !cc.is_null() {
            litert_lm_conversation_config_delete(cc);
        }
        conv
    }

    unsafe fn create_optional_args(
        &self,
        opts: &GenerationOptions,
    ) -> *mut LiteRtLmConversationOptionalArgs {
        let max_tok = if opts.max_tokens > 0 {
            opts.max_tokens
        } else {
            self.max_tokens as i32
        };
        let opt_args = litert_lm_conversation_optional_args_create();
        if !opt_args.is_null() {
            litert_lm_conversation_optional_args_set_max_output_tokens(opt_args, max_tok);
            let t_cfg = litert_lm_thinking_config_create();
            if !t_cfg.is_null() {
                litert_lm_thinking_config_set_enable_thinking(t_cfg, false);
                litert_lm_conversation_optional_args_set_thinking_config(opt_args, t_cfg);
                litert_lm_thinking_config_delete(t_cfg);
            }
            if opts.frequency_penalty != 0.0 || opts.presence_penalty != 0.0 {
                let rep = litert_lm_repetition_penalty_config_create();
                if !rep.is_null() {
                    litert_lm_repetition_penalty_config_set_presence_penalty(
                        rep,
                        opts.presence_penalty,
                    );
                    litert_lm_repetition_penalty_config_set_frequency_penalty(
                        rep,
                        opts.frequency_penalty,
                    );
                    litert_lm_conversation_optional_args_set_repetition_penalty_config(
                        opt_args, rep,
                    );
                    litert_lm_repetition_penalty_config_delete(rep);
                }
            }
        }
        opt_args
    }
}

impl Drop for LiteRtEngine {
    fn drop(&mut self) {
        let _infer = self.infer_mutex.lock().ok();
        let _tok = self.tokenizer_mutex.lock().ok();
        if !self.engine.0.is_null() {
            unsafe { litert_lm_engine_delete(self.engine.0) };
            self.engine.0 = ptr::null_mut();
        }
    }
}

struct StreamCallbackState {
    chunks: VecDeque<String>,
    done: bool,
    error: Option<String>,
}

struct StreamCallbackCtx {
    state: Mutex<StreamCallbackState>,
    cv: Condvar,
}

extern "C" fn litert_stream_trampoline(data: *mut c_void, chunk: *const LiteRtLmStreamChunk) {
    if data.is_null() || chunk.is_null() {
        return;
    }
    let ctx = unsafe { &*(data as *const StreamCallbackCtx) };
    let mut state = match ctx.state.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };

    unsafe {
        let err_ptr = litert_lm_stream_chunk_get_error(chunk);
        if !err_ptr.is_null() {
            let err_str = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
            state.error = Some(err_str);
            state.done = true;
            ctx.cv.notify_one();
            return;
        }
        let text_ptr = litert_lm_stream_chunk_get_text(chunk);
        if !text_ptr.is_null() {
            let raw = CStr::from_ptr(text_ptr).to_string_lossy();
            let extracted = extract_text_from_chunk(&raw);
            if !extracted.is_empty() {
                state.chunks.push_back(extracted);
            }
        }
        if litert_lm_stream_chunk_is_final(chunk) {
            state.done = true;
        }
    }
    ctx.cv.notify_one();
}

impl LlmEngine for LiteRtEngine {
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn supports_system_role(&self) -> bool {
        // LiteRT-LM Conversation API handles system_message natively via set_system_message.
        true
    }

    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let _lock = self.tokenizer_mutex.lock().unwrap_or_else(|e| e.into_inner());
        if self.engine.0.is_null() {
            return (text.len() / 3).max(1);
        }
        let Ok(c_text) = CString::new(text) else {
            return (text.len() / 3).max(1);
        };
        unsafe {
            let res = litert_lm_engine_tokenize(self.engine.0, c_text.as_ptr());
            if !res.is_null() {
                let count = litert_lm_tokenize_result_get_num_tokens(res);
                litert_lm_tokenize_result_delete(res);
                return count;
            }
        }
        (text.len() / 3).max(1)
    }

    fn generate(
        &self,
        system_msg: &str,
        history_json: &str,
        current_msg: &str,
        opts: &GenerationOptions,
    ) -> String {
        let _lock = self.infer_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(msg_c) = CString::new(current_msg) else {
            return String::new();
        };

        let (mut out_text, total_kv_tokens) = unsafe {
            let conv = self.create_conversation(system_msg, history_json, opts);
            if conv.is_null() {
                return String::new();
            }
            let opt_args = self.create_optional_args(opts);
            let resp = litert_lm_conversation_send_message(
                conv,
                msg_c.as_ptr(),
                ptr::null(),
                opt_args,
            );
            if !opt_args.is_null() {
                litert_lm_conversation_optional_args_delete(opt_args);
            }

            let mut text = String::new();
            if !resp.is_null() {
                let ptr = litert_lm_json_response_get_string(resp);
                if !ptr.is_null() {
                    let raw = CStr::from_ptr(ptr).to_string_lossy();
                    text = extract_text_from_chunk(&raw);
                }
                litert_lm_json_response_delete(resp);
            }
            let kv = litert_lm_conversation_get_token_count(conv);
            litert_lm_conversation_delete(conv);
            (text, kv)
        };

        if !opts.stop.is_empty() {
            let mut earliest: Option<usize> = None;
            for sw in &opts.stop {
                if sw.is_empty() {
                    continue;
                }
                if let Some(pos) = out_text.find(sw.as_str()) {
                    earliest = Some(earliest.map_or(pos, |e| e.min(pos)));
                }
            }
            if let Some(pos) = earliest {
                out_text.truncate(pos);
            }
        }

        let out_tokens = self.count_tokens(&out_text);
        println!(
            "[생성 완료] 정상 종료 (출력: {} 토큰, KV 사용량: {} / {} 토큰)",
            out_tokens, total_kv_tokens, self.max_tokens
        );
        out_text
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
        let _lock = self.infer_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(msg_c) = CString::new(current_msg) else {
            let _ = tx.send(StreamEvent::Error("유효하지 않은 메시지 문자열".into()));
            return;
        };

        let conv = unsafe { self.create_conversation(system_msg, history_json, opts) };
        if conv.is_null() {
            let _ = tx.send(StreamEvent::Error("대화 세션 생성 실패".into()));
            return;
        }
        let opt_args = unsafe { self.create_optional_args(opts) };

        let ctx = Box::new(StreamCallbackCtx {
            state: Mutex::new(StreamCallbackState {
                chunks: VecDeque::new(),
                done: false,
                error: None,
            }),
            cv: Condvar::new(),
        });
        let ctx_ptr = &*ctx as *const StreamCallbackCtx as *mut c_void;

        let rc = unsafe {
            let r = litert_lm_conversation_send_message_stream(
                conv,
                msg_c.as_ptr(),
                ptr::null(),
                opt_args,
                litert_stream_trampoline,
                ctx_ptr,
            );
            if !opt_args.is_null() {
                litert_lm_conversation_optional_args_delete(opt_args);
            }
            r
        };

        if rc != 0 {
            let _ = tx.send(StreamEvent::Error(format!("스트리밍 시작 실패 (코드: {})", rc)));
            unsafe { litert_lm_conversation_delete(conv) };
            return;
        }

        let mut accumulated = String::new();
        let mut stop_triggered = false;
        let mut client_cancelled = false;
        let mut final_error: Option<String> = None;

        loop {
            if cancelled.load(Ordering::Relaxed) {
                client_cancelled = true;
                unsafe { litert_lm_conversation_cancel_process(conv) };
                break;
            }

            let mut guard = ctx.state.lock().unwrap_or_else(|e| e.into_inner());
            if guard.chunks.is_empty() && !guard.done {
                let (g, _) = ctx
                    .cv
                    .wait_timeout(guard, Duration::from_millis(50))
                    .unwrap_or_else(|e| e.into_inner());
                guard = g;
            }

            let drained: Vec<String> = guard.chunks.drain(..).collect();
            let is_done = guard.done;
            let err_opt = guard.error.clone();
            drop(guard);

            for c in drained {
                if stop_triggered {
                    continue;
                }
                if !opts.stop.is_empty() {
                    let test_str = format!("{}{}", accumulated, c);
                    let mut earliest: Option<usize> = None;
                    for sw in &opts.stop {
                        if sw.is_empty() {
                            continue;
                        }
                        if let Some(pos) = test_str.find(sw.as_str()) {
                            earliest = Some(earliest.map_or(pos, |e| e.min(pos)));
                        }
                    }
                    if let Some(pos) = earliest {
                        stop_triggered = true;
                        unsafe { litert_lm_conversation_cancel_process(conv) };
                        if pos > accumulated.len() {
                            let part = &test_str[accumulated.len()..pos];
                            if !part.is_empty() {
                                let _ = tx.send(StreamEvent::Chunk(part.as_bytes().to_vec()));
                            }
                        }
                        accumulated = test_str[..pos].to_string();
                        break;
                    }
                }
                accumulated.push_str(&c);
                let _ = tx.send(StreamEvent::Chunk(c.into_bytes()));
            }

            if stop_triggered {
                let _ = tx.send(StreamEvent::Done);
                break;
            }
            if is_done {
                final_error = err_opt;
                if let Some(ref e) = final_error {
                    let _ = tx.send(StreamEvent::Error(e.clone()));
                } else {
                    let _ = tx.send(StreamEvent::Done);
                }
                break;
            }
        }

        let total_kv_tokens = unsafe {
            let kv = litert_lm_conversation_get_token_count(conv);
            litert_lm_conversation_delete(conv);
            kv
        };

        let out_tokens = self.count_tokens(&accumulated);
        if client_cancelled {
            eprintln!(
                "[경고] 클라이언트 연결 종료로 스트리밍 중단 (출력: {} 토큰, KV: {} / {})",
                out_tokens, total_kv_tokens, self.max_tokens
            );
        } else if let Some(err) = final_error {
            eprintln!(
                "[생성 오류] {} (출력: {} 토큰, KV: {} / {})",
                err, out_tokens, total_kv_tokens, self.max_tokens
            );
        } else {
            println!(
                "[생성 완료] 정상 종료 (출력: {} 토큰, KV 사용량: {} / {} 토큰)",
                out_tokens, total_kv_tokens, self.max_tokens
            );
        }
    }
}
