use super::litert_ffi::{
    litert_lm_input_data_create, litert_lm_input_data_delete, LiteRtLmInputData,
    INPUT_DATA_TYPE_TEXT,
};
use libc::{c_char, c_float, c_int, c_void, dlsym, size_t, RTLD_DEFAULT};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr;
use std::sync::{Mutex, OnceLock};

#[repr(C)]
struct LiteRtLmEmbeddingEngine {
    _private: [u8; 0],
}
#[repr(C)]
struct LiteRtLmEmbeddingEngineSettings {
    _private: [u8; 0],
}
#[repr(C)]
struct LiteRtLmEmbeddingOptions {
    _private: [u8; 0],
}
#[repr(C)]
struct LiteRtLmEmbeddingResponse {
    _private: [u8; 0],
}

type FnSettingsCreate = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
) -> *mut LiteRtLmEmbeddingEngineSettings;
type FnSettingsDelete = unsafe extern "C" fn(*mut LiteRtLmEmbeddingEngineSettings);
type FnSettingsSetCacheDir =
    unsafe extern "C" fn(*mut LiteRtLmEmbeddingEngineSettings, *const c_char);
type FnEngineCreate =
    unsafe extern "C" fn(*mut LiteRtLmEmbeddingEngineSettings) -> *mut LiteRtLmEmbeddingEngine;
type FnEngineDelete = unsafe extern "C" fn(*mut LiteRtLmEmbeddingEngine);
type FnOptionsCreate = unsafe extern "C" fn() -> *mut LiteRtLmEmbeddingOptions;
type FnOptionsDelete = unsafe extern "C" fn(*mut LiteRtLmEmbeddingOptions);
type FnOptionsSetNormalize = unsafe extern "C" fn(*mut LiteRtLmEmbeddingOptions, bool);
type FnOptionsSetOutputSize = unsafe extern "C" fn(*mut LiteRtLmEmbeddingOptions, c_int);
type FnComputeEmbedding = unsafe extern "C" fn(
    *mut LiteRtLmEmbeddingEngine,
    *mut *mut LiteRtLmInputData,
    size_t,
    *mut LiteRtLmEmbeddingOptions,
) -> *mut LiteRtLmEmbeddingResponse;
type FnResponseDelete = unsafe extern "C" fn(*mut LiteRtLmEmbeddingResponse);
type FnResponseGetSize = unsafe extern "C" fn(*mut LiteRtLmEmbeddingResponse) -> size_t;
type FnResponseGetValues = unsafe extern "C" fn(*mut LiteRtLmEmbeddingResponse) -> *const c_float;

struct EmbeddingApi {
    settings_create: FnSettingsCreate,
    settings_delete: FnSettingsDelete,
    settings_set_cache_dir: Option<FnSettingsSetCacheDir>,
    engine_create: FnEngineCreate,
    engine_delete: FnEngineDelete,
    options_create: FnOptionsCreate,
    options_delete: FnOptionsDelete,
    options_set_normalize: FnOptionsSetNormalize,
    options_set_output_size: Option<FnOptionsSetOutputSize>,
    compute_embedding: FnComputeEmbedding,
    response_delete: FnResponseDelete,
    response_get_size: FnResponseGetSize,
    response_get_values: FnResponseGetValues,
}

unsafe fn lookup_sym<T>(name: &str) -> Option<T> {
    let c = CString::new(name).ok()?;
    let ptr = dlsym(RTLD_DEFAULT, c.as_ptr());
    if ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&ptr))
    }
}

fn embedding_api() -> Option<&'static EmbeddingApi> {
    static API: OnceLock<Option<EmbeddingApi>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        Some(EmbeddingApi {
            settings_create: lookup_sym("litert_lm_embedding_engine_settings_create")?,
            settings_delete: lookup_sym("litert_lm_embedding_engine_settings_delete")?,
            settings_set_cache_dir: lookup_sym("litert_lm_embedding_engine_settings_set_cache_dir"),
            engine_create: lookup_sym("litert_lm_embedding_engine_create")?,
            engine_delete: lookup_sym("litert_lm_embedding_engine_delete")?,
            options_create: lookup_sym("litert_lm_embedding_options_create")?,
            options_delete: lookup_sym("litert_lm_embedding_options_delete")?,
            options_set_normalize: lookup_sym("litert_lm_embedding_options_set_normalize")?,
            options_set_output_size: lookup_sym("litert_lm_embedding_options_set_output_size"),
            compute_embedding: lookup_sym("litert_lm_embedding_engine_compute_embedding")?,
            response_delete: lookup_sym("litert_lm_embedding_response_delete")?,
            response_get_size: lookup_sym("litert_lm_embedding_response_get_size")?,
            response_get_values: lookup_sym("litert_lm_embedding_response_get_values")?,
        })
    })
    .as_ref()
}

fn is_valid_model_file(p: &Path) -> bool {
    fs::metadata(p).map(|m| m.is_file() && m.len() > 1000).unwrap_or(false)
}

struct SendEmbPtr(*mut LiteRtLmEmbeddingEngine);
unsafe impl Send for SendEmbPtr {}
unsafe impl Sync for SendEmbPtr {}

pub struct EmbeddingEngine {
    engine: Mutex<SendEmbPtr>,
}

impl EmbeddingEngine {
    pub fn ensure_model_available(preferred_path: Option<&str>) -> Option<PathBuf> {
        if let Some(p) = preferred_path.filter(|s| !s.is_empty()) {
            let pb = PathBuf::from(p);
            if is_valid_model_file(&pb) {
                return Some(pb);
            }
        }

        let candidates = [
            "./embeddinggemma-2-text-270m.litertlm",
            "../embeddinggemma-2-text-270m.litertlm",
            "./models/embeddinggemma-2-text-270m.litertlm",
            "../models/embeddinggemma-2-text-270m.litertlm",
        ];
        for cand in candidates {
            let pb = PathBuf::from(cand);
            if is_valid_model_file(&pb) {
                println!("[임베딩] 기존 모델 파일 발견: {}", pb.display());
                return Some(pb);
            }
        }

        let target = PathBuf::from("./embeddinggemma-2-text-270m.litertlm");
        let tmp = PathBuf::from("./embeddinggemma-2-text-270m.litertlm.tmp");
        let url = "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm";
        println!("[임베딩] 모델 파일이 없습니다. HuggingFace에서 자동 다운로드 시작: {}", url);

        let ok = Command::new("curl")
            .args(["-L", "-f", "--progress-bar", "-o"])
            .arg(&tmp)
            .arg(url)
            .stdin(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if ok && fs::rename(&tmp, &target).is_ok() && is_valid_model_file(&target) {
            println!("[임베딩] EmbeddingGemma 2 모델 자동 다운로드 완료: {}", target.display());
            return Some(target);
        }
        let _ = fs::remove_file(&tmp);
        eprintln!("[임베딩] 자동 다운로드 실패. 수동으로 모델을 다운로드해주세요.");
        None
    }

    pub fn new(model_path: &Path, backend: &str) -> Option<Self> {
        let api = embedding_api()?;
        if !is_valid_model_file(model_path) {
            eprintln!("[임베딩] 모델 파일을 찾을 수 없습니다: {}", model_path.display());
            return None;
        }
        println!(
            "[임베딩] 엔진 초기화 중 (백엔드: {}, 모델: {})...",
            backend,
            model_path.display()
        );
        let model_c = CString::new(model_path.to_string_lossy().as_ref()).ok()?;
        let backend_c = CString::new(backend).ok()?;
        let cache_c = CString::new(".").ok()?;

        let eng = unsafe {
            let settings = (api.settings_create)(
                model_c.as_ptr(),
                backend_c.as_ptr(),
                ptr::null(),
                ptr::null(),
            );
            if settings.is_null() {
                return None;
            }
            if let Some(set_cache) = api.settings_set_cache_dir {
                set_cache(settings, cache_c.as_ptr());
            }
            let eng = (api.engine_create)(settings);
            (api.settings_delete)(settings);
            eng
        };

        if eng.is_null() {
            eprintln!("[임베딩] LiteRT-LM EmbeddingEngine 생성 실패");
            return None;
        }
        println!("[임베딩] EmbeddingGemma 2 엔진 준비 완료! (768 차원 벡터 지원)");
        Some(Self {
            engine: Mutex::new(SendEmbPtr(eng)),
        })
    }

    pub fn compute_embedding(&self, text: &str, normalize: bool, output_size: i32) -> Vec<f32> {
        let Some(api) = embedding_api() else {
            return Vec::new();
        };
        let guard = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        if guard.0.is_null() {
            return Vec::new();
        }
        let input_text = if text.is_empty() { " " } else { text };

        unsafe {
            let input_data = litert_lm_input_data_create(
                INPUT_DATA_TYPE_TEXT,
                input_text.as_ptr() as *const c_void,
                input_text.len(),
            );
            if input_data.is_null() {
                return Vec::new();
            }
            let options = (api.options_create)();
            if options.is_null() {
                litert_lm_input_data_delete(input_data);
                return Vec::new();
            }
            (api.options_set_normalize)(options, normalize);
            if output_size > 0 {
                if let Some(set_out) = api.options_set_output_size {
                    set_out(options, output_size);
                }
            }
            let mut inputs = [input_data];
            let resp = (api.compute_embedding)(guard.0, inputs.as_mut_ptr(), 1, options);
            let mut result = Vec::new();
            if !resp.is_null() {
                let dim = (api.response_get_size)(resp);
                let vals = (api.response_get_values)(resp);
                if dim > 0 && !vals.is_null() {
                    result.extend_from_slice(std::slice::from_raw_parts(vals, dim));
                }
                (api.response_delete)(resp);
            }
            (api.options_delete)(options);
            litert_lm_input_data_delete(input_data);
            result
        }
    }
}

impl Drop for EmbeddingEngine {
    fn drop(&mut self) {
        if let Some(api) = embedding_api() {
            let mut guard = self.engine.lock().unwrap_or_else(|e| e.into_inner());
            if !guard.0.is_null() {
                unsafe { (api.engine_delete)(guard.0) };
                guard.0 = ptr::null_mut();
            }
        }
    }
}
