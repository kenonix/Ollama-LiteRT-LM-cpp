use crate::config::GenerationOptions;
use crate::engine::embedding::EmbeddingEngine;
use crate::engine::{LlmEngine, StreamEvent};
use crate::parser::{ToolCallStreamParser, Utf8StreamReassembler};
use crate::utils::{
    generate_random_id, get_iso8601_now, save_audio_source_to_temp_file,
    save_image_source_to_temp_file, TempFilesCleanup,
};
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::stream::StreamExt;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

pub struct AppState {
    pub engine: Arc<dyn LlmEngine>,
    pub emb_engine: Option<Arc<EmbeddingEngine>>,
    pub served_model_name: String,
    pub default_system_prompt: String,
}

/// Automatically marks `cancelled = true` if the HTTP response stream is dropped early.
struct CancelOnDrop {
    cancelled: Arc<AtomicBool>,
    _temp_files: TempFilesCleanup,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn make_openai_model_info(model_name: &str) -> Value {
    json!({
        "id": model_name,
        "object": "model",
        "created": 1728000000u64,
        "owned_by": "litert-lm",
        "permission": [{
            "id": format!("modelperm-{}", model_name),
            "object": "model_permission",
            "created": 1728000000u64,
            "allow_create_engine": false,
            "allow_sampling": true,
            "allow_logprobs": true,
            "allow_search_indices": false,
            "allow_view": true,
            "allow_fine_tuning": false,
            "organization": "*",
            "group": null,
            "is_blocking": false
        }],
        "root": model_name,
        "parent": null
    })
}

fn parse_generation_options(j: &Value) -> GenerationOptions {
    let mut opts = GenerationOptions::default();

    if let Some(v) = j.get("temperature").and_then(Value::as_f64) {
        opts.temperature = v as f32;
    }
    if let Some(v) = j.get("top_p").and_then(Value::as_f64) {
        opts.top_p = v as f32;
    }
    if let Some(v) = j.get("top_k").and_then(Value::as_i64) {
        opts.top_k = v as i32;
    }
    if let Some(v) = j
        .get("max_tokens")
        .or_else(|| j.get("max_completion_tokens"))
        .and_then(Value::as_i64)
    {
        opts.max_tokens = v as i32;
    }
    if let Some(v) = j.get("seed").and_then(Value::as_i64) {
        opts.seed = v as i32;
    }
    if let Some(v) = j.get("frequency_penalty").and_then(Value::as_f64) {
        opts.frequency_penalty = v as f32;
    }
    if let Some(v) = j.get("presence_penalty").and_then(Value::as_f64) {
        opts.presence_penalty = v as f32;
    }

    if let Some(o) = j.get("options").and_then(Value::as_object) {
        if let Some(v) = o.get("temperature").and_then(Value::as_f64) {
            opts.temperature = v as f32;
        }
        if let Some(v) = o.get("top_p").and_then(Value::as_f64) {
            opts.top_p = v as f32;
        }
        if let Some(v) = o.get("top_k").and_then(Value::as_i64) {
            opts.top_k = v as i32;
        }
        if let Some(v) = o.get("num_predict").and_then(Value::as_i64) {
            opts.max_tokens = v as i32;
        }
        if let Some(v) = o.get("seed").and_then(Value::as_i64) {
            opts.seed = v as i32;
        }
        if let Some(v) = o.get("repeat_penalty").and_then(Value::as_f64) {
            opts.repetition_penalty = (v as f32).max(1.0);
        }
        if let Some(stop) = o.get("stop") {
            push_stop_words(stop, &mut opts.stop);
        }
    }

    if let Some(stop) = j.get("stop") {
        push_stop_words(stop, &mut opts.stop);
    }

    opts
}

fn push_stop_words(v: &Value, out: &mut Vec<String>) {
    if let Some(s) = v.as_str() {
        out.push(s.to_string());
    } else if let Some(arr) = v.as_array() {
        for item in arr {
            if let Some(s) = item.as_str() {
                out.push(s.to_string());
            }
        }
    }
}

fn format_tools_system_prompt(tools: &[Value]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut prompt = String::from(
        "\n\n# Available Tools\n\
         You have access to the following tools to interact with the environment, files, and system:\n<tools>\n",
    );
    for tool in tools {
        let Some(tool_obj) = tool.as_object() else {
            continue;
        };
        let fn_val = tool_obj
            .get("function")
            .filter(|v| v.is_object())
            .unwrap_or(tool);

        let mut compact = Map::new();
        compact.insert(
            "name".into(),
            json!(fn_val.get("name").and_then(Value::as_str).unwrap_or("")),
        );
        if let Some(desc) = fn_val.get("description").and_then(Value::as_str) {
            let truncated = if desc.chars().count() > 140 {
                let s: String = desc.chars().take(137).collect();
                format!("{}...", s)
            } else {
                desc.to_string()
            };
            compact.insert("description".into(), json!(truncated));
        }
        if let Some(params) = fn_val.get("parameters").and_then(Value::as_object) {
            let mut compact_params = Map::new();
            if let Some(props) = params.get("properties").and_then(Value::as_object) {
                for (k, v) in props {
                    let p_type = v.get("type").and_then(Value::as_str).unwrap_or("string");
                    let p_desc = v.get("description").and_then(Value::as_str).unwrap_or("");
                    let desc_short = if p_desc.chars().count() > 50 {
                        let s: String = p_desc.chars().take(47).collect();
                        format!("{}...", s)
                    } else {
                        p_desc.to_string()
                    };
                    if !desc_short.is_empty() {
                        compact_params
                            .insert(k.clone(), json!(format!("{} ({})", p_type, desc_short)));
                    } else {
                        compact_params.insert(k.clone(), json!(p_type));
                    }
                }
            }
            compact.insert("parameters".into(), Value::Object(compact_params));
            if let Some(req) = params.get("required").filter(|v| v.is_array()) {
                compact.insert("required".into(), req.clone());
            }
        }
        prompt.push_str(&Value::Object(compact).to_string());
        prompt.push('\n');
    }
    prompt.push_str(
        "</tools>\n\n\
         # Tool Calling Rules\n\
         1. When the user asks you to read, write, edit, search files, execute commands, or manage tasks, you MUST execute the appropriate tool instead of just describing it or outputting code blocks in chat.\n\
         2. To invoke a tool, output the tool call enclosed in <tool_call> tags using valid JSON formatted like this:\n\
         <tool_call>\n\
         {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"val1\"}}\n\
         </tool_call>\n\
         3. You may provide a brief explanation before or after the tool call, but the tool invocation itself MUST be wrapped in <tool_call> and </tool_call> tags.\n\
         4. NEVER output raw markdown code blocks as a substitute for calling write or edit tools when file modification is requested.\n\n",
    );
    prompt
}

fn transform_message_for_engine(msg: &Value, temp_files: &mut TempFilesCleanup) -> Value {
    let mut transformed = msg.clone();
    let role_raw = msg.get("role").and_then(Value::as_str).unwrap_or("user");

    let role = match role_raw {
        "developer" => "system",
        "tool" => {
            let tool_name = msg.get("name").and_then(Value::as_str).unwrap_or("");
            let content_str = match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            let header = if tool_name.is_empty() {
                "[Tool Result]:\n".to_string()
            } else {
                format!("[Tool Result for {}]:\n", tool_name)
            };
            return json!({
                "role": "user",
                "content": format!("{}{}", header, content_str)
            });
        }
        other => other,
    };
    transformed["role"] = json!(role);

    if role == "assistant" {
        let has_text = msg
            .get("content")
            .and_then(Value::as_str)
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if !has_text {
            if let Some(tcs) = msg
                .get("tool_calls")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty())
            {
                let mut calls_text = String::new();
                for tc in tcs {
                    if let Some(f) = tc.get("function").filter(|v| v.is_object()) {
                        calls_text.push_str("<tool_call>\n");
                        calls_text.push_str(&f.to_string());
                        calls_text.push_str("\n</tool_call>\n");
                    }
                }
                transformed["content"] = json!(calls_text);
                return transformed;
            }
        }
    }

    // 1. Ollama-style "images" and/or "audios" arrays
    let ollama_images = msg
        .get("images")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty());
    let ollama_audios = msg
        .get("audios")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty());

    if ollama_images.is_some() || ollama_audios.is_some() {
        let text_content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        let mut content_arr: Vec<Value> = Vec::new();

        if let Some(imgs) = ollama_images {
            for item in imgs {
                if let Some(src) = item.as_str() {
                    if let Some((path, is_temp)) = save_image_source_to_temp_file(src) {
                        if is_temp {
                            temp_files.push(path.clone());
                        }
                        content_arr.push(json!({"type": "image", "path": path.to_string_lossy()}));
                    }
                }
            }
        }
        if let Some(auds) = ollama_audios {
            for item in auds {
                if let Some(src) = item.as_str() {
                    if let Some((path, is_temp)) = save_audio_source_to_temp_file(src, "") {
                        if is_temp {
                            temp_files.push(path.clone());
                        }
                        content_arr.push(json!({"type": "audio", "path": path.to_string_lossy()}));
                    }
                }
            }
        }
        if !text_content.is_empty() {
            content_arr.push(json!({"type": "text", "text": text_content}));
        }
        transformed["content"] = Value::Array(content_arr);
        if let Some(obj) = transformed.as_object_mut() {
            obj.remove("images");
            obj.remove("audios");
        }
        return transformed;
    }

    // 2. OpenAI / multimodal content array (text, image_url, image, input_audio, audio_url, audio)
    if let Some(parts) = msg.get("content").and_then(Value::as_array) {
        let mut content_arr: Vec<Value> = Vec::new();
        for part in parts {
            let Some(pobj) = part.as_object() else {
                continue;
            };
            let ptype = pobj.get("type").and_then(Value::as_str).unwrap_or("");
            match ptype {
                "text" => content_arr.push(part.clone()),
                "image_url" => {
                    let img_src = match pobj.get("image_url") {
                        Some(Value::String(s)) => Some(s.as_str()),
                        Some(Value::Object(o)) => o.get("url").and_then(Value::as_str),
                        _ => None,
                    };
                    if let Some(src) = img_src {
                        if let Some((path, is_temp)) = save_image_source_to_temp_file(src) {
                            if is_temp {
                                temp_files.push(path.clone());
                            }
                            content_arr
                                .push(json!({"type": "image", "path": path.to_string_lossy()}));
                        }
                    }
                }
                "image" => {
                    if let Some(p) = pobj.get("path").and_then(Value::as_str) {
                        content_arr.push(json!({"type": "image", "path": p}));
                    } else if let Some(b) = pobj.get("blob").and_then(Value::as_str) {
                        content_arr.push(json!({"type": "image", "blob": b}));
                    } else if let Some(u) = pobj.get("url").and_then(Value::as_str) {
                        if let Some((path, is_temp)) = save_image_source_to_temp_file(u) {
                            if is_temp {
                                temp_files.push(path.clone());
                            }
                            content_arr
                                .push(json!({"type": "image", "path": path.to_string_lossy()}));
                        }
                    }
                }
                "input_audio" => {
                    let (aud_src, aud_fmt) = if let Some(ia) =
                        pobj.get("input_audio").and_then(Value::as_object)
                    {
                        (
                            ia.get("data").and_then(Value::as_str),
                            ia.get("format").and_then(Value::as_str).unwrap_or(""),
                        )
                    } else {
                        (
                            pobj.get("data").and_then(Value::as_str),
                            pobj.get("format").and_then(Value::as_str).unwrap_or(""),
                        )
                    };
                    if let Some(src) = aud_src {
                        if let Some((path, is_temp)) = save_audio_source_to_temp_file(src, aud_fmt) {
                            if is_temp {
                                temp_files.push(path.clone());
                            }
                            content_arr
                                .push(json!({"type": "audio", "path": path.to_string_lossy()}));
                        }
                    }
                }
                "audio_url" => {
                    let aud_src = match pobj.get("audio_url") {
                        Some(Value::String(s)) => Some(s.as_str()),
                        Some(Value::Object(o)) => o.get("url").and_then(Value::as_str),
                        _ => None,
                    };
                    if let Some(src) = aud_src {
                        if let Some((path, is_temp)) = save_audio_source_to_temp_file(src, "") {
                            if is_temp {
                                temp_files.push(path.clone());
                            }
                            content_arr
                                .push(json!({"type": "audio", "path": path.to_string_lossy()}));
                        }
                    }
                }
                "audio" => {
                    let aud_fmt = pobj.get("format").and_then(Value::as_str).unwrap_or("");
                    let aud_src = pobj
                        .get("path")
                        .or_else(|| pobj.get("blob"))
                        .or_else(|| pobj.get("data"))
                        .or_else(|| pobj.get("url"))
                        .and_then(Value::as_str);
                    if let Some(src) = aud_src {
                        if let Some((path, is_temp)) = save_audio_source_to_temp_file(src, aud_fmt) {
                            if is_temp {
                                temp_files.push(path.clone());
                            }
                            content_arr
                                .push(json!({"type": "audio", "path": path.to_string_lossy()}));
                        }
                    }
                }
                _ => {}
            }
        }
        transformed["content"] = Value::Array(content_arr);
        return transformed;
    }

    if transformed.get("content").is_none_or(Value::is_null) {
        transformed["content"] = json!("");
    }
    transformed
}

fn prepend_system_prompt_to_message(msg: &mut Value, sys_prompt: &str) {
    if sys_prompt.is_empty() {
        return;
    }
    match msg.get_mut("content") {
        Some(Value::String(orig)) => {
            if orig.is_empty() {
                *orig = sys_prompt.to_string();
            } else {
                *orig = format!("{}\n\n{}", sys_prompt, orig);
            }
        }
        Some(Value::Array(parts)) => {
            for part in parts.iter_mut() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        let new_t = if t.is_empty() {
                            sys_prompt.to_string()
                        } else {
                            format!("{}\n\n{}", sys_prompt, t)
                        };
                        part["text"] = json!(new_t);
                        return;
                    }
                }
            }
            parts.insert(0, json!({"type": "text", "text": sys_prompt}));
        }
        _ => {
            msg["content"] = json!(sys_prompt);
        }
    }
}

fn extract_message_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut combined = String::new();
            for p in parts {
                if p.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        if !combined.is_empty() {
                            combined.push('\n');
                        }
                        combined.push_str(t);
                    }
                }
            }
            combined
        }
        _ => String::new(),
    }
}

fn truncate_utf8_to_token_budget(
    engine: &dyn LlmEngine,
    text: &str,
    max_tok_budget: usize,
    keep_tail: bool,
) -> String {
    if text.is_empty() || max_tok_budget == 0 {
        return String::new();
    }
    let cur_tok = engine.count_tokens(text);
    if cur_tok <= max_tok_budget {
        return text.to_string();
    }

    let ratio = (max_tok_budget as f64) / (cur_tok.max(1) as f64) * 0.90;
    let mut target_bytes = ((text.len() as f64) * ratio).max(16.0) as usize;
    if target_bytes >= text.len() {
        target_bytes = text.len() / 2;
    }

    let mut result = text.to_string();
    for _ in 0..6 {
        if keep_tail {
            let mut start = text.len().saturating_sub(target_bytes);
            while start < text.len() && !text.is_char_boundary(start) {
                start += 1;
            }
            result = text[start..].to_string();
        } else {
            let mut end = target_bytes.min(text.len());
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            result = text[..end].to_string();
        }
        if engine.count_tokens(&result) <= max_tok_budget || target_bytes <= 32 {
            break;
        }
        target_bytes = ((target_bytes as f64) * 0.80) as usize;
    }
    result
}

// ============================================================
// Basic & Model Info Endpoints
// ============================================================

async fn handle_root() -> &'static str {
    "Ollama is running"
}

async fn handle_version() -> Json<Value> {
    Json(json!({ "version": "0.3.14" }))
}

async fn handle_tags(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut models = vec![json!({
        "name": state.served_model_name,
        "model": state.served_model_name,
        "modified_at": get_iso8601_now(),
        "size": 0,
        "digest": "000000000000",
        "details": {
            "format": "tflite",
            "family": "litert",
            "families": ["litert"]
        }
    })];
    if state.emb_engine.is_some() {
        for name in ["embeddinggemma-2:latest", "embeddinggemma2"] {
            models.push(json!({
                "name": name,
                "model": name,
                "modified_at": get_iso8601_now(),
                "size": 164626432u64,
                "digest": "embeddinggemma2",
                "details": {
                    "format": "litertlm",
                    "family": "embeddinggemma2",
                    "families": ["embeddinggemma2", "embedding"]
                }
            }));
        }
    }
    Json(json!({ "models": models }))
}

async fn handle_show(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "modelfile": format!("# Modelfile generated by Ollama-LiteRT-LM-rs\nFROM {}", state.served_model_name),
        "parameters": "stop \"<end_of_turn>\"",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "tflite",
            "family": "litert",
            "families": ["litert"],
            "parameter_size": "2B",
            "quantization_level": "none"
        }
    }))
}

async fn handle_models_list(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut data = vec![make_openai_model_info(&state.served_model_name)];
    if state.emb_engine.is_some() {
        data.push(make_openai_model_info("embeddinggemma-2:latest"));
        data.push(make_openai_model_info("embeddinggemma2"));
    }
    Json(json!({ "object": "list", "data": data }))
}

async fn handle_single_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model): AxumPath<String>,
) -> Json<Value> {
    let name = if model.is_empty() {
        state.served_model_name.as_str()
    } else {
        model.as_str()
    };
    Json(make_openai_model_info(name))
}

// ============================================================
// Embedding Endpoints
// ============================================================

fn embedding_disabled_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": "Embedding engine is not loaded or available.",
                "type": "service_unavailable",
                "param": null,
                "code": "embedding_disabled"
            }
        })),
    )
        .into_response()
}

async fn handle_openai_embeddings(
    State(state): State<Arc<AppState>>,
    Json(j): Json<Value>,
) -> Response {
    let Some(emb) = state.emb_engine.clone() else {
        return embedding_disabled_response();
    };
    let model = j
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("embeddinggemma-2")
        .to_string();

    let mut inputs = Vec::new();
    if let Some(inp) = j.get("input") {
        if let Some(s) = inp.as_str() {
            inputs.push(s.to_string());
        } else if let Some(arr) = inp.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    inputs.push(s.to_string());
                }
            }
        }
    }
    if inputs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "Missing or invalid 'input' parameter in embedding request.",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "invalid_input"
                }
            })),
        )
            .into_response();
    }

    let vecs = tokio::task::spawn_blocking(move || {
        inputs
            .into_iter()
            .map(|t| emb.compute_embedding(&t, true, 0))
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    let data: Vec<Value> = vecs
        .into_iter()
        .enumerate()
        .map(|(i, v)| json!({ "object": "embedding", "index": i, "embedding": v }))
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": { "prompt_tokens": 0, "total_tokens": 0 }
    }))
    .into_response()
}

async fn handle_ollama_embeddings(
    State(state): State<Arc<AppState>>,
    Json(j): Json<Value>,
) -> Response {
    let Some(emb) = state.emb_engine.clone() else {
        return embedding_disabled_response();
    };
    let prompt = j
        .get("prompt")
        .or_else(|| j.get("input"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let vec = tokio::task::spawn_blocking(move || emb.compute_embedding(&prompt, true, 0))
        .await
        .unwrap_or_default();
    Json(json!({ "embedding": vec })).into_response()
}

async fn handle_ollama_embed(
    State(state): State<Arc<AppState>>,
    Json(j): Json<Value>,
) -> Response {
    let Some(emb) = state.emb_engine.clone() else {
        return embedding_disabled_response();
    };
    let model = j
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("embeddinggemma-2")
        .to_string();

    let mut inputs = Vec::new();
    if let Some(inp) = j.get("input") {
        if let Some(s) = inp.as_str() {
            inputs.push(s.to_string());
        } else if let Some(arr) = inp.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    inputs.push(s.to_string());
                }
            }
        }
    } else if let Some(p) = j.get("prompt").and_then(Value::as_str) {
        inputs.push(p.to_string());
    }

    let embeddings = tokio::task::spawn_blocking(move || {
        inputs
            .into_iter()
            .map(|t| emb.compute_embedding(&t, true, 0))
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    Json(json!({ "model": model, "embeddings": embeddings })).into_response()
}

async fn handle_llamacpp_embedding(
    State(state): State<Arc<AppState>>,
    Json(j): Json<Value>,
) -> Response {
    let Some(emb) = state.emb_engine.clone() else {
        return embedding_disabled_response();
    };
    let content = j
        .get("content")
        .or_else(|| j.get("input"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let vec = tokio::task::spawn_blocking(move || emb.compute_embedding(&content, true, 0))
        .await
        .unwrap_or_default();
    Json(json!({ "embedding": vec })).into_response()
}

// ============================================================
// Chat Completions (/v1/chat/completions & /api/chat)
// ============================================================

async fn handle_openai_chat(
    State(state): State<Arc<AppState>>,
    Json(j_req): Json<Value>,
) -> Response {
    handle_chat_common(state, j_req, false).await
}

async fn handle_ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(j_req): Json<Value>,
) -> Response {
    handle_chat_common(state, j_req, true).await
}

async fn handle_chat_common(state: Arc<AppState>, j_req: Value, is_ollama: bool) -> Response {
    let want_stream = j_req
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(is_ollama);
    let include_usage = j_req
        .get("stream_options")
        .and_then(|so| so.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let gen_opts = parse_generation_options(&j_req);

    let mut sys_msg = String::new();
    let mut history_arr: Vec<Value> = Vec::new();
    let mut temp_files = TempFilesCleanup::default();

    if let Some(messages) = j_req.get("messages").and_then(Value::as_array) {
        for m in messages {
            if !m.is_object() {
                continue;
            }
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            if role == "system" || role == "developer" {
                let text = extract_message_text(m);
                if !text.is_empty() {
                    if !sys_msg.is_empty() {
                        sys_msg.push_str("\n\n");
                    }
                    sys_msg.push_str(&text);
                }
            } else {
                history_arr.push(transform_message_for_engine(m, &mut temp_files));
            }
        }
    }

    if sys_msg.is_empty() {
        sys_msg = state.default_system_prompt.clone();
    }
    if let Some(tools) = j_req
        .get("tools")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        sys_msg.push_str(&format_tools_system_prompt(tools));
    }

    let mut current_msg_j = history_arr
        .pop()
        .unwrap_or_else(|| json!({"role": "user", "content": ""}));

    // KV cache budget guard
    let max_kv_tokens = state.engine.max_tokens();
    let min_output_reserve = (max_kv_tokens / 5).clamp(256, 1024);
    let max_input_budget = max_kv_tokens.saturating_sub(min_output_reserve).max(256);

    let mut sys_tokens = if sys_msg.is_empty() {
        0
    } else {
        state.engine.count_tokens(&sys_msg) + 6
    };
    let curr_text = extract_message_text(&current_msg_j);
    let mut curr_tokens = if curr_text.is_empty() {
        8
    } else {
        state.engine.count_tokens(&curr_text) + 8
    };

    let mut hist_tokens: Vec<usize> = history_arr
        .iter()
        .map(|hm| {
            let t = extract_message_text(hm);
            if t.is_empty() {
                8
            } else {
                state.engine.count_tokens(&t) + 8
            }
        })
        .collect();
    let hist_total: usize = hist_tokens.iter().sum();
    let mut total_est_input = sys_tokens + hist_total + curr_tokens;

    let mut dropped_msgs = 0usize;
    while !history_arr.is_empty() && total_est_input > max_input_budget {
        total_est_input = total_est_input.saturating_sub(hist_tokens.remove(0));
        history_arr.remove(0);
        dropped_msgs += 1;
    }
    while dropped_msgs > 0
        && !history_arr.is_empty()
        && history_arr[0].get("role").and_then(Value::as_str) != Some("user")
    {
        total_est_input = total_est_input.saturating_sub(hist_tokens.remove(0));
        history_arr.remove(0);
        dropped_msgs += 1;
    }

    if total_est_input > max_input_budget && !sys_msg.is_empty() {
        let allowed_sys = max_input_budget.saturating_sub(curr_tokens + 16).max(128);
        if sys_tokens > allowed_sys {
            sys_msg = truncate_utf8_to_token_budget(state.engine.as_ref(), &sys_msg, allowed_sys, false);
            sys_tokens = state.engine.count_tokens(&sys_msg) + 6;
            total_est_input = sys_tokens + curr_tokens;
        }
    }
    if total_est_input > max_input_budget && current_msg_j.get("content").is_some_and(Value::is_string) {
        let allowed_curr = max_input_budget.saturating_sub(sys_tokens + 16).max(64);
        if curr_tokens > allowed_curr {
            let trunc = truncate_utf8_to_token_budget(state.engine.as_ref(), &curr_text, allowed_curr, true);
            curr_tokens = state.engine.count_tokens(&trunc) + 8;
            current_msg_j["content"] = json!(trunc);
            total_est_input = sys_tokens + curr_tokens;
        }
    }

    // Only prepend system prompt into first user message if the engine does NOT support system role
    if !sys_msg.is_empty() && !state.engine.supports_system_role() {
        let mut injected = false;
        for m in &mut history_arr {
            if m.get("role").and_then(Value::as_str) == Some("user") {
                prepend_system_prompt_to_message(m, &sys_msg);
                injected = true;
                break;
            }
        }
        if !injected {
            prepend_system_prompt_to_message(&mut current_msg_j, &sys_msg);
        }
    }

    let history_json_str = if history_arr.is_empty() {
        String::new()
    } else {
        Value::Array(history_arr).to_string()
    };
    let current_msg_str = current_msg_j.to_string();
    let prompt_tokens = total_est_input.max(1);
    let chat_id = generate_random_id("chatcmpl-");
    let created_ts = unix_now_secs();

    println!(
        "[Server Request] {} (stream={}) | Est. input tokens: ~{} / Max KV: {}",
        if is_ollama { "/api/chat" } else { "/v1/chat/completions" },
        want_stream,
        prompt_tokens,
        max_kv_tokens
    );

    if want_stream {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<StreamEvent>();
        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let cancelled = Arc::new(AtomicBool::new(false));

        // Immediate initial chunk to prevent client TTFT timeout
        if is_ollama {
            let init_j = json!({
                "model": state.served_model_name,
                "created_at": get_iso8601_now(),
                "message": {"role": "assistant", "content": ""},
                "done": false
            });
            let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", init_j))));
        } else {
            let init_j = json!({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": state.served_model_name,
                "system_fingerprint": "fp_litert_lm",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": null,
                    "finish_reason": null
                }]
            });
            let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", init_j))));
        }

        // Spawn blocking engine stream (owned Arc -> zero UAF!)
        let engine_for_worker = state.engine.clone();
        let cancelled_for_worker = cancelled.clone();
        let sys_for_worker = sys_msg.clone();
        tokio::task::spawn_blocking(move || {
            engine_for_worker.stream(
                &sys_for_worker,
                &history_json_str,
                &current_msg_str,
                &gen_opts,
                raw_tx,
                cancelled_for_worker,
            );
        });

        // Async formatter task with 3-second keep-alive during prefill
        let state_fmt = state.clone();
        tokio::spawn(async move {
            let mut utf8_asm = Utf8StreamReassembler::default();
            let mut tool_parser = ToolCallStreamParser::default();
            let mut accumulated = String::new();
            let model_name = &state_fmt.served_model_name;

            let emit_text = |out_tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
                             text: &str| {
                if text.is_empty() {
                    return;
                }
                if is_ollama {
                    let j = json!({
                        "model": model_name,
                        "created_at": get_iso8601_now(),
                        "message": {"role": "assistant", "content": text},
                        "done": false
                    });
                    let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", j))));
                } else {
                    let j = json!({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "system_fingerprint": "fp_litert_lm",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "logprobs": null,
                            "finish_reason": null
                        }]
                    });
                    let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", j))));
                }
            };

            let emit_tool_calls =
                |out_tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
                 calls: &[crate::parser::ParsedToolCall]| {
                    for tc in calls {
                        if is_ollama {
                            let j = json!({
                                "model": model_name,
                                "created_at": get_iso8601_now(),
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments
                                        }
                                    }]
                                },
                                "done": false
                            });
                            let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", j))));
                        } else {
                            let j = json!({
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "system_fingerprint": "fp_litert_lm",
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": tc.index,
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.name,
                                                "arguments": tc.arguments.to_string()
                                            }
                                        }]
                                    },
                                    "logprobs": null,
                                    "finish_reason": null
                                }]
                            });
                            let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", j))));
                        }
                    }
                };

            let mut finish_reason_override: Option<&'static str> = None;

            loop {
                match tokio::time::timeout(Duration::from_secs(3), raw_rx.recv()).await {
                    Ok(Some(StreamEvent::Chunk(bytes))) => {
                        let clean = utf8_asm.process_bytes(&bytes);
                        if clean.is_empty() {
                            continue;
                        }
                        let p = tool_parser.process(&clean);
                        if !p.text_to_emit.is_empty() {
                            accumulated.push_str(&p.text_to_emit);
                            emit_text(&out_tx, &p.text_to_emit);
                        }
                        emit_tool_calls(&out_tx, &p.completed_calls);
                    }
                    Ok(Some(StreamEvent::Done)) | Ok(None) => break,
                    Ok(Some(StreamEvent::Error(err_msg))) => {
                        eprintln!("[Server Stream Error] {}", err_msg);
                        if err_msg.contains("Max number of tokens") {
                            finish_reason_override = Some("length");
                        }
                        break;
                    }
                    Err(_) => {
                        // Prefill keep-alive
                        let ka = if is_ollama {
                            let j = json!({
                                "model": model_name,
                                "created_at": get_iso8601_now(),
                                "message": {"role": "assistant", "content": ""},
                                "done": false
                            });
                            format!("{}\n", j)
                        } else {
                            ": keep-alive\n\n".to_string()
                        };
                        if out_tx.send(Ok(Bytes::from(ka))).is_err() {
                            break;
                        }
                    }
                }
            }

            let leftover = utf8_asm.flush();
            let p1 = tool_parser.process(&leftover);
            let p2 = tool_parser.flush();
            let final_text = format!("{}{}", p1.text_to_emit, p2.text_to_emit);
            if !final_text.is_empty() {
                accumulated.push_str(&final_text);
                emit_text(&out_tx, &final_text);
            }
            emit_tool_calls(&out_tx, &p1.completed_calls);
            emit_tool_calls(&out_tx, &p2.completed_calls);

            let finish_reason = finish_reason_override.unwrap_or_else(|| {
                if tool_parser.has_tool_calls() {
                    "tool_calls"
                } else {
                    "stop"
                }
            });

            if is_ollama {
                let done_j = json!({
                    "model": model_name,
                    "created_at": get_iso8601_now(),
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": finish_reason,
                    "done": true
                });
                let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", done_j))));
            } else {
                let stop_j = json!({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "system_fingerprint": "fp_litert_lm",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": null,
                        "finish_reason": finish_reason
                    }]
                });
                let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", stop_j))));

                if include_usage {
                    let comp_tokens = state_fmt.engine.count_tokens(&accumulated);
                    let usage_j = json!({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "system_fingerprint": "fp_litert_lm",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": comp_tokens,
                            "total_tokens": prompt_tokens + comp_tokens
                        }
                    });
                    let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", usage_j))));
                }
                let _ = out_tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        });

        let guard = CancelOnDrop {
            cancelled,
            _temp_files: temp_files,
        };
        let stream = UnboundedReceiverStream::new(out_rx).map(move |item| {
            let _keep_alive = &guard;
            item
        });

        let content_type = if is_ollama {
            "application/x-ndjson"
        } else {
            "text/event-stream"
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(stream))
            .unwrap();
    }

    // Non-streaming response
    let engine = state.engine.clone();
    let raw_output = tokio::task::spawn_blocking(move || {
        let _keep_temp = temp_files;
        engine.generate(&sys_msg, &history_json_str, &current_msg_str, &gen_opts)
    })
    .await
    .unwrap_or_default();

    let mut parser = ToolCallStreamParser::default();
    let p1 = parser.process(&raw_output);
    let p2 = parser.flush();
    let clean_text = format!("{}{}", p1.text_to_emit, p2.text_to_emit);
    let has_tools = parser.has_tool_calls();
    let finish_reason = if has_tools { "tool_calls" } else { "stop" };

    if is_ollama {
        let mut msg_obj = json!({"role": "assistant", "content": clean_text});
        if has_tools {
            let tcs: Vec<Value> = parser
                .get_all_tool_calls()
                .iter()
                .map(|tc| json!({"function": {"name": tc.name, "arguments": tc.arguments}}))
                .collect();
            msg_obj["tool_calls"] = Value::Array(tcs);
        }
        Json(json!({
            "model": state.served_model_name,
            "created_at": get_iso8601_now(),
            "message": msg_obj,
            "done_reason": finish_reason,
            "done": true
        }))
        .into_response()
    } else {
        let comp_tokens = state.engine.count_tokens(&clean_text);
        let mut msg_obj = json!({"role": "assistant", "content": clean_text});
        if has_tools {
            let tcs: Vec<Value> = parser
                .get_all_tool_calls()
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string()
                        }
                    })
                })
                .collect();
            msg_obj["tool_calls"] = Value::Array(tcs);
        }
        Json(json!({
            "id": chat_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": state.served_model_name,
            "system_fingerprint": "fp_litert_lm",
            "choices": [{
                "index": 0,
                "message": msg_obj,
                "logprobs": null,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens": prompt_tokens + comp_tokens
            }
        }))
        .into_response()
    }
}

// ============================================================
// Legacy OpenAI Completions (/v1/completions)
// ============================================================

async fn handle_legacy_completion(
    State(state): State<Arc<AppState>>,
    Json(j_req): Json<Value>,
) -> Response {
    let want_stream = j_req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let gen_opts = parse_generation_options(&j_req);

    let prompt = match j_req.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .first()
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    };

    let prompt_tokens = state.engine.count_tokens(&prompt).max(1);
    let cmpl_id = generate_random_id("cmpl-");
    let created_ts = unix_now_secs();
    let sys_prompt = state.default_system_prompt.clone();
    let mut current_msg_j = json!({"role": "user", "content": prompt});
    if !sys_prompt.is_empty() && !state.engine.supports_system_role() {
        prepend_system_prompt_to_message(&mut current_msg_j, &sys_prompt);
    }
    let current_msg_str = current_msg_j.to_string();

    if want_stream {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<StreamEvent>();
        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let cancelled = Arc::new(AtomicBool::new(false));
        let _ = out_tx.send(Ok(Bytes::from(": keep-alive\n\n")));

        let engine_worker = state.engine.clone();
        let cancel_worker = cancelled.clone();
        tokio::task::spawn_blocking(move || {
            engine_worker.stream(
                &sys_prompt,
                "",
                &current_msg_str,
                &gen_opts,
                raw_tx,
                cancel_worker,
            );
        });

        let model_name = state.served_model_name.clone();
        tokio::spawn(async move {
            let mut utf8_asm = Utf8StreamReassembler::default();
            let mut finish_reason = "stop";
            loop {
                match tokio::time::timeout(Duration::from_secs(3), raw_rx.recv()).await {
                    Ok(Some(StreamEvent::Chunk(b))) => {
                        let clean = utf8_asm.process_bytes(&b);
                        if !clean.is_empty() {
                            let j = json!({
                                "id": cmpl_id,
                                "object": "text_completion",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [{"text": clean, "index": 0, "logprobs": null, "finish_reason": null}]
                            });
                            let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", j))));
                        }
                    }
                    Ok(Some(StreamEvent::Done)) | Ok(None) => break,
                    Ok(Some(StreamEvent::Error(e))) => {
                        if e.contains("Max number of tokens") {
                            finish_reason = "length";
                        }
                        break;
                    }
                    Err(_) => {
                        if out_tx.send(Ok(Bytes::from(": keep-alive\n\n"))).is_err() {
                            break;
                        }
                    }
                }
            }
            let leftover = utf8_asm.flush();
            if !leftover.is_empty() {
                let j = json!({
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"text": leftover, "index": 0, "logprobs": null, "finish_reason": null}]
                });
                let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", j))));
            }
            let stop_j = json!({
                "id": cmpl_id,
                "object": "text_completion",
                "created": created_ts,
                "model": model_name,
                "choices": [{"text": "", "index": 0, "logprobs": null, "finish_reason": finish_reason}]
            });
            let _ = out_tx.send(Ok(Bytes::from(format!("data: {}\n\n", stop_j))));
            let _ = out_tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
        });

        let guard = CancelOnDrop {
            cancelled,
            _temp_files: TempFilesCleanup::default(),
        };
        let stream = UnboundedReceiverStream::new(out_rx).map(move |item| {
            let _keep = &guard;
            item
        });
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(stream))
            .unwrap();
    }

    let engine = state.engine.clone();
    let text = tokio::task::spawn_blocking(move || {
        engine.generate(&sys_prompt, "", &current_msg_str, &gen_opts)
    })
    .await
    .unwrap_or_default();
    let comp_tokens = state.engine.count_tokens(&text);

    Json(json!({
        "id": cmpl_id,
        "object": "text_completion",
        "created": created_ts,
        "model": state.served_model_name,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens
        }
    }))
    .into_response()
}

// ============================================================
// Ollama Generate (/api/generate)
// ============================================================

async fn handle_ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(j_req): Json<Value>,
) -> Response {
    let want_stream = j_req.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let gen_opts = parse_generation_options(&j_req);
    let prompt = j_req
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let sys_msg = j_req
        .get("system")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_system_prompt.clone());

    let mut temp_files = TempFilesCleanup::default();
    let has_imgs = j_req
        .get("images")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let has_auds = j_req
        .get("audios")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    let mut current_msg_j = if has_imgs || has_auds {
        let mut content_arr = Vec::new();
        if let Some(imgs) = j_req.get("images").and_then(Value::as_array) {
            for item in imgs {
                if let Some(src) = item.as_str() {
                    if let Some((path, is_temp)) = save_image_source_to_temp_file(src) {
                        if is_temp {
                            temp_files.push(path.clone());
                        }
                        content_arr.push(json!({"type": "image", "path": path.to_string_lossy()}));
                    }
                }
            }
        }
        if let Some(auds) = j_req.get("audios").and_then(Value::as_array) {
            for item in auds {
                if let Some(src) = item.as_str() {
                    if let Some((path, is_temp)) = save_audio_source_to_temp_file(src, "") {
                        if is_temp {
                            temp_files.push(path.clone());
                        }
                        content_arr.push(json!({"type": "audio", "path": path.to_string_lossy()}));
                    }
                }
            }
        }
        if !prompt.is_empty() {
            content_arr.push(json!({"type": "text", "text": prompt}));
        }
        json!({"role": "user", "content": content_arr})
    } else {
        json!({"role": "user", "content": prompt})
    };

    if !sys_msg.is_empty() && !state.engine.supports_system_role() {
        prepend_system_prompt_to_message(&mut current_msg_j, &sys_msg);
    }
    let current_msg_str = current_msg_j.to_string();

    if want_stream {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<StreamEvent>();
        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let cancelled = Arc::new(AtomicBool::new(false));

        let init_j = json!({
            "model": state.served_model_name,
            "created_at": get_iso8601_now(),
            "response": "",
            "done": false
        });
        let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", init_j))));

        let engine_worker = state.engine.clone();
        let cancel_worker = cancelled.clone();
        tokio::task::spawn_blocking(move || {
            engine_worker.stream(&sys_msg, "", &current_msg_str, &gen_opts, raw_tx, cancel_worker);
        });

        let model_name = state.served_model_name.clone();
        tokio::spawn(async move {
            let mut utf8_asm = Utf8StreamReassembler::default();
            loop {
                match tokio::time::timeout(Duration::from_secs(3), raw_rx.recv()).await {
                    Ok(Some(StreamEvent::Chunk(b))) => {
                        let clean = utf8_asm.process_bytes(&b);
                        if !clean.is_empty() {
                            let j = json!({
                                "model": model_name,
                                "created_at": get_iso8601_now(),
                                "response": clean,
                                "done": false
                            });
                            let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", j))));
                        }
                    }
                    Ok(Some(StreamEvent::Done)) | Ok(Some(StreamEvent::Error(_))) | Ok(None) => {
                        break
                    }
                    Err(_) => {
                        let ka = json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "response": "",
                            "done": false
                        });
                        if out_tx.send(Ok(Bytes::from(format!("{}\n", ka)))).is_err() {
                            break;
                        }
                    }
                }
            }
            let leftover = utf8_asm.flush();
            if !leftover.is_empty() {
                let j = json!({
                    "model": model_name,
                    "created_at": get_iso8601_now(),
                    "response": leftover,
                    "done": false
                });
                let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", j))));
            }
            let done_j = json!({
                "model": model_name,
                "created_at": get_iso8601_now(),
                "response": "",
                "done": true
            });
            let _ = out_tx.send(Ok(Bytes::from(format!("{}\n", done_j))));
        });

        let guard = CancelOnDrop {
            cancelled,
            _temp_files: temp_files,
        };
        let stream = UnboundedReceiverStream::new(out_rx).map(move |item| {
            let _keep = &guard;
            item
        });
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap();
    }

    let engine = state.engine.clone();
    let resp_text = tokio::task::spawn_blocking(move || {
        let _keep_temp = temp_files;
        engine.generate(&sys_msg, "", &current_msg_str, &gen_opts)
    })
    .await
    .unwrap_or_default();

    Json(json!({
        "model": state.served_model_name,
        "created_at": get_iso8601_now(),
        "response": resp_text,
        "done": true
    }))
    .into_response()
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handle_root))
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/show", post(handle_show))
        .route("/v1/models", get(handle_models_list))
        .route("/models", get(handle_models_list))
        .route("/v1/models/:model", get(handle_single_model))
        .route("/models/:model", get(handle_single_model))
        .route("/v1/embeddings", post(handle_openai_embeddings))
        .route("/embeddings", post(handle_openai_embeddings))
        .route("/api/embeddings", post(handle_ollama_embeddings))
        .route("/api/embed", post(handle_ollama_embed))
        .route("/embedding", post(handle_llamacpp_embedding))
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/chat/completions", post(handle_openai_chat))
        .route("/api/chat", post(handle_ollama_chat))
        .route("/v1/completions", post(handle_legacy_completion))
        .route("/completions", post(handle_legacy_completion))
        .route("/api/generate", post(handle_ollama_generate))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
