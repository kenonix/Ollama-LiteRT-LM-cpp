mod config;
mod engine;
mod parser;
mod server;
mod utils;

use crate::config::{DEFAULT_SYSTEM_PROMPT, PROMPT_FILE};
use crate::engine::embedding::EmbeddingEngine;
use crate::engine::litert::LiteRtEngine;
#[cfg(has_llama_cpp)]
use crate::engine::llama::LlamaEngine;
use crate::engine::LlmEngine;
use crate::server::{build_router, AppState, EngineFactory, EngineRuntimeInfo};
use anyhow::{anyhow, Result};
use clap::Parser;
use std::fs;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

#[derive(Parser, Debug)]
#[command(
    name = "ollama-litert-lm-rs",
    about = "Ollama & OpenAI compatible multimodal LLM server (LiteRT-LM + llama.cpp GGUF) in Rust"
)]
struct Cli {
    /// Model file path (.litertlm or .gguf)
    #[arg(default_value = "./models/multimodal_model.tflite")]
    model_path: String,

    /// Server mode flag (kept for CLI compatibility with C++ binary)
    #[arg(long, default_value_t = false)]
    server: bool,

    /// Enable GPU acceleration (Vulkan/WebGPU or CUDA)
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Enable MTP (Multi-Token Prediction / Speculative Decoding)
    #[arg(long, alias = "speculative", default_value_t = false)]
    mtp: bool,

    /// HTTP server port
    #[arg(long, default_value_t = 11434)]
    port: u16,

    /// Reported model name in API responses
    #[arg(long, default_value = "litert-lm:latest")]
    model_name: String,

    /// Maximum context / KV cache token budget
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// EmbeddingGemma 2 model path (auto-detected if omitted)
    #[arg(long)]
    embedding_model: Option<String>,

    /// Embedding backend ("cpu" or "gpu")
    #[arg(long, default_value = "cpu")]
    embedding_backend: String,

    /// Disable embedding engine
    #[arg(long, default_value_t = false)]
    no_embedding: bool,

    /// External DFlash / speculative draft GGUF model path
    #[arg(long, alias = "dflash-model")]
    draft_model: Option<String>,

    /// Explicit ggml backend device name (e.g. CUDA0, Vulkan0)
    #[arg(long)]
    device: Option<String>,

    /// Keep MoE routed expert weights on CPU (saves VRAM on 6GB GPUs)
    #[arg(long, default_value_t = false)]
    cpu_moe: bool,

    /// Disable offloading host tensor ops to GPU device
    #[arg(long, default_value_t = false)]
    no_op_offload: bool,

    /// Number of CPU threads for llama.cpp
    #[arg(long, default_value_t = 8)]
    threads: i32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.max_tokens == 0 {
        return Err(anyhow!("--max-tokens는 1 이상이어야 합니다."));
    }

    let default_system_prompt = fs::read_to_string(PROMPT_FILE)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    let model_path = cli.model_path.clone();
    let draft_model_path = cli.draft_model.clone();
    let device = cli.device.clone();
    let use_gpu = cli.gpu;
    let enable_mtp = cli.mtp;
    let cpu_moe = cli.cpu_moe;
    let op_offload = !cli.no_op_offload;
    let n_threads = cli.threads;
    let engine_factory: EngineFactory = Arc::new(move |max_tokens: usize| -> Result<Arc<dyn LlmEngine>> {
        let is_gguf = model_path.ends_with(".gguf");
        if is_gguf {
            #[cfg(has_llama_cpp)]
            {
                println!("[시스템] GGUF 모델 감지 -> llama.cpp 엔진 백엔드 활성화 (Max Tokens: {})", max_tokens);
                Ok(Arc::new(LlamaEngine::new(
                    &model_path,
                    draft_model_path.as_deref(),
                    device.as_deref(),
                    max_tokens,
                    use_gpu,
                    99,
                    enable_mtp,
                    cpu_moe,
                    op_offload,
                    n_threads,
                )?))
            }
            #[cfg(not(has_llama_cpp))]
            {
                Err(anyhow!(
                    "GGUF 모델이 지정되었으나 빌드 시 llama.cpp가 감지되지 않았습니다."
                ))
            }
        } else {
            println!("[시스템] LiteRT-LM 모델 감지 -> LiteRT-LM 엔진 백엔드 활성화 (Max Tokens: {})", max_tokens);
            let eng = LiteRtEngine::new(&model_path, use_gpu, max_tokens, enable_mtp)?;
            let _ = eng.audio_enabled();
            Ok(Arc::new(eng))
        }
    });

    let llm_engine = (engine_factory)(cli.max_tokens)?;

    let emb_engine = if !cli.no_embedding {
        if let Some(path) =
            EmbeddingEngine::ensure_model_available(cli.embedding_model.as_deref())
        {
            EmbeddingEngine::new(&path, &cli.embedding_backend).map(Arc::new)
        } else {
            eprintln!("[경고] 임베딩 모델을 찾지 못하여 임베딩 기능이 비활성화됩니다.");
            None
        }
    } else {
        None
    };

    let state = Arc::new(AppState {
        engine: RwLock::new(Some(llm_engine)),
        engine_factory,
        reload_lock: tokio::sync::Mutex::new(()),
        runtime_info: EngineRuntimeInfo {
            model_path: cli.model_path.clone(),
            use_gpu: cli.gpu,
            enable_mtp: cli.mtp,
        },
        emb_engine,
        served_model_name: cli.model_name.clone(),
        default_system_prompt,
    });

    let app = build_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!(
        "[서버] http://0.0.0.0:{} 대기 중 (모델: {})",
        cli.port, cli.model_name
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\n[시스템] 종료 신호 수신, 서버를 안전하게 정리합니다...");
        })
        .await?;

    Ok(())
}
