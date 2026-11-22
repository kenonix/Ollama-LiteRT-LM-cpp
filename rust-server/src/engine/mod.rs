pub mod embedding;
pub mod litert;
pub mod litert_ffi;
#[cfg(has_llama_cpp)]
pub mod llama;
#[cfg(has_llama_cpp)]
pub mod llama_ffi;

use crate::config::GenerationOptions;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Chunk(Vec<u8>),
    Done,
    Error(String),
}

pub trait LlmEngine: Send + Sync {
    fn max_tokens(&self) -> usize;

    /// Whether the engine handles `role: "system"` natively without prepending it into user messages.
    fn supports_system_role(&self) -> bool {
        true
    }

    fn count_tokens(&self, text: &str) -> usize;

    fn generate(
        &self,
        system_msg: &str,
        history_json: &str,
        current_msg: &str,
        opts: &GenerationOptions,
    ) -> String;

    fn stream(
        &self,
        system_msg: &str,
        history_json: &str,
        current_msg: &str,
        opts: &GenerationOptions,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancelled: Arc<AtomicBool>,
    );
}
