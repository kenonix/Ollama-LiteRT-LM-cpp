use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::Utc;
use rand::Rng;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// RAII temp file cleaner (automatically deletes temp files when dropped).
#[derive(Default, Debug)]
pub struct TempFilesCleanup {
    paths: Vec<PathBuf>,
}

impl TempFilesCleanup {
    pub fn push(&mut self, path: PathBuf) {
        self.paths.push(path);
    }
}

impl Drop for TempFilesCleanup {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
    }
}

/// Thread-safe ISO8601 UTC timestamp with microsecond precision.
pub fn get_iso8601_now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

/// Thread-safe random ID generator (e.g. `chatcmpl-...`, `call_...`).
pub fn generate_random_id(prefix: &str) -> String {
    const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut rng = rand::thread_rng();
    let mut id = String::with_capacity(prefix.len() + 28);
    id.push_str(prefix);
    for _ in 0..28 {
        let idx = rng.gen_range(0..CHARSET.len());
        id.push(CHARSET[idx] as char);
    }
    id
}

/// Expands leading `~` to `$HOME`.
pub fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix('~') {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(format!("{}{}", home, rest));
        }
    }
    PathBuf::from(path)
}

/// Decodes Base64 (automatically strips `data:...;base64,` prefix and whitespace).
pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut s = input;
    if let Some(comma) = s.find(',') {
        if s[..comma].contains("base64") {
            s = &s[comma + 1..];
        }
    }
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return None;
    }
    STANDARD.decode(&cleaned).ok()
}

fn make_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let name = format!("{}_{}{}", prefix, generate_random_id(""), ext);
    std::env::temp_dir().join(name)
}

fn write_bytes_to_temp(bytes: &[u8], prefix: &str, ext: &str) -> Option<PathBuf> {
    if bytes.is_empty() {
        return None;
    }
    let path = make_temp_path(prefix, ext);
    let mut f = File::create(&path).ok()?;
    f.write_all(bytes).ok()?;
    Some(path)
}

fn run_command_no_shell(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Resolves an image source (local path, HTTP/HTTPS URL, or Base64/Data URI) to a local file path.
/// Returns `Some((path, is_temp))`.
pub fn save_image_source_to_temp_file(source: &str) -> Option<(PathBuf, bool)> {
    if source.is_empty() {
        return None;
    }

    if source.len() < 4096
        && !source.starts_with("data:")
        && !source.starts_with("http://")
        && !source.starts_with("https://")
    {
        let expanded = expand_path(source);
        if expanded.is_file() {
            return Some((expanded, false));
        }
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        let tmp = make_temp_path("litert_url", ".png");
        let tmp_str = tmp.to_string_lossy().to_string();
        if run_command_no_shell("curl", &["-s", "-L", "--max-time", "15", source, "-o", &tmp_str])
            && tmp.is_file()
        {
            return Some((tmp, true));
        }
        let _ = fs::remove_file(&tmp);
        return None;
    }

    let bytes = base64_decode(source)?;
    let path = write_bytes_to_temp(&bytes, "litert_img", ".png")?;
    Some((path, true))
}

/// Detects whether raw audio bytes match a format natively supported by miniaudio (WAV, FLAC, MP3).
fn detect_native_audio_ext(data: &[u8]) -> Option<&'static str> {
    if data.len() < 4 {
        return None;
    }
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        return Some(".wav");
    }
    if &data[0..4] == b"fLaC" {
        return Some(".flac");
    }
    if &data[0..3] == b"ID3" || (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
        return Some(".mp3");
    }
    None
}

fn normalize_audio_ext(fmt: &str) -> &'static str {
    let lower = fmt.trim().trim_start_matches('.').to_ascii_lowercase();
    match lower.as_str() {
        "wav" | "wave" | "x-wav" => ".wav",
        "mp3" | "mpeg" => ".mp3",
        "flac" => ".flac",
        "webm" | "weba" => ".webm",
        "ogg" | "opus" => ".ogg",
        "m4a" | "mp4" | "aac" => ".m4a",
        _ => ".bin",
    }
}

fn convert_audio_to_wav_temp(input_path: &Path) -> Option<PathBuf> {
    let out_path = make_temp_path("litert_audio_conv", ".wav");
    let in_str = input_path.to_string_lossy().to_string();
    let out_str = out_path.to_string_lossy().to_string();
    let ok = run_command_no_shell(
        "ffmpeg",
        &[
            "-y",
            "-loglevel",
            "error",
            "-i",
            &in_str,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            &out_str,
        ],
    );
    if ok && out_path.is_file() {
        Some(out_path)
    } else {
        let _ = fs::remove_file(&out_path);
        None
    }
}

fn ensure_miniaudio_compatible(path: PathBuf, owns_file: bool) -> (PathBuf, bool) {
    let mut header = [0u8; 16];
    let n = File::open(&path)
        .and_then(|mut f| f.read(&mut header))
        .unwrap_or(0);
    if detect_native_audio_ext(&header[..n]).is_some() {
        return (path, owns_file);
    }
    if let Some(converted) = convert_audio_to_wav_temp(&path) {
        if owns_file {
            let _ = fs::remove_file(&path);
        }
        return (converted, true);
    }
    (path, owns_file)
}

/// Resolves an audio source (local path, HTTP/HTTPS URL, or Base64/Data URI) to a LiteRT-LM
/// compatible local file path (WAV/MP3/FLAC; auto-converts webm/ogg/m4a via ffmpeg if available).
pub fn save_audio_source_to_temp_file(source: &str, format_hint: &str) -> Option<(PathBuf, bool)> {
    if source.is_empty() {
        return None;
    }

    if source.len() < 4096
        && !source.starts_with("data:")
        && !source.starts_with("http://")
        && !source.starts_with("https://")
    {
        let expanded = expand_path(source);
        if expanded.is_file() {
            return Some(ensure_miniaudio_compatible(expanded, false));
        }
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        let ext = if format_hint.is_empty() {
            ".bin"
        } else {
            normalize_audio_ext(format_hint)
        };
        let tmp = make_temp_path("litert_audio_url", ext);
        let tmp_str = tmp.to_string_lossy().to_string();
        if run_command_no_shell("curl", &["-s", "-L", "--max-time", "15", source, "-o", &tmp_str])
            && tmp.is_file()
        {
            return Some(ensure_miniaudio_compatible(tmp, true));
        }
        let _ = fs::remove_file(&tmp);
        return None;
    }

    let bytes = base64_decode(source)?;
    if let Some(native_ext) = detect_native_audio_ext(&bytes) {
        let path = write_bytes_to_temp(&bytes, "litert_audio", native_ext)?;
        return Some((path, true));
    }

    let hint_ext = if format_hint.is_empty() {
        ".bin"
    } else {
        normalize_audio_ext(format_hint)
    };
    let raw_path = write_bytes_to_temp(&bytes, "litert_audio", hint_ext)?;
    Some(ensure_miniaudio_compatible(raw_path, true))
}
