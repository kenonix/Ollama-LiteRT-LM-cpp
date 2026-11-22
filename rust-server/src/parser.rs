use crate::utils::generate_random_id;
use serde_json::{json, Value};

/// Reassembles partial UTF-8 multibyte sequences across streaming byte chunks.
#[derive(Default)]
pub struct Utf8StreamReassembler {
    buf: Vec<u8>,
}

impl Utf8StreamReassembler {
    pub fn process_bytes(&mut self, incoming: &[u8]) -> String {
        self.buf.extend_from_slice(incoming);
        if self.buf.is_empty() {
            return String::new();
        }
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_string();
                self.buf.clear();
                out
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                let mut out = String::new();
                if valid_up_to > 0 {
                    out.push_str(std::str::from_utf8(&self.buf[..valid_up_to]).unwrap_or(""));
                    self.buf.drain(..valid_up_to);
                }
                // If invalid sequence (not just incomplete tail), consume the bad bytes with replacement
                if let Some(err_len) = e.error_len() {
                    out.push('\u{FFFD}');
                    let drain_len = err_len.min(self.buf.len());
                    self.buf.drain(..drain_len);
                }
                out
            }
        }
    }

    pub fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }
}

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Default)]
pub struct ToolParseResult {
    pub text_to_emit: String,
    pub completed_calls: Vec<ParsedToolCall>,
}

/// Extracts `<tool_call>...</tool_call>` and ```` ```tool_call...``` ```` blocks from streaming text.
#[derive(Default)]
pub struct ToolCallStreamParser {
    buffer: String,
    tool_calls: Vec<ParsedToolCall>,
    call_index: usize,
}

impl ToolCallStreamParser {
    fn extract_tag(buf: &str) -> (Option<(usize, usize)>, Option<(usize, usize)>) {
        if let Some(open_pos) = buf.find("<tool_call>") {
            let open_len = "<tool_call>".len();
            if let Some(rel_close) = buf[open_pos + open_len..].find("</tool_call>") {
                return (
                    Some((open_pos, open_len)),
                    Some((open_pos + open_len + rel_close, "</tool_call>".len())),
                );
            }
            return (Some((open_pos, open_len)), None);
        }

        if let Some(open_pos) = buf.find("```tool_call") {
            let open_len = "```tool_call".len();
            if let Some(rel_close) = buf[open_pos + open_len..].find("```") {
                return (
                    Some((open_pos, open_len)),
                    Some((open_pos + open_len + rel_close, "```".len())),
                );
            }
            return (Some((open_pos, open_len)), None);
        }

        (None, None)
    }

    fn parse_and_add_tool_call(&mut self, raw: &str, res: &mut ToolParseResult) -> bool {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return false;
        }
        let Ok(call_j) = serde_json::from_str::<Value>(trimmed) else {
            return false;
        };

        let fn_name = call_j
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| {
                call_j
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("")
            .to_string();

        if fn_name.is_empty() {
            return false;
        }

        let args = if let Some(a) = call_j.get("arguments") {
            if a.is_object() {
                a.clone()
            } else if let Some(s) = a.as_str() {
                serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({ "raw": s }))
            } else {
                json!({})
            }
        } else if let Some(p) = call_j.get("parameters").filter(|v| v.is_object()) {
            p.clone()
        } else {
            json!({})
        };

        let idx = self.call_index;
        self.call_index += 1;
        let tc = ParsedToolCall {
            index: idx,
            id: format!("{}{}", generate_random_id("call_"), idx),
            name: fn_name,
            arguments: args,
        };
        res.completed_calls.push(tc.clone());
        self.tool_calls.push(tc);
        true
    }

    pub fn process(&mut self, chunk: &str) -> ToolParseResult {
        self.buffer.push_str(chunk);
        let mut res = ToolParseResult::default();

        while !self.buffer.is_empty() {
            let (open, close) = Self::extract_tag(&self.buffer);
            match (open, close) {
                (Some((open_pos, open_len)), Some((close_pos, close_len))) => {
                    if open_pos > 0 {
                        res.text_to_emit.push_str(&self.buffer[..open_pos]);
                    }
                    let json_str = self.buffer[open_pos + open_len..close_pos].to_string();
                    self.buffer.drain(..close_pos + close_len);
                    self.parse_and_add_tool_call(&json_str, &mut res);
                }
                (Some((open_pos, _)), None) => {
                    if open_pos > 0 {
                        res.text_to_emit.push_str(&self.buffer[..open_pos]);
                        self.buffer.drain(..open_pos);
                    }
                    break;
                }
                (None, _) => {
                    let tags = ["<tool_call>", "```tool_call"];
                    let mut safe_len = self.buffer.len();
                    for tag in tags {
                        let max_prefix = tag.len().saturating_sub(1).min(self.buffer.len());
                        for plen in 1..=max_prefix {
                            let start = self.buffer.len() - plen;
                            if self.buffer.is_char_boundary(start)
                                && tag.starts_with(&self.buffer[start..])
                                && start < safe_len
                            {
                                safe_len = start;
                            }
                        }
                    }
                    if safe_len > 0 {
                        res.text_to_emit.push_str(&self.buffer[..safe_len]);
                        self.buffer.drain(..safe_len);
                    }
                    break;
                }
            }
        }

        res
    }

    pub fn flush(&mut self) -> ToolParseResult {
        let mut res = ToolParseResult::default();
        if self.buffer.is_empty() {
            return res;
        }
        let buf = std::mem::take(&mut self.buffer);
        let (open, close) = Self::extract_tag(&buf);
        match (open, close) {
            (Some((open_pos, open_len)), Some((close_pos, _))) => {
                if open_pos > 0 {
                    res.text_to_emit.push_str(&buf[..open_pos]);
                }
                self.parse_and_add_tool_call(&buf[open_pos + open_len..close_pos], &mut res);
            }
            (Some((open_pos, open_len)), None) => {
                if open_pos > 0 {
                    res.text_to_emit.push_str(&buf[..open_pos]);
                }
                if !self.parse_and_add_tool_call(&buf[open_pos + open_len..], &mut res) {
                    res.text_to_emit.push_str(&buf[open_pos..]);
                }
            }
            (None, _) => {
                res.text_to_emit.push_str(&buf);
            }
        }
        res
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn get_all_tool_calls(&self) -> &[ParsedToolCall] {
        &self.tool_calls
    }
}
