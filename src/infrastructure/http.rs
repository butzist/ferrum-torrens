// OpenAI-compatible HTTP surface: `POST /v1/completions` (buffered and
// `stream: true`), `POST /v1/chat/completions` (chat template + tool calls),
// `POST /v1/responses` (the Responses wire over the same chat path),
// `GET /v1/models`, `GET /health`. Axum types stay here; the handlers call the
// application layer, which never sees an HTTP type.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::application::dispatcher::{Dispatcher, Job};
use crate::application::engine::{Engine, EngineError};
use crate::application::generation::{self, Request, make_request};
use crate::domain::sampling::SamplingParams;
use crate::domain::scheduler::FinishReason;
use crate::domain::{ChatMessage, ChatTool, ChatToolCall};

use super::tool_call::{ParsedToolCall, parse_tool_calls};

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("io: {0}")]
    Io(String),
    #[error("bind {host}:{port}: {detail}")]
    Bind {
        host: String,
        port: u16,
        detail: String,
    },
}

pub struct AppState {
    pub engine: Arc<Engine>,
    pub served_model_name: String,
    next_uid: AtomicU64,
    /// The single worker every forward runs on; handlers submit jobs and await
    /// their replies. See `application::dispatcher` (TODO.md item 11).
    dispatcher: Dispatcher,
}

impl AppState {
    pub fn new(engine: Arc<Engine>, served_model_name: String) -> Self {
        let dispatcher = Dispatcher::new(Arc::clone(&engine));
        Self {
            engine,
            served_model_name,
            next_uid: AtomicU64::new(1),
            dispatcher,
        }
    }

    fn next_uid(&self) -> u64 {
        self.next_uid.fetch_add(1, Ordering::Relaxed)
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(Arc::new(state))
}

/// Bind and serve until the process is killed.
pub async fn serve_forever(state: AppState, host: &str, port: u16) -> Result<(), ServeError> {
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .map_err(|e| ServeError::Bind {
            host: host.to_string(),
            port,
            detail: e.to_string(),
        })?;
    serve(listener, router(state).into_make_service())
        .await
        .map_err(|e| ServeError::Io(e.to_string()))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub n: Option<u32>,
    // Accepted for OpenAI wire compatibility and ignored, matching the
    // reference: FreeToken declares these fields and never reads them. Without
    // this, `deny_unknown_fields` 400s a drop-in client that sends the defaults.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub best_of: Option<i64>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

impl CompletionRequest {
    /// OpenAI field mapping onto the domain params; omitted fields fall back to
    /// the domain defaults (greedy, no top_k/top_p filtering).
    fn sampling_params(&self) -> SamplingParams {
        CompletionRequest::params_of(
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.stop.clone(),
            self.ignore_eos,
        )
    }

    fn params_of(
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i64>,
        stop: Option<Stop>,
        ignore_eos: Option<bool>,
    ) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            temperature: temperature.unwrap_or(d.temperature),
            top_k: top_k.unwrap_or(d.top_k),
            top_p: top_p.unwrap_or(d.top_p),
            ignore_eos: ignore_eos.unwrap_or(d.ignore_eos),
            max_tokens: max_tokens.unwrap_or(d.max_tokens),
            stop_strs: stop.map(Stop::into_vec).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    /// `null` while the stream is still open; set on the frame that closes it.
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// `POST /v1/chat/completions`. The conversation is rendered with the
/// checkpoint's own chat template, so the body carries turns and not a prompt.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatTurn>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub n: Option<u32>,
    /// Callable functions the model may call; rendered into the system turn.
    #[serde(default)]
    pub tools: Vec<ChatToolSpec>,
    /// `auto`/`none`/`required` or a named function; accepted and ignored (the
    /// model decides from the prompt the template builds).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    // Accepted and ignored; see `CompletionRequest`.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub best_of: Option<i64>,
    #[serde(default)]
    pub user: Option<String>,
}

/// One OpenAI function tool: `{"type":"function","function":{...}}`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatToolSpec {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: FunctionDef,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

impl ChatToolSpec {
    fn to_domain(&self) -> ChatTool {
        ChatTool {
            name: self.function.name.clone(),
            description: self.function.description.clone(),
            parameters: self.function.parameters.as_ref().map(|v| v.to_string()),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatTurn {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Assistant tool calls answered later by a `role: "tool"` turn.
    #[serde(default)]
    pub tool_calls: Option<Vec<TurnToolCall>>,
    /// On a `role: "tool"` turn, the id of the call this result answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: TurnFunction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
}

impl ChatTurn {
    fn to_domain(&self) -> ChatMessage {
        ChatMessage {
            role: self.role.clone(),
            content: self.content.clone(),
            tool_calls: self
                .tool_calls
                .as_ref()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|c| ChatToolCall {
                            id: c.id.clone(),
                            name: c.function.name.clone(),
                            arguments: arguments_text(c.function.arguments.as_ref()),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            tool_call_id: self.tool_call_id.clone(),
        }
    }
}

/// A client sends `arguments` as a JSON string (OpenAI's shape); accept an
/// object too. The domain wants the wire JSON string either way.
fn arguments_text(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    }
}

impl ChatCompletionRequest {
    fn sampling_params(&self) -> SamplingParams {
        CompletionRequest::params_of(
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.stop.clone(),
            self.ignore_eos,
        )
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OutToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct OutToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OutFunction,
}

#[derive(Debug, Serialize)]
pub struct OutFunction {
    pub name: String,
    pub arguments: String,
}

/// One streamed chat chunk: `delta` carries the role on the opening frame and
/// the content after it, `finish_reason` only on the frame that closes it.
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// A `tool_calls` stream delta: the opening frame names `index`/`id`/`type`/
/// `function.name`, later frames carry `function.arguments` in pieces.
#[derive(Debug, Serialize)]
pub struct DeltaToolCall {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<DeltaFunction>,
}

#[derive(Debug, Serialize)]
pub struct DeltaFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// `POST /v1/responses` (the OpenAI Responses wire, used by Codex). The
/// stateless subset: `input` and `instructions` are rendered through the same
/// chat template as `/v1/chat/completions`, so tool calling behaves identically.
/// `store`/`previous_response_id` are accepted and ignored because this server
/// keeps no conversation state; Codex resends full context in `input`.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
    /// Flattened Responses tools: `{"type":"function","name",...}`.
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    // Accepted for wire compatibility and ignored: reasoning is out of scope
    // (decision 74), there is no response store, and the server has no
    // background job queue.
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub truncation: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesItem>),
}

/// One Responses input item. Unknown fields are allowed because Codex replays
/// output items verbatim (`id`, `status`, `annotations`, ...) and sends fields
/// this subset does not read.
#[derive(Debug, Deserialize)]
pub struct ResponsesItem {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<ResponsesContent>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
    #[serde(default)]
    pub output: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ResponsesContentPart {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

impl ResponsesContent {
    fn text(&self) -> String {
        match self {
            ResponsesContent::Text(text) => text.clone(),
            ResponsesContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<String>(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ResponsesTool {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    /// Accepted as well as the flattened shape; Responses itself sends flat.
    #[serde(default)]
    pub function: Option<ResponsesFunction>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

impl ResponsesTool {
    fn to_domain(&self) -> Option<ChatTool> {
        if self.kind.as_deref().is_some_and(|kind| kind != "function") {
            return None;
        }
        let (name, description, parameters) = match &self.function {
            Some(function) => (
                function.name.clone(),
                function.description.clone(),
                function.parameters.clone(),
            ),
            None => (
                self.name.clone().unwrap_or_default(),
                self.description.clone(),
                self.parameters.clone(),
            ),
        };
        (!name.is_empty()).then(|| ChatTool {
            name,
            description,
            parameters: parameters.map(|p| p.to_string()),
        })
    }
}

impl ResponsesRequest {
    fn sampling_params(&self) -> SamplingParams {
        CompletionRequest::params_of(
            self.max_output_tokens,
            self.temperature,
            self.top_p,
            None,
            None,
            None,
        )
    }

    fn chat_tools(&self) -> Vec<ChatTool> {
        let all: Vec<ChatTool> = self
            .tools
            .iter()
            .filter_map(ResponsesTool::to_domain)
            .collect();
        match &self.tool_choice {
            Some(serde_json::Value::String(choice)) if choice == "none" => Vec::new(),
            Some(serde_json::Value::Object(choice))
                if choice.get("type").and_then(|v| v.as_str()) == Some("function") =>
            {
                let name = choice.get("name").and_then(|v| v.as_str()).or_else(|| {
                    choice
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                });
                match name {
                    Some(name) => all.into_iter().filter(|tool| tool.name == name).collect(),
                    None => all,
                }
            }
            _ => all,
        }
    }

    fn chat_messages(&self) -> Result<Vec<ChatMessage>, ApiError> {
        let mut system = Vec::new();
        if let Some(instructions) = &self.instructions
            && !instructions.is_empty()
        {
            system.push(instructions.clone());
        }
        let mut turns: Vec<ChatMessage> = Vec::new();
        match &self.input {
            ResponsesInput::Text(text) => {
                if text.is_empty() {
                    return Err(ApiError::bad_request("input must not be empty"));
                }
                turns.push(ChatMessage::new("user", text));
            }
            ResponsesInput::Items(items) => {
                for item in items {
                    match item.kind.as_deref().unwrap_or("message") {
                        "message" => {
                            let role = item.role.as_deref().unwrap_or("user");
                            let role = if role == "developer" { "system" } else { role };
                            let content = item
                                .content
                                .as_ref()
                                .map(ResponsesContent::text)
                                .unwrap_or_default();
                            if role == "system" {
                                if !content.is_empty() {
                                    system.push(content);
                                }
                            } else if role == "assistant" {
                                merge_assistant_content(&mut turns, content);
                            } else {
                                turns.push(ChatMessage {
                                    role: role.to_string(),
                                    content: Some(content),
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                });
                            }
                        }
                        "function_call" => {
                            let call = ChatToolCall {
                                id: item
                                    .call_id
                                    .clone()
                                    .or_else(|| item.id.clone())
                                    .unwrap_or_default(),
                                name: item.name.clone().unwrap_or_default(),
                                arguments: arguments_text(item.arguments.as_ref()),
                            };
                            merge_assistant_call(&mut turns, call);
                        }
                        "function_call_output" => {
                            turns.push(ChatMessage {
                                role: "tool".to_string(),
                                content: Some(stringify(item.output.as_ref())),
                                tool_calls: Vec::new(),
                                tool_call_id: item.call_id.clone(),
                            });
                        }
                        // Reasoning items are out of scope (decision 74); Codex
                        // replays them, but the model sees only text and calls.
                        "reasoning" => {}
                        _ => {}
                    }
                }
            }
        }
        if !system.is_empty() {
            turns.insert(0, ChatMessage::new("system", &system.join("\n\n")));
        }
        if turns.is_empty() {
            return Err(ApiError::bad_request("input must not be empty"));
        }
        Ok(turns)
    }
}

fn merge_assistant_content(turns: &mut Vec<ChatMessage>, content: String) {
    if let Some(last) = turns.last_mut()
        && last.role == "assistant"
        && last.content.is_none()
        && last.tool_calls.is_empty()
    {
        last.content = Some(content);
        return;
    }
    turns.push(ChatMessage {
        role: "assistant".to_string(),
        content: Some(content),
        tool_calls: Vec::new(),
        tool_call_id: None,
    });
}

fn merge_assistant_call(turns: &mut Vec<ChatMessage>, call: ChatToolCall) {
    if let Some(last) = turns.last_mut()
        && last.role == "assistant"
    {
        last.tool_calls.push(call);
        return;
    }
    turns.push(ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: vec![call],
        tool_call_id: None,
    });
}

fn stringify(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
                .collect();
            if texts.len() == parts.len() {
                texts.concat()
            } else {
                serde_json::Value::Array(parts.clone()).to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: Option<ResponsesUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponsesError>,
    pub parallel_tool_calls: bool,
    pub tool_choice: serde_json::Value,
    pub tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesError {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ResponsesOutputItem {
    Message(ResponsesOutputMessage),
    FunctionCall(ResponsesFunctionCall),
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesOutputMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub status: &'static str,
    pub content: Vec<ResponsesOutputText>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesOutputText {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesFunctionCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    pub input_tokens_details: ResponsesInputTokens,
    pub output_tokens_details: ResponsesOutputTokens,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesInputTokens {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesOutputTokens {
    pub reasoning_tokens: usize,
}

fn responses_usage(g: &generation::Generation) -> ResponsesUsage {
    ResponsesUsage {
        input_tokens: g.prompt_tokens,
        output_tokens: g.tokens.len(),
        total_tokens: g.prompt_tokens + g.tokens.len(),
        input_tokens_details: ResponsesInputTokens { cached_tokens: 0 },
        output_tokens_details: ResponsesOutputTokens {
            reasoning_tokens: 0,
        },
    }
}

fn responses_output(text: &str, calls: Vec<ParsedToolCall>, uid: u64) -> Vec<ResponsesOutputItem> {
    let mut output = Vec::new();
    if !text.is_empty() {
        output.push(ResponsesOutputItem::Message(ResponsesOutputMessage {
            id: format!("msg_{uid:08x}"),
            kind: "message",
            role: "assistant",
            status: "completed",
            content: vec![ResponsesOutputText {
                kind: "output_text",
                text: text.to_string(),
                annotations: Vec::new(),
            }],
        }));
    }
    for (index, call) in calls.into_iter().enumerate() {
        output.push(ResponsesOutputItem::FunctionCall(ResponsesFunctionCall {
            id: format!("fc_{uid:08x}_{index}"),
            kind: "function_call",
            call_id: call.id,
            name: call.name,
            arguments: call.arguments,
            status: "completed",
        }));
    }
    output
}

/// The fields every Responses snapshot repeats. Owned so a stream sink can be
/// `'static`; the non-stream handler builds one too.
#[derive(Debug, Clone)]
struct ResponsesBase {
    id: String,
    model: String,
    created_at: u64,
    parallel_tool_calls: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    instructions: Option<String>,
    metadata: Option<serde_json::Value>,
}

impl ResponsesBase {
    fn new(state: &AppState, req: &ResponsesRequest, uid: u64) -> Self {
        Self {
            id: format!("resp_{uid:08x}"),
            model: state.served_model_name.clone(),
            created_at: unix_now(),
            parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
            temperature: req.temperature,
            top_p: req.top_p,
            instructions: req.instructions.clone(),
            metadata: req.metadata.clone(),
        }
    }

    fn response(
        &self,
        status: &'static str,
        output: Vec<ResponsesOutputItem>,
        usage: Option<ResponsesUsage>,
        error: Option<ResponsesError>,
    ) -> ResponsesResponse {
        ResponsesResponse {
            id: self.id.clone(),
            object: "response",
            created_at: self.created_at,
            model: self.model.clone(),
            status,
            output,
            usage,
            error,
            parallel_tool_calls: self.parallel_tool_calls,
            tool_choice: serde_json::Value::String("auto".to_string()),
            tools: Vec::new(),
            temperature: self.temperature,
            top_p: self.top_p,
            instructions: self.instructions.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// The Responses event stream over the same per-token `ChatStream` the chat
/// endpoint uses. It turns parsed content into `output_text` deltas and a
/// parsed call into one `function_call` item, then closes with a snapshot.
struct ResponsesStream {
    base: ResponsesBase,
    uid: u64,
    seq: u64,
    chat: ChatStream,
    message: Option<(String, String)>,
    output: Vec<ResponsesOutputItem>,
}

impl ResponsesStream {
    fn new(base: ResponsesBase, uid: u64) -> Self {
        Self {
            base,
            uid,
            seq: 0,
            chat: ChatStream::new(),
            message: None,
            output: Vec::new(),
        }
    }

    fn event(&mut self, kind: &str, body: serde_json::Value) -> String {
        let mut body = match body {
            serde_json::Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        body.insert("type".into(), serde_json::Value::String(kind.to_string()));
        body.insert("sequence_number".into(), serde_json::json!(self.seq));
        self.seq += 1;
        let payload = serde_json::Value::Object(body).to_string();
        format!("event: {kind}\ndata: {payload}\n\n")
    }

    fn opened(&mut self) -> Vec<String> {
        let response = self.base.response("in_progress", Vec::new(), None, None);
        let body = serde_json::json!({ "response": response });
        vec![self.event("response.created", body)]
    }

    fn token(&mut self, text: &str) -> Vec<String> {
        let actions = self.chat.feed(text);
        self.actions(actions)
    }

    fn end_of_text(&mut self) -> Vec<String> {
        let (_emitted, flush) = self.chat.finish();
        match flush {
            Some(text) => self.actions(vec![ChatAction::Content(text)]),
            None => Vec::new(),
        }
    }

    fn actions(&mut self, actions: Vec<ChatAction>) -> Vec<String> {
        let mut frames = Vec::new();
        for action in actions {
            match action {
                ChatAction::Content(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    let index = self.output.len();
                    if self.message.is_none() {
                        let id = format!("msg_{:08x}_{index}", self.uid);
                        frames.push(self.event(
                            "response.output_item.added",
                            serde_json::json!({
                                "output_index": index,
                                "item": {
                                    "id": id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            }),
                        ));
                        frames.push(self.event(
                            "response.content_part.added",
                            serde_json::json!({
                                "item_id": id,
                                "output_index": index,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            }),
                        ));
                        self.message = Some((id, String::new()));
                    }
                    let id = {
                        let (id, full) = self.message.as_mut().expect("a message is open");
                        full.push_str(&text);
                        id.clone()
                    };
                    frames.push(self.event(
                        "response.output_text.delta",
                        serde_json::json!({
                            "item_id": id,
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                            "logprobs": [],
                        }),
                    ));
                }
                ChatAction::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } => {
                    self.close_message(&mut frames);
                    let index = self.output.len();
                    let item_id = format!("fc_{:08x}_{index}", self.uid);
                    frames.push(self.event(
                        "response.output_item.added",
                        serde_json::json!({
                            "output_index": index,
                            "item": {
                                "id": item_id,
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        }),
                    ));
                    frames.push(self.event(
                        "response.function_call_arguments.delta",
                        serde_json::json!({
                            "item_id": item_id,
                            "output_index": index,
                            "delta": arguments,
                        }),
                    ));
                    frames.push(self.event(
                        "response.function_call_arguments.done",
                        serde_json::json!({
                            "item_id": item_id,
                            "output_index": index,
                            "name": name,
                            "arguments": arguments,
                        }),
                    ));
                    let item = ResponsesFunctionCall {
                        id: item_id,
                        kind: "function_call",
                        call_id: id,
                        name,
                        arguments,
                        status: "completed",
                    };
                    frames.push(self.event(
                        "response.output_item.done",
                        serde_json::json!({ "output_index": index, "item": item }),
                    ));
                    self.output.push(ResponsesOutputItem::FunctionCall(item));
                }
            }
        }
        frames
    }

    fn close_message(&mut self, frames: &mut Vec<String>) {
        let Some((id, text)) = self.message.take() else {
            return;
        };
        let index = self.output.len();
        frames.push(self.event(
            "response.output_text.done",
            serde_json::json!({
                "item_id": id,
                "output_index": index,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            }),
        ));
        frames.push(self.event(
            "response.content_part.done",
            serde_json::json!({
                "item_id": id,
                "output_index": index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                },
            }),
        ));
        let item = ResponsesOutputMessage {
            id,
            kind: "message",
            role: "assistant",
            status: "completed",
            content: vec![ResponsesOutputText {
                kind: "output_text",
                text,
                annotations: Vec::new(),
            }],
        };
        frames.push(self.event(
            "response.output_item.done",
            serde_json::json!({ "output_index": index, "item": item }),
        ));
        self.output.push(ResponsesOutputItem::Message(item));
    }

    fn finish(&mut self, g: &generation::Generation) -> Vec<String> {
        let mut frames = self.end_of_text();
        self.close_message(&mut frames);
        let response = self.base.response(
            "completed",
            self.output.clone(),
            Some(responses_usage(g)),
            None,
        );
        frames.push(self.event(
            "response.completed",
            serde_json::json!({ "response": response }),
        ));
        frames
    }

    fn failed(&mut self, message: &str) -> String {
        let response = self.base.response(
            "failed",
            self.output.clone(),
            None,
            Some(ResponsesError {
                code: "server_error",
                message: message.to_string(),
            }),
        );
        self.event(
            "response.failed",
            serde_json::json!({ "response": response }),
        )
    }
}

/// `stream: true` for `/v1/responses`: the same dispatcher job and per-token
/// sink as the chat stream, formatted as Responses events. Frames ride a
/// bounded channel, so a slow client drops its own stream instead of stalling
/// the batch it rides in.
fn responses_stream_body(
    state: &Arc<AppState>,
    request: generation::Request,
    req: &ResponsesRequest,
) -> Response {
    let (tx, rx) = mpsc::channel::<String>(32);
    let base = ResponsesBase::new(state, req, request.uid);
    let stream = Arc::new(Mutex::new(ResponsesStream::new(base, request.uid)));
    for frame in stream.lock().expect("responses stream lock").opened() {
        let _ = tx.try_send(frame);
    }

    let sink_tx = tx.clone();
    let sink_stream = Arc::clone(&stream);
    let sink: crate::application::dispatcher::Sink = Box::new(move |_uid, text, _id, _index| {
        let mut stream = sink_stream.lock().expect("responses stream lock");
        for frame in stream.token(text) {
            if sink_tx.try_send(frame).is_err() {
                return false;
            }
        }
        true
    });

    let (reply, rx_reply) = tokio::sync::oneshot::channel();
    if state
        .dispatcher
        .submit(Job {
            request,
            sink: Some(sink),
            reply,
        })
        .is_err()
    {
        let frame = stream
            .lock()
            .expect("responses stream lock")
            .failed("the engine worker is not running");
        let _ = tx.try_send(frame);
        let _ = tx.try_send(sse_data("[DONE]"));
        drop(tx);
        return sse_response(rx);
    }

    let final_tx = tx.clone();
    let final_stream = Arc::clone(&stream);
    tokio::spawn(async move {
        let reply = rx_reply.await;
        let frames = {
            let mut stream = final_stream.lock().expect("responses stream lock");
            match reply {
                Ok(Ok(g)) if g.finish_reason == FinishReason::Abort => {
                    vec![stream.failed("the generation was aborted")]
                }
                Ok(Ok(g)) => stream.finish(&g),
                Ok(Err(msg)) => vec![stream.failed(&msg)],
                Err(_) => vec![stream.failed("the engine worker dropped the request")],
            }
        };
        for frame in frames {
            let _ = final_tx.try_send(frame);
        }
        let _ = final_tx.try_send(sse_data("[DONE]"));
        drop(final_tx);
    });
    drop(tx);
    sse_response(rx)
}

async fn responses(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) =
        payload.map_err(|e| ApiError::bad_request(format!("invalid json body: {e}")))?;

    if !req.model.is_empty() && req.model != state.served_model_name {
        return Err(ApiError::not_found(format!(
            "model {} not found, this server serves {}",
            req.model, state.served_model_name
        )));
    }
    if req.max_output_tokens == Some(0) {
        return Err(ApiError::bad_request(
            "max_output_tokens must be a positive integer",
        ));
    }
    if req.background == Some(true) {
        return Err(ApiError::bad_request("background mode is not supported"));
    }

    let messages = req.chat_messages()?;
    let tools = req.chat_tools();
    let uid = state.next_uid();
    let engine = Arc::clone(&state.engine);
    let params = req.sampling_params();
    let streaming = req.stream.unwrap_or(false);

    let worker = Arc::clone(&engine);
    let prepared =
        tokio::task::spawn_blocking(move || -> Result<generation::Request, EngineError> {
            let prompt = worker.render_chat(&messages, &tools)?;
            let ids = worker.encode(&prompt)?;
            make_request(&worker, uid, ids, params, uid)
        })
        .await
        .expect("preparation task is not cancellable");
    let request = match prepared {
        Ok(r) => r,
        Err(e) => return Err(ApiError::from(e)),
    };

    if streaming {
        return Ok(responses_stream_body(&state, request, &req));
    }

    let generated = dispatch(&state, request, None).await?;
    let (content, calls) = parse_tool_calls(&generated.text);
    let output = responses_output(&content, calls, uid);
    let base = ResponsesBase::new(&state, &req, uid);
    Ok(
        Json(base.response("completed", output, Some(responses_usage(&generated)), None))
            .into_response(),
    )
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model: String,
}

/// OpenAI-shaped error body: `{"error": {"message", "type", "code"}}`.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub code: &'static str,
}

pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "invalid_request",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: "model_not_found",
            message: message.into(),
        }
    }

    fn unprocessable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            kind: "invalid_request_error",
            code: "unsupported_value",
            message: message.into(),
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        use EngineError::*;
        match e {
            EmptyPrompt => ApiError::bad_request("prompt must not be empty"),
            PromptTooLong { len, max } => {
                ApiError::unprocessable(format!("prompt has {len} tokens, over the {max} limit"))
            }
            Tokenize(m) => ApiError::bad_request(format!("prompt could not be tokenized: {m}")),
            Chat(m) => ApiError::bad_request(m),
            BadLogits { got, want } => {
                ApiError::unprocessable(format!("model returned {got} logits, expected {want}"))
            }
            NoParams(uid) => ApiError::unprocessable(format!("no sampling parameters for {uid}")),
            // Engine-side faults, not anything the caller can fix: the cache is
            // a server resource and its bookkeeping failing is a server bug.
            e @ (NoKvCache
            | EmptyStep { .. }
            | CacheOutOfStep { .. }
            | KvCache(_)
            | NoCapturedDecode
            | DecodeStepNotOne { .. }
            | Captured(_)) => ApiError::unprocessable(format!("engine kv cache: {e}")),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.message,
                kind: self.kind,
                code: self.code,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finish_reason_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Abort => "abort",
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) =
        payload.map_err(|e| ApiError::bad_request(format!("invalid json body: {e}")))?;

    if !req.model.is_empty() && req.model != state.served_model_name {
        return Err(ApiError::not_found(format!(
            "model {} not found, this server serves {}",
            req.model, state.served_model_name
        )));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::unprocessable("only n=1 is supported"));
    }

    let uid = state.next_uid();
    let engine = Arc::clone(&state.engine);
    let params = req.sampling_params();
    let seed = req.seed.unwrap_or(uid);
    let stream = req.stream.unwrap_or(false);
    let prompt = req.prompt;

    // Tokenize off the runtime, then hand the request to the dispatcher, which
    // owns the single model worker and batches whatever is queued behind it.
    let ids = {
        let tokenizer = Arc::clone(&engine);
        match tokio::task::spawn_blocking(move || tokenizer.encode(&prompt))
            .await
            .expect("encoding task is not cancellable")
        {
            Ok(ids) => ids,
            Err(e) => return Err(ApiError::from(e)),
        }
    };
    let request = make_request(&engine, uid, ids, params, seed)?;

    if stream {
        return Ok(stream_body(&state, request, Wire::Text));
    }

    let generated = dispatch(&state, request, None).await?;
    let usage = usage_of(&generated);
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", generated.uid),
        object: "text_completion",
        created: unix_now(),
        model: state.served_model_name.clone(),
        choices: vec![CompletionChoice {
            text: generated.text,
            index: 0,
            logprobs: None,
            finish_reason: Some(finish_reason_str(generated.finish_reason)),
        }],
        usage: Some(usage),
    })
    .into_response())
}

/// Submit one request to the engine worker and await its `Generation`. The
/// worker batches whatever it queued behind it (TODO.md item 11); a per-token
/// `sink` rides along when the caller wants a stream.
async fn dispatch(
    state: &Arc<AppState>,
    request: Request,
    sink: Option<crate::application::dispatcher::Sink>,
) -> Result<generation::Generation, ApiError> {
    let (reply, rx) = tokio::sync::oneshot::channel();
    if state
        .dispatcher
        .submit(Job {
            request,
            sink,
            reply,
        })
        .is_err()
    {
        return Err(ApiError::unprocessable("the engine worker is not running"));
    }
    match rx.await {
        Ok(Ok(g)) => Ok(g),
        Ok(Err(msg)) => Err(ApiError::unprocessable(format!("generation failed: {msg}"))),
        Err(_) => Err(ApiError::unprocessable(
            "the engine worker dropped the request",
        )),
    }
}

fn usage_of(g: &generation::Generation) -> Usage {
    Usage {
        prompt_tokens: g.prompt_tokens,
        completion_tokens: g.tokens.len(),
        total_tokens: g.prompt_tokens + g.tokens.len(),
    }
}

/// `stream: true`: one `data: {...}` frame per token as it is sampled, a frame
/// carrying the finish reason plus the usage, then `data: [DONE]`.
///
/// The prompt is tokenized and clamped before the stream starts, so a bad
/// request is still an ordinary JSON error with its status code. Everything
/// after that has already sent headers, so a failure mid-run rides on the stream
/// as an `error` frame. The generation runs on a blocking task and pushes frames
/// through a bounded channel: a slow client holds the run back instead of
/// buffering it, and a client that hangs up closes the channel, which stops the
/// run.
/// Which wire shape a stream speaks: `/v1/completions` frames carry `text`,
/// `/v1/chat/completions` frames carry a `delta`.
#[derive(Clone, Copy)]
enum Wire {
    Text,
    Chat,
}

impl Wire {
    fn id(self, uid: u64) -> String {
        match self {
            Wire::Text => format!("cmpl-{uid}"),
            Wire::Chat => format!("chatcmpl-{uid}"),
        }
    }
}

/// `stream: true`: one `data: {...}` frame per token as it is sampled, a frame
/// carrying the finish reason plus the usage, then `data: [DONE]`.
///
/// The prompt is tokenized and clamped before the stream starts, so a bad
/// request is still an ordinary JSON error with its status code. Everything
/// after that has already sent headers, so a failure mid-run rides on the stream
/// as an `error` frame. The generation runs on a blocking task and pushes frames
/// through a bounded channel: a slow client holds the run back instead of
/// buffering it, and a client that hangs up closes the channel, which stops the
/// run.
/// Builds the SSE frames a stream sends. Owns its ids so the per-token sink can
/// be a `'static + Send` closure the engine worker calls.
#[derive(Clone)]
struct Frames {
    wire: Wire,
    id: String,
    created: u64,
    model: String,
}

impl Frames {
    fn chunk(&self, text: &str, finish: Option<&'static str>, usage: Option<Usage>) -> String {
        let payload = match self.wire {
            Wire::Text => serde_json::to_string(&CompletionResponse {
                id: self.id.clone(),
                object: "text_completion",
                created: self.created,
                model: self.model.clone(),
                choices: vec![CompletionChoice {
                    text: text.to_string(),
                    index: 0,
                    logprobs: None,
                    finish_reason: finish,
                }],
                usage,
            }),
            Wire::Chat => serde_json::to_string(&ChatChunk {
                id: self.id.clone(),
                object: "chat.completion.chunk",
                created: self.created,
                model: self.model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: (finish.is_none() && text.is_empty()).then_some("assistant"),
                        content: text.to_string(),
                        tool_calls: None,
                    },
                    finish_reason: finish,
                }],
                usage,
            }),
        };
        sse_data(&payload.expect("chunk serializes"))
    }

    /// The opening chat frame: names the role before any content arrives.
    fn role(&self) -> String {
        self.chat_frame(ChatDelta {
            role: Some("assistant"),
            content: String::new(),
            tool_calls: None,
        })
    }

    /// The first frame of a tool call: names `index`, `id`, `type` and the
    /// function, with empty arguments.
    fn tool_open(&self, index: usize, id: &str, name: &str) -> String {
        self.chat_frame(ChatDelta {
            role: None,
            content: String::new(),
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: Some(id.to_string()),
                kind: Some("function"),
                function: Some(DeltaFunction {
                    name: Some(name.to_string()),
                    arguments: None,
                }),
            }]),
        })
    }

    /// A follow-up frame carrying the call's `arguments` JSON.
    fn tool_args(&self, index: usize, arguments: &str) -> String {
        self.chat_frame(ChatDelta {
            role: None,
            content: String::new(),
            tool_calls: Some(vec![DeltaToolCall {
                index,
                id: None,
                kind: None,
                function: Some(DeltaFunction {
                    name: None,
                    arguments: Some(arguments.to_string()),
                }),
            }]),
        })
    }

    fn chat_frame(&self, delta: ChatDelta) -> String {
        let payload = serde_json::to_string(&ChatChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta,
                finish_reason: None,
            }],
            usage: None,
        });
        sse_data(&payload.expect("chunk serializes"))
    }

    fn error(&self, message: &str) -> String {
        let payload = serde_json::json!({
            "error": { "message": message, "type": "engine_error" }
        });
        sse_data(&payload.to_string())
    }
}

fn sse_response(rx: mpsc::Receiver<String>) -> Response {
    let body = Body::from_stream(SseBody { rx });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("a static response builder succeeds")
}

/// One thing a chat stream should emit as tokens arrive: either assistant
/// content, or a fully-parsed tool call.
enum ChatAction {
    Content(String),
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
}

/// Turns the per-token chat stream into content up to the first `<tool_call>`
/// and tool-call events after it. Text that could still grow into the opener
/// (`<`, `<tool`, ...) is held back so a token boundary cannot leak a partial
/// tag as content. An unterminated block at the end is flushed as content, so a
/// malformed answer is not silently dropped.
struct ChatStream {
    buffer: String,
    in_tool: bool,
    emitted: usize,
}

impl ChatStream {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            in_tool: false,
            emitted: 0,
        }
    }

    fn feed(&mut self, piece: &str) -> Vec<ChatAction> {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        let mut actions = Vec::new();
        self.buffer.push_str(piece);
        loop {
            if !self.in_tool {
                match self.buffer.find(OPEN) {
                    Some(pos) => {
                        if pos > 0 {
                            actions.push(ChatAction::Content(self.buffer[..pos].to_string()));
                        }
                        self.buffer.drain(..pos + OPEN.len());
                        self.in_tool = true;
                    }
                    None => {
                        let keep = partial_opener(&self.buffer, OPEN);
                        let emit = self.buffer.len() - keep;
                        if emit > 0 {
                            actions.push(ChatAction::Content(self.buffer.drain(..emit).collect()));
                        }
                        break;
                    }
                }
            } else if let Some(pos) = self.buffer.find(CLOSE) {
                let inner = self.buffer[..pos].to_string();
                self.buffer.drain(..pos + CLOSE.len());
                self.in_tool = false;
                match super::tool_call::parse_block(&inner) {
                    Some(call) => {
                        self.emitted += 1;
                        actions.push(ChatAction::ToolCall {
                            index: self.emitted - 1,
                            id: call.id,
                            name: call.name,
                            arguments: call.arguments,
                        });
                    }
                    None => actions.push(ChatAction::Content(format!("{OPEN}{inner}{CLOSE}"))),
                }
            } else {
                break;
            }
        }
        actions
    }

    /// The run is over: any still-open block or held-back suffix becomes
    /// content, but only when no complete call was emitted.
    fn finish(&mut self) -> (usize, Option<String>) {
        let pending = if self.in_tool {
            format!("<tool_call>{}", self.buffer)
        } else {
            std::mem::take(&mut self.buffer)
        };
        self.buffer.clear();
        let flush = (self.emitted == 0 && !pending.is_empty()).then_some(pending);
        (self.emitted, flush)
    }
}

/// Longest suffix of `buffer` that is a proper prefix of `open`, so it might
/// still complete into the opener on a later token.
fn partial_opener(buffer: &str, open: &str) -> usize {
    let max = open.len().saturating_sub(1).min(buffer.len());
    for keep in (1..=max).rev() {
        let start = buffer.len() - keep;
        if buffer.is_char_boundary(start) && open.starts_with(&buffer[start..]) {
            return keep;
        }
    }
    0
}

/// `stream: true`: one `data: {...}` frame per token as it is sampled, a frame
/// carrying the finish reason plus the usage, then `data: [DONE]`.
///
/// The prompt is tokenized and clamped before the stream starts, so a bad
/// request is still an ordinary JSON error with its status code. Everything
/// after that has already sent headers, so a failure mid-run rides on the stream
/// as an `error` frame. The tokens come back through the dispatcher's per-request
/// sink: frames go over a bounded channel, so a slow client drops its own stream
/// (a full buffer aborts the sink) instead of stalling the batch it rides in.
fn stream_body(state: &Arc<AppState>, request: generation::Request, wire: Wire) -> Response {
    let (tx, rx) = mpsc::channel::<String>(32);
    let frames = Frames {
        wire,
        id: wire.id(request.uid),
        created: unix_now(),
        model: state.served_model_name.clone(),
    };
    let tool_stream = Arc::new(Mutex::new(ChatStream::new()));
    if let Wire::Chat = wire {
        let _ = tx.try_send(frames.role());
    }

    let sink: crate::application::dispatcher::Sink = match wire {
        Wire::Text => {
            let sink_tx = tx.clone();
            let sink_frames = frames.clone();
            Box::new(move |_uid, text, _id, _index| {
                sink_tx
                    .try_send(sink_frames.chunk(text, None, None))
                    .is_ok()
            })
        }
        Wire::Chat => {
            let sink_tx = tx.clone();
            let sink_frames = frames.clone();
            let stream = Arc::clone(&tool_stream);
            Box::new(move |_uid, text, _id, _index| {
                let actions = stream.lock().expect("tool stream lock").feed(text);
                for action in actions {
                    let frame = match action {
                        ChatAction::Content(content) => sink_frames.chunk(&content, None, None),
                        ChatAction::ToolCall {
                            index,
                            id,
                            name,
                            arguments,
                        } => {
                            if sink_tx
                                .try_send(sink_frames.tool_open(index, &id, &name))
                                .is_err()
                            {
                                return false;
                            }
                            sink_frames.tool_args(index, &arguments)
                        }
                    };
                    if sink_tx.try_send(frame).is_err() {
                        return false;
                    }
                }
                true
            })
        }
    };

    let (reply, rx_reply) = tokio::sync::oneshot::channel();
    if state
        .dispatcher
        .submit(Job {
            request,
            sink: Some(sink),
            reply,
        })
        .is_err()
    {
        let _ = tx.try_send(frames.error("the engine worker is not running"));
        let _ = tx.try_send(sse_data("[DONE]"));
        drop(tx);
        return sse_response(rx);
    }

    let final_tx = tx.clone();
    let final_frames = frames.clone();
    let final_tools = Arc::clone(&tool_stream);
    tokio::spawn(async move {
        match rx_reply.await {
            Ok(Ok(g)) => {
                if g.finish_reason != FinishReason::Abort {
                    let usage = usage_of(&g);
                    let mut reason = finish_reason_str(g.finish_reason);
                    if let Wire::Chat = final_frames.wire {
                        let (emitted, leftover) =
                            final_tools.lock().expect("tool stream lock").finish();
                        if let Some(text) = leftover {
                            let _ = final_tx.try_send(final_frames.chunk(&text, None, None));
                        }
                        if emitted > 0 {
                            reason = "tool_calls";
                        }
                    }
                    let _ = final_tx.try_send(final_frames.chunk("", Some(reason), Some(usage)));
                    let _ = final_tx.try_send(sse_data("[DONE]"));
                }
            }
            Ok(Err(msg)) => {
                let _ = final_tx.try_send(final_frames.error(&msg));
                let _ = final_tx.try_send(sse_data("[DONE]"));
            }
            Err(_) => {
                let _ =
                    final_tx.try_send(final_frames.error("the engine worker dropped the request"));
                let _ = final_tx.try_send(sse_data("[DONE]"));
            }
        }
        drop(final_tx);
    });
    drop(tx);
    sse_response(rx)
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) =
        payload.map_err(|e| ApiError::bad_request(format!("invalid json body: {e}")))?;

    if !req.model.is_empty() && req.model != state.served_model_name {
        return Err(ApiError::not_found(format!(
            "model {} not found, this server serves {}",
            req.model, state.served_model_name
        )));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::unprocessable("only n=1 is supported"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let uid = state.next_uid();
    let engine = Arc::clone(&state.engine);
    let params = req.sampling_params();
    let seed = req.seed.unwrap_or(uid);
    let streaming = req.stream.unwrap_or(false);
    let messages: Vec<ChatMessage> = req.messages.iter().map(ChatTurn::to_domain).collect();
    let tools: Vec<ChatTool> = req.tools.iter().map(ChatToolSpec::to_domain).collect();

    // Rendering the template and tokenizing it both run off the runtime, and both
    // still have to answer with a status code rather than a frame.
    let worker = Arc::clone(&engine);
    let prepared =
        tokio::task::spawn_blocking(move || -> Result<generation::Request, EngineError> {
            let prompt = worker.render_chat(&messages, &tools)?;
            let ids = worker.encode(&prompt)?;
            make_request(&worker, uid, ids, params, seed)
        })
        .await
        .expect("preparation task is not cancellable");
    let request = match prepared {
        Ok(r) => r,
        Err(e) => return Err(ApiError::from(e)),
    };

    if streaming {
        return Ok(stream_body(&state, request, Wire::Chat));
    }

    let generated = dispatch(&state, request, None).await?;
    let usage = usage_of(&generated);
    let (content, calls) = parse_tool_calls(&generated.text);
    let (message, finish_reason) = if calls.is_empty() {
        (
            ChatMessageOut {
                role: "assistant",
                content: generated.text,
                tool_calls: None,
            },
            finish_reason_str(generated.finish_reason),
        )
    } else {
        (
            ChatMessageOut {
                role: "assistant",
                content,
                tool_calls: Some(
                    calls
                        .into_iter()
                        .map(|c| OutToolCall {
                            id: c.id,
                            kind: "function",
                            function: OutFunction {
                                name: c.name,
                                arguments: c.arguments,
                            },
                        })
                        .collect(),
                ),
            },
            "tool_calls",
        )
    };
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", generated.uid),
        object: "chat.completion",
        created: unix_now(),
        model: state.served_model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason: Some(finish_reason),
        }],
        usage: Some(usage),
    })
    .into_response())
}

fn sse_data(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// The SSE body: whole frames off the channel, in order, as they arrive.
struct SseBody {
    rx: mpsc::Receiver<String>,
}

impl Stream for SseBody {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx
            .poll_recv(cx)
            .map(|frame| frame.map(|f| Ok(Bytes::from(f))))
    }
}

async fn models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.served_model_name.clone(),
            object: "model",
            created: unix_now(),
            owned_by: "ferrum-torrens".to_string(),
        }],
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.served_model_name.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::engine::fake::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_router() -> Router {
        test_router_with_rec().0
    }

    /// The fake tokenizer renders `"{role}\n{content}\n"` per turn plus the
    /// `assistant\n` generation prompt, and encodes bytes one to one.
    fn rendered(system: &str, user: &str) -> String {
        format!("system\n{system}\nuser\n{user}\nassistant\n")
    }

    /// The fake model answers with the counter one past the prompt's last id, so
    /// the expected tokens follow from the rendered prompt.
    fn fake_tokens(prompt: &str, n: usize) -> Vec<u32> {
        let last = u32::from(*prompt.as_bytes().last().expect("a prompt"));
        (1..=n as u32).map(|i| last + i).collect()
    }

    fn fake_text(tokens: &[u32]) -> String {
        tokens.iter().map(|&b| char::from(b as u8)).collect()
    }

    fn test_router_with_rec() -> (Router, Recorder) {
        let (engine, rec) = test_engine(limits());
        let state = AppState::new(Arc::new(engine), "test-model".to_string());
        (router(state), rec)
    }

    async fn post_chat(router: Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let response = router
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn post_chat_sse(body: serde_json::Value) -> Vec<String> {
        post_chat_sse_with(test_router(), body).await
    }

    async fn post_chat_sse_with(router: Router, body: serde_json::Value) -> Vec<String> {
        let response = router
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .map(str::to_string)
            .collect()
    }

    fn chunk_of(frame: &str) -> serde_json::Value {
        serde_json::from_str(&frame[6..]).expect("a data frame holds one json")
    }

    async fn post_responses_with(
        router: Router,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let response = router
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn post_responses(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        post_responses_with(test_router(), body).await
    }

    /// Responses SSE frames pair an `event:` line with a `data:` line, so they
    /// are split whole; `[DONE]` is a bare data frame.
    async fn post_responses_sse_with(router: Router, body: serde_json::Value) -> Vec<String> {
        let response = router
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .map(str::to_string)
            .collect()
    }

    fn responses_event(frame: &str) -> (String, serde_json::Value) {
        let mut kind = String::new();
        let mut data = String::new();
        for line in frame.split('\n') {
            if let Some(rest) = line.strip_prefix("event: ") {
                kind = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = rest.to_string();
            }
        }
        let value = serde_json::from_str(&data).expect("a data line holds one json");
        (kind, value)
    }

    #[tokio::test]
    async fn responses_text_input_answers_a_message_item() {
        let (router, rec) = test_router_with_rec();
        let (status, body) = post_responses_with(
            router,
            serde_json::json!({
                "model": "test-model",
                "instructions": "be brief",
                "input": "ab",
                "max_output_tokens": 3,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "response");
        assert!(body["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["role"], "assistant");
        assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(
            body["output"][0]["content"][0]["text"],
            fake_text(&fake_tokens(&rendered("be brief", "ab"), 3))
        );
        assert_eq!(
            body["usage"]["input_tokens"],
            rendered("be brief", "ab").len()
        );
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(
            body["usage"]["total_tokens"],
            rendered("be brief", "ab").len() + 3
        );

        let prompt: Vec<u32> = rendered("be brief", "ab").bytes().map(u32::from).collect();
        assert_eq!(rec.calls()[0].input_ids, prompt);
    }

    #[tokio::test]
    async fn responses_function_call_output_renders_the_tool_turn() {
        let (router, rec) = test_router_with_rec();
        let (status, _) = post_responses_with(
            router,
            serde_json::json!({
                "input": [
                    { "role": "user", "content": "weather?" },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    },
                    { "type": "function_call_output", "call_id": "call_1", "output": "sunny" }
                ],
                "max_output_tokens": 1,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let prompt = "user\nweather?\nassistant\n\ntool\nsunny\nassistant\n";
        assert_eq!(
            rec.calls()[0].input_ids,
            prompt.bytes().map(u32::from).collect::<Vec<u32>>(),
            "the function call and its output become assistant/tool turns"
        );
    }

    #[tokio::test]
    async fn responses_tool_call_returns_a_function_call_item() {
        let (status, body) = post_responses_with(
            tool_router(),
            serde_json::json!({
                "model": "test-model",
                "input": "weather in Paris?",
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "look up weather",
                    "parameters": { "type": "object" }
                }],
                "max_output_tokens": TOOL_SCRIPT.len(),
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "completed");
        let output = body["output"].as_array().expect("output array");
        assert_eq!(output.len(), 1, "no text, one call: {output:?}");
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["name"], "get_weather");
        assert_eq!(output[0]["status"], "completed");
        assert!(output[0]["call_id"].as_str().unwrap().starts_with("call_"));
        let args: serde_json::Value =
            serde_json::from_str(output[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[tokio::test]
    async fn responses_streams_text_deltas_and_completes() {
        let frames = post_responses_sse_with(
            test_router(),
            serde_json::json!({
                "input": "ab",
                "instructions": "be brief",
                "max_output_tokens": 3,
                "stream": true,
            }),
        )
        .await;
        let events: Vec<(String, serde_json::Value)> = frames
            .iter()
            .filter(|f| *f != "data: [DONE]")
            .map(|f| responses_event(f))
            .collect();
        assert_eq!(events[0].0, "response.created");
        assert_eq!(events[0].1["response"]["status"], "in_progress");
        assert_eq!(
            events
                .iter()
                .filter(|(kind, _)| kind == "response.output_text.delta")
                .count(),
            3
        );
        let text: String = events
            .iter()
            .filter(|(kind, _)| kind == "response.output_text.delta")
            .map(|(_, data)| data["delta"].as_str().unwrap())
            .collect();
        assert_eq!(
            text,
            fake_text(&fake_tokens(&rendered("be brief", "ab"), 3))
        );
        let done = events
            .iter()
            .find(|(kind, _)| kind == "response.output_text.done")
            .expect("a text done event");
        assert_eq!(done.1["text"], text);
        let completed = events
            .iter()
            .find(|(kind, _)| kind == "response.completed")
            .expect("a completed event");
        assert_eq!(completed.1["response"]["status"], "completed");
        assert_eq!(
            completed.1["response"]["output"][0]["content"][0]["text"],
            text
        );
        assert_eq!(completed.1["response"]["usage"]["output_tokens"], 3);
        assert_eq!(frames.last().unwrap(), "data: [DONE]");
    }

    #[tokio::test]
    async fn responses_streams_function_call_arguments() {
        let frames = post_responses_sse_with(
            tool_router(),
            serde_json::json!({
                "input": "weather in Paris?",
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "parameters": { "type": "object" }
                }],
                "max_output_tokens": TOOL_SCRIPT.len(),
                "stream": true,
            }),
        )
        .await;
        let events: Vec<(String, serde_json::Value)> = frames
            .iter()
            .filter(|f| *f != "data: [DONE]")
            .map(|f| responses_event(f))
            .collect();
        let added = events
            .iter()
            .find(|(kind, _)| kind == "response.output_item.added")
            .expect("a call item added");
        assert_eq!(added.1["item"]["type"], "function_call");
        assert_eq!(added.1["item"]["name"], "get_weather");
        let args_done = events
            .iter()
            .find(|(kind, _)| kind == "response.function_call_arguments.done")
            .expect("arguments done");
        let args: serde_json::Value =
            serde_json::from_str(args_done.1["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
        let completed = events
            .iter()
            .find(|(kind, _)| kind == "response.completed")
            .expect("a completed event");
        assert_eq!(
            completed.1["response"]["output"][0]["type"],
            "function_call"
        );
        assert_eq!(frames.last().unwrap(), "data: [DONE]");
    }

    #[tokio::test]
    async fn responses_ignores_stateful_fields_and_rejects_an_empty_input() {
        let (status, _) = post_responses(serde_json::json!({
            "input": "hi",
            "store": true,
            "previous_response_id": "resp_other",
            "metadata": { "k": "v" },
            "text": { "format": { "type": "text" } },
            "reasoning": { "effort": "low" },
            "max_output_tokens": 1,
        }))
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = post_responses(serde_json::json!({ "input": [] })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].as_str().unwrap().contains("input"));

        let (status, _) =
            post_responses(serde_json::json!({ "input": "hi", "max_output_tokens": 0 })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_completions_answer_with_the_rendered_template() {
        let (router, rec) = test_router_with_rec();
        let (status, body) = post_chat(
            router,
            serde_json::json!({
                "model": "test-model",
                "messages": [
                    { "role": "system", "content": "be brief" },
                    { "role": "user", "content": "ab" }
                ],
                "max_tokens": 3,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(
            body["choices"][0]["message"]["content"],
            fake_text(&fake_tokens(&rendered("be brief", "ab"), 3))
        );
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(
            body["usage"]["prompt_tokens"],
            rendered("be brief", "ab").len()
        );
        assert_eq!(body["usage"]["completion_tokens"], 3);

        let prompt: Vec<u32> = rendered("be brief", "ab").bytes().map(u32::from).collect();
        assert_eq!(
            rec.calls()[0].input_ids,
            prompt,
            "the model must see the template-rendered prompt"
        );
    }

    #[tokio::test]
    async fn a_chat_stream_opens_with_the_role_then_streams_content() {
        let frames = post_chat_sse(serde_json::json!({
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "ab" }
            ],
            "max_tokens": 3,
            "stream": true,
        }))
        .await;
        assert_eq!(frames.len(), 6, "role, 3 tokens, the final frame, [DONE]");

        let opening = chunk_of(&frames[0]);
        assert_eq!(opening["object"], "chat.completion.chunk");
        assert_eq!(opening["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(opening["choices"][0]["delta"]["content"], "");
        assert!(opening["choices"][0]["finish_reason"].is_null());

        let contents: Vec<String> = frames[1..4]
            .iter()
            .map(|f| {
                chunk_of(f)["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            contents,
            fake_tokens(&rendered("be brief", "ab"), 3)
                .iter()
                .map(|&b| char::from(b as u8).to_string())
                .collect::<Vec<String>>()
        );
        assert!(
            frames[1..4]
                .iter()
                .all(|f| chunk_of(f)["choices"][0]["delta"].get("role").is_none()),
            "only the opening frame names the role"
        );

        let last = chunk_of(&frames[4]);
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["choices"][0]["delta"]["content"], "");
        assert_eq!(last["usage"]["completion_tokens"], 3);
        assert_eq!(frames[5], "data: [DONE]");
    }

    #[tokio::test]
    async fn an_empty_conversation_is_400() {
        let (status, body) = post_chat(test_router(), serde_json::json!({ "messages": [] })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("messages")
        );
    }

    #[tokio::test]
    async fn a_turn_with_an_unknown_field_is_400() {
        let (status, _) = post_chat(
            test_router(),
            serde_json::json!({
                "messages": [{ "role": "user", "content": "a", "name": "bob" }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    async fn post_json(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let response = test_router()
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn get(path: &str) -> (StatusCode, serde_json::Value) {
        let response = test_router()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn completions_returns_a_greedy_text_completion() {
        let (status, body) = post_json(serde_json::json!({
            "model": "test-model",
            "prompt": "ab",
            "max_tokens": 3,
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["text"], "cde");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["choices"][0]["logprobs"], serde_json::Value::Null);
        assert_eq!(body["usage"]["prompt_tokens"], 2);
        assert_eq!(body["usage"]["completion_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 5);
        assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
        assert!(body["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn eos_maps_to_the_stop_finish_reason() {
        // The reserved "<before-eos>" prompt makes the counter sample eos next.
        let (status, body) = post_json(serde_json::json!({
            "prompt": "<before-eos>", "max_tokens": 4
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["choices"][0]["finish_reason"], "stop",
            "eos is a content stop"
        );
        assert_eq!(
            body["usage"]["completion_tokens"], 1,
            "generation stops on the eos token"
        );
    }

    #[tokio::test]
    async fn omitted_sampling_fields_stay_greedy() {
        let (status, body) =
            post_json(serde_json::json!({ "prompt": "ab", "max_tokens": 2 })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["choices"][0]["text"], "cd",
            "no temperature means greedy regardless of top_k/top_p defaults"
        );
    }

    #[tokio::test]
    async fn model_mismatch_is_404() {
        let (status, body) = post_json(serde_json::json!({
            "model": "other", "prompt": "ab", "max_tokens": 1
        }))
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(body["error"]["message"].as_str().unwrap().contains("other"));
    }

    #[tokio::test]
    async fn empty_prompt_is_400() {
        let (status, body) = post_json(serde_json::json!({ "prompt": "" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn unknown_field_is_400() {
        let (status, _) = post_json(serde_json::json!({ "prompt": "a", "nucleus": 0.9 })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "deny_unknown_fields");
    }

    #[tokio::test]
    async fn openai_compat_fields_are_accepted_and_ignored() {
        // FreeToken declares these and never reads them; accepting them keeps a
        // drop-in client that sends the defaults from getting a 400.
        for (field, value) in [
            ("presence_penalty", serde_json::json!(0.0)),
            ("frequency_penalty", serde_json::json!(0.0)),
            ("best_of", serde_json::json!(1)),
            ("user", serde_json::json!("u-1")),
        ] {
            let mut comp = serde_json::json!({ "prompt": "ab", "max_tokens": 2 });
            comp[field] = value.clone();
            let (status, body) = post_json(comp).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "completions must accept OpenAI compat field '{field}'"
            );
            assert_eq!(body["choices"][0]["text"], "cd");

            let mut chat = serde_json::json!({
                "messages": [{ "role": "user", "content": "ab" }],
                "max_tokens": 2
            });
            chat[field] = value;
            let (status, _) = post_chat(test_router(), chat).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "chat completions must accept OpenAI compat field '{field}'"
            );
        }
    }

    #[tokio::test]
    async fn streaming_sends_one_frame_per_token_then_the_final_frame_and_done() {
        let response = test_router()
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "prompt": "ab", "max_tokens": 3, "stream": true })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "text/event-stream"
        );

        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let frames: Vec<&str> = body.split("\n\n").filter(|f| !f.is_empty()).collect();
        assert_eq!(frames.len(), 5, "3 tokens, the final frame, [DONE]");
        assert!(frames[0].starts_with("data: "));

        let chunk = |frame: &str| serde_json::from_str::<serde_json::Value>(&frame[6..]).unwrap();
        let texts: Vec<String> = frames[..3]
            .iter()
            .map(|f| chunk(f)["choices"][0]["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(texts, ["c", "d", "e"], "one frame per token, in order");
        assert_eq!(texts.concat(), "cde", "the frames carry the whole text");
        assert!(
            frames[..3]
                .iter()
                .all(|f| chunk(f)["choices"][0]["finish_reason"].is_null()),
            "only the final frame closes the choice"
        );
        assert!(
            frames[..3].iter().all(|f| chunk(f).get("usage").is_none()),
            "usage rides on the final frame only"
        );
        assert!(
            chunk(frames[0])["id"]
                .as_str()
                .unwrap()
                .starts_with("cmpl-")
        );
        assert_eq!(chunk(frames[0])["object"], "text_completion");

        let last = chunk(frames[3]);
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["choices"][0]["text"], "");
        assert_eq!(last["usage"]["prompt_tokens"], 2);
        assert_eq!(last["usage"]["completion_tokens"], 3);
        assert_eq!(last["usage"]["total_tokens"], 5);
        assert_eq!(frames[4], "data: [DONE]");
    }

    #[tokio::test]
    async fn a_stream_with_a_bad_prompt_fails_before_any_frame() {
        let (status, body) = post_json(serde_json::json!({ "prompt": "", "stream": true })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn a_stream_stops_at_eos_and_says_so() {
        let response = test_router()
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "prompt": "<before-eos>", "max_tokens": 4, "stream": true })
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let frames: Vec<&str> = body.split("\n\n").filter(|f| !f.is_empty()).collect();
        assert_eq!(frames.len(), 3, "the eos token, the final frame, [DONE]");
        let last: serde_json::Value = serde_json::from_str(&frames[1][6..]).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["usage"]["completion_tokens"], 1);
    }

    #[tokio::test]
    async fn n_greater_than_one_is_rejected() {
        let (status, _) = post_json(serde_json::json!({ "prompt": "a", "n": 2 })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn stop_accepts_a_string_or_a_list() {
        let one = Stop::One("END".to_string()).into_vec();
        let many = Stop::Many(vec!["A".to_string(), "B".to_string()]).into_vec();
        assert_eq!(one, vec!["END".to_string()]);
        assert_eq!(many, vec!["A".to_string(), "B".to_string()]);

        let (_, body) = post_json(serde_json::json!({
            "prompt": "a", "max_tokens": 6, "stop": "cd"
        }))
        .await;
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["text"], "b");
    }

    #[tokio::test]
    async fn models_lists_the_served_model() {
        let (status, body) = get("/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "test-model");
        assert_eq!(body["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn health_is_ok() {
        let (status, body) = get("/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["model"], "test-model");
    }

    #[test]
    fn sampling_params_mapping_uses_domain_defaults() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "a"}"#).unwrap();
        let p = req.sampling_params();
        assert_eq!(p, SamplingParams::default());

        let req: CompletionRequest = serde_json::from_str(
            r#"{"prompt": "a", "temperature": 0.5, "top_k": 5, "top_p": 0.9, "max_tokens": 3, "stop": ["X", "Y"], "ignore_eos": true}"#,
        )
        .unwrap();
        let p = req.sampling_params();
        assert_eq!(p.temperature, 0.5);
        assert_eq!(p.top_k, 5);
        assert_eq!(p.top_p, 0.9);
        assert_eq!(p.max_tokens, 3);
        assert_eq!(p.stop_strs, vec!["X".to_string(), "Y".to_string()]);
        assert!(p.ignore_eos);
    }

    /// The tool-call text the scripted model below emits verbatim (its byte ids
    /// start at `SCRIPT_BASE`). Kept under `FAKE_VOCAB - SCRIPT_BASE` bytes so a
    /// run of consecutive ids never wraps.
    const TOOL_SCRIPT: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    const REASONED_TOOL_SCRIPT: &str = "Checking it.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    const SCRIPT_BASE: u32 = 400;

    /// Encodes every prompt to one id, so the counter model then walks the ids
    /// that decode to `script` byte by byte. A deterministic model output for
    /// the tool-call path, which the plain fake (consecutive printable bytes)
    /// cannot produce.
    struct ScriptTokenizer(&'static str);

    impl crate::domain::Tokenizer for ScriptTokenizer {
        fn encode(&self, _text: &str) -> Result<Vec<u32>, crate::domain::TokenizerError> {
            Ok(vec![SCRIPT_BASE - 1])
        }

        fn decode(&self, ids: &[u32]) -> Result<String, crate::domain::TokenizerError> {
            Ok(ids
                .iter()
                .map(|&id| {
                    let i = id.wrapping_sub(SCRIPT_BASE) as usize;
                    if id >= SCRIPT_BASE && i < self.0.len() {
                        char::from(self.0.as_bytes()[i])
                    } else {
                        char::from(id as u8)
                    }
                })
                .collect())
        }

        fn render_chat(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ChatTool],
        ) -> Result<String, crate::domain::TokenizerError> {
            Ok(String::new())
        }
    }

    fn script_router(script: &'static str) -> Router {
        use crate::application::engine::EngineLimits;
        use crate::infrastructure::torch::Device;
        let (model, _rec) = FakeModel::new();
        let limits = EngineLimits {
            max_seq_len: 4096,
            max_output_tokens: 256,
            max_running_requests: 4,
            max_prefill_length: 16,
        };
        let engine = Engine::new(
            Arc::new(model),
            Arc::new(ScriptTokenizer(script)),
            vec![FAKE_EOS],
            limits,
            Device::Cpu,
        );
        router(AppState::new(Arc::new(engine), "test-model".to_string()))
    }

    fn tool_router() -> Router {
        script_router(TOOL_SCRIPT)
    }

    fn tool_request(script: &'static str, stream: bool) -> serde_json::Value {
        serde_json::json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "weather in Paris?" }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "look up weather",
                    "parameters": { "type": "object" }
                }
            }],
            "max_tokens": script.len(),
            "stream": stream,
        })
    }

    #[tokio::test]
    async fn a_tool_call_answer_returns_a_tool_calls_message() {
        let (status, body) = post_chat(tool_router(), tool_request(TOOL_SCRIPT, false)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let message = &body["choices"][0]["message"];
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "");
        assert!(
            message.get("content").is_some(),
            "the message still carries a content field"
        );
        let calls = message["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[tokio::test]
    async fn a_content_only_answer_still_has_no_tool_calls() {
        let (router, _rec) = test_router_with_rec();
        let (status, body) = post_chat(
            router,
            serde_json::json!({
                "messages": [{ "role": "user", "content": "ab" }],
                "max_tokens": 3,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn a_tool_call_stream_emits_tool_calls_deltas() {
        let frames = post_chat_sse_with(tool_router(), tool_request(TOOL_SCRIPT, true)).await;
        assert_eq!(
            frames.len(),
            5,
            "role, open, args, final, [DONE]: {frames:?}"
        );
        assert_eq!(
            chunk_of(&frames[0])["choices"][0]["delta"]["role"],
            "assistant"
        );

        let open = chunk_of(&frames[1]);
        let call = &open["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["type"], "function");
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["function"]["name"], "get_weather");

        let args_frame = chunk_of(&frames[2]);
        let call = &args_frame["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        let args: serde_json::Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");

        let last = chunk_of(&frames[3]);
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(last["usage"]["completion_tokens"], TOOL_SCRIPT.len());
        assert_eq!(frames[4], "data: [DONE]");
    }

    #[tokio::test]
    async fn reasoning_before_a_tool_call_streams_as_content() {
        let frames = post_chat_sse_with(
            script_router(REASONED_TOOL_SCRIPT),
            tool_request(REASONED_TOOL_SCRIPT, true),
        )
        .await;
        let content: String = frames
            .iter()
            .filter(|f| *f != "data: [DONE]")
            .map(|f| chunk_of(f))
            .map(|chunk| {
                let delta = &chunk["choices"][0]["delta"];
                match delta.get("tool_calls") {
                    Some(_) => String::new(),
                    None => delta["content"].as_str().unwrap_or("").to_string(),
                }
            })
            .collect();
        assert!(
            content.contains("Checking it."),
            "the reasoning before the call must stream, got {content:?}"
        );
        assert_eq!(frames.last().unwrap(), "data: [DONE]");
        let last = chunk_of(&frames[frames.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn tool_choice_and_a_tool_result_turn_deserialize() {
        let (status, _) = post_chat(
            test_router(),
            serde_json::json!({
                "messages": [
                    { "role": "user", "content": "weather?" },
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
                        }]
                    },
                    { "role": "tool", "tool_call_id": "call_1", "content": "sunny" }
                ],
                "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": {"type":"object"} } }],
                "tool_choice": "auto",
                "parallel_tool_calls": false,
                "max_tokens": 1,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[test]
    fn a_tool_call_split_across_tokens_is_held_until_complete() {
        let mut stream = ChatStream::new();
        let mut calls = Vec::new();
        let mut content = String::new();
        for ch in TOOL_SCRIPT.chars() {
            for action in stream.feed(&ch.to_string()) {
                match action {
                    ChatAction::Content(c) => content.push_str(&c),
                    ChatAction::ToolCall {
                        name, arguments, ..
                    } => calls.push((name, arguments)),
                }
            }
        }
        let (emitted, flush) = stream.finish();
        assert_eq!(emitted, 1);
        assert!(flush.is_none());
        assert_eq!(content, "");
        assert_eq!(calls[0].0, "get_weather");
        assert!(calls[0].1.contains("Paris"));
    }

    #[test]
    fn an_unterminated_tool_call_is_flushed_as_content() {
        let mut stream = ChatStream::new();
        let mut out = String::new();
        for ch in "<tool_call><function=x>".chars() {
            for action in stream.feed(&ch.to_string()) {
                if let ChatAction::Content(c) = action {
                    out.push_str(&c);
                }
            }
        }
        let (emitted, flush) = stream.finish();
        assert_eq!(emitted, 0);
        out.push_str(flush.as_deref().unwrap_or(""));
        assert_eq!(out, "<tool_call><function=x>");
    }
}
