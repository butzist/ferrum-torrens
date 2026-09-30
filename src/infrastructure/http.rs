// OpenAI-compatible HTTP surface: `POST /v1/completions` (non-stream),
// `GET /v1/models`, `GET /health`. Axum types stay here; the handlers call the
// application layer, which never sees an HTTP type. Streaming and
// `/v1/chat/completions` land later (TODO.md section "Server").

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use serde::{Deserialize, Serialize};

use crate::application::engine::{Engine, EngineError};
use crate::application::generation::{self, make_request};
use crate::domain::sampling::SamplingParams;
use crate::domain::scheduler::FinishReason;

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
}

impl AppState {
    pub fn new(engine: Arc<Engine>, served_model_name: String) -> Self {
        Self {
            engine,
            served_model_name,
            next_uid: AtomicU64::new(1),
        }
    }

    fn next_uid(&self) -> u64 {
        self.next_uid.fetch_add(1, Ordering::Relaxed)
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
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
        let d = SamplingParams::default();
        SamplingParams {
            temperature: self.temperature.unwrap_or(d.temperature),
            top_k: self.top_k.unwrap_or(d.top_k),
            top_p: self.top_p.unwrap_or(d.top_p),
            ignore_eos: self.ignore_eos.unwrap_or(d.ignore_eos),
            max_tokens: self.max_tokens.unwrap_or(d.max_tokens),
            stop_strs: self.stop.clone().map(Stop::into_vec).unwrap_or_default(),
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
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<()>,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
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

    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "invalid_request_error",
            code: "not_implemented",
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
            BadLogits { got, want } => {
                ApiError::unprocessable(format!("model returned {got} logits, expected {want}"))
            }
            NoParams(uid) => ApiError::unprocessable(format!("no sampling parameters for {uid}")),
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
) -> Result<Json<CompletionResponse>, ApiError> {
    let Json(req) =
        payload.map_err(|e| ApiError::bad_request(format!("invalid json body: {e}")))?;

    if !req.model.is_empty() && req.model != state.served_model_name {
        return Err(ApiError::not_found(format!(
            "model {} not found, this server serves {}",
            req.model, state.served_model_name
        )));
    }
    if req.stream.unwrap_or(false) {
        return Err(ApiError::not_implemented(
            "streaming is not implemented yet",
        ));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::unprocessable("only n=1 is supported"));
    }

    let uid = state.next_uid();
    let engine = Arc::clone(&state.engine);
    let params = req.sampling_params();
    let seed = req.seed.unwrap_or(uid);
    let prompt = req.prompt;

    // The forward pass is CPU-bound and long-running; never on the runtime.
    let outcome = tokio::task::spawn_blocking(move || {
        let ids = engine.encode(&prompt)?;
        let request = make_request(&engine, uid, ids, params)?;
        generation::generate(&engine, request, seed)
    })
    .await;

    let generated = match outcome {
        Ok(Ok(g)) => g,
        Ok(Err(e)) => return Err(e.into()),
        Err(join) => {
            return Err(ApiError::unprocessable(format!(
                "generation task failed: {join}"
            )));
        }
    };
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", generated.uid),
        object: "text_completion",
        created: unix_now(),
        model: state.served_model_name.clone(),
        choices: vec![CompletionChoice {
            text: generated.text,
            index: 0,
            logprobs: None,
            finish_reason: finish_reason_str(generated.finish_reason),
        }],
        usage: Usage {
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.tokens.len(),
            total_tokens: generated.prompt_tokens + generated.tokens.len(),
        },
    }))
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
        let (engine, _) = test_engine(limits());
        let state = AppState::new(Arc::new(engine), "test-model".to_string());
        router(state)
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
    async fn streaming_is_501() {
        let (status, body) = post_json(serde_json::json!({ "prompt": "a", "stream": true })).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["code"], "not_implemented");
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
}
