use super::*;
pub use crate::application::engine::fake::*;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

pub(crate) fn test_router() -> Router {
    test_router_with_rec().0
}

/// The fake tokenizer renders `"{role}\n{content}\n"` per turn plus the
/// `assistant\n` generation prompt, and encodes bytes one to one.
pub(crate) fn rendered(system: &str, user: &str) -> String {
    format!("system\n{system}\nuser\n{user}\nassistant\n")
}

/// The fake model answers with the counter one past the prompt's last id, so
/// the expected tokens follow from the rendered prompt.
pub(crate) fn fake_tokens(prompt: &str, n: usize) -> Vec<u32> {
    let last = u32::from(*prompt.as_bytes().last().expect("a prompt"));
    (1..=n as u32).map(|i| last + i).collect()
}

pub(crate) fn fake_text(tokens: &[u32]) -> String {
    tokens.iter().map(|&b| char::from(b as u8)).collect()
}

pub(crate) fn test_router_with_rec() -> (Router, Recorder) {
    let (engine, rec) = test_engine(limits());
    let state = AppState::new(Arc::new(engine), "test-model".to_string());
    (router(state), rec)
}

pub(crate) async fn post_chat(
    router: Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
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

pub(crate) async fn post_chat_sse(body: serde_json::Value) -> Vec<String> {
    post_chat_sse_with(test_router(), body).await
}

pub(crate) async fn post_chat_sse_with(router: Router, body: serde_json::Value) -> Vec<String> {
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

pub(crate) fn chunk_of(frame: &str) -> serde_json::Value {
    serde_json::from_str(&frame[6..]).expect("a data frame holds one json")
}

pub(crate) async fn post_responses_with(
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

pub(crate) async fn post_responses(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    post_responses_with(test_router(), body).await
}

/// Responses SSE frames pair an `event:` line with a `data:` line, so they
/// are split whole; `[DONE]` is a bare data frame.
pub(crate) async fn post_responses_sse_with(
    router: Router,
    body: serde_json::Value,
) -> Vec<String> {
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

pub(crate) fn responses_event(frame: &str) -> (String, serde_json::Value) {
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

pub(crate) const TOOL_SCRIPT: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

pub(crate) const REASONED_TOOL_SCRIPT: &str = "Checking it.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

pub(crate) fn script_router(script: &'static str) -> Router {
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

pub(crate) fn tool_router() -> Router {
    script_router(TOOL_SCRIPT)
}

pub(crate) struct ScriptTokenizer(&'static str);

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

pub(crate) const SCRIPT_BASE: u32 = 400;
