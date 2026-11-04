use super::*;

pub(crate) fn responses_stream_body(
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

pub(crate) async fn responses(
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

    let generated = completions::dispatch(&state, request, None).await?;
    let (content, calls) = parse_tool_calls(&generated.text);
    let output = responses_output(&content, calls, uid);
    let base = ResponsesBase::new(&state, &req, uid);
    Ok(
        Json(base.response("completed", output, Some(responses_usage(&generated)), None))
            .into_response(),
    )
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

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        use EngineError::*;
        match e {
            EmptyPrompt => ApiError::bad_request("prompt must not be empty"),
            PromptTooLong { len, max } => {
                ApiError::unprocessable(format!("prompt has {len} tokens, over the {max} limit"))
            }
            Busy { retry_after_ms } => ApiError::busy(
                "the server is finishing another request; retry when it is idle",
                retry_after_ms.div_ceil(1000),
            ),
            Tokenize(m) => ApiError::bad_request(format!("prompt could not be tokenized: {m}")),
            Chat(m) => ApiError::bad_request(m),
            Vision(m) => ApiError::unprocessable(m),
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
            | Captured(_)
            | NotResident
            | Rebuild(_)
            | Moe(_)) => ApiError::unprocessable(format!("engine kv cache: {e}")),
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
        let retry_after = self.retry_after;
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = retry_after
            && let Ok(value) = secs.to_string().parse()
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::http::test_support::*;

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
}
