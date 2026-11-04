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
