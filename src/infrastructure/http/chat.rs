use super::*;

/// `stream: true`: one `data: {...}` frame per token as it is sampled, a frame
/// carrying the finish reason plus the usage, then `data: [DONE]`.
///
/// The prompt is tokenized and clamped before the stream starts, so a bad
/// request is still an ordinary JSON error with its status code. Everything
/// after that has already sent headers, so a failure mid-run rides on the stream
/// as an `error` frame. The tokens come back through the dispatcher's per-request
/// sink: frames go over a bounded channel, so a slow client drops its own stream
/// (a full buffer aborts the sink) instead of stalling the batch it rides in.
pub(crate) fn stream_body(
    state: &Arc<AppState>,
    request: generation::Request,
    wire: Wire,
) -> Response {
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
                    let usage = completions::usage_of(&g);
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
            Ok(Err(e)) => {
                let msg = e.message();
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

pub(crate) async fn chat_completions(
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
    let mut images: Vec<Vec<u8>> = Vec::new();
    for turn in &req.messages {
        for img in turn.image_parts()? {
            images.push(img.bytes);
        }
    }
    // An image request borrows the device exclusively for its tower pass, so it
    // must not start while a batch (or another vision pass) owns it.
    if state.engine.exclusive() {
        return Err(ApiError::busy(
            "the device is serving another request; retry when it is idle",
            state.dispatcher.retry_after_ms().div_ceil(1000),
        ));
    }
    if !images.is_empty() && state.dispatcher.busy() {
        return Err(ApiError::busy(
            "the device is serving another request; retry when it is idle",
            state.dispatcher.retry_after_ms().div_ceil(1000),
        ));
    }

    // Rendering the template and tokenizing it both run off the runtime, and both
    // still have to answer with a status code rather than a frame. An image
    // request's tower pass is the heaviest piece of preparation, which is exactly
    // what spawn_blocking is here for. It takes the exclusive lease so the
    // expert banks can be freed for the encode and no batch races it.
    let worker = Arc::clone(&engine);
    let prepared =
        tokio::task::spawn_blocking(move || -> Result<generation::Request, EngineError> {
            let prompt = worker.render_chat(&messages, &tools)?;
            let ids = worker.encode(&prompt)?;
            if images.is_empty() {
                return make_request(&worker, uid, ids, params, seed);
            }
            let lease = worker.try_exclusive().ok_or(EngineError::Busy {
                retry_after_ms: 1_000,
            })?;
            // The vision tower is streamed per block from the host cache, so an
            // image request no longer frees or rebuilds the MoE expert cache;
            // the captured decode graphs and the cache survive untouched.
            let result = worker.make_request_with_images(uid, ids, images, params, seed);
            drop(lease);
            result
        })
        .await
        .expect("preparation task is not cancellable");
    let request = match prepared {
        Ok(r) => r,
        Err(e) => return Err(ApiError::from(e)),
    };

    if streaming {
        state.ensure_resident().await?;
        return Ok(stream_body(&state, request, Wire::Chat));
    }

    let generated = completions::dispatch(&state, request, None).await?;
    let usage = completions::usage_of(&generated);
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
