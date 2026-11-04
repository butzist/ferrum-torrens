use super::*;

pub(crate) async fn completions(
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
        state.ensure_resident().await?;
        return Ok(chat::stream_body(&state, request, Wire::Text));
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
pub(crate) async fn dispatch(
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
        Ok(Err(e)) => Err(ApiError::from(e)),
        Err(_) => Err(ApiError::unprocessable(
            "the engine worker dropped the request",
        )),
    }
}

pub(crate) fn usage_of(g: &generation::Generation) -> Usage {
    Usage {
        prompt_tokens: g.prompt_tokens,
        completion_tokens: g.tokens.len(),
        total_tokens: g.prompt_tokens + g.tokens.len(),
    }
}
