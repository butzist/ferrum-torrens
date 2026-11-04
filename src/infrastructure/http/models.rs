use super::*;

pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
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

pub(crate) async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.served_model_name.clone(),
    })
}
