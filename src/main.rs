// Binary entry point: read the TOML config, build the engine (FTW load, model,
// tokenizer) and serve the OpenAI-compatible API until killed.

use std::sync::Arc;

use ferrum_torrens::application::engine::load_engine;
use ferrum_torrens::config::Config;
use ferrum_torrens::infrastructure::http::{AppState, serve_forever};

const DEFAULT_CONFIG: &str = "ferrum-torrens.toml";

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("ferrum-torrens: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG.to_string());
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let config: Config = text.parse().map_err(|e| format!("{path}: {e}"))?;

    let engine = Arc::new(load_engine(&config).map_err(|e| e.to_string())?);
    let params = engine.params();
    eprintln!(
        "loaded {} on {} ({} layers, {} experts, vocab {}, eos {:?})",
        config.served_model_name(),
        engine.device(),
        params.num_layers,
        params.num_experts,
        params.vocab_size,
        engine.eos_ids()
    );

    let (host, port) = (config.serve.host.clone(), config.serve.port);
    eprintln!("serving on http://{host}:{port}");
    serve_forever(
        AppState::with_idle_timeout(
            engine,
            config.served_model_name().to_string(),
            config.serve.idle_timeout_secs,
        ),
        &host,
        port,
    )
    .await
    .map_err(|e| e.to_string())
}
