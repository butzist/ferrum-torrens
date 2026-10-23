pub mod application;
pub mod config;
pub mod converter;
pub mod domain;
pub mod infrastructure;
pub mod models;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
