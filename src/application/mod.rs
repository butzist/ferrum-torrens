// Application layer: use-cases orchestrating the domain ports. No compute here,
// just workflow (registry dispatch, engine step loop, generation advancement).

pub mod engine;
pub mod generation;
pub mod registry;
pub mod run_experts;
