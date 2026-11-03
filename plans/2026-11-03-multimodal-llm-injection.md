# Plan: inject vision tokens into the LLM forward pass

Goal: run a downloaded image through the ViT tower, splice the resulting
visual tokens into the LLM embedding stream at image-placeholder positions,
and sample a decoded answer.

## Steps
1. Extend `ForwardInput` with `image_embeddings: Option<Vec<Vec<f32>>>` and
   `image_positions: Option<Vec<usize>>` (positions in the flattened token
   stream that are image placeholders). Update all 17 construction sites.
2. In `models/qwen3_6/forward.rs`, when image embeddings are present, replace
   the embedding lookup at those positions with the provided vectors.
3. Add a `Qwen3_6::forward_with_images` convenience (or just use the field).
4. Update the engine to accept image bytes, run `encode_block0`, and build the
   ForwardInput.
5. Add a gated integration test mirroring `engine_real_generate.rs` but with an
   image and a prompt "describe this image".

## Status
Steps 1-5 not started. Vision tower side is complete (encode_block0).
