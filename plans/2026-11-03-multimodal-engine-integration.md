# Plan: multimodal engine integration (remaining steps)

Remaining after `2026-11-03-multimodal-llm-injection.md` steps 1-3.

## Step 4 — engine wiring
- Add a `VisionTower` (or its weights) to `Engine` behind an Option, loaded when a
  vision_config is present.
- Add image bytes handling to the request path: decode via `image_prep`, run
  `encode_block0`, produce `Vec<Vec<f32>>` of visual tokens.
- Thread `image_embeddings`/`image_positions` through `BatchRow` and `extend_cached`
  / `run_forward` into `ForwardInput`. Image rows must not be batched with
  text rows (different embedding shapes); process them as single-sequence steps.
- Insert image placeholder token ids (e.g. a chosen image token id) into
  `input_ids` at the positions that will be replaced; the splice in
  `forward_reference` overwrites their embeddings.

## Step 5 — gated test
- Mirror `tests/engine_real_generate.rs`: load the real checkpoint under prlimit,
  feed a small image + "describe this image", assert the decoded output is
  non-empty and contains a relevant word.

## Resource notes
- One model-loading process at a time, prlimit --as=68719476736.
- Vision tower adds ~0.83 GiB transiently during image prefill; borrow from
  expert cache (reduce slots temporarily) as discussed.
