// HF `tokenizers` adapter implementing the domain Tokenizer port. Complections
// encode raw prompts with no special tokens. Load from the tokenizer.json the
// FTW conversion bundles in the model dir.

use tokenizers::Tokenizer as Hf;

use crate::domain::{Tokenizer, TokenizerError};

pub struct HfTokenizer {
    inner: Hf,
}

impl HfTokenizer {
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        Hf::from_file(path)
            .map(|inner| Self { inner })
            .map_err(|e| TokenizerError::Message(e.to_string()))
    }

    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        Hf::from_bytes(json.as_bytes())
            .map(|inner| Self { inner })
            .map_err(|e| TokenizerError::Message(e.to_string()))
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| TokenizerError::Message(e.to_string()))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| TokenizerError::Message(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal byte-level BPE tokenizer fixture, hermetic for `cargo test`.
    const FIXTURE: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "unk_token": "<unk>",
        "vocab": {
          "hello": 0,
          "world": 1,
          "the": 2,
          "quick": 3
        }
      }
    }"#;

    #[test]
    fn encode_round_trips_through_decode() {
        let t = HfTokenizer::from_json(FIXTURE).unwrap();
        let ids = t.encode("hello world").unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(t.decode(&ids).unwrap(), "hello world");
    }

    #[test]
    fn encode_is_deterministic() {
        let t = HfTokenizer::from_json(FIXTURE).unwrap();
        assert_eq!(
            t.encode("the quick").unwrap(),
            t.encode("the quick").unwrap()
        );
    }

    #[test]
    fn empty_prompt_encodes_to_empty() {
        let t = HfTokenizer::from_json(FIXTURE).unwrap();
        assert!(t.encode("").unwrap().is_empty());
    }

    #[test]
    fn malformed_json_errors() {
        assert!(HfTokenizer::from_json("not json [").is_err());
        assert!(HfTokenizer::from_file("/nonexistent/tokenizer.json").is_err());
    }

    fn real_dir() -> Option<String> {
        let dir = ".model-cache/Qwen3.6-35B-A3B-NVFP4.ftw/tokenizer.json";
        std::path::Path::new(dir).exists().then(|| dir.to_string())
    }

    #[test]
    fn real_checkpoint_tokenizer_encodes_prompt() {
        let Some(dir) = real_dir() else {
            eprintln!("skipped: no converted checkpoint in .model-cache");
            return;
        };
        let t = HfTokenizer::from_file(&dir).unwrap();
        let text = "The quick brown fox jumps over the lazy dog";
        let ids = t.encode(text).unwrap();
        assert!(!ids.is_empty());
        assert!(ids.len() < text.len(), "BPE must not inflate short prompts");
        assert_eq!(
            t.encode(text).unwrap(),
            ids,
            "encoding must be deterministic"
        );
        let back = t.decode(&ids).unwrap();
        assert!(
            back.contains("quick"),
            "decode should recover the prompt, got {back:?}"
        );
    }

    #[test]
    fn real_checkpoint_has_expected_specials() {
        let Some(dir) = real_dir() else {
            eprintln!("skipped: no converted checkpoint in .model-cache");
            return;
        };
        let t = HfTokenizer::from_file(&dir).unwrap();
        let eos = t.encode("<|endoftext|>").unwrap();
        assert_eq!(eos, vec![248044], "eos id must match the arch constant");
        assert_eq!(t.decode(&eos).unwrap(), "<|endoftext|>");
        let im_end = t.encode("<|im_end|>").unwrap();
        assert_eq!(im_end, vec![248046], "im_end must be a single id");
    }
}
