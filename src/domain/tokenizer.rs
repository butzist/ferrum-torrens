// Tokenizer port. The domain core depends only on this trait; the HF adapter
// (infrastructure::tokenizers_hf) loads the tokenizer.json shipped in the FTW
// model dir. encode/decode only: chat template rendering arrives with M2.

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("tokenizer error: {0}")]
    Message(String),
}
