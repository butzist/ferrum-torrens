// Tokenizer port. The domain core depends only on this trait; the HF adapter
// (infrastructure::tokenizers_hf) loads the tokenizer.json shipped in the FTW
// model dir, and renders chat with the chat template the same dir ships.

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;

    /// Render a conversation through the checkpoint's own chat template, ending
    /// on the assistant's generation prompt. v0 is reasoning-free and tool-free,
    /// so a message is a role plus flat text. Adapters with no template to render
    /// (the fake one, a bare tokenizer.json) say so rather than guessing a format.
    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        let _ = messages;
        Err(TokenizerError::NoChatTemplate)
    }
}

/// One chat turn as the API carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("tokenizer error: {0}")]
    Message(String),
    #[error("this tokenizer ships no chat template")]
    NoChatTemplate,
}
