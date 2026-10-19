// Tokenizer port. The domain core depends only on this trait; the HF adapter
// (infrastructure::tokenizers_hf) loads the tokenizer.json shipped in the FTW
// model dir, and renders chat with the chat template the same dir ships.

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;

    /// Render a conversation through the checkpoint's own chat template, ending
    /// on the assistant's generation prompt. A message is a role plus flat text,
    /// optionally carrying the tool calls an assistant made and the id a tool
    /// result answers. `tools` are the callable functions, rendered into the
    /// system turn. Adapters with no template to render (the fake one, a bare
    /// tokenizer.json) say so rather than guessing a format.
    fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
    ) -> Result<String, TokenizerError> {
        let _ = (messages, tools);
        Err(TokenizerError::NoChatTemplate)
    }
}

/// One chat turn as the API carries it. `content` is absent for a pure tool-call
/// assistant turn; `tool_call_id` names the call a `role: "tool"` result answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ChatToolCall>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// One tool call an assistant turn made. `arguments` is the wire JSON string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One callable function offered to the model. `parameters` is the JSON-schema
/// text (absent when the caller sent none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenizerError {
    #[error("tokenizer error: {0}")]
    Message(String),
    #[error("this tokenizer ships no chat template")]
    NoChatTemplate,
}
