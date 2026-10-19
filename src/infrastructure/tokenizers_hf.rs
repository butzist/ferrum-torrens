// HF `tokenizers` adapter implementing the domain Tokenizer port. Completions
// encode raw prompts with no special tokens. Load from the tokenizer.json the
// FTW conversion bundles in the model dir, next to which the same dir carries
// the checkpoint's chat template.

use std::path::Path;

use minijinja::{Environment, Value};
use tokenizers::Tokenizer as Hf;

use crate::domain::{ChatMessage, ChatTool, Tokenizer, TokenizerError};

pub struct HfTokenizer {
    inner: Hf,
    chat: Option<ChatTemplate>,
}

impl HfTokenizer {
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let inner = Hf::from_file(path).map_err(|e| TokenizerError::Message(e.to_string()))?;
        let chat = match Path::new(path).parent() {
            Some(dir) => load_chat_template(dir)?,
            None => None,
        };
        Ok(Self { inner, chat })
    }

    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let inner =
            Hf::from_bytes(json.as_bytes()).map_err(|e| TokenizerError::Message(e.to_string()))?;
        Ok(Self { inner, chat: None })
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

    fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
    ) -> Result<String, TokenizerError> {
        match &self.chat {
            Some(chat) => chat.render(messages, tools),
            None => Err(TokenizerError::NoChatTemplate),
        }
    }
}

/// Read the chat template the model dir ships: the standalone
/// `chat_template.jinja`, else the `chat_template` string inside
/// `tokenizer_config.json` (older conversions inline it).
fn load_chat_template(dir: &Path) -> Result<Option<ChatTemplate>, TokenizerError> {
    let standalone = dir.join("chat_template.jinja");
    let source = if standalone.exists() {
        Some(
            std::fs::read_to_string(&standalone)
                .map_err(|e| TokenizerError::Message(format!("{}: {e}", standalone.display())))?,
        )
    } else {
        let raw = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok();
        raw.and_then(|raw| {
            serde_json::from_str::<serde_json::Value>(&raw)
                .ok()?
                .get("chat_template")?
                .as_str()
                .map(str::to_string)
        })
    };
    source.map(|src| ChatTemplate::new(&src)).transpose()
}

/// A parsed chat template: the checkpoint's own Jinja, rendered by minijinja.
///
/// HF templates are rendered by Python's jinja2, where `content` is a Python
/// `str` and so carries methods. minijinja strings do not, so the handful of
/// methods a template may call become filters of the same name (`startswith`,
/// `endswith`, `lstrip`, `rstrip`) and their call sites are rewritten to filter
/// syntax. Whatever minijinja still cannot do fails the render with its own
/// message, which is what the API reports.
struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    fn new(source: &str) -> Result<Self, TokenizerError> {
        let mut env = Environment::new();
        env.add_global("raise_exception", Value::from_function(raise_exception));
        env.add_filter("startswith", |v: String, p: String| v.starts_with(&p));
        env.add_filter("endswith", |v: String, p: String| v.ends_with(&p));
        env.add_filter("lstrip", |v: String, chars: Option<String>| {
            trim_ends(&v, chars.as_deref(), true)
        });
        env.add_filter("rstrip", |v: String, chars: Option<String>| {
            trim_ends(&v, chars.as_deref(), false)
        });
        env.add_template_owned("chat", rewrite_python_methods(source))
            .map_err(|e| TokenizerError::Message(format!("chat template: {e}")))?;
        Ok(Self { env })
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
    ) -> Result<String, TokenizerError> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| TokenizerError::Message(format!("chat template: {e}")))?;
        let turns: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut turn = serde_json::Map::new();
                turn.insert("role".into(), serde_json::json!(m.role));
                turn.insert(
                    "content".into(),
                    match &m.content {
                        Some(content) => serde_json::json!(content),
                        None => serde_json::Value::Null,
                    },
                );
                if !m.tool_calls.is_empty() {
                    turn.insert(
                        "tool_calls".into(),
                        serde_json::Value::Array(
                            m.tool_calls
                                .iter()
                                .map(|c| {
                                    serde_json::json!({
                                        "id": c.id,
                                        "type": "function",
                                        "function": {
                                            "name": c.name,
                                            "arguments": args_object(&c.arguments),
                                        },
                                    })
                                })
                                .collect(),
                        ),
                    );
                }
                if let Some(id) = &m.tool_call_id {
                    turn.insert("tool_call_id".into(), serde_json::json!(id));
                }
                serde_json::Value::Object(turn)
            })
            .collect();
        let tools = tool_specs(tools);
        let ctx = serde_json::json!({
            "messages": turns,
            "add_generation_prompt": true,
            "enable_thinking": true,
            "tools": tools,
        });
        template
            .render(ctx)
            .map_err(|e| TokenizerError::Message(format!("chat template: {e:#}")))
    }
}

/// The template iterates `tool_call.arguments|items`, so arguments must be a
/// mapping. The wire string is parsed to an object; a parse failure (a client
/// sent a non-object) falls back to an empty object so the render still runs.
fn args_object(arguments: &str) -> serde_json::Value {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v @ serde_json::Value::Object(_)) => v,
        _ => serde_json::json!({}),
    }
}

/// The `tools` context value: `null` when there are none (what the template's
/// `if tools` guard expects), else the OpenAI function list.
fn tool_specs(tools: &[ChatTool]) -> serde_json::Value {
    if tools.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::Value::Array(
        tools
            .iter()
            .map(|t| {
                let mut function = serde_json::Map::new();
                function.insert("name".into(), serde_json::json!(t.name));
                if let Some(description) = &t.description {
                    function.insert("description".into(), serde_json::json!(description));
                }
                function.insert(
                    "parameters".into(),
                    t.parameters
                        .as_deref()
                        .and_then(|p| serde_json::from_str(p).ok())
                        .unwrap_or_else(|| serde_json::json!({})),
                );
                serde_json::json!({ "type": "function", "function": function })
            })
            .collect(),
    )
}

/// `raise_exception` is a transformers jinja global. minijinja has no Python
/// exception to raise, so it is an error that fails the render.
fn raise_exception(message: String) -> Result<(), minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        message,
    ))
}

/// Python `str.strip(chars)` semantics: `chars` is a set of characters, and an
/// absent argument means whitespace.
fn trim_ends(value: &str, chars: Option<&str>, start: bool) -> String {
    let set: Vec<char> = match chars {
        Some(chars) if !chars.is_empty() => chars.chars().collect(),
        _ => vec![' ', '\t', '\n', '\r', '\u{b}', '\u{c}'],
    };
    if start {
        value.trim_start_matches(|c| set.contains(&c)).to_string()
    } else {
        value.trim_end_matches(|c| set.contains(&c)).to_string()
    }
}

/// Python string methods a HF chat template may call, rewritten to filters:
/// `content.startswith('<x>')` becomes `(content)|startswith('<x>')`. The
/// receiver is parenthesized when it is an expression rather than a name, so
/// `a.split('x')[0].rstrip('\n')` keeps Python's chaining. Calls inside string
/// literals are plain text and stay as they are.
fn rewrite_python_methods(source: &str) -> String {
    const METHODS: [&str; 5] = ["startswith", "endswith", "split", "lstrip", "rstrip"];
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    // marks[i] is `out.len()` before source byte i was copied, so a receiver that
    // was already rewritten (and is no longer the same length as the source it
    // came from) can still be cut back out.
    let mut marks: Vec<usize> = vec![0; bytes.len() + 1];
    let mut i = 0;
    let mut literal: Option<u8> = None;
    while i < bytes.len() {
        marks[i] = out.len();
        let c = bytes[i];
        if let Some(q) = literal {
            out.push(c as char);
            if c == b'\\' && i + 1 < bytes.len() {
                i += 1;
                out.push(bytes[i] as char);
            } else if c == q {
                literal = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => {
                literal = Some(c);
                out.push(c as char);
                i += 1;
            }
            b'.' => {
                let name_end = ident_end(bytes, i + 1);
                let name = &source[i + 1..name_end];
                let after = *bytes.get(name_end).unwrap_or(&0);
                if METHODS.contains(&name) && after == b'(' {
                    let start = receiver_start(bytes, i);
                    // The receiver was copied out verbatim as the scan walked over
                    // it; drop that copy and rewrite the expression instead, since
                    // it may itself hold method calls (`a.split('x')[0].rstrip()`).
                    out.truncate(marks[start]);
                    let recv = rewrite_python_methods(&source[start..i]);
                    let call_end = call_end(bytes, name_end).min(source.len());
                    // The whole call is parenthesized: a `[0]` chained after it has
                    // to index the filter's result, not the filter itself.
                    let nested = recv.contains(['.', '[', '(', '|']);
                    out.push('(');
                    if nested {
                        out.push('(');
                    }
                    out.push_str(&recv);
                    if nested {
                        out.push(')');
                    }
                    out.push('|');
                    out.push_str(name);
                    out.push_str(&source[name_end..call_end]);
                    out.push(')');
                    i = call_end;
                } else {
                    out.push('.');
                    i += 1;
                }
            }
            _ => {
                let ch = source[i..].chars().next().unwrap_or('\0');
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    out
}

/// Index just past the `)` closing the `(` at `from`, stepping over string
/// literals so a bracket inside one does not count.
fn call_end(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    let mut depth = 0usize;
    let mut literal: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = literal {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                literal = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => literal = Some(c),
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    bytes.len()
}

fn ident_end(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    i
}

/// Walk back from a method call's `.` over its receiver: a name, an attribute
/// chain, an indexing or a balanced group, stepping over string literals as
/// units. Stops at the first byte that cannot be part of the expression, which
/// for `{% if c.startswith(x) %}` is the space before `c`.
fn receiver_start(bytes: &[u8], dot: usize) -> usize {
    let mut i = dot;
    loop {
        if i == 0 {
            return 0;
        }
        let last = bytes[i - 1];
        match last {
            b'\'' | b'"' => {
                // A closing quote: step back over the literal, then keep going
                // from before its opening quote.
                let mut j = i - 1;
                while j > 0 {
                    j -= 1;
                    if bytes[j] == last {
                        break;
                    }
                }
                i = j;
            }
            b')' | b']' | b'}' => {
                let open = match last {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                let mut depth = 0usize;
                while i > 0 {
                    i -= 1;
                    if bytes[i] == last {
                        depth += 1;
                    } else if bytes[i] == open {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            // An index is a postfix on what came before it, so it belongs to the
            // receiver: `a.split('x')[0]`. An unclosed paren belongs to an
            // enclosing construct instead: in `not(c.startswith(x))` the receiver
            // is just `c`.
            b'[' => i -= 1,
            b'(' | b'{' => return i,
            c if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' => i -= 1,
            _ => return i,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ChatMessage;

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

    fn chat(text: &str) -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", "be brief"),
            ChatMessage::new("user", text),
        ]
    }

    fn render(src: &str, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        render_with_tools(src, messages, &[])
    }

    fn render_with_tools(
        src: &str,
        messages: &[ChatMessage],
        tools: &[ChatTool],
    ) -> Result<String, TokenizerError> {
        let template = ChatTemplate::new(src).unwrap();
        template.render(messages, tools)
    }

    #[test]
    fn the_rewriter_only_touches_method_calls() {
        assert_eq!(
            rewrite_python_methods("{{ a.b }} {{ 'x' }} {% if c.startswith(d) %}"),
            "{{ a.b }} {{ 'x' }} {% if (c|startswith(d)) %}"
        );
        assert_eq!(
            rewrite_python_methods("{{ m.split('x')[0].rstrip('\\n') }}"),
            "{{ (((m|split('x'))[0])|rstrip('\\n')) }}"
        );
        assert_eq!(
            rewrite_python_methods("{% if not(c.startswith('<')) %}"),
            "{% if not((c|startswith('<'))) %}"
        );
    }

    #[test]
    fn a_plain_template_renders_the_conversation() {
        let out = render(
            "{{ messages[0].role }}|{{ messages[1].content }}",
            &chat("hi"),
        )
        .unwrap();
        assert_eq!(out, "system|hi");
    }

    #[test]
    fn python_string_methods_become_minijinja_filters() {
        let src = "{% set content = messages[1].content %}{% if content.startswith('<tool_response>') and content.endswith('</tool_response>') %}match{% else %}plain{% endif %}";
        let t = ChatTemplate::new(src).unwrap();
        assert_eq!(
            t.render(&chat("<tool_response>x</tool_response>"), &[])
                .unwrap(),
            "match"
        );
        assert_eq!(t.render(&chat("plain"), &[]).unwrap(), "plain");
    }

    #[test]
    fn a_chained_split_strip_keeps_the_python_meaning() {
        // `content.split('</think>')[0]` is "a\n", then rstrip/lstrip drop the newline.
        let src = "{{ messages[1].content.split('</think>')[0].rstrip('\n').lstrip('\n') }}";
        assert_eq!(render(src, &chat("a\n</think>\n")).unwrap(), "a");
    }

    #[test]
    fn a_method_call_inside_a_string_literal_is_left_alone() {
        assert_eq!(
            render("{{ 'content.startswith(x)' }}", &chat("hi")).unwrap(),
            "content.startswith(x)"
        );
    }

    #[test]
    fn tools_reach_the_template_as_json() {
        let src = "{% for t in tools %}{{ t.function.name }}={{ t.function.parameters|tojson }}{% endfor %}";
        let tools = vec![ChatTool {
            name: "get_weather".to_string(),
            description: Some("look up weather".to_string()),
            parameters: Some(r#"{"type":"object"}"#.to_string()),
        }];
        assert_eq!(
            render_with_tools(src, &chat("hi"), &tools).unwrap(),
            r#"get_weather={"type":"object"}"#
        );
    }

    #[test]
    fn an_assistant_tool_call_renders_its_parsed_arguments() {
        let src = "{% for tc in messages[1].tool_calls %}{{ tc.function.name }}:{{ tc.function.arguments.city }}{% endfor %}";
        let messages = vec![
            ChatMessage::new("user", "weather?"),
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: vec![crate::domain::ChatToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                }],
                tool_call_id: None,
            },
        ];
        assert_eq!(
            render_with_tools(src, &messages, &[]).unwrap(),
            "get_weather:Paris"
        );
    }

    #[test]
    fn real_checkpoint_chat_template_renders_tools() {
        let Some(path) = real_dir() else {
            eprintln!("skipped: no converted checkpoint in .model-cache");
            return;
        };
        let t = HfTokenizer::from_file(&path).unwrap();
        let tools = vec![ChatTool {
            name: "get_weather".to_string(),
            description: None,
            parameters: Some(
                r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.to_string(),
            ),
        }];
        let prompt = t.render_chat(&chat("weather in Paris?"), &tools).unwrap();
        assert!(prompt.contains("<tools>"), "tools block is opened");
        assert!(
            prompt.contains(r#""get_weather""#),
            "the tool name is in the prompt"
        );
        assert!(
            prompt.contains("<tool_call>"),
            "the format instructions are in the prompt"
        );
    }

    #[test]
    fn raise_exception_surfaces_the_template_message() {
        let err = render(
            "{{ raise_exception('No user query found in messages.') }}",
            &chat("hi"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("No user query found"), "got {err}");
    }

    #[test]
    fn a_template_minijinja_cannot_parse_is_an_error_not_a_panic() {
        assert!(ChatTemplate::new("{{ unclosed ").is_err());
    }

    #[test]
    fn a_bare_tokenizer_json_has_no_chat_template() {
        let t = HfTokenizer::from_json(FIXTURE).unwrap();
        assert_eq!(
            t.render_chat(&chat("hi"), &[]),
            Err(TokenizerError::NoChatTemplate)
        );
    }

    /// The turn opener of this checkpoint's template: one per rendered turn,
    /// counting the generation prompt's own turn.
    fn count_turn_openers(ids: &[u32]) -> usize {
        ids.iter().filter(|id| **id == 248045).count()
    }

    #[test]
    fn real_checkpoint_chat_template_matches_the_transformers_oracle() {
        let Some(path) = real_dir() else {
            eprintln!("skipped: no converted checkpoint in .model-cache");
            return;
        };
        let t = HfTokenizer::from_file(&path).unwrap();
        let prompt = t
            .render_chat(&chat("The capital of France is"), &[])
            .unwrap();
        let ids = t.encode(&prompt).unwrap();
        assert_eq!(
            count_turn_openers(&ids),
            3,
            "the template opens a turn for the system, the user and the reply"
        );
        assert!(
            prompt.ends_with("<think>\n"),
            "the generation prompt opens the thinking block, got {:?}",
            prompt.rsplit('\n').next()
        );
        // tokenizer.apply_chat_template(msgs, add_generation_prompt=True) on the
        // same checkpoint: the rendered prompt must encode to these ids.
        assert_eq!(
            ids,
            vec![
                248045, 8678, 198, 1323, 9522, 248046, 198, 248045, 846, 198, 760, 6511, 314, 9338,
                369, 248046, 198, 248045, 74455, 198, 248068, 198
            ],
            "must match the transformers oracle"
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
