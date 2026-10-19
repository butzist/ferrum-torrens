// Parses the checkpoint's plain-text tool-call dialect out of a generated
// answer. The model emits, with no special tokens:
//
//   <tool_call>
//   <function=NAME>
//   <parameter=ARG>
//   VALUE
//   </parameter>
//   </function>
//   </tool_call>
//
// One answer can carry several blocks; any natural-language reasoning before
// the first block is the message content. `arguments` comes back as the JSON
// object string OpenAI clients expect.

use std::sync::atomic::{AtomicU64, Ordering};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    let mut first_open = None;
    while let Some(rel) = text[cursor..].find(OPEN) {
        let open = cursor + rel;
        if first_open.is_none() {
            first_open = Some(open);
        }
        let inner_start = open + OPEN.len();
        let Some(rel_close) = text[inner_start..].find(CLOSE) else {
            break;
        };
        let close = inner_start + rel_close;
        match parse_block(&text[inner_start..close]) {
            Some(call) => {
                calls.push(call);
                cursor = close + CLOSE.len();
            }
            None => break,
        }
    }
    let content = match first_open {
        Some(open) if !calls.is_empty() => text[..open].trim().to_string(),
        _ => text.to_string(),
    };
    (content, calls)
}

pub(crate) fn parse_block(inner: &str) -> Option<ParsedToolCall> {
    let name = function_name(inner)?;
    let mut args = serde_json::Map::new();
    let mut cursor = 0usize;
    while let Some(rel) = inner[cursor..].find(PARAM_OPEN) {
        let key_start = cursor + rel + PARAM_OPEN.len();
        let key_end = key_start + inner[key_start..].find('>')?;
        let key = inner[key_start..key_end].trim();
        if key.is_empty() {
            return None;
        }
        let value_start = key_end + 1;
        let value_end = value_start + inner[value_start..].find(PARAM_CLOSE)?;
        args.insert(
            key.to_string(),
            coerce(inner[value_start..value_end].trim()),
        );
        cursor = value_end + PARAM_CLOSE.len();
    }
    Some(ParsedToolCall {
        id: next_id(),
        name,
        arguments: serde_json::Value::Object(args).to_string(),
    })
}

fn function_name(inner: &str) -> Option<String> {
    let start = inner.find(FN_OPEN)? + FN_OPEN.len();
    let end = start + inner[start..].find('>')?;
    let name = inner[start..end].trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// A parameter value is embedded as JSON when it already reads as JSON, and as
/// a JSON string otherwise, so `true`, `42`, `[1,2]` and `{...}` keep their type
/// while free text stays text.
fn coerce(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

fn next_id() -> String {
    format!("call_{:08x}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("arguments must be a JSON object")
    }

    #[test]
    fn a_plain_answer_has_no_calls() {
        let (content, calls) = parse_tool_calls("the capital is Paris");
        assert_eq!(content, "the capital is Paris");
        assert!(calls.is_empty());
    }

    #[test]
    fn one_call_is_parsed_into_arguments() {
        let text = "I will look it up.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "I will look it up.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(json(&calls[0].arguments)["city"], "Paris");
    }

    #[test]
    fn argument_values_coerce_to_json_scalars_and_containers() {
        let text = "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=z>\nnull\n</parameter>\n<parameter=o>\n{\"a\":1}\n</parameter>\n<parameter=a>\n[1,2]\n</parameter>\n<parameter=s>\nhello\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(text);
        let args = json(&calls[0].arguments);
        assert_eq!(args["n"], 42);
        assert_eq!(args["b"], true);
        assert!(args["z"].is_null());
        assert_eq!(args["o"]["a"], 1);
        assert_eq!(args["a"][1], 2);
        assert_eq!(args["s"], "hello");
    }

    #[test]
    fn a_multiline_value_stays_a_string() {
        let text = "<tool_call>\n<function=f>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(json(&calls[0].arguments)["body"], "line one\nline two");
    }

    #[test]
    fn several_blocks_produce_several_calls() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn an_unterminated_block_is_content_with_no_call() {
        let text = "sure thing\n<tool_call>\n<function=f>\n<parameter=x>\n1";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn a_block_without_a_function_is_content_with_no_call() {
        let text = "<tool_call>\n<parameter=x>\n1\n</parameter>\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn content_after_a_complete_block_is_dropped() {
        let text = "lead\n<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\ntrailing";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "lead");
        assert_eq!(calls.len(), 1);
    }
}
