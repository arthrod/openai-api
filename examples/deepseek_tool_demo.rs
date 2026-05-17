//! DeepSeek tool-calling demo (OpenAI-compatible API).
//!
//! Mirrors `gemini-sdk/examples/tool_demo.rs` but talks to DeepSeek's
//! OpenAI-compatible endpoint. The single tool is `lookup(kind)` where
//! `kind` is `"color"` or `"fruit"` — local execution returns `"red"` or
//! `"apple"` respectively.
//!
//! Run with:
//!
//!   DEEPSEEK_API_KEY=sk-… cargo run --example deepseek_tool_demo
//!
//! Optional env:
//!
//!   DEEPSEEK_BASE_URL    default https://api.deepseek.com/v1/
//!   DEEPSEEK_MODEL       default deepseek-v4-flash (per user request)

use openai_api_rust::{
    chat::{ChatApi, ChatBody},
    Auth, FunctionDeclaration, Message, OpenAI, Role, Tool, ToolCall,
};
use serde_json::{json, Value};
use std::env;

fn run_lookup(kind: &str) -> Value {
    match kind {
        "color" => json!({ "result": "red" }),
        "fruit" => json!({ "result": "apple" }),
        other => json!({ "error": format!("unknown kind: {other}") }),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("DEEPSEEK_API_KEY")
        .map_err(|_| "DEEPSEEK_API_KEY must be set")?;
    let base_url =
        env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1/".to_string());
    let model = env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_string());

    let client = OpenAI::new(Auth::new(&api_key), &base_url);

    // The single tool the model is allowed to call.
    let tools = vec![Tool::function(FunctionDeclaration {
        name: "lookup".to_string(),
        description: Some(
            "Look up a canonical example for a category. \
             Pass kind=\"color\" to get a color, or kind=\"fruit\" to get a fruit."
                .to_string(),
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["color", "fruit"],
                    "description": "Which kind of example to look up.",
                }
            },
            "required": ["kind"],
            "additionalProperties": false,
        }),
    })];

    // Running chat history. We append assistant + tool messages as the loop
    // progresses until the model finally produces a plain-text answer.
    let mut messages = vec![Message::new(
        Role::User,
        "Use the lookup tool to fetch the canonical color, then the canonical fruit. \
         After both tool calls, reply with one sentence naming them.",
    )];

    println!("model: {model}");
    println!("> sending initial prompt with `lookup` tool registered\n");

    for turn in 1..=6 {
        let body = ChatBody {
            model: model.clone(),
            max_tokens: Some(512),
            temperature: Some(0.0),
            top_p: None,
            n: Some(1),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            messages: messages.clone(),
            tools: Some(tools.clone()),
            tool_choice: Some(json!("auto")),
        };

        let completion = client
            .chat_completion_create(&body)
            .map_err(|e| format!("DeepSeek call failed: {e:?}"))?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or("no choices returned")?;
        let msg = choice.message.ok_or("choice missing message")?;

        // Echo the assistant's turn back into history so the next round
        // sees it.
        messages.push(msg.clone());

        match msg.tool_calls.as_ref() {
            Some(calls) if !calls.is_empty() => {
                for call in calls {
                    let args: Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or(Value::Null);
                    let kind = args
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<missing>");
                    let result = run_lookup(kind);
                    println!(
                        "[turn {turn}] model called {}({{kind: {:?}}}) -> {}",
                        call.function.name, kind, result
                    );
                    messages.push(tool_response(call, result));
                }
            }
            _ => {
                println!("\n[turn {turn}] final reply:\n{}", msg.content);
                return Ok(());
            }
        }
    }

    Err("gave up after 6 turns without a plain-text answer".into())
}

/// Wraps a tool's result as a `role: tool` message tied to the call by id —
/// the DeepSeek/OpenAI 1.x shape.
fn tool_response(call: &ToolCall, result: Value) -> Message {
    Message {
        role: Role::Tool,
        content: result.to_string(),
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
        reasoning_content: None,
    }
}
