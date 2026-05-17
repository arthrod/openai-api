use serde::{Deserialize, Deserializer, Serialize};

pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod images;
pub mod models;

// Models API
const MODELS_LIST: &str = "models";
const MODELS_RETRIEVE: &str = "models/";
// Completions API
const COMPLETION_CREATE: &str = "completions";
// Chat API
const CHAT_COMPLETION_CREATE: &str = "chat/completions";
// Images API
const IMAGES_CREATE: &str = "images/generations";
const IMAGES_EDIT: &str = "images/edits";
const IMAGES_VARIATIONS: &str = "images/variations";
// Embeddings API
const EMBEDDINGS_CREATE: &str = "embeddings";
// Audio API
const AUDIO_TRANSCRIPTION_CREATE: &str = "audio/transcriptions";
const AUDIO_TRANSLATIONS_CREATE: &str = "audio/translations";

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
	pub prompt_tokens: Option<u32>,
	pub completion_tokens: Option<u32>,
	pub total_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
	pub text: Option<String>,
	pub index: u32,
	pub logprobs: Option<String>,
	pub finish_reason: Option<String>,
	pub message: Option<Message>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
	pub role: Role,
	/// `null` from the wire (e.g. an assistant message that's tool-calls only)
	/// is deserialized as `""` so existing callers that index `.content`
	/// keep working. New callers that care about tool calls should look at
	/// `tool_calls` instead.
	#[serde(default, deserialize_with = "deserialize_content_lenient")]
	pub content: String,
	/// Present when the assistant returns one or more tool calls instead of
	/// (or alongside) text. OpenAI 1.x / DeepSeek style.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_calls: Option<Vec<ToolCall>>,
	/// Set on a `role: tool` message that carries the tool's result back to
	/// the model. Pairs with `ToolCall.id`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_call_id: Option<String>,
	/// DeepSeek "thinking mode" trace. The server returns this on assistant
	/// turns and **requires** it to be echoed back on the next turn, otherwise
	/// the next request fails with
	/// `The reasoning_content in the thinking mode must be passed back to the API`.
	/// Not part of vanilla OpenAI 1.x; safe to leave `None` for OpenAI.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning_content: Option<String>,
}

impl Clone for Message {
	fn clone(&self) -> Self {
		Self {
			role: self.role.clone(),
			content: self.content.clone(),
			tool_calls: self.tool_calls.clone(),
			tool_call_id: self.tool_call_id.clone(),
			reasoning_content: self.reasoning_content.clone(),
		}
	}
}

impl Message {
	/// Convenience builder for the common case (no tool fields).
	pub fn new(role: Role, content: impl Into<String>) -> Self {
		Self {
			role,
			content: content.into(),
			tool_calls: None,
			tool_call_id: None,
			reasoning_content: None,
		}
	}
}

fn deserialize_content_lenient<'de, D>(d: D) -> Result<String, D::Error>
where
	D: Deserializer<'de>,
{
	let v: Option<String> = Option::deserialize(d)?;
	Ok(v.unwrap_or_default())
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	System,
	Assistant,
	User,
	/// Carries a tool's result back to the model (paired with `tool_call_id`).
	/// OpenAI 1.x / DeepSeek style. Not used in the legacy `function` form.
	Tool,
}

impl Clone for Role {
	fn clone(&self) -> Self {
		match self {
			Self::System => Self::System,
			Self::Assistant => Self::Assistant,
			Self::User => Self::User,
			Self::Tool => Self::Tool,
		}
	}
}

// --- Tool calling (OpenAI 1.x / DeepSeek shape) -----------------------------
//
// Sent in `ChatBody.tools`:
//
//   { "type": "function", "function": { "name": "...", "description": "...",
//     "parameters": <JSON schema> } }
//
// Returned in `Message.tool_calls`:
//
//   { "id": "call_…", "type": "function",
//     "function": { "name": "...", "arguments": "<JSON-encoded string>" } }
//
// Note `arguments` is a JSON string, not a JSON object — that's the OpenAI
// convention. Callers parse it with `serde_json::from_str`.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
	#[serde(rename = "type")]
	pub kind: String,
	pub function: FunctionDeclaration,
}

impl Tool {
	pub fn function(decl: FunctionDeclaration) -> Self {
		Self { kind: "function".to_string(), function: decl }
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	/// JSON Schema describing the function's parameters object.
	pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
	pub id: String,
	#[serde(rename = "type", default = "default_tool_call_type")]
	pub kind: String,
	pub function: FunctionCall,
}

fn default_tool_call_type() -> String {
	"function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
	pub name: String,
	/// JSON-encoded string of the call's arguments. Parse with
	/// `serde_json::from_str::<Value>(&call.arguments)`.
	pub arguments: String,
}
