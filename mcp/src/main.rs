//! osgrep MCP Server
//!
//! Model Context Protocol server for Claude Code integration.
//! Provides semantic code search as MCP tools.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::process::Command;

const SERVER_NAME: &str = "osgrep";
const SERVER_VERSION: &str = "0.1.0";

// MCP Protocol Types
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// Tool definitions
fn get_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "semantic_search",
                "description": "Search code semantically using natural language. Use this instead of grep for conceptual searches like 'authentication logic' or 'error handling'. Returns relevant code snippets with file paths and line numbers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'how is user authentication handled?')"
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional: limit search to a specific directory path"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index_directory",
                "description": "Index a directory for semantic search. Run this before searching a new codebase or after major changes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to index (default: current directory)"
                        }
                    }
                }
            },
            {
                "name": "get_simd_info",
                "description": "Get system SIMD capabilities and osgrep version info.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

// Execute osgrep CLI command
fn run_osgrep(args: &[&str]) -> Result<String> {
    let output = Command::new("osgrep")
        .args(args)
        .output()
        .context("Failed to execute osgrep CLI")?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow::anyhow!("osgrep failed: {}", stderr))
    }
}

// Tool handlers
fn handle_semantic_search(params: &Value) -> Result<Value> {
    let query = params["query"]
        .as_str()
        .context("Missing required parameter: query")?;

    let limit = params["limit"].as_i64().unwrap_or(10);
    let limit_str = limit.to_string();
    let path = params["path"].as_str();

    let mut args = vec!["search", query, "-k", &limit_str];

    if let Some(p) = path {
        args.push("--path");
        args.push(p);
    }

    let output = run_osgrep(&args)?;

    // Parse results and format for Claude
    Ok(json!({
        "content": [{
            "type": "text",
            "text": output
        }]
    }))
}

fn handle_index_directory(params: &Value) -> Result<Value> {
    let path = params["path"].as_str().unwrap_or(".");

    let output = run_osgrep(&["index", path])?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("Indexed directory: {}\n{}", path, output)
        }]
    }))
}

fn handle_get_info(_params: &Value) -> Result<Value> {
    let output = run_osgrep(&["info"])?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": output
        }]
    }))
}

// MCP Protocol handlers
fn handle_initialize(_params: &Value) -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    })
}

fn handle_list_tools(_params: &Value) -> Value {
    get_tools()
}

fn handle_call_tool(params: &Value) -> Value {
    let tool_name = params["name"].as_str().unwrap_or("");
    let arguments = &params["arguments"];

    let result = match tool_name {
        "semantic_search" => handle_semantic_search(arguments),
        "index_directory" => handle_index_directory(arguments),
        "get_simd_info" => handle_get_info(arguments),
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
    };

    match result {
        Ok(content) => content,
        Err(e) => json!({
            "content": [{
                "type": "text",
                "text": format!("Error: {}", e)
            }],
            "isError": true
        }),
    }
}

fn process_request(request: JsonRpcRequest) -> JsonRpcResponse {
    let result = match request.method.as_str() {
        "initialize" => Some(handle_initialize(&request.params)),
        "initialized" => None, // Notification, no response
        "tools/list" => Some(handle_list_tools(&request.params)),
        "tools/call" => Some(handle_call_tool(&request.params)),
        _ => Some(json!({
            "error": {
                "code": -32601,
                "message": format!("Method not found: {}", request.method)
            }
        })),
    };

    if let Some(result) = result {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(result),
            error: None,
        }
    } else {
        // For notifications, return empty response
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: None,
        }
    }
}

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.context("Failed to read line")?;

        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                writeln!(stdout, "{}", serde_json::to_string(&error_response)?)?;
                stdout.flush()?;
                continue;
            }
        };

        let response = process_request(request);

        // Only send response if it has an id (not a notification)
        if response.id.is_some() || response.result.is_some() {
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }
    }

    Ok(())
}
