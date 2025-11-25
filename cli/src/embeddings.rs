//! Embeddings module with remote API support
//!
//! Uses OpenAI-compatible APIs (OpenRouter, OpenAI, etc.) for embeddings.
//!
//! Configuration via ~/.osgrep/config.json or environment variables:
//! - provider: "openrouter" (default), "openai", or "remote"
//! - api_key: API key for remote provider (required)
//! - model: Model name (default: openai/text-embedding-3-small)
//! - base_url: Base URL (default: https://openrouter.ai/api/v1)

use crate::config;
use anyhow::Result;
use std::sync::OnceLock;

// Remote embedding provider configuration
struct RemoteConfig {
    api_key: String,
    base_url: String,
    model: String,
    dimensions: usize,
}

static REMOTE_CONFIG: OnceLock<Option<RemoteConfig>> = OnceLock::new();

fn get_remote_config() -> Option<&'static RemoteConfig> {
    REMOTE_CONFIG
        .get_or_init(|| {
            let cfg = config::load();
            let provider = cfg.embedding.provider.as_deref().unwrap_or("");

            if provider != "openrouter" && provider != "openai" && provider != "remote" {
                return None;
            }

            let api_key = cfg.embedding.api_key.clone()?;
            if api_key.is_empty() {
                return None;
            }

            Some(RemoteConfig {
                api_key,
                base_url: cfg
                    .embedding
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
                model: cfg
                    .embedding
                    .model
                    .clone()
                    .unwrap_or_else(|| "openai/text-embedding-3-small".to_string()),
                dimensions: cfg.embedding.dimensions.unwrap_or(1536),
            })
        })
        .as_ref()
}

/// Get embedding dimensions (for remote provider info)
pub fn get_dimensions() -> usize {
    get_remote_config().map(|c| c.dimensions).unwrap_or(1536) // Default to OpenAI dimensions
}

/// Get provider info for display
pub fn get_provider_info() -> String {
    if let Some(config) = get_remote_config() {
        format!("remote ({} via {})", config.model, config.base_url)
    } else {
        "not configured (run: osgrep config --init)".to_string()
    }
}

// ============================================================================
// Remote embeddings via OpenAI-compatible API
// ============================================================================

fn remote_embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let config =
        get_remote_config().ok_or_else(|| anyhow::anyhow!("Remote config not available"))?;

    if texts.is_empty() {
        return Ok(vec![]);
    }

    // OpenAI-compatible embedding request
    // Note: Some models only accept a single string, others accept an array
    // We try array first, fall back to single string if needed
    let input_value: serde_json::Value = if texts.len() == 1 {
        serde_json::json!(texts[0])
    } else {
        serde_json::json!(texts)
    };

    let request_body = serde_json::json!({
        "model": config.model,
        "input": input_value
    });

    let url = format!("{}/embeddings", config.base_url);

    // Use agent with timeout to prevent hanging
    let agent = ureq::AgentBuilder::new()
        .timeout_read(std::time::Duration::from_secs(60))
        .timeout_write(std::time::Duration::from_secs(30))
        .build();

    let response = agent
        .post(&url)
        .set("Authorization", &format!("Bearer {}", config.api_key))
        .set("Content-Type", "application/json")
        .set("HTTP-Referer", "https://github.com/wwicak/osgrep")
        .set("X-Title", "osgrep")
        .send_json(&request_body);

    // Handle HTTP errors
    let response = match response {
        Ok(r) => r,
        Err(ureq::Error::Status(code, response)) => {
            let body = response
                .into_string()
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(anyhow::anyhow!(
                "API error (HTTP {}): {}",
                code,
                body.chars().take(1000).collect::<String>()
            ));
        }
        Err(e) => return Err(anyhow::anyhow!("API request failed: {}", e)),
    };

    let response_body: serde_json::Value = response
        .into_json()
        .map_err(|e| anyhow::anyhow!("Failed to parse API response: {}", e))?;

    // Check for error in response body
    if let Some(error) = response_body.get("error") {
        let msg = if error.is_string() {
            error.as_str().unwrap_or("Unknown error").to_string()
        } else {
            error["message"]
                .as_str()
                .or_else(|| error["msg"].as_str())
                .unwrap_or("Unknown error")
                .to_string()
        };

        // Add helpful context for common errors
        let detailed_msg = if msg.contains("No successful provider responses") {
            format!(
                "{}\n\n\
                Possible causes:\n\
                1. API key may not have credits or proper permissions\n\
                2. Check your OpenRouter dashboard: https://openrouter.ai/activity\n\
                3. Verify your API key has access to the model: {}\n\
                4. Try a smaller batch by reducing chunks (current: {} items)\n\n\
                Full error response:\n{}",
                msg,
                config.model,
                texts.len(),
                truncate_json(&response_body, 500)
            )
        } else {
            format!(
                "{}\n\nFull error response:\n{}",
                msg,
                truncate_json(&response_body, 500)
            )
        };

        return Err(anyhow::anyhow!("API error: {}", detailed_msg));
    }

    // Try to extract embeddings - handle different response formats
    let embeddings = if let Some(data) = response_body["data"].as_array() {
        // Standard OpenAI format: { "data": [{"embedding": [...]}] }
        let mut result = Vec::with_capacity(texts.len());
        for item in data {
            let embedding = item["embedding"]
                .as_array()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Invalid API response: missing 'embedding' in data item. Response: {}",
                        truncate_json(&response_body, 500)
                    )
                })?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect::<Vec<f32>>();
            result.push(embedding);
        }
        result
    } else if let Some(embedding) = response_body["embedding"].as_array() {
        // Some APIs return embedding directly: { "embedding": [...] }
        vec![embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect()]
    } else if let Some(embeddings) = response_body["embeddings"].as_array() {
        // Some APIs use "embeddings" (plural): { "embeddings": [[...], [...]] }
        embeddings
            .iter()
            .map(|emb| {
                emb.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect()
    } else {
        return Err(anyhow::anyhow!(
            "Invalid API response format. Expected 'data', 'embedding', or 'embeddings' field.\n\
             Response: {}",
            truncate_json(&response_body, 1000)
        ));
    };

    Ok(embeddings)
}

/// Truncate JSON for error messages
fn truncate_json(value: &serde_json::Value, max_len: usize) -> String {
    let s = serde_json::to_string_pretty(value).unwrap_or_else(|_| "unparseable".to_string());
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Initialize the embedding provider (verify remote config)
pub fn init() -> Result<()> {
    let config = get_remote_config().ok_or_else(|| {
        anyhow::anyhow!(
            "Remote embedding provider not configured.\n\
             \n\
             To configure:\n\
             1. Run: osgrep config --init\n\
             2. Edit ~/.osgrep/config.json with your API key\n\
             \n\
             Or use environment variables:\n\
             export OSGREP_EMBEDDING_PROVIDER=openrouter\n\
             export OSGREP_EMBEDDING_API_KEY=sk-or-v1-...\n\
             \n\
             Get API keys at: https://openrouter.ai"
        )
    })?;

    eprintln!(
        "  Using remote embeddings: {} ({})",
        config.model, config.base_url
    );

    // Warn about models that may have compatibility issues
    if config.model.contains("gemini") {
        eprintln!("  Warning: Gemini models may have API compatibility issues.");
        eprintln!("  Consider using: osgrep config --model openai/text-embedding-3-small");
    }

    Ok(())
}

/// Embed a single text
pub fn embed(text: &str) -> Result<Vec<f32>> {
    let embeddings = embed_batch(&[text.to_string()])?;
    Ok(embeddings.into_iter().next().unwrap_or_default())
}

/// Embed multiple texts (batch processing)
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    embed_batch_with_progress(texts, |_, _| {})
}

/// Embed multiple texts with progress callback
pub fn embed_batch_with_progress<F>(texts: &[String], mut progress: F) -> Result<Vec<Vec<f32>>>
where
    F: FnMut(usize, usize), // (completed, total)
{
    // Use small batches for maximum reliability with OpenRouter
    // OpenRouter sometimes has issues with larger batches
    const BATCH_SIZE: usize = 10;
    let mut all_embeddings = Vec::with_capacity(texts.len());
    let total = texts.len();

    for chunk in texts.chunks(BATCH_SIZE) {
        let chunk_embeddings = remote_embed_batch(&chunk.to_vec())?;
        all_embeddings.extend(chunk_embeddings);
        progress(all_embeddings.len().min(total), total);

        // Delay between batches to avoid overwhelming OpenRouter
        // OpenRouter can be sensitive to burst requests
        if all_embeddings.len() < total {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    Ok(all_embeddings)
}
