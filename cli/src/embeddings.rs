//! Embeddings module with support for local and remote providers
//!
//! Supports two modes:
//! 1. Local: Uses Candle ML with BAAI/bge-base-en-v1.5 (requires `embeddings` feature)
//! 2. Remote: Uses OpenAI-compatible API (OpenRouter, OpenAI, etc.)
//!
//! Configuration via ~/.osgrep/config.json or environment variables:
//! - provider: "local" (default) or "openrouter"
//! - api_key: API key for remote provider
//! - model: Model name (default: google/gemini-embedding-001)
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
                    .unwrap_or_else(|| "google/gemini-embedding-001".to_string()),
                dimensions: cfg.embedding.dimensions.unwrap_or(768),
            })
        })
        .as_ref()
}

/// Check if using remote embeddings
pub fn is_remote() -> bool {
    get_remote_config().is_some()
}

/// Get embedding dimensions (for remote provider info)
pub fn get_dimensions() -> usize {
    get_remote_config().map(|c| c.dimensions).unwrap_or(768)
}

/// Get provider info for display
pub fn get_provider_info() -> String {
    if let Some(config) = get_remote_config() {
        format!("remote ({} via {})", config.model, config.base_url)
    } else {
        #[cfg(feature = "embeddings")]
        {
            "local (BAAI/bge-base-en-v1.5)".to_string()
        }
        #[cfg(not(feature = "embeddings"))]
        {
            "disabled".to_string()
        }
    }
}

// ============================================================================
// Remote embeddings via OpenAI-compatible API
// ============================================================================

fn remote_embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let config = get_remote_config().ok_or_else(|| anyhow::anyhow!("Remote config not available"))?;

    if texts.is_empty() {
        return Ok(vec![]);
    }

    // OpenAI-compatible embedding request
    let request_body = serde_json::json!({
        "model": config.model,
        "input": texts,
        "encoding_format": "float"
    });

    let response = ureq::post(&format!("{}/embeddings", config.base_url))
        .set("Authorization", &format!("Bearer {}", config.api_key))
        .set("Content-Type", "application/json")
        .set("HTTP-Referer", "https://github.com/osgrep/osgrep")
        .set("X-Title", "osgrep")
        .send_json(&request_body);

    // Handle HTTP errors
    let response = match response {
        Ok(r) => r,
        Err(ureq::Error::Status(code, response)) => {
            let body = response.into_string().unwrap_or_else(|_| "unknown".to_string());
            return Err(anyhow::anyhow!(
                "API error (HTTP {}): {}",
                code,
                body.chars().take(500).collect::<String>()
            ));
        }
        Err(e) => return Err(anyhow::anyhow!("API request failed: {}", e)),
    };

    let response_body: serde_json::Value = response
        .into_json()
        .map_err(|e| anyhow::anyhow!("Failed to parse API response: {}", e))?;

    // Check for error in response
    if let Some(error) = response_body.get("error") {
        let msg = error["message"].as_str().unwrap_or("Unknown error");
        return Err(anyhow::anyhow!("API error: {}", msg));
    }

    // Extract embeddings from response
    let data = response_body["data"]
        .as_array()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid API response: missing 'data' array. Response: {}",
                serde_json::to_string_pretty(&response_body)
                    .unwrap_or_else(|_| "unparseable".to_string())
                    .chars()
                    .take(500)
                    .collect::<String>()
            )
        })?;

    let mut embeddings = Vec::with_capacity(texts.len());
    for item in data {
        let embedding = item["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid API response: missing 'embedding'"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<f32>>();
        embeddings.push(embedding);
    }

    Ok(embeddings)
}

// ============================================================================
// Local embeddings via Candle ML
// ============================================================================

#[cfg(feature = "embeddings")]
use std::sync::Mutex;

#[cfg(feature = "embeddings")]
use {
    candle_core::{Device, Tensor},
    candle_nn::VarBuilder,
    candle_transformers::models::bert::{BertModel, Config, DTYPE},
    hf_hub::{api::sync::ApiBuilder, Repo, RepoType},
    tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams},
};

#[cfg(feature = "embeddings")]
static MODEL: OnceLock<Mutex<EmbeddingModel>> = OnceLock::new();

#[cfg(feature = "embeddings")]
struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

#[cfg(feature = "embeddings")]
impl EmbeddingModel {
    fn new() -> Result<Self> {
        let device = {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            Device::Cpu
        };

        let model_id = "BAAI/bge-base-en-v1.5";
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|e| anyhow::anyhow!("HF API error: {}", e))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .map_err(|e| anyhow::anyhow!("Failed to get config: {}", e))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to get tokenizer: {}", e))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| anyhow::anyhow!("Failed to get weights: {}", e))?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| anyhow::anyhow!("Failed to read config: {}", e))?,
        )
        .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

        let dim = config.hidden_size;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Tokenizer config error: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
        })
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let input_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
        let attention_mask: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| e.get_attention_mask().to_vec())
            .collect();

        let batch_size = input_ids.len();
        let seq_len = input_ids[0].len();

        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().collect();
        let attention_mask_flat: Vec<u32> = attention_mask.into_iter().flatten().collect();

        let input_ids_tensor =
            Tensor::from_vec(input_ids_flat, (batch_size, seq_len), &self.device)
                .map_err(|e| anyhow::anyhow!("Input tensor error: {}", e))?;

        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_flat, (batch_size, seq_len), &self.device)
                .map_err(|e| anyhow::anyhow!("Attention mask tensor error: {}", e))?;

        let token_type_ids =
            Tensor::zeros((batch_size, seq_len), candle_core::DType::U32, &self.device)
                .map_err(|e| anyhow::anyhow!("Token type tensor error: {}", e))?;

        let embeddings = self
            .model
            .forward(
                &input_ids_tensor,
                &token_type_ids,
                Some(&attention_mask_tensor),
            )
            .map_err(|e| anyhow::anyhow!("Model forward error: {}", e))?;

        // Mean pooling
        let mask = attention_mask_tensor
            .unsqueeze(2)
            .map_err(|e| anyhow::anyhow!("Unsqueeze error: {}", e))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| anyhow::anyhow!("Dtype error: {}", e))?;

        let masked = embeddings
            .broadcast_mul(&mask)
            .map_err(|e| anyhow::anyhow!("Broadcast mul error: {}", e))?;
        let summed = masked
            .sum(1)
            .map_err(|e| anyhow::anyhow!("Sum error: {}", e))?;
        let counts = mask
            .sum(1)
            .map_err(|e| anyhow::anyhow!("Count error: {}", e))?;
        let pooled = summed
            .broadcast_div(&counts)
            .map_err(|e| anyhow::anyhow!("Div error: {}", e))?;

        // L2 normalize
        let norm = pooled
            .sqr()
            .map_err(|e| anyhow::anyhow!("Sqr error: {}", e))?
            .sum_keepdim(1)
            .map_err(|e| anyhow::anyhow!("Sum keepdim error: {}", e))?
            .sqrt()
            .map_err(|e| anyhow::anyhow!("Sqrt error: {}", e))?;
        let normalized = pooled
            .broadcast_div(&norm)
            .map_err(|e| anyhow::anyhow!("Normalize error: {}", e))?;

        let flat: Vec<f32> = normalized
            .to_vec2::<f32>()
            .map_err(|e| anyhow::anyhow!("To vec error: {}", e))?
            .into_iter()
            .flatten()
            .collect();
        Ok(flat.chunks(self.dim).map(|c| c.to_vec()).collect())
    }
}

#[cfg(feature = "embeddings")]
fn local_embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let model = MODEL
        .get()
        .ok_or_else(|| anyhow::anyhow!("Model not initialized"))?
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock model: {}", e))?;

    const CHUNK_SIZE: usize = 32;
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(CHUNK_SIZE) {
        let chunk_embeddings = model.embed_batch(chunk)?;
        all_embeddings.extend(chunk_embeddings);
    }

    Ok(all_embeddings)
}

// ============================================================================
// Public API
// ============================================================================

/// Initialize the embedding provider (local model or verify remote config)
pub fn init() -> Result<()> {
    if is_remote() {
        // Remote provider - just verify config is valid
        let config = get_remote_config().unwrap();
        eprintln!(
            "  Using remote embeddings: {} ({})",
            config.model, config.base_url
        );
        Ok(())
    } else {
        #[cfg(feature = "embeddings")]
        {
            if MODEL.get().is_none() {
                let model = EmbeddingModel::new()?;
                let _ = MODEL.set(Mutex::new(model));
            }
            Ok(())
        }
        #[cfg(not(feature = "embeddings"))]
        {
            anyhow::bail!(
                "No embedding provider configured. Either:\n\
                 1. Build with --features embeddings for local model\n\
                 2. Set OSGREP_EMBEDDING_PROVIDER=openrouter and OSGREP_EMBEDDING_API_KEY"
            )
        }
    }
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
    if is_remote() {
        // Use remote API with small batches (some APIs like Gemini have strict limits)
        const BATCH_SIZE: usize = 20; // Small batch for API rate limits
        let mut all_embeddings = Vec::with_capacity(texts.len());
        let total = texts.len();

        for (i, chunk) in texts.chunks(BATCH_SIZE).enumerate() {
            let chunk_embeddings = remote_embed_batch(&chunk.to_vec())?;
            all_embeddings.extend(chunk_embeddings);
            progress((i + 1) * BATCH_SIZE.min(total - i * BATCH_SIZE), total);
        }

        Ok(all_embeddings)
    } else {
        #[cfg(feature = "embeddings")]
        {
            local_embed_batch(texts)
        }
        #[cfg(not(feature = "embeddings"))]
        {
            // Return zero vectors if no provider available
            Ok(texts.iter().map(|_| vec![0.0f32; 768]).collect())
        }
    }
}
