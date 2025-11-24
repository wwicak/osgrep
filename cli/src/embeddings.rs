//! Native embeddings using Candle ML framework
//!
//! Uses BAAI/bge-base-en-v1.5 for high-quality code embeddings:
//! - 768 dimensions, trained on diverse data including code
//! - Uses Metal GPU acceleration on Apple Silicon
//! - Processes batches in chunks of 32
//! - L2 normalized outputs for cosine similarity

use anyhow::Result;

#[cfg(feature = "embeddings")]
use std::sync::{Mutex, OnceLock};

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

        // BGE model - excellent for code retrieval, BERT-compatible
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

        // L2 normalize the embeddings (required for BGE model)
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

/// Initialize the embedding model
#[cfg(feature = "embeddings")]
pub fn init() -> Result<()> {
    if MODEL.get().is_none() {
        let model = EmbeddingModel::new()?;
        let _ = MODEL.set(Mutex::new(model));
    }
    Ok(())
}

#[cfg(not(feature = "embeddings"))]
#[allow(dead_code)]
pub fn init() -> Result<()> {
    Ok(())
}

/// Embed a single text
#[cfg(feature = "embeddings")]
pub fn embed(text: &str) -> Result<Vec<f32>> {
    let model = MODEL
        .get()
        .ok_or_else(|| anyhow::anyhow!("Model not initialized"))?
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock model: {}", e))?;
    let embeddings = model.embed_batch(&[text.to_string()])?;
    Ok(embeddings.into_iter().next().unwrap_or_default())
}

#[cfg(not(feature = "embeddings"))]
#[allow(dead_code)]
pub fn embed(_text: &str) -> Result<Vec<f32>> {
    Ok(vec![0.0f32; 768]) // BGE base dimension
}

/// Embed multiple texts (batch processing)
#[cfg(feature = "embeddings")]
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
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

#[cfg(not(feature = "embeddings"))]
#[allow(dead_code)]
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    Ok(texts.iter().map(|_| vec![0.0f32; 768]).collect()) // BGE base dimension
}
