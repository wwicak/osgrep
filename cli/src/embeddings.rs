//! Native embeddings using Candle ML framework
//!
//! Memory-optimized for 8GB RAM:
//! - Uses Metal GPU acceleration on Apple Silicon
//! - Processes batches in chunks of 32

use anyhow::Result;

#[cfg(feature = "embeddings")]
use std::sync::OnceLock;

#[cfg(feature = "embeddings")]
use {
    candle_core::{DType, Device, Tensor},
    candle_nn::VarBuilder,
    candle_transformers::models::bert::{BertModel, Config, DTYPE},
    hf_hub::{api::sync::Api, Repo, RepoType},
    tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams},
};

#[cfg(feature = "embeddings")]
static MODEL: OnceLock<EmbeddingModel> = OnceLock::new();

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

        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        let api = Api::new().context("HF API error")?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").context("Failed to get config")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to get tokenizer")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("Failed to get weights")?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config")?,
        )
        .context("Failed to parse config")?;

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
                .context("Failed to load weights")?
        };

        let model = BertModel::load(vb, &config).context("Failed to load model")?;

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

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&x| x as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&x| x as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&x| x as i64));
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, seq_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, seq_len), &self.device)?;

        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling
        let mask_expanded = attention_mask
            .unsqueeze(2)?
            .broadcast_as(embeddings.shape())?
            .to_dtype(embeddings.dtype())?;

        let masked = (&embeddings * &mask_expanded)?;
        let sum = masked.sum(1)?;
        let count = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;
        let pooled = (&sum / &count)?;

        // L2 normalize
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-9, f64::MAX)?;
        let normalized = (&pooled / &norm)?;

        let flat: Vec<f32> = normalized
            .to_dtype(DType::F32)?
            .to_vec2()?
            .into_iter()
            .flatten()
            .collect();
        Ok(flat.chunks(self.dim).map(|c| c.to_vec()).collect())
    }
}

/// Initialize the embedding model
#[cfg(feature = "embeddings")]
pub fn init() -> Result<()> {
    MODEL.get_or_try_init(EmbeddingModel::new)?;
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
    use anyhow::Context;
    let model = MODEL.get().context("Model not initialized")?;
    let embeddings = model.embed_batch(&[text.to_string()])?;
    Ok(embeddings.into_iter().next().unwrap_or_default())
}

#[cfg(not(feature = "embeddings"))]
#[allow(dead_code)]
pub fn embed(_text: &str) -> Result<Vec<f32>> {
    Ok(vec![0.0f32; 384])
}

/// Embed multiple texts (batch processing)
#[cfg(feature = "embeddings")]
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    use anyhow::Context;
    let model = MODEL.get().context("Model not initialized")?;

    const CHUNK_SIZE: usize = 32;
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(CHUNK_SIZE) {
        let chunk_embeddings = model.embed_batch(&chunk.to_vec())?;
        all_embeddings.extend(chunk_embeddings);
    }

    Ok(all_embeddings)
}

#[cfg(not(feature = "embeddings"))]
#[allow(dead_code)]
pub fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    Ok(texts.iter().map(|_| vec![0.0f32; 384]).collect())
}
