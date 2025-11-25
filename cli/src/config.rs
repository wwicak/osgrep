//! Configuration management for osgrep
//!
//! Config file location: ~/.osgrep/config.json
//!
//! Example config:
//! ```json
//! {
//!   "embedding": {
//!     "provider": "openrouter",
//!     "api_key": "sk-or-v1-...",
//!     "model": "google/gemini-embedding-001",
//!     "base_url": "https://openrouter.ai/api/v1"
//!   }
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingConfig {
    /// Provider: "openrouter", "openai", or "remote"
    #[serde(default)]
    pub provider: Option<String>,

    /// API key for remote providers (required)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model name (default: openai/text-embedding-3-small)
    #[serde(default)]
    pub model: Option<String>,

    /// Base URL for API (default: https://openrouter.ai/api/v1)
    #[serde(default)]
    pub base_url: Option<String>,

    /// Vector dimensions (default: 1536)
    #[serde(default)]
    pub dimensions: Option<usize>,
}

/// Get the config directory path (~/.osgrep)
pub fn get_config_dir() -> Result<PathBuf> {
    let home = get_home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    Ok(home.join(".osgrep"))
}

/// Get the config file path (~/.osgrep/config.json)
pub fn get_config_path() -> Result<PathBuf> {
    Ok(get_config_dir()?.join("config.json"))
}

fn get_home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Load config from file, falling back to environment variables
pub fn load() -> &'static Config {
    CONFIG.get_or_init(|| {
        // Try to load from file first
        if let Ok(config_path) = get_config_path() {
            if config_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<Config>(&content) {
                        return config;
                    }
                }
            }
        }

        // Fall back to environment variables
        Config {
            embedding: EmbeddingConfig {
                provider: std::env::var("OSGREP_EMBEDDING_PROVIDER").ok(),
                api_key: std::env::var("OSGREP_EMBEDDING_API_KEY").ok(),
                model: std::env::var("OSGREP_EMBEDDING_MODEL").ok(),
                base_url: std::env::var("OSGREP_EMBEDDING_BASE_URL").ok(),
                dimensions: std::env::var("OSGREP_EMBEDDING_DIMENSIONS")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            },
        }
    })
}

/// Save config to file
pub fn save(config: &Config) -> Result<()> {
    let config_dir = get_config_dir()?;
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.json");
    let content = serde_json::to_string_pretty(config)?;
    std::fs::write(&config_path, content)?;

    Ok(())
}

/// Create a sample config file if it doesn't exist
pub fn create_sample() -> Result<PathBuf> {
    let config_path = get_config_path()?;

    if config_path.exists() {
        return Ok(config_path);
    }

    let sample = Config {
        embedding: EmbeddingConfig {
            provider: Some("openrouter".to_string()),
            api_key: Some("sk-or-v1-YOUR_API_KEY_HERE".to_string()),
            model: Some("openai/text-embedding-3-small".to_string()),
            base_url: Some("https://openrouter.ai/api/v1".to_string()),
            dimensions: Some(1536),
        },
    };

    save(&sample)?;
    Ok(config_path)
}

/// Update a specific config value
pub fn set_embedding_config(
    provider: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
) -> Result<()> {
    let mut config = load().clone();

    if let Some(p) = provider {
        config.embedding.provider = Some(p);
    }
    if let Some(k) = api_key {
        config.embedding.api_key = Some(k);
    }
    if let Some(m) = model {
        config.embedding.model = Some(m);
    }
    if let Some(u) = base_url {
        config.embedding.base_url = Some(u);
    }

    save(&config)
}
