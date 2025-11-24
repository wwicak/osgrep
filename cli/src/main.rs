//! osgrep - Ultra-fast semantic code search
//!
//! A native Rust CLI for semantic code search with:
//! - SIMD-optimized vector operations (AVX-512/AVX2/NEON)
//! - Native embeddings via Candle (optional Metal acceleration)
//! - SQLite-Vec vector storage
//! - Tree-sitter code chunking
//! - File watching for live index updates

#![allow(clippy::ptr_arg)]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

mod chunker;
mod embeddings;
mod simd;
mod store;

#[derive(Parser)]
#[command(name = "osgrep")]
#[command(author = "osgrep")]
#[command(version)]
#[command(about = "Ultra-fast semantic code search", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a directory for semantic search
    Index {
        /// Path to directory to index
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Store name (default: directory name)
        #[arg(short, long)]
        name: Option<String>,

        /// Watch for file changes
        #[arg(short, long)]
        watch: bool,

        /// Number of parallel jobs (default: 2, use 1 for low-memory systems)
        #[arg(short, long, default_value = "2")]
        jobs: usize,
    },

    /// Search indexed code
    Search {
        /// Search query
        query: String,

        /// Store name to search
        #[arg(short, long)]
        name: Option<String>,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Filter by path prefix
        #[arg(short, long)]
        path: Option<String>,

        /// Output JSON
        #[arg(long)]
        json: bool,

        /// Output TOON (Token-Oriented Object Notation) - most efficient for LLMs
        #[arg(long)]
        toon: bool,
    },

    /// List indexed stores
    List,

    /// Show system info
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index {
            path,
            name,
            watch,
            jobs,
        } => cmd_index(path, name, watch, jobs),
        Commands::Search {
            query,
            name,
            top_k,
            path,
            json,
            toon,
        } => cmd_search(query, name, top_k, path, json, toon),
        Commands::List => cmd_list(),
        Commands::Info => cmd_info(),
    }
}

fn cmd_index(path: PathBuf, name: Option<String>, watch: bool, jobs: usize) -> Result<()> {
    let path = path.canonicalize().context("Invalid path")?;
    let store_name = name.unwrap_or_else(|| {
        path.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "default".to_string())
    });

    println!(
        "{} Indexing {} as '{}'",
        style("→").cyan(),
        path.display(),
        store_name
    );

    // Initialize store
    let db_path = get_db_path()?;
    store::open(&db_path, &store_name)?;

    // Collect files
    let files = collect_files(&path)?;
    println!("{} Found {} files", style("→").cyan(), files.len());

    // Phase 1: Collect all chunks (parallel file I/O, no GPU)
    println!("{} Chunking files...", style("→").cyan());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    // Struct to hold chunk data before embedding
    struct PendingChunk {
        path: String,
        text: String,
        start_line: i32,
        end_line: i32,
        chunk_index: i32,
        is_anchor: bool,
    }

    let all_chunks: Vec<PendingChunk> = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            use std::sync::Mutex;

            let chunks = Mutex::new(Vec::new());
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(jobs.max(1))
                .build()
                .context("Failed to create thread pool")?;

            pool.install(|| {
                files.par_iter().for_each(|file| {
                    if let Ok(content) = std::fs::read_to_string(file) {
                        let rel_path = file.strip_prefix(&path).unwrap_or(file);
                        let rel_path_str = rel_path.to_string_lossy().to_string();

                        // Delete existing chunks for this file
                        let _ = store::delete_by_path(&db_path, &store_name, &rel_path_str);

                        if let Ok(file_chunks) = chunker::chunk(file, &content) {
                            let pending: Vec<PendingChunk> = file_chunks
                                .into_iter()
                                .enumerate()
                                .map(|(i, c)| PendingChunk {
                                    path: rel_path_str.clone(),
                                    text: c.text,
                                    start_line: c.start_line as i32,
                                    end_line: c.end_line as i32,
                                    chunk_index: i as i32,
                                    is_anchor: c.is_anchor,
                                })
                                .collect();
                            chunks.lock().unwrap().extend(pending);
                        }
                    }
                    pb.inc(1);
                });
            });

            chunks.into_inner().unwrap()
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut chunks = Vec::new();
            for file in &files {
                if let Ok(content) = std::fs::read_to_string(file) {
                    let rel_path = file.strip_prefix(&path).unwrap_or(file);
                    let rel_path_str = rel_path.to_string_lossy().to_string();

                    let _ = store::delete_by_path(&db_path, &store_name, &rel_path_str);

                    if let Ok(file_chunks) = chunker::chunk(file, &content) {
                        let pending: Vec<PendingChunk> = file_chunks
                            .into_iter()
                            .enumerate()
                            .map(|(i, c)| PendingChunk {
                                path: rel_path_str.clone(),
                                text: c.text,
                                start_line: c.start_line as i32,
                                end_line: c.end_line as i32,
                                chunk_index: i as i32,
                                is_anchor: c.is_anchor,
                            })
                            .collect();
                        chunks.extend(pending);
                    }
                }
                pb.inc(1);
            }
            chunks
        }
    };

    pb.finish_with_message("done");
    println!("{} Found {} chunks", style("→").cyan(), all_chunks.len());

    if all_chunks.is_empty() {
        println!("{} No chunks to index", style("!").yellow());
        return Ok(());
    }

    // Phase 2: Batch embed all chunks (single GPU operation)
    #[cfg(feature = "embeddings")]
    {
        println!("{} Loading embedding model...", style("→").cyan());
        embeddings::init()?;

        println!("{} Generating embeddings...", style("→").cyan());
        let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = embeddings::embed_batch(&texts)?;

        // Phase 3: Insert all to database
        println!("{} Storing vectors...", style("→").cyan());
        let records: Vec<store::Record> = all_chunks
            .iter()
            .map(|c| store::Record {
                id: uuid::Uuid::new_v4().to_string(),
                path: c.path.clone(),
                content: c.text.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_index: c.chunk_index,
                is_anchor: c.is_anchor,
            })
            .collect();

        // Insert in batches to avoid memory issues
        const BATCH_SIZE: usize = 500;
        for (batch_records, batch_vectors) in records
            .chunks(BATCH_SIZE)
            .zip(vectors.chunks(BATCH_SIZE))
        {
            store::insert_batch(&db_path, &store_name, batch_records, batch_vectors)?;
        }
    }

    #[cfg(not(feature = "embeddings"))]
    {
        let records: Vec<store::Record> = all_chunks
            .iter()
            .map(|c| store::Record {
                id: uuid::Uuid::new_v4().to_string(),
                path: c.path.clone(),
                content: c.text.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_index: c.chunk_index,
                is_anchor: c.is_anchor,
            })
            .collect();

        let vectors: Vec<Vec<f32>> = records.iter().map(|_| vec![0.0f32; 768]).collect();
        store::insert_batch(&db_path, &store_name, &records, &vectors)?;
    }

    let count = store::count(&db_path, &store_name)?;
    println!("{} Indexed {} chunks", style("✓").green(), count);

    if watch {
        println!("{} Watching for changes...", style("→").cyan());
        watch_directory(&path, &db_path, &store_name)?;
    }

    Ok(())
}

#[allow(unused_variables)]
fn cmd_search(
    query: String,
    name: Option<String>,
    top_k: usize,
    path_filter: Option<String>,
    json_output: bool,
    toon_output: bool,
) -> Result<()> {
    let store_name = name.unwrap_or_else(|| "default".to_string());
    let db_path = get_db_path()?;

    // Open store connection (required before search)
    store::open(&db_path, &store_name)?;

    // Generate query embedding
    #[cfg(feature = "embeddings")]
    let query_vec = {
        embeddings::init()?;
        embeddings::embed(&format!(
            "Represent this sentence for searching relevant passages: {}",
            query
        ))?
    };

    #[cfg(not(feature = "embeddings"))]
    let query_vec = vec![0.0f32; 768]; // BGE base dimension

    // Search
    let results = store::search(
        &db_path,
        &store_name,
        &query_vec,
        top_k,
        path_filter.as_deref(),
    )?;

    // TOON output - Token-Oriented Object Notation (most efficient for LLMs)
    if toon_output {
        if results.is_empty() {
            println!("results[0]{{path,score,lines,content}}:");
            return Ok(());
        }
        println!("results[{}]{{path,score,lines,content}}:", results.len());
        for r in &results {
            // Escape commas and newlines in content for TOON format
            let content_escaped = r
                .content
                .lines()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
                .replace(',', "\\,")
                .chars()
                .take(200)
                .collect::<String>();
            println!(
                "  {},{:.2},{}-{},{}",
                r.path, r.score, r.start_line, r.end_line, content_escaped
            );
        }
        return Ok(());
    }

    // JSON output
    if json_output {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "path": r.path,
                    "score": r.score,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "content": r.content
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&json_results)?);
        return Ok(());
    }

    if results.is_empty() {
        println!("{} No results found", style("!").yellow());
        return Ok(());
    }

    // Display results (human-readable)
    for (i, result) in results.iter().enumerate() {
        println!(
            "\n{} {} (score: {:.3})",
            style(format!("[{}]", i + 1)).cyan(),
            style(&result.path).green(),
            result.score
        );
        println!("   Lines {}-{}", result.start_line, result.end_line);

        // Show content preview
        let preview: String = result
            .content
            .lines()
            .take(5)
            .map(|l| format!("   {}", l))
            .collect::<Vec<_>>()
            .join("\n");
        println!("{}", style(preview).dim());
    }

    Ok(())
}

fn cmd_list() -> Result<()> {
    let _db_path = get_db_path()?;

    println!("{} Indexed stores:", style("→").cyan());

    // List all stores (simplified - would need metadata table)
    println!("  Use 'osgrep search --name <store>' to search a specific store");

    Ok(())
}

fn cmd_info() -> Result<()> {
    println!("{}", style("osgrep system info").bold());
    println!();

    // SIMD level
    println!("SIMD: {}", simd::get_level());

    // Embeddings
    #[cfg(feature = "embeddings")]
    {
        println!("Embeddings: candle (native)");
        #[cfg(feature = "metal")]
        println!("  Metal: enabled");
        #[cfg(not(feature = "metal"))]
        println!("  Metal: disabled");
    }
    #[cfg(not(feature = "embeddings"))]
    println!("Embeddings: disabled");

    // Storage
    #[cfg(feature = "sqlite")]
    println!("Storage: SQLite-Vec");
    #[cfg(not(feature = "sqlite"))]
    println!("Storage: disabled");

    // Parallel
    #[cfg(feature = "parallel")]
    println!("Parallel: enabled (rayon)");
    #[cfg(not(feature = "parallel"))]
    println!("Parallel: disabled");

    Ok(())
}

fn get_db_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not find home directory")?;
    let db_dir = home.join(".osgrep").join("data");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("osgrep.db"))
}

fn collect_files(path: &PathBuf) -> Result<Vec<PathBuf>> {
    use ignore::WalkBuilder;

    let mut files = Vec::new();

    for entry in WalkBuilder::new(path).hidden(true).git_ignore(true).build() {
        let entry = entry?;
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            let path = entry.path();
            if is_code_file(path) {
                files.push(path.to_path_buf());
            }
        }
    }

    Ok(files)
}

fn is_code_file(path: &std::path::Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "rs" | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "py"
            | "go"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "java"
            | "kt"
            | "swift"
            | "rb"
            | "php"
            | "cs"
            | "scala"
            | "zig"
            | "lua"
            | "sh"
            | "bash"
            | "zsh"
            | "fish"
            | "ps1"
            | "bat"
            | "cmd"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "xml"
            | "html"
            | "css"
            | "scss"
            | "md"
    )
}

fn index_file(
    db_path: &PathBuf,
    store_name: &str,
    file: &std::path::Path,
    base: &std::path::Path,
) -> Result<()> {
    let content = std::fs::read_to_string(file)?;
    let rel_path = file.strip_prefix(base).unwrap_or(file);
    let rel_path_str = rel_path.to_string_lossy().to_string();

    // Delete existing chunks for this file
    store::delete_by_path(db_path, store_name, &rel_path_str)?;

    // Chunk the file
    let chunks = chunker::chunk(file, &content)?;

    if chunks.is_empty() {
        return Ok(());
    }

    // Generate embeddings and insert
    #[cfg(feature = "embeddings")]
    {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = embeddings::embed_batch(&texts)?;

        let records: Vec<store::Record> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| store::Record {
                id: uuid::Uuid::new_v4().to_string(),
                path: rel_path_str.clone(),
                content: c.text.clone(),
                start_line: c.start_line as i32,
                end_line: c.end_line as i32,
                chunk_index: i as i32,
                is_anchor: c.is_anchor,
            })
            .collect();

        store::insert_batch(db_path, store_name, &records, &vectors)?;
    }

    #[cfg(not(feature = "embeddings"))]
    {
        // Without embeddings, just store the chunks with zero vectors
        let records: Vec<store::Record> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| store::Record {
                id: uuid::Uuid::new_v4().to_string(),
                path: rel_path_str.clone(),
                content: c.text.clone(),
                start_line: c.start_line as i32,
                end_line: c.end_line as i32,
                chunk_index: i as i32,
                is_anchor: c.is_anchor,
            })
            .collect();

        let vectors: Vec<Vec<f32>> = records.iter().map(|_| vec![0.0f32; 768]).collect(); // BGE base dimension
        store::insert_batch(db_path, store_name, &records, &vectors)?;
    }

    Ok(())
}

fn watch_directory(path: &PathBuf, db_path: &PathBuf, store_name: &str) -> Result<()> {
    use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
    use std::sync::mpsc::channel;

    let (tx, rx) = channel();

    let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
    watcher.watch(path, RecursiveMode::Recursive)?;

    for res in rx {
        match res {
            Ok(event) => {
                for path in event.paths {
                    if is_code_file(&path) {
                        println!("{} Changed: {}", style("→").cyan(), path.display());
                        if let Err(e) =
                            index_file(db_path, store_name, &path, path.parent().unwrap_or(&path))
                        {
                            eprintln!("Error re-indexing: {}", e);
                        }
                    }
                }
            }
            Err(e) => eprintln!("Watch error: {}", e),
        }
    }

    Ok(())
}

// Add dirs crate for home directory
mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            std::env::var("USERPROFILE").ok().map(PathBuf::from)
        }
        #[cfg(not(target_os = "windows"))]
        {
            std::env::var("HOME").ok().map(PathBuf::from)
        }
    }
}
