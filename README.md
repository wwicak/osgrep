<div align="center">
  <h1>osgrep</h1>
  <p><em>Ultra-fast semantic code search with native SIMD acceleration.</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a><br>
</div>

Natural-language search that works like `grep`. Native Rust implementation with SIMD-optimized vector operations.

- **Semantic:** Finds concepts ("auth logic"), not just strings.
- **Native Performance:** Pure Rust with AVX-512/AVX2/NEON SIMD acceleration.
- **Cloud-Powered:** Remote embeddings via OpenAI-compatible APIs (OpenRouter, OpenAI).
- **Lightweight:** SQLite-Vec storage, no heavy dependencies.
- **Agent-Ready:** MCP server for Claude Code integration.

## Installation

### macOS

#### Apple Silicon (M1/M2/M3/M4) - Recommended
```bash
# Download and install
curl -LO https://github.com/wwicak/osgrep/releases/latest/download/osgrep-darwin-arm64.tar.gz
tar xzf osgrep-darwin-arm64.tar.gz
sudo mv osgrep osgrep-mcp /usr/local/bin/

# Verify installation
osgrep info
```

#### Intel Mac
```bash
curl -LO https://github.com/wwicak/osgrep/releases/latest/download/osgrep-darwin-x64.tar.gz
tar xzf osgrep-darwin-x64.tar.gz
sudo mv osgrep osgrep-mcp /usr/local/bin/
```

#### Build from Source (macOS)
```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and build
git clone https://github.com/wwicak/osgrep
cd osgrep
cargo build --release -p osgrep --features sqlite,parallel

# Install
sudo cp target/release/osgrep /usr/local/bin/
```

### Windows

#### Pre-built Binary (Recommended)
```powershell
# Create install directory
New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\osgrep"

# Download and extract
Invoke-WebRequest -Uri "https://github.com/wwicak/osgrep/releases/latest/download/osgrep-win32-x64.zip" -OutFile "$env:TEMP\osgrep.zip"
Expand-Archive -Path "$env:TEMP\osgrep.zip" -DestinationPath "$env:LOCALAPPDATA\osgrep" -Force

# Add to PATH permanently (run as Administrator or add manually)
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:LOCALAPPDATA\osgrep", "User")

# Restart terminal and verify
osgrep info
```

#### Build from Source (Windows)
```powershell
# Install Rust from https://rustup.rs (follow installer prompts)
# Install Visual Studio Build Tools with "Desktop development with C++"

# Clone and build
git clone https://github.com/wwicak/osgrep
cd osgrep
cargo build --release -p osgrep --features sqlite,parallel

# Install
Copy-Item target\release\osgrep.exe $env:LOCALAPPDATA\osgrep\
```

### Linux

```bash
# Download and install
curl -LO https://github.com/wwicak/osgrep/releases/latest/download/osgrep-linux-x64.tar.gz
tar xzf osgrep-linux-x64.tar.gz
sudo mv osgrep osgrep-mcp /usr/local/bin/

# Verify installation
osgrep info
```

#### Build from Source (Linux)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and build
git clone https://github.com/wwicak/osgrep
cd osgrep
cargo build --release -p osgrep --features sqlite,parallel

# Install
sudo cp target/release/osgrep /usr/local/bin/
```

## Quick Start

```bash
# Configure remote embeddings (required)
osgrep config --init
# Edit ~/.osgrep/config.json with your API key

# Index your codebase
cd my-repo
osgrep index

# Search semantically
osgrep search "where do we handle authentication?"

# Show system info (SIMD level, features)
osgrep info
```

## Claude Code Integration

osgrep integrates with Claude Code as an MCP (Model Context Protocol) server, providing semantic search tools that replace traditional grep/find operations.

### Installation

**Option 1: Automatic Setup**
```bash
osgrep install-claude-code
```

**Option 2: Manual Setup**

1. Build the MCP server:
```bash
cargo build --release -p osgrep-mcp
```

2. Add to your Claude Code MCP settings (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "osgrep": {
      "command": "/usr/local/bin/osgrep-mcp",
      "args": []
    }
  }
}
```

On Windows, use the full path:
```json
{
  "mcpServers": {
    "osgrep": {
      "command": "C:\\Users\\YOU\\AppData\\Local\\osgrep\\osgrep-mcp.exe",
      "args": []
    }
  }
}
```

### Available Tools

Once configured, Claude Code has access to:

| Tool | Description |
|------|-------------|
| `semantic_search` | Search code by meaning (replaces grep for conceptual queries) |
| `index_directory` | Index a codebase for semantic search |
| `get_simd_info` | Show SIMD capabilities and version |

### Usage in Claude Code

Ask Claude natural language questions about your codebase:

```
"Where is authentication handled?"
"How does the rate limiting work?"
"Find the database connection pooling logic"
```

Claude will automatically use osgrep's semantic search instead of traditional grep when appropriate.

## Commands

### `osgrep search`

Searches the indexed codebase using semantic meaning.

```bash
osgrep search "how is the database connection pooled?"
```

**Options:**
| Flag | Description | Default |
| --- | --- | --- |
| `-k`, `--top-k <n>` | Number of results to return | `10` |
| `-n`, `--name <store>` | Store name to search | Current directory |
| `-p`, `--path <prefix>` | Filter by path prefix | None |
| `--json` | Output JSON format | `false` |
| `--toon` | Output TOON format (efficient for LLMs) | `false` |

**Examples:**

```bash
# Basic search
osgrep search "API rate limiting logic"

# Get more results
osgrep search "error handling" --top-k 20

# Search specific store
osgrep search "user validation" --name my-project

# Filter by path
osgrep search "database queries" --path src/db/

# JSON output for tools
osgrep search "authentication" --json
```

### `osgrep index`

Indexes a directory for semantic search. Creates embeddings for all code files.

- Respects `.gitignore` patterns
- Uses tree-sitter for smart code chunking
- Supports parallel processing for faster indexing

```bash
osgrep index              # Index current directory
osgrep index /path/to/repo    # Index specific directory
osgrep index --name my-project  # Use custom store name
osgrep index --watch      # Watch for file changes (live updates)
osgrep index --jobs 4     # Use 4 parallel workers
```

### `osgrep list`

Lists all indexed repositories (stores).

```bash
osgrep list
```

### `osgrep info`

Shows system information including SIMD capabilities and embedding configuration.

```bash
osgrep info
```

### `osgrep config`

Manages configuration settings for embedding providers.

```bash
osgrep config                    # Show current config
osgrep config --init             # Create sample config file
osgrep config --path             # Show config file path
osgrep config --show             # Show current config
osgrep config --reset            # Delete config file
osgrep config --provider openai  # Set provider
osgrep config --api-key sk-...   # Set API key
osgrep config --model text-embedding-3-small  # Set model
```

## Performance & Architecture

osgrep is designed to be a "good citizen" on your machine:

1.  **The Thermostat:** Indexing adjusts concurrency in real-time based on memory pressure and CPU speed. It won't freeze your laptop.
2.  **Smart Chunking:** Uses `tree-sitter` to split code by function/class boundaries, ensuring embeddings capture complete logical blocks.
3.  **Deduplication:** Identical code blocks (boilerplate, license headers) are embedded once and cached, saving space and time.
4.  **Hybrid Search:** Uses Reciprocal Rank Fusion (RRF) to combine Vector Search (semantic) with FTS (keyword) for best-of-both-worlds accuracy.

## Configuration

### Embedding Providers

osgrep uses remote embeddings via cloud APIs like OpenRouter, OpenAI, or any OpenAI-compatible endpoint.

#### Configuring Remote Embeddings (Required)

You must configure remote embeddings before using osgrep:

**1. Create config file:**
```bash
osgrep config --init
```

**2. Edit `~/.osgrep/config.json`:**
```json
{
  "embedding": {
    "provider": "openrouter",
    "api_key": "sk-or-v1-YOUR_API_KEY_HERE",
    "model": "openai/text-embedding-3-small",
    "base_url": "https://openrouter.ai/api/v1"
  }
}
```

**3. Or use CLI commands:**
```bash
osgrep config --provider openrouter
osgrep config --api-key sk-or-v1-...
osgrep config --model openai/text-embedding-3-small
```

#### Config Commands

| Command | Description |
|---------|-------------|
| `osgrep config` | Show current configuration |
| `osgrep config --init` | Create sample config file |
| `osgrep config --path` | Print config file path |
| `osgrep config --reset` | Delete config file |
| `osgrep config --provider X` | Set embedding provider |
| `osgrep config --api-key X` | Set API key |
| `osgrep config --model X` | Set model name |
| `osgrep config --base-url X` | Set API base URL |

#### Supported Providers

| Provider | `provider` value | `base_url` |
|----------|------------------|------------|
| OpenRouter | `openrouter` | `https://openrouter.ai/api/v1` (default) |
| OpenAI | `openai` | `https://api.openai.com/v1` |

#### Recommended Models

| Model | Provider | Dimensions | Notes |
|-------|----------|------------|-------|
| `openai/text-embedding-3-small` | OpenRouter | 1536 | Default, good balance |
| `openai/text-embedding-3-large` | OpenRouter | 3072 | Higher quality |
| `text-embedding-3-small` | OpenAI | 1536 | Direct OpenAI API |

### Automatic Repository Isolation

osgrep automatically creates a unique index for each repository based on:

1. **Git Remote URL** (e.g., `github.com/facebook/react` → `facebook-react`)
2. **Git Repo without Remote** → directory name + hash (e.g., `utils-7f8a2b3c`)
3. **Non-Git Directory** → directory name + hash for collision safety

**Examples:**
```bash
cd ~/work/myproject        # Auto-detected: owner-myproject
osgrep "API handlers"

cd ~/personal/utils        # Auto-detected: utils-abc12345
osgrep "helper functions"
```

Stores are isolated automatically — no manual `--store` flags needed!

### Ignoring Files

osgrep respects both `.gitignore` and `.osgrepignore` files when indexing. Create a `.osgrepignore` file in your repository root to exclude additional files or patterns from indexing.

**`.osgrepignore` syntax:**
- Uses the same pattern syntax as `.gitignore`
- Patterns are relative to the repository root
- Supports glob patterns, negation (`!`), and directory patterns (`/`)


### Manual Store Management

  - **View all stores:** `osgrep list`
  - **Override auto-detection:** `osgrep --store custom-name "query"`
  - **Clean up old stores:** `rm -rf ~/.osgrep/data/store-name`
  - **Data location:** `~/.osgrep/data`
  - **Env Vars:**
      - `MXBAI_STORE`: Override default store name
      - `OSGREP_PROFILE=1`: Enable performance profiling logs

## Development

```bash
# Build all packages
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt --all

# Lint
cargo clippy --all-targets --all-features
```

## Troubleshooting

  - **Index feels stale?** Run `osgrep index` to refresh.
  - **No results?** Check that indexing completed successfully and the store exists (`osgrep list`).
  - **API errors?** Verify your API key has credits (`osgrep config --show`).
  - **Need a fresh start?** Delete `~/.osgrep/data` and re-index.

## Attribution

osgrep is built upon the foundation of [mgrep](https://github.com/mixedbread-ai/mgrep) by MixedBread. While approximately 90% of the current codebase has been rewritten or substantially modified, we acknowledge and appreciate the original architectural concepts and design decisions that informed this work.

Key transformations in osgrep include:
- Transition to remote-only embeddings via OpenAI-compatible APIs
- SQLite-Vec storage architecture for efficient vector operations
- Enhanced chunking, indexing, and watch mode capabilities
- Extensive tooling for benchmarking and evaluation

See the [NOTICE](NOTICE) file for detailed attribution information.

## License

Licensed under the Apache License, Version 2.0.  
See [LICENSE](LICENSE) and [Apache-2.0](https://opensource.org/licenses/Apache-2.0) for details.

