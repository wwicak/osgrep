<div align="center">
  <h1>osgrep</h1>
  <p><em>Ultra-fast semantic code search with native SIMD acceleration.</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a><br>
</div>

Natural-language search that works like `grep`. Native Rust implementation with SIMD-optimized vector operations.

- **Semantic:** Finds concepts ("auth logic"), not just strings.
- **Native Performance:** Pure Rust with AVX-512/AVX2/NEON SIMD acceleration.
- **Local & Private:** 100% local embeddings via Candle ML (with Metal GPU on Apple Silicon).
- **Lightweight:** SQLite-Vec storage, no heavy dependencies.
- **Agent-Ready:** MCP server for Claude Code integration.

## Installation

### macOS

#### Apple Silicon (M1/M2/M3/M4) - Recommended
```bash
# Download and install
curl -LO https://github.com/Ryandonofrio3/osgrep/releases/latest/download/osgrep-darwin-arm64.tar.gz
tar xzf osgrep-darwin-arm64.tar.gz
sudo mv osgrep osgrep-mcp /usr/local/bin/

# Verify installation
osgrep info
```

#### Intel Mac
```bash
curl -LO https://github.com/Ryandonofrio3/osgrep/releases/latest/download/osgrep-darwin-x64.tar.gz
tar xzf osgrep-darwin-x64.tar.gz
sudo mv osgrep osgrep-mcp /usr/local/bin/
```

#### Build from Source (macOS)
```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and build
git clone https://github.com/Ryandonofrio3/osgrep
cd osgrep

# Apple Silicon: Build with Metal GPU acceleration
cargo build --release -p osgrep --features embeddings,metal,sqlite,parallel

# Intel Mac: Build without Metal
cargo build --release -p osgrep --features embeddings,sqlite,parallel

# Install
sudo cp target/release/osgrep /usr/local/bin/
```

### Windows

#### Pre-built Binary (Recommended)
```powershell
# Create install directory
New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\osgrep"

# Download and extract
Invoke-WebRequest -Uri "https://github.com/Ryandonofrio3/osgrep/releases/latest/download/osgrep-win32-x64.zip" -OutFile "$env:TEMP\osgrep.zip"
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
git clone https://github.com/Ryandonofrio3/osgrep
cd osgrep
cargo build --release -p osgrep --features embeddings,sqlite,parallel

# Install
Copy-Item target\release\osgrep.exe $env:LOCALAPPDATA\osgrep\
```

### Linux

```bash
# Download and install
curl -LO https://github.com/Ryandonofrio3/osgrep/releases/latest/download/osgrep-linux-x64.tar.gz
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
git clone https://github.com/Ryandonofrio3/osgrep
cd osgrep
cargo build --release -p osgrep --features embeddings,sqlite,parallel

# Install
sudo cp target/release/osgrep /usr/local/bin/
```

## Quick Start

```bash
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

The default command. Searches the current directory using semantic meaning.

The CLI prefers the hot server when available (via `.osgrep/server.json`), falling back to standalone search automatically.

```bash
osgrep "how is the database connection pooled?"
```

**Options:**
| Flag | Description | Default |
| --- | --- | --- |
| `-m <n>` | Max total results to return. | `25` |
| `--per-file <n>` | Max matches to show per file. | `1` |
| `-c`, `--content` | Show full chunk content instead of snippets. | `false` |
| `--scores` | Show relevance scores (0.0-1.0). | `false` |
| `--compact` | Show file paths only (like `grep -l`). | `false` |
| `-s`, `--sync` | Force re-index changed files before searching. | `false` |
| `--json` | Dense output for agents. | `false` |

**Examples:**

```bash
# General concept search
osgrep "API rate limiting logic"

# Deep dive (show more matches per file)
osgrep "error handling" --per-file 5

# Just give me the files
osgrep "user validation" --compact
```

### `osgrep index`

Manually indexes the repository. Useful if you want to pre-warm the cache or if you've made massive changes outside of the editor.

- Respects `.gitignore` and `.osgrepignore` (see [Configuration](#ignoring-files) section).
- **Smart Indexing:** Only embeds code and config files. Skips binaries, lockfiles, and minified assets.
- **Adaptive Throttling:** Monitors your RAM and CPU usage. If your system gets hot, indexing slows down automatically.

```bash
osgrep index              # Index current dir
osgrep index --dry-run    # See what would be indexed
```

### `osgrep serve`

Runs a lightweight HTTP server with live file watching so searches stay hot in RAM.

- Keeps LanceDB and the embedding worker resident for <50ms responses.
- Watches the repo (via chokidar) and incrementally re-indexes on change.
- Health endpoint: `GET /health`
- Search endpoint: `POST /search` with `{ query, limit, path, rerank }`
- Writes lock: `.osgrep/server.json` with `port`/`pid`

Usage:

```bash
osgrep serve             # defaults to port 4444
OSGREP_PORT=5555 osgrep serve
```

Claude Code hooks start/stop this automatically; you rarely need to run it manually.

### `osgrep list`

Lists all indexed repositories (stores) and their metadata.

```bash
osgrep list
```

Shows store names, sizes, and last modified times. Useful for seeing what's indexed and cleaning up old stores.

### `osgrep doctor`

Checks installation health, model paths, and database integrity.

```bash
osgrep doctor
```

## Performance & Architecture

osgrep is designed to be a "good citizen" on your machine:

1.  **The Thermostat:** Indexing adjusts concurrency in real-time based on memory pressure and CPU speed. It won't freeze your laptop.
2.  **Smart Chunking:** Uses `tree-sitter` to split code by function/class boundaries, ensuring embeddings capture complete logical blocks.
3.  **Deduplication:** Identical code blocks (boilerplate, license headers) are embedded once and cached, saving space and time.
4.  **Hybrid Search:** Uses Reciprocal Rank Fusion (RRF) to combine Vector Search (semantic) with FTS (keyword) for best-of-both-worlds accuracy.

## Configuration

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
  - **Weird results?** Run `osgrep doctor` to verify models.
  - **Need a fresh start?** Delete `~/.osgrep/data` and re-index.

## Attribution

osgrep is built upon the foundation of [mgrep](https://github.com/mixedbread-ai/mgrep) by MixedBread. While approximately 90% of the current codebase has been rewritten or substantially modified to enable fully local operation, we acknowledge and appreciate the original architectural concepts and design decisions that informed this work.

Key transformations in osgrep include:
- Complete transition to local-only embeddings (no remote APIs)
- New local storage architecture with LanceDB
- Enhanced chunking, indexing, and watch mode capabilities
- Extensive tooling for benchmarking and evaluation

See the [NOTICE](NOTICE) file for detailed attribution information.

## License

Licensed under the Apache License, Version 2.0.  
See [LICENSE](LICENSE) and [Apache-2.0](https://opensource.org/licenses/Apache-2.0) for details.

