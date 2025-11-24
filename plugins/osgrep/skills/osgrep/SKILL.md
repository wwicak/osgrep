---
name: osgrep
description: Ultra-fast semantic code search with SIMD acceleration. Native Rust implementation. Use instead of grep/find for conceptual searches.
license: Apache-2.0
---

## When to use

Use `osgrep` for all **conceptual** code discovery:
- "Where is authentication handled?"
- "How does rate limiting work?"
- "Find error handling logic"

Use `grep` only for **exact string matches** when osgrep doesn't find results.

## How to use

**Always use `--toon` flag.** TOON (Token-Oriented Object Notation) uses ~40% fewer tokens than JSON.

### Basic Search

```bash
osgrep search --toon "How are user authentication tokens validated?"
osgrep search --toon "Where do we handle retries or backoff?"
```

### Scoped Search

```bash
osgrep search --toon "auth middleware" --path src/api
```

### First-time Setup

Index the codebase first (one-time, ~30s for medium repo):

```bash
osgrep index
```

### Flags

| Flag | Description |
|------|-------------|
| `--toon` | **Recommended.** TOON output (40% fewer tokens than JSON) |
| `--json` | JSON output (fallback) |
| `-k <n>` | Max results (default: 10) |
| `--path <prefix>` | Filter by path prefix |
| `--name <store>` | Use specific store name |

### TOON Output Format

```
results[3]{path,score,lines,content}:
  src/auth.rs,0.92,45-78,fn authenticate() { ... }
  src/user.rs,0.85,12-34,fn validate_user() { ... }
  src/login.rs,0.81,100-120,fn login() { ... }
```

Parse by:
1. First line: `results[N]{fields}:` - N = result count, fields = column names
2. Following lines: CSV-like rows with 2-space indent

### Strategy

1. Run `osgrep search --toon "<question>"`.
2. Parse TOON header for result count and fields.
3. Use `Read` only if you need more context from a specific file.
4. Increase `-k` if results seem incomplete.

### Performance

- **SIMD-accelerated**: AVX-512/AVX2/NEON vector operations
- **Native embeddings**: Candle ML (Metal GPU on Apple Silicon)
- **SQLite-Vec**: Lightweight vector storage
- **Hot index**: Sub-50ms searches after first index
- **TOON output**: 40% fewer tokens than JSON

## Keywords
semantic search, code search, local search, grep alternative, find code, explore codebase, understand code, search by meaning, SIMD, native, fast, TOON
