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

**Always use `--json` flag.** Results are fast (<50ms with hot index).

### Basic Search

```bash
osgrep search --json "How are user authentication tokens validated?"
osgrep search --json "Where do we handle retries or backoff?"
```

### Scoped Search

```bash
osgrep search --json "auth middleware" --path src/api
```

### First-time Setup

Index the codebase first (one-time, ~30s for medium repo):

```bash
osgrep index
```

### Flags

| Flag | Description |
|------|-------------|
| `--json` | **Required.** JSON output for parsing |
| `-k <n>` | Max results (default: 10) |
| `--path <prefix>` | Filter by path prefix |
| `--name <store>` | Use specific store name |

### Strategy

1. Run `osgrep search --json "<question>"`.
2. Parse JSON results: `[{path, score, start_line, end_line, content}]`
3. Use `Read` only if you need more context from a specific file.
4. Increase `-k` if results seem incomplete.

### Performance

- **SIMD-accelerated**: AVX-512/AVX2/NEON vector operations
- **Native embeddings**: Candle ML (Metal GPU on Apple Silicon)
- **SQLite-Vec**: Lightweight vector storage
- **Hot index**: Sub-50ms searches after first index

## Keywords
semantic search, code search, local search, grep alternative, find code, explore codebase, understand code, search by meaning, SIMD, native, fast
