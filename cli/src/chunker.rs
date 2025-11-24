//! Tree-sitter based code chunking
//!
//! Chunks code files into semantic units (functions, classes, etc.)

#![allow(clippy::only_used_in_recursion)]

use anyhow::{Context, Result};
use std::path::Path;
use tree_sitter::{Language, Parser};

pub struct Chunk {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub is_anchor: bool,
}

const MAX_CHUNK_LINES: usize = 50;
const MIN_CHUNK_LINES: usize = 5;

/// Chunk a file based on its language
pub fn chunk(path: &Path, content: &str) -> Result<Vec<Chunk>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let language = get_language(ext);

    if let Some(lang) = language {
        chunk_with_treesitter(content, lang)
    } else {
        // Fallback to line-based chunking
        chunk_by_lines(content)
    }
}

fn get_language(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "js" | "jsx" | "mjs" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" | "hpp" | "cc" | "cxx" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        _ => None,
    }
}

fn chunk_with_treesitter(content: &str, language: Language) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .context("Failed to set language")?;

    let tree = parser.parse(content, None).context("Failed to parse")?;
    let root = tree.root_node();

    let mut chunks = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    // Add anchor chunk (file summary)
    if !content.is_empty() {
        let anchor_text = create_anchor(content, &lines);
        chunks.push(Chunk {
            text: anchor_text,
            start_line: 1,
            end_line: lines.len().min(10),
            is_anchor: true,
        });
    }

    // Extract semantic chunks
    extract_chunks(&root, content, &lines, &mut chunks);

    // If no semantic chunks found, fall back to line-based
    if chunks.len() <= 1 {
        chunks.extend(chunk_by_lines(content)?);
    }

    Ok(chunks)
}

fn extract_chunks(
    node: &tree_sitter::Node,
    content: &str,
    lines: &[&str],
    chunks: &mut Vec<Chunk>,
) {
    let kind = node.kind();

    // Check if this is a significant node (function, class, etc.)
    let is_significant = matches!(
        kind,
        "function_definition"
            | "function_item"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "class_definition"
            | "class_declaration"
            | "struct_item"
            | "impl_item"
            | "interface_declaration"
            | "trait_item"
            | "module"
            | "namespace_definition"
    );

    if is_significant {
        let start_line = node.start_position().row;
        let end_line = node.end_position().row;
        let num_lines = end_line - start_line + 1;

        if (MIN_CHUNK_LINES..=MAX_CHUNK_LINES).contains(&num_lines) {
            let text: String = lines[start_line..=end_line.min(lines.len() - 1)].join("\n");

            chunks.push(Chunk {
                text,
                start_line: start_line + 1, // 1-indexed
                end_line: end_line + 1,
                is_anchor: false,
            });
            return; // Don't recurse into children
        }
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_chunks(&child, content, lines, chunks);
    }
}

fn create_anchor(_content: &str, lines: &[&str]) -> String {
    // Create a summary of the file structure
    let mut anchor = String::new();

    // Add imports/includes (first 20 lines or until first non-import)
    let import_lines: Vec<&str> = lines
        .iter()
        .take(20)
        .filter(|l| {
            let l = l.trim();
            l.starts_with("import")
                || l.starts_with("from")
                || l.starts_with("use ")
                || l.starts_with("#include")
                || l.starts_with("require")
                || l.starts_with("package ")
        })
        .copied()
        .collect();

    if !import_lines.is_empty() {
        anchor.push_str(&import_lines.join("\n"));
        anchor.push_str("\n...\n");
    }

    // Add function/class signatures
    let signatures: Vec<&str> = lines
        .iter()
        .filter(|l| {
            let l = l.trim();
            (l.starts_with("fn ")
                || l.starts_with("pub fn ")
                || l.starts_with("def ")
                || l.starts_with("async def ")
                || l.starts_with("function ")
                || l.starts_with("async function ")
                || l.starts_with("class ")
                || l.starts_with("struct ")
                || l.starts_with("interface ")
                || l.starts_with("trait ")
                || l.starts_with("impl ")
                || l.starts_with("type "))
                && l.len() < 200
        })
        .take(20)
        .copied()
        .collect();

    if !signatures.is_empty() {
        anchor.push_str(&signatures.join("\n"));
    }

    if anchor.is_empty() {
        // Use first 10 lines as fallback
        anchor = lines
            .iter()
            .take(10)
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
    }

    anchor
}

fn chunk_by_lines(content: &str) -> Result<Vec<Chunk>> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let chunk_size = 30;
    let overlap = 5;

    let mut i = 0;
    while i < lines.len() {
        let end = (i + chunk_size).min(lines.len());
        let text = lines[i..end].join("\n");

        chunks.push(Chunk {
            text,
            start_line: i + 1,
            end_line: end,
            is_anchor: false,
        });

        if end >= lines.len() {
            break;
        }
        i += chunk_size - overlap;
    }

    Ok(chunks)
}
