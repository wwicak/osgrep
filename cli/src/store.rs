//! SQLite-Vec vector storage
//!
//! Lightweight vector storage with FTS5 full-text search

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "sqlite")]
use {
    rusqlite::{params, Connection},
    sqlite_vec::sqlite3_vec_init,
};

#[cfg(feature = "sqlite")]
static CONNECTIONS: OnceLock<Mutex<HashMap<String, Connection>>> = OnceLock::new();

#[cfg(feature = "sqlite")]
fn get_connections() -> &'static Mutex<HashMap<String, Connection>> {
    CONNECTIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct Record {
    pub id: String,
    pub path: String,
    pub content: String,
    pub start_line: i32,
    pub end_line: i32,
    pub chunk_index: i32,
    pub is_anchor: bool,
}

pub struct SearchResult {
    pub path: String,
    pub content: String,
    pub start_line: i32,
    pub end_line: i32,
    pub score: f32,
}

/// Open or create a vector store
#[cfg(feature = "sqlite")]
pub fn open(db_path: &PathBuf, store_id: &str) -> Result<()> {
    let conn = Connection::open(db_path).context("Failed to open database")?;

    // Load sqlite-vec extension
    unsafe {
        sqlite3_vec_init();
    }

    // Create chunks table
    conn.execute(
        &format!(
            r#"
            CREATE TABLE IF NOT EXISTS "{store_id}" (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                is_anchor INTEGER NOT NULL DEFAULT 0
            )
            "#
        ),
        [],
    )?;

    // Create vector table
    conn.execute(
        &format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS "{store_id}_vec" USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[384]
            )
            "#
        ),
        [],
    )?;

    // Create FTS5 table
    conn.execute(
        &format!(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS "{store_id}_fts" USING fts5(
                id, content, path
            )
            "#
        ),
        [],
    )?;

    // Create index
    conn.execute(
        &format!(r#"CREATE INDEX IF NOT EXISTS "{store_id}_path_idx" ON "{store_id}"(path)"#),
        [],
    ).ok();

    let mut connections = get_connections().lock().unwrap();
    connections.insert(format!("{}:{}", db_path.display(), store_id), conn);

    Ok(())
}

#[cfg(not(feature = "sqlite"))]
pub fn open(_db_path: &PathBuf, _store_id: &str) -> Result<()> {
    anyhow::bail!("SQLite feature not enabled")
}

/// Insert batch of records
#[cfg(feature = "sqlite")]
pub fn insert_batch(db_path: &PathBuf, store_id: &str, records: &[Record], vectors: &[Vec<f32>]) -> Result<()> {
    let mut connections = get_connections().lock().unwrap();
    let key = format!("{}:{}", db_path.display(), store_id);
    let conn = connections.get_mut(&key).context("Store not opened")?;

    let tx = conn.transaction()?;

    for (record, vector) in records.iter().zip(vectors.iter()) {
        tx.execute(
            &format!(
                r#"INSERT OR REPLACE INTO "{store_id}"
                (id, path, content, start_line, end_line, chunk_index, is_anchor)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#
            ),
            params![
                &record.id,
                &record.path,
                &record.content,
                record.start_line,
                record.end_line,
                record.chunk_index,
                record.is_anchor as i32,
            ],
        )?;

        let vector_blob = bytemuck_cast_slice(vector);
        tx.execute(
            &format!(r#"INSERT OR REPLACE INTO "{store_id}_vec" (id, embedding) VALUES (?1, ?2)"#),
            params![&record.id, vector_blob],
        )?;

        tx.execute(
            &format!(r#"INSERT OR REPLACE INTO "{store_id}_fts" (id, content, path) VALUES (?1, ?2, ?3)"#),
            params![&record.id, &record.content, &record.path],
        ).ok();
    }

    tx.commit()?;
    Ok(())
}

#[cfg(not(feature = "sqlite"))]
pub fn insert_batch(_db_path: &PathBuf, _store_id: &str, _records: &[Record], _vectors: &[Vec<f32>]) -> Result<()> {
    anyhow::bail!("SQLite feature not enabled")
}

/// Delete records by path
#[cfg(feature = "sqlite")]
pub fn delete_by_path(db_path: &PathBuf, store_id: &str, path: &str) -> Result<()> {
    let connections = get_connections().lock().unwrap();
    let key = format!("{}:{}", db_path.display(), store_id);
    let conn = connections.get(&key).context("Store not opened")?;

    // Get IDs to delete
    let mut stmt = conn.prepare(&format!(r#"SELECT id FROM "{store_id}" WHERE path = ?1"#))?;
    let ids: Vec<String> = stmt.query_map([path], |row| row.get(0))?.filter_map(|r| r.ok()).collect();

    for id in &ids {
        conn.execute(&format!(r#"DELETE FROM "{store_id}_vec" WHERE id = ?1"#), [id]).ok();
        conn.execute(&format!(r#"DELETE FROM "{store_id}_fts" WHERE id = ?1"#), [id]).ok();
    }

    conn.execute(&format!(r#"DELETE FROM "{store_id}" WHERE path = ?1"#), [path])?;
    Ok(())
}

#[cfg(not(feature = "sqlite"))]
pub fn delete_by_path(_db_path: &PathBuf, _store_id: &str, _path: &str) -> Result<()> {
    anyhow::bail!("SQLite feature not enabled")
}

/// Vector search
#[cfg(feature = "sqlite")]
pub fn search(db_path: &PathBuf, store_id: &str, query_vec: &[f32], limit: usize, path_prefix: Option<&str>) -> Result<Vec<SearchResult>> {
    let connections = get_connections().lock().unwrap();
    let key = format!("{}:{}", db_path.display(), store_id);
    let conn = connections.get(&key).context("Store not opened")?;

    let query_blob = bytemuck_cast_slice(query_vec);

    let sql = if path_prefix.is_some() {
        format!(
            r#"
            SELECT v.id, v.distance, c.path, c.content, c.start_line, c.end_line
            FROM "{store_id}_vec" v
            JOIN "{store_id}" c ON v.id = c.id
            WHERE v.embedding MATCH ?1
            AND k = ?2
            AND c.path LIKE ?3
            ORDER BY v.distance
            "#
        )
    } else {
        format!(
            r#"
            SELECT v.id, v.distance, c.path, c.content, c.start_line, c.end_line
            FROM "{store_id}_vec" v
            JOIN "{store_id}" c ON v.id = c.id
            WHERE v.embedding MATCH ?1
            AND k = ?2
            ORDER BY v.distance
            "#
        )
    };

    let mut stmt = conn.prepare(&sql)?;

    let results: Vec<SearchResult> = if let Some(prefix) = path_prefix {
        let rows = stmt.query_map(params![query_blob, limit, format!("{}%", prefix)], |row| {
            Ok(SearchResult {
                path: row.get(2)?,
                content: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
                score: 1.0 - row.get::<_, f32>(1)?,
            })
        })?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        let rows = stmt.query_map(params![query_blob, limit], |row| {
            Ok(SearchResult {
                path: row.get(2)?,
                content: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
                score: 1.0 - row.get::<_, f32>(1)?,
            })
        })?;
        rows.filter_map(|r| r.ok()).collect()
    };

    Ok(results)
}

#[cfg(not(feature = "sqlite"))]
pub fn search(_db_path: &PathBuf, _store_id: &str, _query_vec: &[f32], _limit: usize, _path_prefix: Option<&str>) -> Result<Vec<SearchResult>> {
    anyhow::bail!("SQLite feature not enabled")
}

/// Get record count
#[cfg(feature = "sqlite")]
pub fn count(db_path: &PathBuf, store_id: &str) -> Result<u32> {
    let connections = get_connections().lock().unwrap();
    let key = format!("{}:{}", db_path.display(), store_id);
    let conn = connections.get(&key).context("Store not opened")?;

    let count: i64 = conn.query_row(&format!(r#"SELECT COUNT(*) FROM "{store_id}""#), [], |row| row.get(0))?;
    Ok(count as u32)
}

#[cfg(not(feature = "sqlite"))]
pub fn count(_db_path: &PathBuf, _store_id: &str) -> Result<u32> {
    anyhow::bail!("SQLite feature not enabled")
}

#[cfg(feature = "sqlite")]
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
    }
}
