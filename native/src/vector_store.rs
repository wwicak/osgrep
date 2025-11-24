//! SQLite-Vec based vector store (replaces LanceDB)
//!
//! Lightweight vector storage with:
//! - SQLite FTS5 for full-text search
//! - sqlite-vec for vector similarity search
//! - Memory-efficient for 8GB RAM devices
//!
//! Single-file database, no external dependencies.

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[cfg(feature = "sqlite")]
use {
    rusqlite::{params, Connection},
    sqlite_vec::sqlite3_vec_init,
    std::collections::HashMap,
    std::path::PathBuf,
    std::sync::Mutex,
    uuid::Uuid,
};

#[cfg(feature = "sqlite")]
use std::sync::OnceLock;

// Global connection pool
#[cfg(feature = "sqlite")]
static CONNECTIONS: OnceLock<Mutex<HashMap<String, Connection>>> = OnceLock::new();

#[cfg(feature = "sqlite")]
fn get_connections() -> &'static Mutex<HashMap<String, Connection>> {
    CONNECTIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Vector record stored in SQLite
#[napi(object)]
#[derive(Clone)]
pub struct VecRecord {
    pub id: String,
    pub path: String,
    pub hash: String,
    pub content: String,
    pub start_line: i32,
    pub end_line: i32,
    pub chunk_index: i32,
    pub is_anchor: bool,
    pub context_prev: Option<String>,
    pub context_next: Option<String>,
}

/// Search result with score
#[napi(object)]
#[derive(Clone)]
pub struct VecSearchResult {
    pub record: VecRecord,
    pub score: f64,
}

// ============================================================================
// Store Management
// ============================================================================

/// Check if SQLite-Vec is available
#[napi]
pub fn is_sqlite_available() -> bool {
    #[cfg(feature = "sqlite")]
    {
        true
    }
    #[cfg(not(feature = "sqlite"))]
    {
        false
    }
}

/// Open or create a vector store
/// Returns the store ID for subsequent operations
#[napi]
pub fn open_store(db_path: String, store_id: String) -> Result<bool> {
    #[cfg(feature = "sqlite")]
    {
        let path = PathBuf::from(&db_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::from_reason(format!("Failed to create directory: {}", e)))?;
        }

        let conn = Connection::open(&path)
            .map_err(|e| Error::from_reason(format!("Failed to open database: {}", e)))?;

        // Load sqlite-vec extension (global initialization)
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
                    hash TEXT,
                    content TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    is_anchor INTEGER NOT NULL DEFAULT 0,
                    context_prev TEXT,
                    context_next TEXT
                )
                "#
            ),
            [],
        )
        .map_err(|e| Error::from_reason(format!("Failed to create table: {}", e)))?;

        // Create vector table using sqlite-vec
        conn.execute(
            &format!(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS "{store_id}_vec" USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[768]
                )
                "#
            ),
            [],
        )
        .map_err(|e| Error::from_reason(format!("Failed to create vector table: {}", e)))?;

        // Create FTS5 table for full-text search
        conn.execute(
            &format!(
                r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS "{store_id}_fts" USING fts5(
                    id,
                    content,
                    path,
                    content='{store_id}',
                    content_rowid='rowid'
                )
                "#
            ),
            [],
        )
        .map_err(|e| Error::from_reason(format!("Failed to create FTS table: {}", e)))?;

        // Create indexes
        conn.execute(
            &format!(r#"CREATE INDEX IF NOT EXISTS "{store_id}_path_idx" ON "{store_id}"(path)"#),
            [],
        )
        .ok();

        // Store connection
        let mut connections = get_connections().lock().unwrap();
        connections.insert(format!("{}:{}", db_path, store_id), conn);

        Ok(true)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// Close a store connection
#[napi]
pub fn close_store(db_path: String, store_id: String) -> Result<bool> {
    #[cfg(feature = "sqlite")]
    {
        let mut connections = get_connections().lock().unwrap();
        connections.remove(&format!("{}:{}", db_path, store_id));
        Ok(true)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

// ============================================================================
// CRUD Operations
// ============================================================================

/// Insert a record with its vector
#[napi]
pub fn insert_record(
    db_path: String,
    store_id: String,
    record: VecRecord,
    vector: Vec<f64>,
) -> Result<String> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        let id = if record.id.is_empty() { Uuid::new_v4().to_string() } else { record.id.clone() };

        // Insert main record
        conn.execute(
            &format!(
                r#"
                INSERT OR REPLACE INTO "{store_id}"
                (id, path, hash, content, start_line, end_line, chunk_index, is_anchor, context_prev, context_next)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#
            ),
            params![
                &id,
                &record.path,
                &record.hash,
                &record.content,
                record.start_line,
                record.end_line,
                record.chunk_index,
                record.is_anchor as i32,
                &record.context_prev,
                &record.context_next,
            ],
        )
        .map_err(|e| Error::from_reason(format!("Failed to insert record: {}", e)))?;

        // Insert vector
        let vector_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let vector_blob = bytemuck_cast_slice(&vector_f32);

        conn.execute(
            &format!(r#"INSERT OR REPLACE INTO "{store_id}_vec" (id, embedding) VALUES (?1, ?2)"#),
            params![&id, vector_blob],
        )
        .map_err(|e| Error::from_reason(format!("Failed to insert vector: {}", e)))?;

        // Update FTS index
        conn.execute(
            &format!(r#"INSERT INTO "{store_id}_fts" (id, content, path) VALUES (?1, ?2, ?3)"#),
            params![&id, &record.content, &record.path],
        )
        .ok(); // FTS insert may fail if already exists

        Ok(id)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// Insert multiple records in a batch (more efficient)
#[napi]
pub fn insert_batch(
    db_path: String,
    store_id: String,
    records: Vec<VecRecord>,
    vectors: Vec<Vec<f64>>,
) -> Result<Vec<String>> {
    #[cfg(feature = "sqlite")]
    {
        if records.len() != vectors.len() {
            return Err(Error::from_reason("Records and vectors must have same length"));
        }

        let mut connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn =
            connections.get_mut(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        let tx = conn
            .transaction()
            .map_err(|e| Error::from_reason(format!("Failed to start transaction: {}", e)))?;

        let mut ids = Vec::with_capacity(records.len());

        for (record, vector) in records.iter().zip(vectors.iter()) {
            let id =
                if record.id.is_empty() { Uuid::new_v4().to_string() } else { record.id.clone() };

            tx.execute(
                &format!(
                    r#"
                    INSERT OR REPLACE INTO "{store_id}"
                    (id, path, hash, content, start_line, end_line, chunk_index, is_anchor, context_prev, context_next)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                    "#
                ),
                params![
                    &id,
                    &record.path,
                    &record.hash,
                    &record.content,
                    record.start_line,
                    record.end_line,
                    record.chunk_index,
                    record.is_anchor as i32,
                    &record.context_prev,
                    &record.context_next,
                ],
            )
            .map_err(|e| Error::from_reason(format!("Failed to insert record: {}", e)))?;

            let vector_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
            let vector_blob = bytemuck_cast_slice(&vector_f32);

            tx.execute(
                &format!(
                    r#"INSERT OR REPLACE INTO "{store_id}_vec" (id, embedding) VALUES (?1, ?2)"#
                ),
                params![&id, vector_blob],
            )
            .map_err(|e| Error::from_reason(format!("Failed to insert vector: {}", e)))?;

            tx.execute(
                &format!(
                    r#"INSERT OR REPLACE INTO "{store_id}_fts" (id, content, path) VALUES (?1, ?2, ?3)"#
                ),
                params![&id, &record.content, &record.path],
            )
            .ok();

            ids.push(id);
        }

        tx.commit().map_err(|e| Error::from_reason(format!("Failed to commit: {}", e)))?;

        Ok(ids)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// Delete records by path
#[napi]
pub fn delete_by_path(db_path: String, store_id: String, path: String) -> Result<u32> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        // Get IDs to delete
        let mut stmt = conn
            .prepare(&format!(r#"SELECT id FROM "{store_id}" WHERE path = ?1"#))
            .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

        let ids: Vec<String> = stmt
            .query_map([&path], |row| row.get(0))
            .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        // Delete from all tables
        for id in &ids {
            conn.execute(&format!(r#"DELETE FROM "{store_id}_vec" WHERE id = ?1"#), [id]).ok();
            conn.execute(&format!(r#"DELETE FROM "{store_id}_fts" WHERE id = ?1"#), [id]).ok();
        }

        let deleted = conn
            .execute(&format!(r#"DELETE FROM "{store_id}" WHERE path = ?1"#), [&path])
            .map_err(|e| Error::from_reason(format!("Delete error: {}", e)))?;

        Ok(deleted as u32)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

// ============================================================================
// Search Operations
// ============================================================================

#[cfg(feature = "sqlite")]
fn parse_result_row(row: &rusqlite::Row) -> rusqlite::Result<VecSearchResult> {
    Ok(VecSearchResult {
        score: 1.0 - row.get::<_, f64>(1)?, // Convert distance to similarity
        record: VecRecord {
            id: row.get(2)?,
            path: row.get(3)?,
            hash: row.get(4)?,
            content: row.get(5)?,
            start_line: row.get(6)?,
            end_line: row.get(7)?,
            chunk_index: row.get(8)?,
            is_anchor: row.get::<_, i32>(9)? != 0,
            context_prev: row.get(10)?,
            context_next: row.get(11)?,
        },
    })
}

#[cfg(feature = "sqlite")]
fn parse_fts_result_row(row: &rusqlite::Row) -> rusqlite::Result<VecSearchResult> {
    let bm25_score: f64 = row.get(1)?;
    Ok(VecSearchResult {
        score: -bm25_score, // BM25 is negative, lower is better
        record: VecRecord {
            id: row.get(2)?,
            path: row.get(3)?,
            hash: row.get(4)?,
            content: row.get(5)?,
            start_line: row.get(6)?,
            end_line: row.get(7)?,
            chunk_index: row.get(8)?,
            is_anchor: row.get::<_, i32>(9)? != 0,
            context_prev: row.get(10)?,
            context_next: row.get(11)?,
        },
    })
}

/// Vector similarity search
#[napi]
pub fn vector_search(
    db_path: String,
    store_id: String,
    query_vector: Vec<f64>,
    limit: u32,
    path_prefix: Option<String>,
) -> Result<Vec<VecSearchResult>> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        let query_f32: Vec<f32> = query_vector.iter().map(|&v| v as f32).collect();
        let query_blob = bytemuck_cast_slice(&query_f32);

        let results: Vec<VecSearchResult> = if let Some(prefix) = &path_prefix {
            let sql = format!(
                r#"
                SELECT v.id, v.distance, c.*
                FROM "{store_id}_vec" v
                JOIN "{store_id}" c ON v.id = c.id
                WHERE v.embedding MATCH ?1
                AND k = ?2
                AND c.path LIKE ?3
                ORDER BY v.distance
                "#
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

            let rows = stmt
                .query_map(params![query_blob, limit, format!("{}%", prefix)], parse_result_row)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        } else {
            let sql = format!(
                r#"
                SELECT v.id, v.distance, c.*
                FROM "{store_id}_vec" v
                JOIN "{store_id}" c ON v.id = c.id
                WHERE v.embedding MATCH ?1
                AND k = ?2
                ORDER BY v.distance
                "#
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

            let rows = stmt
                .query_map(params![query_blob, limit], parse_result_row)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        Ok(results)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// Full-text search using FTS5
#[napi]
pub fn fts_search(
    db_path: String,
    store_id: String,
    query: String,
    limit: u32,
    path_prefix: Option<String>,
) -> Result<Vec<VecSearchResult>> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        // Escape FTS query
        let fts_query = query.replace('"', "\"\"");

        let results: Vec<VecSearchResult> = if let Some(prefix) = &path_prefix {
            let sql = format!(
                r#"
                SELECT f.id, bm25("{store_id}_fts") as score, c.*
                FROM "{store_id}_fts" f
                JOIN "{store_id}" c ON f.id = c.id
                WHERE "{store_id}_fts" MATCH ?1
                AND c.path LIKE ?3
                ORDER BY score
                LIMIT ?2
                "#
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

            let rows = stmt
                .query_map(params![&fts_query, limit, format!("{}%", prefix)], parse_fts_result_row)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        } else {
            let sql = format!(
                r#"
                SELECT f.id, bm25("{store_id}_fts") as score, c.*
                FROM "{store_id}_fts" f
                JOIN "{store_id}" c ON f.id = c.id
                WHERE "{store_id}_fts" MATCH ?1
                ORDER BY score
                LIMIT ?2
                "#
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

            let rows = stmt
                .query_map(params![&fts_query, limit], parse_fts_result_row)
                .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        Ok(results)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// List unique file paths in the store
#[napi]
pub fn list_files(db_path: String, store_id: String) -> Result<Vec<String>> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        let mut stmt = conn
            .prepare(&format!(r#"SELECT DISTINCT path FROM "{store_id}""#))
            .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(paths)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

/// Get record count
#[napi]
pub fn count_records(db_path: String, store_id: String) -> Result<u32> {
    #[cfg(feature = "sqlite")]
    {
        let connections = get_connections().lock().unwrap();
        let key = format!("{}:{}", db_path, store_id);
        let conn = connections.get(&key).ok_or_else(|| Error::from_reason("Store not opened"))?;

        let count: i64 = conn
            .query_row(&format!(r#"SELECT COUNT(*) FROM "{store_id}""#), [], |row| row.get(0))
            .map_err(|e| Error::from_reason(format!("Query error: {}", e)))?;

        Ok(count as u32)
    }
    #[cfg(not(feature = "sqlite"))]
    {
        Err(Error::from_reason("SQLite feature not enabled"))
    }
}

// Helper function to cast f32 slice to bytes
#[cfg(feature = "sqlite")]
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}
