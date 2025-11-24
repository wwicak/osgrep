/**
 * SIMD-optimized native addon for osgrep
 * Provides extreme performance for vector operations
 *
 * Supports:
 * - AVX-512 (16 floats/cycle) on modern Intel/AMD CPUs
 * - AVX2+FMA (8 floats/cycle) on most x86-64 CPUs
 * - SSE4.1 (4 floats/cycle) fallback
 * - NEON (4 floats/cycle) on ARM64
 * - Parallel batch processing with Rayon
 */

/**
 * Get the detected SIMD capability level
 */
export function getSimdLevel(): string;

/**
 * Compute dot product using AVX-512/AVX2/NEON SIMD intrinsics
 * ~8-16x faster than JavaScript for 384-dim vectors
 */
export function dotProduct(a: Float32Array, b: Float32Array): number;

/**
 * Cosine similarity (for normalized vectors = dot product)
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number;

/**
 * Batch compute dot products against a single query
 * Uses parallel processing for batches >= 32
 */
export function batchDotProduct(
  query: Float32Array,
  vectors: Float32Array[],
): number[];

/**
 * RRF score computation: score = 1/(k + rank)
 */
export function computeRrfScores(ranks: number[], k: number): number[];

/**
 * Fuse two RRF score arrays by summing
 */
export function fuseRrfScores(scoresA: number[], scoresB: number[]): number[];

/**
 * Blend scores: final = weightRerank * rerank + weightRrf * rrf
 */
export function blendScores(
  rerankScores: number[],
  rrfScores: number[],
  weightRerank: number,
  weightRrf: number,
): number[];

/**
 * Normalize scores to [0, 1] range
 */
export function normalizeScores(scores: number[]): number[];

/**
 * Fast sigmoid approximation (~10x faster than exp-based)
 */
export function fastSigmoid(x: number): number;

/**
 * Batch sigmoid computation with parallel processing
 */
export function batchSigmoid(values: number[]): number[];

/**
 * Fast descending argsort with parallel sort for large arrays
 */
export function argsortDesc(values: number[]): number[];

/**
 * L2 normalize a vector in-place using SIMD
 */
export function l2Normalize(vec: Float32Array): Float32Array;

/**
 * Batch L2 normalize with parallel processing
 */
export function batchL2Normalize(vectors: Float32Array[]): Float32Array[];

/**
 * Compute pairwise distance matrix (cosine distance)
 * Uses parallel processing for O(n²) computation
 */
export function computeDistanceMatrix(vectors: Float32Array[]): number[];

// ============================================================================
// Native Embeddings (requires --features embeddings)
// ============================================================================

/**
 * Check if native embeddings are available
 * Returns false if built without --features embeddings
 */
export function isEmbeddingsAvailable(): boolean;

/**
 * Get the embedding backend being used
 * Returns: "candle-metal" | "candle-cpu" | "none"
 */
export function getEmbeddingBackend(): string;

/**
 * Initialize the embedding model (lazy loaded)
 * Downloads model from HuggingFace Hub on first call
 * @param modelId - HuggingFace model ID (default: "sentence-transformers/all-MiniLM-L6-v2")
 */
export function initEmbeddings(modelId?: string): boolean;

/**
 * Embed a single text
 * @param text - Text to embed
 * @returns 384-dimensional embedding vector (for all-MiniLM-L6-v2)
 */
export function embed(text: string): number[];

/**
 * Embed multiple texts in a batch (more efficient)
 * For 8GB RAM: processes in chunks of 32 to avoid memory spikes
 * @param texts - Array of texts to embed
 * @returns Array of embedding vectors
 */
export function embedBatch(texts: string[]): number[][];

/**
 * Get embedding dimension (384 for all-MiniLM-L6-v2)
 */
export function getEmbeddingDim(): number;

/**
 * Compute similarity between two texts
 * @returns Similarity score between -1 and 1
 */
export function textSimilarity(textA: string, textB: string): number;

/**
 * Search for most similar texts from candidates
 * @param query - Query text
 * @param candidates - Array of candidate texts
 * @param topK - Number of results to return (default: 10)
 * @returns Indices of most similar candidates, sorted by similarity
 */
export function searchSimilar(
  query: string,
  candidates: string[],
  topK?: number,
): number[];

// ============================================================================
// SQLite-Vec Vector Store (requires --features sqlite)
// Replaces LanceDB with lightweight SQLite-based vector storage
// ============================================================================

/**
 * Vector record stored in SQLite
 */
export interface VecRecord {
  id: string;
  path: string;
  hash: string;
  content: string;
  startLine: number;
  endLine: number;
  chunkIndex: number;
  isAnchor: boolean;
  contextPrev?: string;
  contextNext?: string;
}

/**
 * Search result with score
 */
export interface VecSearchResult {
  record: VecRecord;
  score: number;
}

/**
 * Check if SQLite-Vec is available
 * Returns false if built without --features sqlite
 */
export function isSqliteAvailable(): boolean;

/**
 * Open or create a vector store
 * @param dbPath - Path to SQLite database file
 * @param storeId - Unique identifier for the store (table prefix)
 */
export function openStore(dbPath: string, storeId: string): boolean;

/**
 * Close a store connection
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 */
export function closeStore(dbPath: string, storeId: string): boolean;

/**
 * Insert a single record with its vector
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @param record - Record to insert
 * @param vector - 384-dimensional embedding vector
 * @returns Generated or provided record ID
 */
export function insertRecord(
  dbPath: string,
  storeId: string,
  record: VecRecord,
  vector: number[],
): string;

/**
 * Insert multiple records in a batch (more efficient)
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @param records - Records to insert
 * @param vectors - Embedding vectors (must match records length)
 * @returns Generated record IDs
 */
export function insertBatch(
  dbPath: string,
  storeId: string,
  records: VecRecord[],
  vectors: number[][],
): string[];

/**
 * Delete all records for a file path
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @param path - File path to delete records for
 * @returns Number of deleted records
 */
export function deleteByPath(
  dbPath: string,
  storeId: string,
  path: string,
): number;

/**
 * Vector similarity search using sqlite-vec
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @param queryVector - 384-dimensional query embedding
 * @param limit - Maximum results to return
 * @param pathPrefix - Optional path prefix filter
 * @returns Search results sorted by similarity
 */
export function vectorSearch(
  dbPath: string,
  storeId: string,
  queryVector: number[],
  limit: number,
  pathPrefix?: string,
): VecSearchResult[];

/**
 * Full-text search using SQLite FTS5
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @param query - Text query
 * @param limit - Maximum results to return
 * @param pathPrefix - Optional path prefix filter
 * @returns Search results sorted by BM25 score
 */
export function ftsSearch(
  dbPath: string,
  storeId: string,
  query: string,
  limit: number,
  pathPrefix?: string,
): VecSearchResult[];

/**
 * List unique file paths in the store
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 * @returns Array of file paths
 */
export function listFiles(dbPath: string, storeId: string): string[];

/**
 * Get total record count
 * @param dbPath - Path to SQLite database file
 * @param storeId - Store identifier
 */
export function countRecords(dbPath: string, storeId: string): number;
