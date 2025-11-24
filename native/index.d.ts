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
