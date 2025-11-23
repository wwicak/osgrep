/**
 * SIMD-optimized native addon for osgrep
 * Provides extreme performance for vector operations
 */

/**
 * Compute dot product using AVX2/NEON SIMD intrinsics
 * ~8x faster than JavaScript for 384-dim vectors
 */
export function dotProduct(a: Float32Array, b: Float32Array): number;

/**
 * Cosine similarity (for normalized vectors = dot product)
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number;

/**
 * Batch compute dot products against a single query
 */
export function batchDotProduct(query: Float32Array, vectors: Float32Array[]): number[];

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
 * Batch sigmoid computation
 */
export function batchSigmoid(values: number[]): number[];

/**
 * Fast descending argsort
 */
export function argsortDesc(values: number[]): number[];

/**
 * L2 normalize a vector in-place
 */
export function l2Normalize(vec: Float32Array): Float32Array;
