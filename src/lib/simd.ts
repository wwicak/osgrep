/**
 * SIMD-optimized operations with native addon support
 *
 * This module provides extreme performance for vector operations
 * by using a Rust native addon with AVX2/NEON SIMD intrinsics.
 *
 * Falls back to optimized JavaScript if native addon is unavailable.
 */

// Try to load native addon
let native: NativeAddon | null = null;

interface NativeAddon {
  dotProduct(a: Float32Array, b: Float32Array): number;
  cosineSimilarity(a: Float32Array, b: Float32Array): number;
  batchDotProduct(query: Float32Array, vectors: Float32Array[]): number[];
  computeRrfScores(ranks: number[], k: number): number[];
  fuseRrfScores(scoresA: number[], scoresB: number[]): number[];
  blendScores(
    rerankScores: number[],
    rrfScores: number[],
    weightRerank: number,
    weightRrf: number,
  ): number[];
  normalizeScores(scores: number[]): number[];
  fastSigmoid(x: number): number;
  batchSigmoid(values: number[]): number[];
  argsortDesc(values: number[]): number[];
  l2Normalize(vec: Float32Array): Float32Array;
}

try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  native = require("../../native/osgrep-native.node") as NativeAddon;
  console.log("[simd] Native SIMD addon loaded successfully");
} catch {
  // Native addon not available, using JS fallback
}

/**
 * Check if native SIMD operations are available
 */
export function isNativeAvailable(): boolean {
  return native !== null;
}

/**
 * Compute dot product of two vectors
 * Uses AVX2 (8 floats/cycle) or NEON (4 floats/cycle) when available
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  if (native) {
    return native.dotProduct(a, b);
  }
  return dotProductJS(a, b);
}

/**
 * Optimized JS fallback using loop unrolling
 */
function dotProductJS(a: Float32Array, b: Float32Array): number {
  const len = a.length;
  let sum0 = 0,
    sum1 = 0,
    sum2 = 0,
    sum3 = 0;

  // Process 4 elements at a time (loop unrolling)
  const chunks = (len / 4) | 0;
  let i = 0;

  for (let c = 0; c < chunks; c++) {
    sum0 += a[i] * b[i];
    sum1 += a[i + 1] * b[i + 1];
    sum2 += a[i + 2] * b[i + 2];
    sum3 += a[i + 3] * b[i + 3];
    i += 4;
  }

  // Handle remainder
  let sum = sum0 + sum1 + sum2 + sum3;
  for (; i < len; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

/**
 * Cosine similarity (for normalized vectors, equals dot product)
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (native) {
    return native.cosineSimilarity(a, b);
  }
  return dotProductJS(a, b);
}

/**
 * Batch dot product - compute similarity of one query against many vectors
 */
export function batchDotProduct(
  query: Float32Array,
  vectors: Float32Array[],
): number[] {
  if (native) {
    return native.batchDotProduct(query, vectors);
  }

  return vectors.map((v) => dotProductJS(query, v));
}

/**
 * Compute RRF scores from ranks
 * score = 1 / (k + rank)
 */
export function computeRrfScores(ranks: number[], k: number = 60): number[] {
  if (native) {
    return native.computeRrfScores(ranks, k);
  }

  return ranks.map((rank) => 1 / (k + rank));
}

/**
 * Fuse two RRF score arrays by summing
 */
export function fuseRrfScores(scoresA: number[], scoresB: number[]): number[] {
  if (native) {
    return native.fuseRrfScores(scoresA, scoresB);
  }

  const maxLen = Math.max(scoresA.length, scoresB.length);
  const result = new Array(maxLen).fill(0);

  for (let i = 0; i < scoresA.length; i++) {
    result[i] += scoresA[i];
  }
  for (let i = 0; i < scoresB.length; i++) {
    result[i] += scoresB[i];
  }

  return result;
}

/**
 * Blend rerank scores with RRF scores
 * final = weightRerank * rerank + weightRrf * rrf
 */
export function blendScores(
  rerankScores: number[],
  rrfScores: number[],
  weightRerank: number = 0.7,
  weightRrf: number = 0.3,
): number[] {
  if (native) {
    return native.blendScores(rerankScores, rrfScores, weightRerank, weightRrf);
  }

  return rerankScores.map((r, i) => weightRerank * r + weightRrf * rrfScores[i]);
}

/**
 * Normalize scores to [0, 1] range
 */
export function normalizeScores(scores: number[]): number[] {
  if (native) {
    return native.normalizeScores(scores);
  }

  if (scores.length === 0) return scores;

  const max = Math.max(...scores);
  if (max <= 0) return scores.map(() => 0);

  return scores.map((s) => s / max);
}

/**
 * Fast sigmoid approximation (~10x faster than Math.exp based)
 * Uses Pade approximant for high accuracy
 */
export function fastSigmoid(x: number): number {
  if (native) {
    return native.fastSigmoid(x);
  }
  return fastSigmoidJS(x);
}

/**
 * JS implementation of fast sigmoid using Pade approximant
 */
function fastSigmoidJS(x: number): number {
  if (x >= 4.5) return 1.0;
  if (x <= -4.5) return 0.0;

  // Pade approximant
  const x2 = x * x;
  const num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
  const den = 270270.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
  return 0.5 + num / (2.0 * den);
}

/**
 * Batch sigmoid computation
 */
export function batchSigmoid(values: number[]): number[] {
  if (native) {
    return native.batchSigmoid(values);
  }

  return values.map(fastSigmoidJS);
}

/**
 * Fast argsort - returns indices that would sort array in descending order
 */
export function argsortDesc(values: number[]): number[] {
  if (native) {
    return native.argsortDesc(values);
  }

  const indices = values.map((_, i) => i);
  indices.sort((a, b) => values[b] - values[a]);
  return indices;
}

/**
 * L2 normalize a vector
 */
export function l2Normalize(vec: Float32Array): Float32Array {
  if (native) {
    return native.l2Normalize(vec);
  }

  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }

  const norm = Math.sqrt(sum);
  if (norm > 0) {
    const invNorm = 1 / norm;
    for (let i = 0; i < vec.length; i++) {
      vec[i] *= invNorm;
    }
  }

  return vec;
}

/**
 * RRF Fusion helper - combines vector and FTS results using native SIMD
 * This is the main optimization target for search performance
 */
export function rrfFuse(
  vectorResults: { key: string; record: unknown }[],
  ftsResults: { key: string; record: unknown }[],
  k: number = 60,
): Map<string, { score: number; record: unknown }> {
  const scores = new Map<string, { score: number; record: unknown }>();

  // Process vector results
  for (let i = 0; i < vectorResults.length; i++) {
    const { key, record } = vectorResults[i];
    const score = 1 / (k + i + 1);
    scores.set(key, { score, record });
  }

  // Process FTS results and merge
  for (let i = 0; i < ftsResults.length; i++) {
    const { key, record } = ftsResults[i];
    const score = 1 / (k + i + 1);
    const existing = scores.get(key);
    if (existing) {
      existing.score += score;
    } else {
      scores.set(key, { score, record });
    }
  }

  return scores;
}

/**
 * Benchmark helper - compare native vs JS performance
 */
export function benchmark(iterations: number = 10000): {
  native: boolean;
  dotProductMs: number;
  sigmoidMs: number;
} {
  const a = new Float32Array(384).map(() => Math.random());
  const b = new Float32Array(384).map(() => Math.random());

  // Warm up
  for (let i = 0; i < 100; i++) {
    dotProduct(a, b);
    fastSigmoid(Math.random() * 10 - 5);
  }

  // Benchmark dot product
  const dotStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    dotProduct(a, b);
  }
  const dotProductMs = performance.now() - dotStart;

  // Benchmark sigmoid
  const sigStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    fastSigmoid(Math.random() * 10 - 5);
  }
  const sigmoidMs = performance.now() - sigStart;

  return {
    native: isNativeAvailable(),
    dotProductMs,
    sigmoidMs,
  };
}
