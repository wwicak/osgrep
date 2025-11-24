/**
 * Native embeddings using Candle ML framework
 *
 * Memory-optimized for M2 MacBook with 8GB RAM:
 * - Uses Metal GPU acceleration on Apple Silicon
 * - Processes batches in chunks of 32 to avoid memory spikes
 * - Lazy model loading - only loads when first embedding is requested
 *
 * Falls back to transformers.js if native addon is not available
 */

// Try to load native addon
let native: NativeEmbeddings | null = null;

interface NativeEmbeddings {
  isEmbeddingsAvailable(): boolean;
  getEmbeddingBackend(): string;
  initEmbeddings(modelId?: string): boolean;
  embed(text: string): number[];
  embedBatch(texts: string[]): number[][];
  getEmbeddingDim(): number;
  textSimilarity(textA: string, textB: string): number;
  searchSimilar(query: string, candidates: string[], topK?: number): number[];
}

try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  native = require("../../native/index.node") as NativeEmbeddings;
} catch {
  // Native addon not available - will use fallback
}

/**
 * Check if native embeddings are available
 */
export function isNativeEmbeddingsAvailable(): boolean {
  try {
    return native?.isEmbeddingsAvailable() ?? false;
  } catch {
    return false;
  }
}

/**
 * Get the embedding backend being used
 * Returns: "candle-metal" | "candle-cpu" | "transformers.js" | "none"
 */
export function getEmbeddingBackend(): string {
  if (native) {
    try {
      return native.getEmbeddingBackend();
    } catch {
      return "none";
    }
  }
  return "transformers.js";
}

/**
 * Initialize the embedding model
 * @param modelId - HuggingFace model ID (default: "sentence-transformers/all-MiniLM-L6-v2")
 */
export async function initEmbeddings(modelId?: string): Promise<boolean> {
  if (!native || !isNativeEmbeddingsAvailable()) {
    // Fall back to transformers.js initialization
    return false;
  }

  try {
    return native.initEmbeddings(modelId);
  } catch (e) {
    console.error("Failed to initialize native embeddings:", e);
    return false;
  }
}

/**
 * Embed a single text using native candle
 * @param text - Text to embed
 * @returns 384-dimensional embedding vector (for all-MiniLM-L6-v2)
 */
export function embed(text: string): Float32Array {
  if (!native || !isNativeEmbeddingsAvailable()) {
    throw new Error("Native embeddings not available. Use transformers.js fallback.");
  }

  const result = native.embed(text);
  return new Float32Array(result);
}

/**
 * Embed multiple texts in a batch (more efficient)
 * For 8GB RAM: automatically processes in chunks of 32
 * @param texts - Array of texts to embed
 * @returns Array of 384-dimensional embedding vectors
 */
export function embedBatch(texts: string[]): Float32Array[] {
  if (!native || !isNativeEmbeddingsAvailable()) {
    throw new Error("Native embeddings not available. Use transformers.js fallback.");
  }

  const results = native.embedBatch(texts);
  return results.map((r) => new Float32Array(r));
}

/**
 * Get embedding dimension (384 for all-MiniLM-L6-v2)
 */
export function getEmbeddingDim(): number {
  if (native && isNativeEmbeddingsAvailable()) {
    return native.getEmbeddingDim();
  }
  return 384;
}

/**
 * Compute similarity between two texts (cosine similarity)
 * @param textA - First text
 * @param textB - Second text
 * @returns Similarity score between -1 and 1
 */
export function textSimilarity(textA: string, textB: string): number {
  if (!native || !isNativeEmbeddingsAvailable()) {
    throw new Error("Native embeddings not available. Use transformers.js fallback.");
  }

  return native.textSimilarity(textA, textB);
}

/**
 * Search for most similar texts from candidates
 * @param query - Query text
 * @param candidates - Array of candidate texts
 * @param topK - Number of results to return (default: 10)
 * @returns Indices of most similar candidates, sorted by similarity
 */
export function searchSimilar(query: string, candidates: string[], topK = 10): number[] {
  if (!native || !isNativeEmbeddingsAvailable()) {
    throw new Error("Native embeddings not available. Use transformers.js fallback.");
  }

  return native.searchSimilar(query, candidates, topK);
}

/**
 * Embedding info for debugging
 */
export function getEmbeddingInfo(): {
  backend: string;
  available: boolean;
  dimension: number;
  model: string;
} {
  return {
    backend: getEmbeddingBackend(),
    available: isNativeEmbeddingsAvailable(),
    dimension: getEmbeddingDim(),
    model: "sentence-transformers/all-MiniLM-L6-v2",
  };
}
