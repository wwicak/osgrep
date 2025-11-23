import { describe, expect, it } from "vitest";
import {
  argsortDesc,
  batchL2Normalize,
  benchmark,
  blendScores,
  computeDistanceMatrix,
  computeRrfScores,
  cosineSimilarity,
  dotProduct,
  fastSigmoid,
  fuseRrfScores,
  getSimdLevel,
  isNativeAvailable,
  l2Normalize,
  normalizeScores,
} from "../src/lib/simd";

describe("SIMD optimized operations", () => {
  describe("dotProduct", () => {
    it("computes dot product correctly for small vectors", () => {
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([5, 6, 7, 8]);
      // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
      expect(dotProduct(a, b)).toBeCloseTo(70, 5);
    });

    it("computes dot product correctly for 384-dim vectors (embedding size)", () => {
      const a = new Float32Array(384).fill(0.5);
      const b = new Float32Array(384).fill(0.5);
      // 384 * 0.5 * 0.5 = 96
      expect(dotProduct(a, b)).toBeCloseTo(96, 5);
    });

    it("handles vectors with remainder elements (not multiple of 8)", () => {
      const a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const b = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
      // Sum of 1-10 = 55
      expect(dotProduct(a, b)).toBeCloseTo(55, 5);
    });
  });

  describe("cosineSimilarity", () => {
    it("returns 1 for identical normalized vectors", () => {
      const a = new Float32Array([0.6, 0.8, 0, 0]);
      const b = new Float32Array([0.6, 0.8, 0, 0]);
      expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
    });

    it("returns 0 for orthogonal vectors", () => {
      const a = new Float32Array([1, 0, 0, 0]);
      const b = new Float32Array([0, 1, 0, 0]);
      expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
    });
  });

  describe("fastSigmoid", () => {
    it("returns 0.5 for input 0", () => {
      expect(fastSigmoid(0)).toBeCloseTo(0.5, 5);
    });

    it("returns ~1 for large positive values", () => {
      expect(fastSigmoid(10)).toBeCloseTo(1, 3);
    });

    it("returns ~0 for large negative values", () => {
      expect(fastSigmoid(-10)).toBeCloseTo(0, 3);
    });

    it("approximates standard sigmoid within 2% error", () => {
      for (const x of [-4, -2, -1, 0, 1, 2, 4]) {
        const expected = 1 / (1 + Math.exp(-x));
        const actual = fastSigmoid(x);
        // Allow 2% error for speed tradeoff - still accurate enough for ranking
        expect(Math.abs(expected - actual)).toBeLessThan(0.02);
      }
    });
  });

  describe("normalizeScores", () => {
    it("normalizes scores to [0, 1] range", () => {
      const scores = [10, 20, 30, 40, 50];
      const normalized = normalizeScores(scores);
      expect(normalized).toEqual([0.2, 0.4, 0.6, 0.8, 1.0]);
    });

    it("handles empty array", () => {
      expect(normalizeScores([])).toEqual([]);
    });

    it("handles all zeros", () => {
      const result = normalizeScores([0, 0, 0]);
      expect(result).toEqual([0, 0, 0]);
    });
  });

  describe("blendScores", () => {
    it("blends scores with weights", () => {
      const rerank = [1, 0.5, 0];
      const rrf = [0, 0.5, 1];
      const blended = blendScores(rerank, rrf, 0.7, 0.3);
      // [0.7*1 + 0.3*0, 0.7*0.5 + 0.3*0.5, 0.7*0 + 0.3*1] = [0.7, 0.5, 0.3]
      expect(blended[0]).toBeCloseTo(0.7, 5);
      expect(blended[1]).toBeCloseTo(0.5, 5);
      expect(blended[2]).toBeCloseTo(0.3, 5);
    });
  });

  describe("argsortDesc", () => {
    it("returns indices sorted by descending values", () => {
      const values = [3, 1, 4, 1, 5, 9, 2, 6];
      const indices = argsortDesc(values);
      // Sorted desc: [9, 6, 5, 4, 3, 2, 1, 1] at indices [5, 7, 4, 2, 0, 6, 1, 3]
      expect(indices[0]).toBe(5); // 9
      expect(indices[1]).toBe(7); // 6
      expect(indices[2]).toBe(4); // 5
    });
  });

  describe("computeRrfScores", () => {
    it("computes RRF scores correctly", () => {
      const ranks = [1, 2, 3];
      const k = 60;
      const scores = computeRrfScores(ranks, k);
      expect(scores[0]).toBeCloseTo(1 / 61, 5);
      expect(scores[1]).toBeCloseTo(1 / 62, 5);
      expect(scores[2]).toBeCloseTo(1 / 63, 5);
    });
  });

  describe("fuseRrfScores", () => {
    it("fuses scores by summing", () => {
      const a = [0.1, 0.2, 0.3];
      const b = [0.3, 0.2, 0.1];
      const fused = fuseRrfScores(a, b);
      expect(fused).toEqual([0.4, 0.4, 0.4]);
    });

    it("handles arrays of different lengths", () => {
      const a = [0.1, 0.2];
      const b = [0.3, 0.2, 0.1];
      const fused = fuseRrfScores(a, b);
      expect(fused.length).toBe(3);
      expect(fused[2]).toBe(0.1);
    });
  });

  describe("l2Normalize", () => {
    it("normalizes vector to unit length", () => {
      const vec = new Float32Array([3, 4, 0, 0]);
      const normalized = l2Normalize(vec);
      // L2 norm of [3, 4, 0, 0] = 5, so normalized = [0.6, 0.8, 0, 0]
      expect(normalized[0]).toBeCloseTo(0.6, 5);
      expect(normalized[1]).toBeCloseTo(0.8, 5);
    });
  });

  describe("isNativeAvailable", () => {
    it("returns a boolean", () => {
      expect(typeof isNativeAvailable()).toBe("boolean");
    });
  });

  describe("benchmark", () => {
    it("returns benchmark results", () => {
      const result = benchmark(100);
      expect(typeof result.native).toBe("boolean");
      expect(typeof result.dotProductMs).toBe("number");
      expect(typeof result.sigmoidMs).toBe("number");
      expect(typeof result.simdLevel).toBe("string");
      expect(typeof result.opsPerSecond).toBe("number");
    });
  });

  describe("getSimdLevel", () => {
    it("returns a string describing SIMD level", () => {
      const level = getSimdLevel();
      expect(typeof level).toBe("string");
      expect(level.length).toBeGreaterThan(0);
    });
  });

  describe("batchL2Normalize", () => {
    it("normalizes multiple vectors", () => {
      const vectors = [
        new Float32Array([3, 4, 0, 0]),
        new Float32Array([0, 5, 12, 0]),
      ];
      const normalized = batchL2Normalize(vectors);
      expect(normalized.length).toBe(2);
      // First vector: [3, 4, 0, 0] / 5 = [0.6, 0.8, 0, 0]
      expect(normalized[0][0]).toBeCloseTo(0.6, 5);
      expect(normalized[0][1]).toBeCloseTo(0.8, 5);
      // Second vector: [0, 5, 12, 0] / 13 = [0, 0.385, 0.923, 0]
      expect(normalized[1][1]).toBeCloseTo(5 / 13, 4);
      expect(normalized[1][2]).toBeCloseTo(12 / 13, 4);
    });
  });

  describe("computeDistanceMatrix", () => {
    it("computes pairwise distances", () => {
      // Normalized vectors for cosine distance
      const vectors = [
        new Float32Array([1, 0, 0, 0]),
        new Float32Array([0, 1, 0, 0]),
        new Float32Array([1, 0, 0, 0]), // Same as first
      ];
      const distances = computeDistanceMatrix(vectors);
      // 3x3 matrix flattened
      expect(distances.length).toBe(9);
      // Distance between identical vectors should be 0
      expect(distances[0 * 3 + 2]).toBeCloseTo(0, 5); // [0][2]
      expect(distances[2 * 3 + 0]).toBeCloseTo(0, 5); // [2][0]
      // Distance between orthogonal vectors should be 1
      expect(distances[0 * 3 + 1]).toBeCloseTo(1, 5); // [0][1]
      expect(distances[1 * 3 + 0]).toBeCloseTo(1, 5); // [1][0]
    });
  });
});
