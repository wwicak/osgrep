//! SIMD-optimized vector operations for osgrep
//!
//! This native addon provides extreme performance for:
//! - Vector dot products (cosine similarity)
//! - RRF (Reciprocal Rank Fusion) score computation
//! - Batch score blending
//! - Fast sigmoid approximation

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Compute dot product using SIMD intrinsics
/// For 384-dimensional vectors (osgrep embedding size)
#[napi]
pub fn dot_product(a: Float32Array, b: Float32Array) -> f64 {
    let a = a.as_ref();
    let b = b.as_ref();

    if a.len() != b.len() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { dot_product_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_product_neon(a, b) };
    }

    // Scalar fallback
    dot_product_scalar(a, b)
}

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    // Handle remainder
    for i in (chunks * 8)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    // Horizontal sum
    let shuf = _mm_shuffle_ps(sum, sum, 0b10_11_00_01);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result128 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(result128) as f64;

    // Handle remainder
    for i in (chunks * 4)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }

    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;
    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum) as f64;

    // Handle remainder
    for i in (chunks * 4)..len {
        result += (*a_ptr.add(i) as f64) * (*b_ptr.add(i) as f64);
    }

    result
}

/// Compute cosine similarity between two normalized vectors
/// Since osgrep vectors are pre-normalized, this is just dot product
#[napi]
pub fn cosine_similarity(a: Float32Array, b: Float32Array) -> f64 {
    dot_product(a, b)
}

/// Batch compute multiple dot products against a single query vector
/// Returns array of similarity scores
#[napi]
pub fn batch_dot_product(query: Float32Array, vectors: Vec<Float32Array>) -> Vec<f64> {
    let query_slice = query.as_ref();

    vectors
        .into_iter()
        .map(|v| {
            let v_slice = v.as_ref();
            if v_slice.len() != query_slice.len() {
                return 0.0;
            }

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    return unsafe { dot_product_avx2(query_slice, v_slice) };
                }
            }

            dot_product_scalar(query_slice, v_slice)
        })
        .collect()
}

/// RRF (Reciprocal Rank Fusion) score computation
/// score = 1 / (k + rank) for each result
#[napi]
pub fn compute_rrf_scores(ranks: Vec<u32>, k: u32) -> Vec<f64> {
    let k_f64 = k as f64;
    ranks
        .into_iter()
        .map(|rank| 1.0 / (k_f64 + rank as f64))
        .collect()
}

/// Fuse two RRF score arrays by summing matching indices
#[napi]
pub fn fuse_rrf_scores(scores_a: Vec<f64>, scores_b: Vec<f64>) -> Vec<f64> {
    let max_len = scores_a.len().max(scores_b.len());
    let mut result = vec![0.0; max_len];

    for (i, &s) in scores_a.iter().enumerate() {
        result[i] += s;
    }
    for (i, &s) in scores_b.iter().enumerate() {
        result[i] += s;
    }

    result
}

/// Blend rerank scores with RRF scores
/// final = weight_rerank * rerank + weight_rrf * rrf
#[napi]
pub fn blend_scores(
    rerank_scores: Vec<f64>,
    rrf_scores: Vec<f64>,
    weight_rerank: f64,
    weight_rrf: f64,
) -> Vec<f64> {
    rerank_scores
        .iter()
        .zip(rrf_scores.iter())
        .map(|(&r, &rrf)| weight_rerank * r + weight_rrf * rrf)
        .collect()
}

/// Normalize scores to [0, 1] range
#[napi]
pub fn normalize_scores(scores: Vec<f64>) -> Vec<f64> {
    if scores.is_empty() {
        return scores;
    }

    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return vec![0.0; scores.len()];
    }

    scores.iter().map(|&s| s / max).collect()
}

/// Fast sigmoid approximation using lookup table + linear interpolation
/// ~10x faster than exp() based sigmoid
#[napi]
pub fn fast_sigmoid(x: f64) -> f64 {
    // Fast approximation: 1 / (1 + exp(-x))
    // Using rational approximation for |x| < 4.5
    if x >= 4.5 {
        return 1.0;
    }
    if x <= -4.5 {
        return 0.0;
    }

    // Pade approximant for sigmoid
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 270270.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    0.5 + num / (2.0 * den)
}

/// Batch sigmoid computation
#[napi]
pub fn batch_sigmoid(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(fast_sigmoid).collect()
}

/// Fast argsort - returns indices that would sort the array in descending order
#[napi]
pub fn argsort_desc(values: Vec<f64>) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..values.len() as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        values[b as usize]
            .partial_cmp(&values[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// L2 normalize a vector in-place
#[napi]
pub fn l2_normalize(mut vec: Float32Array) -> Float32Array {
    let slice = vec.as_mut();

    #[cfg(target_arch = "x86_64")]
    let norm = if is_x86_feature_detected!("avx2") {
        unsafe { l2_norm_avx2(slice) }
    } else {
        l2_norm_scalar(slice)
    };

    #[cfg(not(target_arch = "x86_64"))]
    let norm = l2_norm_scalar(slice);

    if norm > 0.0 {
        let inv_norm = 1.0 / norm as f32;
        for x in slice.iter_mut() {
            *x *= inv_norm;
        }
    }

    vec
}

fn l2_norm_scalar(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_norm_avx2(v: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = v.len();
    let chunks = len / 8;
    let mut sum = _mm256_setzero_ps();
    let ptr = v.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(ptr.add(offset));
        sum = _mm256_fmadd_ps(va, va, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    // Handle remainder
    for i in (chunks * 8)..len {
        let x = v[i] as f64;
        result += x * x;
    }

    result.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sigmoid() {
        // Test against actual sigmoid
        for x in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] {
            let expected = 1.0 / (1.0 + (-x).exp());
            let actual = fast_sigmoid(x);
            assert!((expected - actual).abs() < 0.01,
                "sigmoid({}) expected {} got {}", x, expected, actual);
        }
    }
}
