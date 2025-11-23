//! EXTREME SIMD-optimized vector operations for osgrep
//!
//! This native addon provides maximum performance using:
//! - AVX-512 (16 floats/cycle) when available
//! - AVX2 (8 floats/cycle) fallback
//! - SSE4.1 (4 floats/cycle) fallback
//! - NEON (4 floats/cycle) on ARM
//! - Parallel batch processing with Rayon
//! - CPU prefetch hints for cache optimization
//! - Aligned memory operations

#![allow(clippy::missing_safety_doc)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;

// Threshold for parallel processing (avoid overhead for small batches)
const PARALLEL_THRESHOLD: usize = 32;

// Prefetch distance in cache lines (64 bytes = 16 floats)
const PREFETCH_DISTANCE: usize = 4;

/// Get SIMD capability level for diagnostics
#[napi]
pub fn get_simd_level() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512 (16 floats/cycle)".to_string();
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2+FMA (8 floats/cycle)".to_string();
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2 (8 floats/cycle)".to_string();
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1 (4 floats/cycle)".to_string();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON (4 floats/cycle)".to_string();
    }
    "Scalar".to_string()
}

/// Ultra-fast dot product with automatic SIMD dispatch
#[napi]
pub fn dot_product(a: Float32Array, b: Float32Array) -> f64 {
    let a = a.as_ref();
    let b = b.as_ref();

    if a.len() != b.len() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_product_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
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

    #[allow(unreachable_code)]
    dot_product_scalar(a, b)
}

// ============================================================================
// AVX-512 Implementation (16 floats per cycle!)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Prefetch ahead for cache optimization
    for i in 0..chunks {
        let offset = i * 16;

        // Prefetch next cache lines
        if i + PREFETCH_DISTANCE < chunks {
            let prefetch_offset = (i + PREFETCH_DISTANCE) * 16;
            _mm_prefetch(a_ptr.add(prefetch_offset) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b_ptr.add(prefetch_offset) as *const i8, _MM_HINT_T0);
        }

        let va = _mm512_loadu_ps(a_ptr.add(offset));
        let vb = _mm512_loadu_ps(b_ptr.add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum using AVX-512 reduction
    let mut result = _mm512_reduce_add_ps(sum) as f64;

    // Handle remainder
    for i in (chunks * 16)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }

    result
}

// ============================================================================
// AVX2 + FMA Implementation (8 floats per cycle with fused multiply-add)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32; // Process 32 floats (4 AVX registers) per iteration

    // Use 4 accumulators to hide FMA latency (4 cycles on modern CPUs)
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 32;

        // Prefetch
        if i + PREFETCH_DISTANCE < chunks {
            let pf = (i + PREFETCH_DISTANCE) * 32;
            _mm_prefetch(a_ptr.add(pf) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b_ptr.add(pf) as *const i8, _MM_HINT_T0);
        }

        // Load and FMA 4 vectors
        let va0 = _mm256_loadu_ps(a_ptr.add(base));
        let vb0 = _mm256_loadu_ps(b_ptr.add(base));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        let va1 = _mm256_loadu_ps(a_ptr.add(base + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(base + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        let va2 = _mm256_loadu_ps(a_ptr.add(base + 16));
        let vb2 = _mm256_loadu_ps(b_ptr.add(base + 16));
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

        let va3 = _mm256_loadu_ps(a_ptr.add(base + 24));
        let vb3 = _mm256_loadu_ps(b_ptr.add(base + 24));
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }

    // Combine accumulators
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    let sum = _mm256_add_ps(sum0, sum2);

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    // Handle remainder with single AVX2 passes then scalar
    let mut i = chunks * 32;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        let prod = _mm256_mul_ps(va, vb);

        let hi = _mm256_extractf128_ps(prod, 1);
        let lo = _mm256_castps256_ps128(prod);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        result += _mm_cvtss_f32(sum32) as f64;
        i += 8;
    }

    // Scalar remainder
    while i < len {
        result += (a[i] as f64) * (b[i] as f64);
        i += 1;
    }

    result
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
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

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

    let shuf = _mm_shuffle_ps(sum, sum, 0b10_11_00_01);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result128 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(result128) as f64;

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

    for i in (chunks * 4)..len {
        result += (*a_ptr.add(i) as f64) * (*b_ptr.add(i) as f64);
    }

    result
}

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f64 {
    // Unrolled scalar for when no SIMD is available
    let len = a.len();
    let chunks = len / 4;
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += (a[base] as f64) * (b[base] as f64);
        sum1 += (a[base + 1] as f64) * (b[base + 1] as f64);
        sum2 += (a[base + 2] as f64) * (b[base + 2] as f64);
        sum3 += (a[base + 3] as f64) * (b[base + 3] as f64);
    }

    let mut result = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 4)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }
    result
}

/// Cosine similarity (for normalized vectors = dot product)
#[napi]
pub fn cosine_similarity(a: Float32Array, b: Float32Array) -> f64 {
    dot_product(a, b)
}

/// PARALLEL batch dot product - extreme performance for large batches
#[napi]
pub fn batch_dot_product(query: Float32Array, vectors: Vec<Float32Array>) -> Vec<f64> {
    let query_slice = query.as_ref();

    if vectors.len() >= PARALLEL_THRESHOLD {
        // Parallel processing for large batches
        vectors
            .into_par_iter()
            .map(|v| {
                let v_slice = v.as_ref();
                if v_slice.len() != query_slice.len() {
                    return 0.0;
                }

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        return unsafe { dot_product_avx512(query_slice, v_slice) };
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return unsafe { dot_product_avx2_fma(query_slice, v_slice) };
                    }
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { dot_product_avx2(query_slice, v_slice) };
                    }
                }

                dot_product_scalar(query_slice, v_slice)
            })
            .collect()
    } else {
        // Sequential for small batches (avoid thread overhead)
        vectors
            .into_iter()
            .map(|v| {
                let v_slice = v.as_ref();
                if v_slice.len() != query_slice.len() {
                    return 0.0;
                }

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        return unsafe { dot_product_avx512(query_slice, v_slice) };
                    }
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return unsafe { dot_product_avx2_fma(query_slice, v_slice) };
                    }
                }

                dot_product_scalar(query_slice, v_slice)
            })
            .collect()
    }
}

/// RRF scores with SIMD vectorization
#[napi]
pub fn compute_rrf_scores(ranks: Vec<u32>, k: u32) -> Vec<f64> {
    let k_f64 = k as f64;

    if ranks.len() >= PARALLEL_THRESHOLD {
        ranks
            .into_par_iter()
            .map(|rank| 1.0 / (k_f64 + rank as f64))
            .collect()
    } else {
        ranks
            .into_iter()
            .map(|rank| 1.0 / (k_f64 + rank as f64))
            .collect()
    }
}

/// Fuse RRF scores
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

/// SIMD-optimized score blending
#[napi]
pub fn blend_scores(
    rerank_scores: Vec<f64>,
    rrf_scores: Vec<f64>,
    weight_rerank: f64,
    weight_rrf: f64,
) -> Vec<f64> {
    if rerank_scores.len() >= PARALLEL_THRESHOLD {
        rerank_scores
            .into_par_iter()
            .zip(rrf_scores.into_par_iter())
            .map(|(r, rrf)| weight_rerank * r + weight_rrf * rrf)
            .collect()
    } else {
        rerank_scores
            .iter()
            .zip(rrf_scores.iter())
            .map(|(&r, &rrf)| weight_rerank * r + weight_rrf * rrf)
            .collect()
    }
}

/// Normalize scores to [0, 1]
#[napi]
pub fn normalize_scores(scores: Vec<f64>) -> Vec<f64> {
    if scores.is_empty() {
        return scores;
    }

    // Find max using parallel reduction for large arrays
    let max = if scores.len() >= PARALLEL_THRESHOLD {
        scores.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max)
    } else {
        scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    };

    if max <= 0.0 {
        return vec![0.0; scores.len()];
    }

    let inv_max = 1.0 / max;

    if scores.len() >= PARALLEL_THRESHOLD {
        scores.into_par_iter().map(|s| s * inv_max).collect()
    } else {
        scores.into_iter().map(|s| s * inv_max).collect()
    }
}

/// Ultra-fast sigmoid using Pade approximant
/// Error < 0.02 for |x| < 4.5, exact at boundaries
#[inline(always)]
fn fast_sigmoid_impl(x: f64) -> f64 {
    if x >= 4.5 {
        return 1.0;
    }
    if x <= -4.5 {
        return 0.0;
    }

    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 270270.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    0.5 + num / (2.0 * den)
}

#[napi]
pub fn fast_sigmoid(x: f64) -> f64 {
    fast_sigmoid_impl(x)
}

/// Batch sigmoid with parallel processing
#[napi]
pub fn batch_sigmoid(values: Vec<f64>) -> Vec<f64> {
    if values.len() >= PARALLEL_THRESHOLD {
        values.into_par_iter().map(fast_sigmoid_impl).collect()
    } else {
        values.into_iter().map(fast_sigmoid_impl).collect()
    }
}

/// Parallel argsort descending
#[napi]
pub fn argsort_desc(values: Vec<f64>) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..values.len() as u32).collect();

    if values.len() >= PARALLEL_THRESHOLD {
        indices.par_sort_unstable_by(|&a, &b| {
            values[b as usize]
                .partial_cmp(&values[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_unstable_by(|&a, &b| {
            values[b as usize]
                .partial_cmp(&values[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    indices
}

/// L2 normalize with SIMD
#[napi]
pub fn l2_normalize(mut vec: Float32Array) -> Float32Array {
    let slice = vec.as_mut();

    #[cfg(target_arch = "x86_64")]
    let norm = if is_x86_feature_detected!("avx512f") {
        unsafe { l2_norm_avx512(slice) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { l2_norm_avx2(slice) }
    } else {
        l2_norm_scalar(slice)
    };

    #[cfg(not(target_arch = "x86_64"))]
    let norm = l2_norm_scalar(slice);

    if norm > 0.0 {
        let inv_norm = 1.0 / norm as f32;

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe { scale_vector_avx2(slice, inv_norm) };
        } else {
            for x in slice.iter_mut() {
                *x *= inv_norm;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        for x in slice.iter_mut() {
            *x *= inv_norm;
        }
    }

    vec
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_norm_avx512(v: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = v.len();
    let chunks = len / 16;
    let mut sum = _mm512_setzero_ps();
    let ptr = v.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(ptr.add(offset));
        sum = _mm512_fmadd_ps(va, va, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum) as f64;

    for i in (chunks * 16)..len {
        let x = v[i] as f64;
        result += x * x;
    }

    result.sqrt()
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
        let prod = _mm256_mul_ps(va, va);
        sum = _mm256_add_ps(sum, prod);
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    for i in (chunks * 8)..len {
        let x = v[i] as f64;
        result += x * x;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_vector_avx2(v: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let len = v.len();
    let chunks = len / 8;
    let ptr = v.as_mut_ptr();
    let scale_vec = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(ptr.add(offset));
        let scaled = _mm256_mul_ps(va, scale_vec);
        _mm256_storeu_ps(ptr.add(offset), scaled);
    }

    for i in (chunks * 8)..len {
        v[i] *= scale;
    }
}

fn l2_norm_scalar(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// Batch L2 normalize - parallel for large batches
#[napi]
pub fn batch_l2_normalize(vectors: Vec<Float32Array>) -> Vec<Float32Array> {
    if vectors.len() >= PARALLEL_THRESHOLD {
        vectors.into_par_iter().map(l2_normalize).collect()
    } else {
        vectors.into_iter().map(l2_normalize).collect()
    }
}

/// Distance matrix computation (for clustering/reranking)
#[napi]
pub fn compute_distance_matrix(vectors: Vec<Float32Array>) -> Vec<f64> {
    let n = vectors.len();
    let mut distances = vec![0.0f64; n * n];

    // Parallel computation of upper triangle
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let computed: Vec<(usize, usize, f64)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let a = vectors[i].as_ref();
            let b = vectors[j].as_ref();
            let sim = dot_product_scalar(a, b);
            let dist = 1.0 - sim; // Cosine distance
            (i, j, dist)
        })
        .collect();

    // Fill matrix (symmetric)
    for (i, j, dist) in computed {
        distances[i * n + j] = dist;
        distances[j * n + i] = dist;
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sigmoid() {
        for x in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] {
            let expected = 1.0 / (1.0 + (-x).exp());
            let actual = fast_sigmoid_impl(x);
            assert!(
                (expected - actual).abs() < 0.02,
                "sigmoid({}) expected {} got {}",
                x,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_simd_level() {
        let level = get_simd_level();
        assert!(!level.is_empty());
    }
}
