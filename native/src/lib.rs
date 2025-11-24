//! Ultra-lightweight SIMD-optimized vector operations for osgrep
//!
//! Zero-cost abstractions with automatic CPU feature detection:
//! - AVX-512: 16 floats/cycle
//! - AVX2+FMA: 8 floats/cycle
//! - SSE4.1: 4 floats/cycle
//! - NEON: 4 floats/cycle (ARM64)
//!
//! Optional features:
//! - --features parallel: Parallel processing with Rayon
//! - --features embeddings: Native sentence embeddings with Candle
//! - --features metal: Metal GPU acceleration for Apple Silicon
//! - --features sqlite: SQLite-Vec vector storage (replaces LanceDB)

#![allow(clippy::missing_safety_doc)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Native embeddings module (optional)
#[cfg(feature = "embeddings")]
mod embeddings;

#[cfg(feature = "embeddings")]
pub use embeddings::*;

// SQLite-Vec vector store module (optional)
#[cfg(feature = "sqlite")]
mod vector_store;

#[cfg(feature = "sqlite")]
pub use vector_store::*;

// ============================================================================
// SIMD Level Detection
// ============================================================================

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

// ============================================================================
// Core Dot Product - The Hot Path
// ============================================================================

#[napi]
pub fn dot_product(a: Float32Array, b: Float32Array) -> f64 {
    let a = a.as_ref();
    let b = b.as_ref();
    if a.len() != b.len() {
        return 0.0;
    }
    dot_product_dispatch(a, b)
}

#[inline(always)]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2_fma(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_neon(a, b) };
    }

    dot_scalar(a, b)
}

// AVX-512: 16 floats per cycle
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (len, mut sum) = (a.len(), _mm512_setzero_ps());
    let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());

    for i in 0..(len / 16) {
        let va = _mm512_loadu_ps(a_ptr.add(i * 16));
        let vb = _mm512_loadu_ps(b_ptr.add(i * 16));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum) as f64;
    for i in ((len / 16) * 16)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }
    result
}

// AVX2+FMA: 8 floats with 4-way accumulator
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let len = a.len();
    let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());

    let (mut s0, mut s1, mut s2, mut s3) =
        (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());

    for i in 0..(len / 32) {
        let base = i * 32;
        s0 =
            _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base)), _mm256_loadu_ps(b_ptr.add(base)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(base + 8)),
            _mm256_loadu_ps(b_ptr.add(base + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(base + 16)),
            _mm256_loadu_ps(b_ptr.add(base + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a_ptr.add(base + 24)),
            _mm256_loadu_ps(b_ptr.add(base + 24)),
            s3,
        );
    }

    let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    for i in ((len / 32) * 32)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }
    result
}

// AVX2: 8 floats per cycle
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (len, mut sum) = (a.len(), _mm256_setzero_ps());
    let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());

    for i in 0..(len / 8) {
        let va = _mm256_loadu_ps(a_ptr.add(i * 8));
        let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    for i in ((len / 8) * 8)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }
    result
}

// NEON: 4 floats per cycle (ARM64)
#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::aarch64::*;
    let (len, mut sum) = (a.len(), vdupq_n_f32(0.0));
    let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());

    for i in 0..(len / 4) {
        sum = vfmaq_f32(sum, vld1q_f32(a_ptr.add(i * 4)), vld1q_f32(b_ptr.add(i * 4)));
    }

    let mut result = vaddvq_f32(sum) as f64;
    for i in ((len / 4) * 4)..len {
        result += (*a_ptr.add(i) as f64) * (*b_ptr.add(i) as f64);
    }
    result
}

// Scalar fallback with loop unrolling
#[inline(always)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f64 {
    let (len, mut s0, mut s1, mut s2, mut s3) = (a.len(), 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..(len / 4) {
        let base = i * 4;
        s0 += (a[base] as f64) * (b[base] as f64);
        s1 += (a[base + 1] as f64) * (b[base + 1] as f64);
        s2 += (a[base + 2] as f64) * (b[base + 2] as f64);
        s3 += (a[base + 3] as f64) * (b[base + 3] as f64);
    }
    let mut result = s0 + s1 + s2 + s3;
    for i in ((len / 4) * 4)..len {
        result += (a[i] as f64) * (b[i] as f64);
    }
    result
}

#[napi]
pub fn cosine_similarity(a: Float32Array, b: Float32Array) -> f64 {
    dot_product(a, b)
}

// ============================================================================
// Batch Operations (parallel when feature enabled)
// ============================================================================

#[napi]
pub fn batch_dot_product(query: Float32Array, vectors: Vec<Float32Array>) -> Vec<f64> {
    let q = query.as_ref();

    #[cfg(feature = "parallel")]
    if vectors.len() >= 32 {
        return vectors
            .into_par_iter()
            .map(|v| {
                if v.as_ref().len() == q.len() {
                    dot_product_dispatch(q, v.as_ref())
                } else {
                    0.0
                }
            })
            .collect();
    }

    vectors
        .into_iter()
        .map(
            |v| if v.as_ref().len() == q.len() { dot_product_dispatch(q, v.as_ref()) } else { 0.0 },
        )
        .collect()
}

// ============================================================================
// Score Operations
// ============================================================================

#[napi]
pub fn compute_rrf_scores(ranks: Vec<u32>, k: u32) -> Vec<f64> {
    let k = k as f64;
    ranks.into_iter().map(|r| 1.0 / (k + r as f64)).collect()
}

#[napi]
pub fn fuse_rrf_scores(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; a.len().max(b.len())];
    for (i, &s) in a.iter().enumerate() {
        result[i] += s;
    }
    for (i, &s) in b.iter().enumerate() {
        result[i] += s;
    }
    result
}

#[napi]
pub fn blend_scores(rerank: Vec<f64>, rrf: Vec<f64>, w_rerank: f64, w_rrf: f64) -> Vec<f64> {
    rerank.iter().zip(rrf.iter()).map(|(&r, &f)| w_rerank * r + w_rrf * f).collect()
}

#[napi]
pub fn normalize_scores(scores: Vec<f64>) -> Vec<f64> {
    if scores.is_empty() {
        return scores;
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.into_iter().map(|s| s / max).collect()
}

// ============================================================================
// Fast Sigmoid (~10x faster than exp)
// ============================================================================

#[inline(always)]
fn sigmoid_impl(x: f64) -> f64 {
    if x >= 4.5 {
        return 1.0;
    }
    if x <= -4.5 {
        return 0.0;
    }
    let x2 = x * x;
    0.5 + x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
        / (2.0 * (270270.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0))))
}

#[napi]
pub fn fast_sigmoid(x: f64) -> f64 {
    sigmoid_impl(x)
}

#[napi]
pub fn batch_sigmoid(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(sigmoid_impl).collect()
}

// ============================================================================
// Sorting & Normalization
// ============================================================================

#[napi]
pub fn argsort_desc(values: Vec<f64>) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..values.len() as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        values[b as usize].partial_cmp(&values[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

#[napi]
pub fn l2_normalize(mut vec: Float32Array) -> Float32Array {
    let slice = vec.as_mut();
    let norm = l2_norm(slice);
    if norm > 0.0 {
        let inv = 1.0 / norm as f32;
        slice.iter_mut().for_each(|x| *x *= inv);
    }
    vec
}

fn l2_norm(v: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { l2_norm_avx2(v) };
    }
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_norm_avx2(v: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (len, mut sum) = (v.len(), _mm256_setzero_ps());
    let ptr = v.as_ptr();

    for i in 0..(len / 8) {
        let va = _mm256_loadu_ps(ptr.add(i * 8));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, va));
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32) as f64;

    for i in ((len / 8) * 8)..len {
        let x = v[i] as f64;
        result += x * x;
    }
    result.sqrt()
}

#[napi]
pub fn batch_l2_normalize(vectors: Vec<Float32Array>) -> Vec<Float32Array> {
    vectors.into_iter().map(l2_normalize).collect()
}

#[napi]
pub fn compute_distance_matrix(vectors: Vec<Float32Array>) -> Vec<f64> {
    let n = vectors.len();
    let mut distances = vec![0.0f64; n * n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - dot_scalar(vectors[i].as_ref(), vectors[j].as_ref());
            distances[i * n + j] = dist;
            distances[j * n + i] = dist;
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid_impl(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid_impl(10.0) > 0.99);
        assert!(sigmoid_impl(-10.0) < 0.01);
    }
}
