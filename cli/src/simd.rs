//! SIMD-optimized vector operations

/// Get the detected SIMD capability level
pub fn get_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512 (16 floats/cycle)";
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2+FMA (8 floats/cycle)";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2 (8 floats/cycle)";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1 (4 floats/cycle)";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON (4 floats/cycle)";
    }
    "Scalar"
}

/// Compute dot product of two vectors
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2_fma(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len());
    let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());

    let (mut s0, mut s1, mut s2, mut s3) = (
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    );

    for i in 0..(len / 32) {
        let base = i * 32;
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base)), _mm256_loadu_ps(b_ptr.add(base)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base+8)), _mm256_loadu_ps(b_ptr.add(base+8)), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base+16)), _mm256_loadu_ps(b_ptr.add(base+16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base+24)), _mm256_loadu_ps(b_ptr.add(base+24)), s3);
    }

    let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    for i in ((len / 32) * 32)..len {
        result += a[i] * b[i];
    }
    result
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..(len / 4) {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base+1] * b[base+1];
        s2 += a[base+2] * b[base+2];
        s3 += a[base+3] * b[base+3];
    }
    let mut result = s0 + s1 + s2 + s3;
    for i in ((len / 4) * 4)..len {
        result += a[i] * b[i];
    }
    result
}

/// L2 normalize a vector
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        vec.iter_mut().for_each(|x| *x *= inv);
    }
}
