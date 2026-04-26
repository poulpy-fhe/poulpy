// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

/// Converts `i64` ring element coefficients to `f64` using IEEE 754 bit manipulation.
///
/// This function performs exact conversion from signed 64-bit integers to 64-bit floats
/// for inputs bounded by `|x| < 2^50`. The conversion uses bitwise operations to avoid
/// floating-point rounding modes, ensuring deterministic results across platforms.
///
/// # Preconditions
///
/// - **CPU features**: AVX2 and FMA must be supported (enforced via `#[target_feature]`).
/// - **Slice lengths**: `res.len() == a.len()` (validated in debug builds).
/// - **Numeric bounds**: `|a[i]| <= 2^50 - 1` for all `i` (validated in debug builds).
///
/// # Correctness
///
/// The IEEE 754 bit manipulation relies on the input bound `|x| < 2^50`. Inputs exceeding
/// this bound will produce **silent wrong results** without panicking. Debug builds validate
/// this invariant; release builds assume the caller has ensured correctness upstream.
///
/// # Algorithm
///
/// 1. Add `2^51` to each input (shift into the positive range).
/// 2. Reinterpret bits as `f64` and OR with exponent bits to set mantissa.
/// 3. Subtract `3 * 2^51` to restore correct signed value.
///
/// This approach avoids FP rounding and ensures bit-exact determinism.
///
/// # Performance
///
/// - **Vectorization**: Processes 4 elements per AVX2 iteration.
/// - **Tail handling**: Scalar fallback for `len % 4 != 0` (negligible overhead).
/// - **Complexity**: O(n) with ~1.5 cycles per element on modern CPUs.
///
/// # Panics
///
/// In debug builds, panics if:
/// - Slice lengths mismatch.
/// - Any input element exceeds the bound `|x| > 2^50 - 1`.
///
/// # Safety
///
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma")`).
/// Calling this function on incompatible CPUs results in `SIGILL`.
#[target_feature(enable = "fma")]
pub fn reim_from_znx_i64_bnd50_fma(res: &mut [f64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        const BOUND: i64 = (1i64 << 50) - 1;
        for (i, &val) in a.iter().enumerate() {
            assert!(
                val.abs() <= BOUND,
                "Input a[{}] = {} exceeds bound 2^50-1 ({})",
                i,
                val,
                BOUND
            );
        }
    }

    let n: usize = res.len();

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_epi64, _mm256_castsi256_pd, _mm256_loadu_si256, _mm256_or_pd, _mm256_set1_epi64x,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        };

        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_256: __m256d = _mm256_set1_pd(expo);
        let add_cst_256: __m256i = _mm256_set1_epi64x(add_cst);
        let sub_cst_256: __m256d = _mm256_set1_pd(sub_cst);

        let mut res_ptr: *mut f64 = res.as_mut_ptr();
        let mut a_ptr: *const __m256i = a.as_ptr() as *const __m256i;

        let span: usize = n >> 2;

        for _ in 0..span {
            let mut ai64_256: __m256i = _mm256_loadu_si256(a_ptr);

            ai64_256 = _mm256_add_epi64(ai64_256, add_cst_256);

            let mut af64_256: __m256d = _mm256_castsi256_pd(ai64_256);
            af64_256 = _mm256_or_pd(af64_256, expo_256);
            af64_256 = _mm256_sub_pd(af64_256, sub_cst_256);

            _mm256_storeu_pd(res_ptr, af64_256);

            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(1);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_from_znx_i64_ref;
            reim_from_znx_i64_ref(&mut res[span << 2..], &a[span << 2..])
        }
    }
}

/// Masked AVX2/FMA variant of [`reim_from_znx_i64_bnd50_fma`].
///
/// Converts `(a[i] & mask)` into `f64` exactly for values bounded by `|x| < 2^50`.
#[target_feature(enable = "fma")]
pub fn reim_from_znx_i64_masked_bnd50_fma(res: &mut [f64], a: &[i64], mask: i64) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        const BOUND: i64 = (1i64 << 50) - 1;
        for (i, &val) in a.iter().enumerate() {
            let masked = val & mask;
            assert!(
                masked.abs() <= BOUND,
                "Masked input (a[{}] & mask) = {} exceeds bound 2^50-1 ({})",
                i,
                masked,
                BOUND
            );
        }
    }

    let n: usize = res.len();

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_pd, _mm256_loadu_si256, _mm256_or_pd,
            _mm256_set1_epi64x, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        };

        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_256: __m256d = _mm256_set1_pd(expo);
        let add_cst_256: __m256i = _mm256_set1_epi64x(add_cst);
        let mask_256: __m256i = _mm256_set1_epi64x(mask);
        let sub_cst_256: __m256d = _mm256_set1_pd(sub_cst);

        let mut res_ptr: *mut f64 = res.as_mut_ptr();
        let mut a_ptr: *const __m256i = a.as_ptr() as *const __m256i;

        let span: usize = n >> 2;

        for _ in 0..span {
            let mut ai64_256: __m256i = _mm256_loadu_si256(a_ptr);
            ai64_256 = _mm256_and_si256(ai64_256, mask_256);
            ai64_256 = _mm256_add_epi64(ai64_256, add_cst_256);

            let mut af64_256: __m256d = _mm256_castsi256_pd(ai64_256);
            af64_256 = _mm256_or_pd(af64_256, expo_256);
            af64_256 = _mm256_sub_pd(af64_256, sub_cst_256);

            _mm256_storeu_pd(res_ptr, af64_256);

            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(1);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_from_znx_i64_masked_ref;
            reim_from_znx_i64_masked_ref(&mut res[span << 2..], &a[span << 2..], mask)
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma,avx2")`);
#[allow(dead_code)]
#[target_feature(enable = "avx2,fma")]
pub fn reim_to_znx_i64_bnd63_avx2_fma(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    let sign_mask: u64 = 0x8000000000000000u64;
    let expo_mask: u64 = 0x7FF0000000000000u64;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010000000000000u64;
    let divi_bits: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.;

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_pd,
            _mm256_loadu_pd, _mm256_or_pd, _mm256_or_si256, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_sllv_epi64,
            _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_sub_epi64, _mm256_xor_si256,
        };

        let sign_mask_256: __m256d = _mm256_castsi256_pd(_mm256_set1_epi64x(sign_mask as i64));
        let expo_mask_256: __m256i = _mm256_set1_epi64x(expo_mask as i64);
        let mantissa_mask_256: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);
        let mantissa_msb_256: __m256i = _mm256_set1_epi64x(mantissa_msb as i64);
        let offset_256 = _mm256_set1_pd(offset);
        let divi_bits_256 = _mm256_castpd_si256(_mm256_set1_pd(divi_bits));

        let mut res_ptr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut a_ptr: *const f64 = a.as_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            use std::arch::x86_64::_mm256_storeu_si256;
            let mut a: __m256d = _mm256_loadu_pd(a_ptr);

            // a += sign(a) * m/2
            let asign: __m256d = _mm256_and_pd(a, sign_mask_256);
            a = _mm256_add_pd(a, _mm256_or_pd(asign, offset_256));

            // sign: either 0 or -1
            let mut sign_mask: __m256i = _mm256_castpd_si256(asign);
            sign_mask = _mm256_sub_epi64(_mm256_set1_epi64x(0), _mm256_srli_epi64(sign_mask, 63));

            // compute the exponents
            let a0exp: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), expo_mask_256);
            let mut a0lsh: __m256i = _mm256_sub_epi64(a0exp, divi_bits_256);
            let mut a0rsh: __m256i = _mm256_sub_epi64(divi_bits_256, a0exp);
            a0lsh = _mm256_srli_epi64(a0lsh, 52);
            a0rsh = _mm256_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), mantissa_mask_256);
            a0pos = _mm256_or_si256(a0pos, mantissa_msb_256);
            a0lsh = _mm256_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm256_srlv_epi64(a0pos, a0rsh);
            let mut out: __m256i = _mm256_or_si256(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm256_xor_si256(out, sign_mask);
            out = _mm256_sub_epi64(out, sign_mask);

            // stores
            _mm256_storeu_si256(res_ptr, out);

            res_ptr = res_ptr.add(1);
            a_ptr = a_ptr.add(4);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_to_znx_i64_ref;
            reim_to_znx_i64_ref(&mut res[span << 2..], divisor, &a[span << 2..])
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma,avx2")`);
#[target_feature(enable = "avx2,fma")]
pub fn reim_to_znx_i64_assign_bnd63_avx2_fma(res: &mut [f64], divisor: f64) {
    let sign_mask: u64 = 0x8000000000000000u64;
    let expo_mask: u64 = 0x7FF0000000000000u64;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010000000000000u64;
    let divi_bits: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.;

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_pd,
            _mm256_loadu_pd, _mm256_or_pd, _mm256_or_si256, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_sllv_epi64,
            _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_sub_epi64, _mm256_xor_si256,
        };

        use poulpy_cpu_ref::reference::fft64::reim::reim_to_znx_i64_assign_ref;

        let sign_mask_256: __m256d = _mm256_castsi256_pd(_mm256_set1_epi64x(sign_mask as i64));
        let expo_mask_256: __m256i = _mm256_set1_epi64x(expo_mask as i64);
        let mantissa_mask_256: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);
        let mantissa_msb_256: __m256i = _mm256_set1_epi64x(mantissa_msb as i64);
        let offset_256: __m256d = _mm256_set1_pd(offset);
        let divi_bits_256: __m256i = _mm256_castpd_si256(_mm256_set1_pd(divi_bits));

        let mut res_ptr_4xi64: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut res_ptr_1xf64: *mut f64 = res.as_mut_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            use std::arch::x86_64::_mm256_storeu_si256;
            let mut a: __m256d = _mm256_loadu_pd(res_ptr_1xf64);

            // a += sign(a) * m/2
            let asign: __m256d = _mm256_and_pd(a, sign_mask_256);
            a = _mm256_add_pd(a, _mm256_or_pd(asign, offset_256));

            // sign: either 0 or -1
            let mut sign_mask: __m256i = _mm256_castpd_si256(asign);
            sign_mask = _mm256_sub_epi64(_mm256_set1_epi64x(0), _mm256_srli_epi64(sign_mask, 63));

            // compute the exponents
            let a0exp: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), expo_mask_256);
            let mut a0lsh: __m256i = _mm256_sub_epi64(a0exp, divi_bits_256);
            let mut a0rsh: __m256i = _mm256_sub_epi64(divi_bits_256, a0exp);
            a0lsh = _mm256_srli_epi64(a0lsh, 52);
            a0rsh = _mm256_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m256i = _mm256_and_si256(_mm256_castpd_si256(a), mantissa_mask_256);
            a0pos = _mm256_or_si256(a0pos, mantissa_msb_256);
            a0lsh = _mm256_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm256_srlv_epi64(a0pos, a0rsh);
            let mut out: __m256i = _mm256_or_si256(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm256_xor_si256(out, sign_mask);
            out = _mm256_sub_epi64(out, sign_mask);

            // stores
            _mm256_storeu_si256(res_ptr_4xi64, out);

            res_ptr_4xi64 = res_ptr_4xi64.add(1);
            res_ptr_1xf64 = res_ptr_1xf64.add(4);
        }

        if !res.len().is_multiple_of(4) {
            reim_to_znx_i64_assign_ref(&mut res[span << 2..], divisor)
        }
    }
}

/// # Correctness
/// Only ensured for inputs absoluate value bounded by 2^50-1
/// # Safety
/// Caller must ensure the CPU supports FMA (e.g., via `is_x86_feature_detected!("fma")`);
#[target_feature(enable = "fma")]
#[allow(dead_code)]
pub fn reim_to_znx_i64_avx2_bnd50_fma(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    unsafe {
        use std::arch::x86_64::{
            __m256d, __m256i, _mm256_add_pd, _mm256_and_si256, _mm256_castpd_si256, _mm256_loadu_pd, _mm256_set1_epi64x,
            _mm256_set1_pd, _mm256_storeu_si256, _mm256_sub_epi64,
        };

        let mantissa_mask: u64 = 0x000FFFFFFFFFFFFFu64;
        let sub_cst: i64 = 1i64 << 51;
        let add_cst: f64 = divisor * (3i64 << 51) as f64;

        let sub_cst_4: __m256i = _mm256_set1_epi64x(sub_cst);
        let add_cst_4: std::arch::x86_64::__m256d = _mm256_set1_pd(add_cst);
        let mantissa_mask_4: __m256i = _mm256_set1_epi64x(mantissa_mask as i64);

        let mut res_ptr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut a_ptr = a.as_ptr();

        let span: usize = res.len() >> 2;

        for _ in 0..span {
            // read the next value
            let mut a: __m256d = _mm256_loadu_pd(a_ptr);
            a = _mm256_add_pd(a, add_cst_4);
            let mut ai: __m256i = _mm256_castpd_si256(a);
            ai = _mm256_and_si256(ai, mantissa_mask_4);
            ai = _mm256_sub_epi64(ai, sub_cst_4);
            // store the next value
            _mm256_storeu_si256(res_ptr, ai);

            res_ptr = res_ptr.add(1);
            a_ptr = a_ptr.add(4);
        }

        if !res.len().is_multiple_of(4) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_to_znx_i64_ref;
            reim_to_znx_i64_ref(&mut res[span << 2..], divisor, &a[span << 2..])
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim::{reim_from_znx_i64_ref, reim_to_znx_i64_ref};

    use super::*;

    /// AVX2 `reim_from_znx_i64_bnd50_fma` matches reference for bounded i64 inputs.
    #[test]
    fn reim_from_znx_i64_avx2_vs_ref() {
        let n = 64usize;
        let a: Vec<i64> = (0..n as i64).map(|i| i * 997 - 32000).collect();

        let mut res_avx = vec![0f64; n];
        let mut res_ref = vec![0f64; n];

        unsafe { reim_from_znx_i64_bnd50_fma(&mut res_avx, &a) };
        reim_from_znx_i64_ref(&mut res_ref, &a);

        assert_eq!(res_avx, res_ref, "reim_from_znx_i64: AVX2 vs ref mismatch");
    }

    /// AVX2 `reim_to_znx_i64_bnd63_avx2_fma` matches reference for exact-float inputs.
    #[test]
    fn reim_to_znx_i64_avx2_vs_ref() {
        let n = 64usize;
        let divisor = 4.0f64;
        // Exact multiples of divisor so rounding is unambiguous
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 100.0 - 3000.0) * divisor).collect();

        let mut res_avx = vec![0i64; n];
        let mut res_ref = vec![0i64; n];

        unsafe { reim_to_znx_i64_bnd63_avx2_fma(&mut res_avx, divisor, &a) };
        reim_to_znx_i64_ref(&mut res_ref, divisor, &a);

        assert_eq!(res_avx, res_ref, "reim_to_znx_i64: AVX2 vs ref mismatch");
    }
}
