// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code adapted from the AVX2 / FMA C kernels of the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The 256-bit AVX2 originals were widened to 512-bit AVX-512 and translated
// to Rust intrinsics; algorithmic structure is preserved one-to-one with the
// spqlios sources to keep semantics identical.
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
/// - **CPU features**: AVX-512F must be supported (enforced via `#[target_feature]`).
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
/// - **Vectorization**: Processes 8 elements per AVX-512 iteration.
/// - **Tail handling**: Scalar fallback for `len % 8 != 0` (negligible overhead).
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
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
/// Calling this function on incompatible CPUs results in `SIGILL`.
#[target_feature(enable = "avx512f")]
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
            __m512d, __m512i, _mm512_add_epi64, _mm512_castsi512_pd, _mm512_loadu_si512, _mm512_or_si512, _mm512_set1_epi64,
            _mm512_storeu_pd, _mm512_sub_pd,
        };

        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_512: __m512i = _mm512_castpd_si512(_mm512_set1_pd(expo));
        let add_cst_512: __m512i = _mm512_set1_epi64(add_cst);
        let sub_cst_512: __m512d = _mm512_set1_pd(sub_cst);

        // Need these imports for the additional intrinsics used above
        use std::arch::x86_64::{_mm512_castpd_si512, _mm512_set1_pd};

        let mut res_ptr: *mut f64 = res.as_mut_ptr();
        let mut a_ptr: *const i64 = a.as_ptr();

        let span: usize = n >> 3;

        for _ in 0..span {
            let mut ai64_512: __m512i = _mm512_loadu_si512(a_ptr as *const __m512i);

            ai64_512 = _mm512_add_epi64(ai64_512, add_cst_512);

            // Bitcast i64 -> f64, then OR with exponent bits.
            // _mm512_or_pd requires AVX512DQ, so we stay in the integer domain:
            // cast to int, OR, cast back to double.
            let af64_as_int: __m512i = ai64_512; // already integer
            let ored: __m512i = _mm512_or_si512(af64_as_int, expo_512);
            let mut af64_512: __m512d = _mm512_castsi512_pd(ored);

            af64_512 = _mm512_sub_pd(af64_512, sub_cst_512);

            _mm512_storeu_pd(res_ptr, af64_512);

            res_ptr = res_ptr.add(8);
            a_ptr = a_ptr.add(8);
        }

        if !res.len().is_multiple_of(8) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_from_znx_i64_ref;
            reim_from_znx_i64_ref(&mut res[span << 3..], &a[span << 3..])
        }
    }
}

/// # Correctness
/// Only ensured for inputs absolute value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim_to_znx_i64_bnd63_avx512(res: &mut [i64], divisor: f64, a: &[f64]) {
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
            __m512i, _mm512_add_pd, _mm512_and_si512, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_loadu_pd, _mm512_or_si512,
            _mm512_set1_epi64, _mm512_set1_pd, _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_si512,
            _mm512_sub_epi64, _mm512_xor_si512,
        };

        let sign_mask_512: __m512i = _mm512_set1_epi64(sign_mask as i64);
        let expo_mask_512: __m512i = _mm512_set1_epi64(expo_mask as i64);
        let mantissa_mask_512: __m512i = _mm512_set1_epi64(mantissa_mask as i64);
        let mantissa_msb_512: __m512i = _mm512_set1_epi64(mantissa_msb as i64);
        let offset_512: __m512i = _mm512_castpd_si512(_mm512_set1_pd(offset));
        let divi_bits_512: __m512i = _mm512_castpd_si512(_mm512_set1_pd(divi_bits));
        let zero_512: __m512i = _mm512_set1_epi64(0);

        let mut res_ptr: *mut i64 = res.as_mut_ptr();
        let mut a_ptr: *const f64 = a.as_ptr();

        let span: usize = res.len() >> 3;

        for _ in 0..span {
            // read the next value
            let a_int: __m512i = _mm512_castpd_si512(_mm512_loadu_pd(a_ptr));

            // a += sign(a) * m/2
            // Extract sign bits, OR with offset, then add (all in integer domain
            // to avoid _mm512_and_pd / _mm512_or_pd which need AVX512DQ).
            let asign: __m512i = _mm512_and_si512(a_int, sign_mask_512);
            let signed_offset: __m512i = _mm512_or_si512(asign, offset_512);
            let a_adj: __m512i =
                _mm512_castpd_si512(_mm512_add_pd(_mm512_castsi512_pd(a_int), _mm512_castsi512_pd(signed_offset)));

            // sign: either 0 or -1
            let sign_shift: __m512i = _mm512_srli_epi64(asign, 63);
            let sign_extended: __m512i = _mm512_sub_epi64(zero_512, sign_shift);

            // compute the exponents
            let a0exp: __m512i = _mm512_and_si512(a_adj, expo_mask_512);
            let mut a0lsh: __m512i = _mm512_sub_epi64(a0exp, divi_bits_512);
            let mut a0rsh: __m512i = _mm512_sub_epi64(divi_bits_512, a0exp);
            a0lsh = _mm512_srli_epi64(a0lsh, 52);
            a0rsh = _mm512_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m512i = _mm512_and_si512(a_adj, mantissa_mask_512);
            a0pos = _mm512_or_si512(a0pos, mantissa_msb_512);
            a0lsh = _mm512_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm512_srlv_epi64(a0pos, a0rsh);
            let mut out: __m512i = _mm512_or_si512(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm512_xor_si512(out, sign_extended);
            out = _mm512_sub_epi64(out, sign_extended);

            // stores
            _mm512_storeu_si512(res_ptr as *mut __m512i, out);

            res_ptr = res_ptr.add(8);
            a_ptr = a_ptr.add(8);
        }

        if !res.len().is_multiple_of(8) {
            use poulpy_cpu_ref::reference::fft64::reim::reim_to_znx_i64_ref;
            reim_to_znx_i64_ref(&mut res[span << 3..], divisor, &a[span << 3..])
        }
    }
}

/// # Correctness
/// Only ensured for inputs absolute value bounded by 2^63-1
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
#[target_feature(enable = "avx512f")]
pub fn reim_to_znx_i64_assign_bnd63_avx512(res: &mut [f64], divisor: f64) {
    let sign_mask: u64 = 0x8000000000000000u64;
    let expo_mask: u64 = 0x7FF0000000000000u64;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010000000000000u64;
    let divi_bits: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.;

    unsafe {
        use std::arch::x86_64::{
            __m512i, _mm512_add_pd, _mm512_and_si512, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_loadu_pd, _mm512_or_si512,
            _mm512_set1_epi64, _mm512_set1_pd, _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_si512,
            _mm512_sub_epi64, _mm512_xor_si512,
        };

        use poulpy_cpu_ref::reference::fft64::reim::reim_to_znx_i64_assign_ref;

        let sign_mask_512: __m512i = _mm512_set1_epi64(sign_mask as i64);
        let expo_mask_512: __m512i = _mm512_set1_epi64(expo_mask as i64);
        let mantissa_mask_512: __m512i = _mm512_set1_epi64(mantissa_mask as i64);
        let mantissa_msb_512: __m512i = _mm512_set1_epi64(mantissa_msb as i64);
        let offset_512: __m512i = _mm512_castpd_si512(_mm512_set1_pd(offset));
        let divi_bits_512: __m512i = _mm512_castpd_si512(_mm512_set1_pd(divi_bits));
        let zero_512: __m512i = _mm512_set1_epi64(0);

        let mut res_ptr_8xi64: *mut __m512i = res.as_mut_ptr() as *mut __m512i;
        let mut res_ptr_1xf64: *mut f64 = res.as_mut_ptr();

        let span: usize = res.len() >> 3;

        for _ in 0..span {
            // read the next value
            let a_int: __m512i = _mm512_castpd_si512(_mm512_loadu_pd(res_ptr_1xf64));

            // a += sign(a) * m/2
            let asign: __m512i = _mm512_and_si512(a_int, sign_mask_512);
            let signed_offset: __m512i = _mm512_or_si512(asign, offset_512);
            let a_adj: __m512i =
                _mm512_castpd_si512(_mm512_add_pd(_mm512_castsi512_pd(a_int), _mm512_castsi512_pd(signed_offset)));

            // sign: either 0 or -1
            let sign_shift: __m512i = _mm512_srli_epi64(asign, 63);
            let sign_extended: __m512i = _mm512_sub_epi64(zero_512, sign_shift);

            // compute the exponents
            let a0exp: __m512i = _mm512_and_si512(a_adj, expo_mask_512);
            let mut a0lsh: __m512i = _mm512_sub_epi64(a0exp, divi_bits_512);
            let mut a0rsh: __m512i = _mm512_sub_epi64(divi_bits_512, a0exp);
            a0lsh = _mm512_srli_epi64(a0lsh, 52);
            a0rsh = _mm512_srli_epi64(a0rsh, 52);

            // compute the new mantissa
            let mut a0pos: __m512i = _mm512_and_si512(a_adj, mantissa_mask_512);
            a0pos = _mm512_or_si512(a0pos, mantissa_msb_512);
            a0lsh = _mm512_sllv_epi64(a0pos, a0lsh);
            a0rsh = _mm512_srlv_epi64(a0pos, a0rsh);
            let mut out: __m512i = _mm512_or_si512(a0lsh, a0rsh);

            // negate if the sign was negative
            out = _mm512_xor_si512(out, sign_extended);
            out = _mm512_sub_epi64(out, sign_extended);

            // stores
            _mm512_storeu_si512(res_ptr_8xi64, out);

            res_ptr_8xi64 = res_ptr_8xi64.add(1); // 1 __m512i = 8 i64s = 64 bytes
            res_ptr_1xf64 = res_ptr_1xf64.add(8);
        }

        if !res.len().is_multiple_of(8) {
            reim_to_znx_i64_assign_ref(&mut res[span << 3..], divisor)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim::{reim_from_znx_i64_ref, reim_to_znx_i64_ref};

    use super::*;

    /// AVX-512 `reim_from_znx_i64_bnd50_fma` matches reference for bounded i64 inputs.
    #[test]
    fn reim_from_znx_i64_avx512_vs_ref() {
        let n = 64usize;
        let a: Vec<i64> = (0..n as i64).map(|i| i * 997 - 32000).collect();

        let mut res_avx512 = vec![0f64; n];
        let mut res_ref = vec![0f64; n];

        unsafe { reim_from_znx_i64_bnd50_fma(&mut res_avx512, &a) };
        reim_from_znx_i64_ref(&mut res_ref, &a);

        assert_eq!(res_avx512, res_ref, "reim_from_znx_i64: AVX-512 vs ref mismatch");
    }

    /// AVX-512 `reim_to_znx_i64_bnd63_avx512` matches reference for exact-float inputs.
    #[test]
    fn reim_to_znx_i64_avx512_vs_ref() {
        let n = 64usize;
        let divisor = 4.0f64;
        // Exact multiples of divisor so rounding is unambiguous
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 100.0 - 3000.0) * divisor).collect();

        let mut res_avx512 = vec![0i64; n];
        let mut res_ref = vec![0i64; n];

        unsafe { reim_to_znx_i64_bnd63_avx512(&mut res_avx512, divisor, &a) };
        reim_to_znx_i64_ref(&mut res_ref, divisor, &a);

        assert_eq!(res_avx512, res_ref, "reim_to_znx_i64: AVX-512 vs ref mismatch");
    }
}
