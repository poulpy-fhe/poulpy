//! CPU normalization kernels and optimization hooks shared by CPU backends.

use poulpy_hal::reference::znx::{ZnxExtractDigitAddMul, get_carry_i64, get_carry_i128, get_digit_i64, get_digit_i128};

/// CPU extraction hooks used by the shared normalization loops.
/// Implementations may override the scalar defaults to reduce memory passes.
/// Source and carry lengths must cover `res`; shorter slices panic before writing.
/// Arithmetic bounds are those of the HAL extraction and normalization primitives.
pub trait I64NormalizeOps: ZnxExtractDigitAddMul {
    #[inline(always)]
    fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_mul_ref(base2k, lsh, res, src);
    }

    /// Requires `1 <= res_base2k <= 62` and representable accumulated sums
    /// and centered subtractions in i64.
    #[inline(always)]
    fn znx_extract_digit_addmul_normalize<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i64],
        carry: &mut [i64],
    ) {
        znx_extract_digit_addmul_normalize_ref::<OVERWRITE>(base2k, lsh, res_base2k, res, src, carry);
    }
}

#[inline(always)]
pub fn znx_extract_digit_addmul_impl_ref<const OVERWRITE: bool>(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
    assert!(src.len() >= res.len());
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let digit: i64 = get_digit_i64(base2k, *s);
        *s = get_carry_i64(base2k, *s, digit);
        *r = if OVERWRITE { digit << lsh } else { *r + (digit << lsh) };
    }
}

#[inline(always)]
pub fn znx_extract_digit_mul_ref(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
    znx_extract_digit_addmul_impl_ref::<true>(base2k, lsh, res, src);
}

#[inline(always)]
pub fn znx_extract_digit_addmul_normalize_ref<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    res_base2k: usize,
    res: &mut [i64],
    src: &mut [i64],
    carry: &mut [i64],
) {
    assert!(src.len() >= res.len());
    assert!(carry.len() >= res.len());
    for ((r, s), c) in res.iter_mut().zip(src.iter_mut()).zip(carry.iter_mut()) {
        let digit = get_digit_i64(base2k, *s);
        *s = get_carry_i64(base2k, *s, digit);
        let partial = if OVERWRITE { digit << lsh } else { *r + (digit << lsh) };
        let sum = partial + *c;
        *r = get_digit_i64(res_base2k, sum);
        *c = get_carry_i64(res_base2k, sum, *r);
    }
}

/// Fused wide extraction and destination carry normalization.
pub fn znx_extract_digit_addmul_normalize_i128_ref<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    res_base2k: usize,
    res: &mut [i64],
    src: &mut [i128],
    carry: &mut [i128],
) {
    assert!(src.len() >= res.len());
    assert!(carry.len() >= res.len());
    for ((r, s), c) in res.iter_mut().zip(src).zip(carry) {
        let digit = get_digit_i128(base2k, *s);
        *s = get_carry_i128(base2k, *s, digit);
        let previous = if OVERWRITE { 0i64 } else { *r };
        let accum = previous.wrapping_add((digit as i64).wrapping_shl(lsh as u32)) as i128;
        let d = get_digit_i128(res_base2k, accum);
        let q = get_carry_i128(res_base2k, accum, d);
        let sum = d + *c;
        let out = get_digit_i128(res_base2k, sum);
        *r = out as i64;
        *c = q + get_carry_i128(res_base2k, sum, out);
    }
}

/// Extract a wide source digit and initialize a destination limb.
pub fn znx_extract_digit_mul_i128_ref(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
    assert!(src.len() >= res.len());
    for (r, s) in res.iter_mut().zip(src) {
        let digit = get_digit_i128(base2k, *s);
        *s = get_carry_i128(base2k, *s, digit);
        *r = (digit as i64).wrapping_shl(lsh as u32);
    }
}
