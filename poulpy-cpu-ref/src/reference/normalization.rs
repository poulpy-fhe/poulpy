//! CPU normalization kernels and optimization hooks shared by CPU backends.

use poulpy_hal::reference::znx::{ZnxExtractDigitAddMul, get_carry_i64, get_carry_i128, get_digit_i64, get_digit_i128};

/// CPU extraction hooks used by the shared normalization loops.
/// Implementations may override the scalar defaults to reduce memory passes.
/// Source and carry lengths must cover `res`; shorter slices panic before writing.
/// Arithmetic bounds are those of the HAL extraction and normalization primitives.
pub trait I64NormalizeOps: ZnxExtractDigitAddMul {
    /// Floor-carry and boundary operations use the bounds of the corresponding reference kernels.
    #[inline(always)]
    fn znx_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i64], carry: &mut [i64]) {
        znx_normalize_floor_ref::<CARRY_IN, ROUND>(base2k, lsh, a, carry);
    }

    #[inline(always)]
    fn znx_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i64],
        carry: &mut [i64],
    ) {
        znx_normalize_round_ref::<CARRY_IN, PAD>(base2k, lsh, padding, res, a, carry);
    }

    #[inline(always)]
    fn znx_normalize_round_assign<const CARRY_IN: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        carry: &mut [i64],
    ) {
        znx_normalize_round_assign_ref::<CARRY_IN>(base2k, lsh, padding, res, carry);
    }

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

struct NormalizationBoundary {
    base2k: usize,
    lsh: usize,
    padding: usize,
    width: usize,
    mask: u64,
    source_mask: u64,
    half: u64,
}

impl NormalizationBoundary {
    fn new(base2k: usize, lsh: usize, padding: usize) -> Self {
        assert!((1..=63).contains(&base2k) && lsh < base2k && padding < base2k);
        Self {
            base2k,
            lsh,
            padding,
            width: base2k - padding,
            mask: (1u64 << base2k) - 1,
            source_mask: (1u64 << (base2k - lsh)) - 1,
            half: if padding == 0 { 0 } else { 1u64 << (padding - 1) },
        }
    }

    #[inline(always)]
    fn floor(&self, a: i64, carry: i64) -> (u64, i64) {
        let low = ((a as u64 & self.source_mask) << self.lsh) + (carry as u64 & self.mask);
        let high = (a >> (self.base2k - self.lsh)) + (carry >> self.base2k) + (low >> self.base2k) as i64;
        (low & self.mask, high)
    }

    #[inline(always)]
    fn round(&self, a: i64, carry: i64) -> (i64, i64) {
        let (low, high) = self.floor(a, carry);
        let rounded = (low + self.half) >> self.padding;
        let digit = ((rounded << (64 - self.width)) as i64) >> (64 - self.width);
        (
            digit << self.padding,
            high + (rounded.wrapping_sub(digit as u64) >> self.width) as i64,
        )
    }
}

// Decompose a shifted source and incoming floor carry without shifting the signed word.
#[inline(always)]
fn normalization_floor(base2k: usize, lsh: usize, a: i128, carry: i128) -> (u64, i128) {
    let mask = (1u64 << base2k) - 1;
    let source_bits = base2k - lsh;
    let low = ((a as u64 & ((1u64 << source_bits) - 1)) << lsh) + (carry as u64 & mask);
    let high = (a >> source_bits) + (carry >> base2k);
    (low & mask, high + (low >> base2k) as i128)
}

/// Floor the shifted source plus carry; `ROUND` retains the top discarded bit.
/// Radices are in `1..=63`, `lsh < base2k`, and input/carry magnitudes are at most `2^62`.
pub fn znx_normalize_floor_ref<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i64], carry: &mut [i64]) {
    assert!(a.len() >= carry.len());
    let step = NormalizationBoundary::new(base2k, lsh, 0);
    for (&source, carry) in a.iter().zip(carry.iter_mut()) {
        let (low, high) = step.floor(source, if CARRY_IN { *carry } else { 0 });
        *carry = high + if ROUND { (low >> (base2k - 1)) as i64 } else { 0 };
    }
}

/// Round at `padding` low bits, center the retained digit and update its carry.
/// `PAD` restores the cleared low bits; source and carry must cover `res`.
pub fn znx_normalize_round_ref<const CARRY_IN: bool, const PAD: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    assert!(a.len() >= res.len() && carry.len() >= res.len());
    let step = NormalizationBoundary::new(base2k, lsh, padding);
    for ((out, &source), carry) in res.iter_mut().zip(a).zip(carry.iter_mut()) {
        let (digit, high) = step.round(source, if CARRY_IN { *carry } else { 0 });
        *out = if PAD { digit } else { digit >> padding };
        *carry = high;
    }
}

/// In-place form of `znx_normalize_round_ref` with padded output.
pub fn znx_normalize_round_assign_ref<const CARRY_IN: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    carry: &mut [i64],
) {
    assert!(carry.len() >= res.len());
    let step = NormalizationBoundary::new(base2k, lsh, padding);
    for (out, carry) in res.iter_mut().zip(carry.iter_mut()) {
        (*out, *carry) = step.round(*out, if CARRY_IN { *carry } else { 0 });
    }
}

/// Wide floor-carry step; input and carry magnitudes are at most `2^126`.
/// Radices are in `1..=63`, `lsh < base2k`; source covers carry.
pub fn nfc_normalize_floor_ref<const CARRY_IN: bool, const ROUND: bool>(
    base2k: usize,
    lsh: usize,
    a: &[i128],
    carry: &mut [i128],
) {
    assert!(a.len() >= carry.len());
    assert!((1..=63).contains(&base2k) && lsh < base2k);
    for (&source, carry) in a.iter().zip(carry.iter_mut()) {
        let (low, high) = normalization_floor(base2k, lsh, source, if CARRY_IN { *carry } else { 0 });
        *carry = high + if ROUND { (low >> (base2k - 1)) as i128 } else { 0 };
    }
}

/// Wide boundary rounding with the same digit contract as `znx_normalize_round_ref`.
pub fn nfc_normalize_round_ref<const CARRY_IN: bool, const PAD: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    a: &[i128],
    carry: &mut [i128],
) {
    assert!(a.len() >= res.len() && carry.len() >= res.len());
    let step = NormalizationBoundary::new(base2k, lsh, padding);
    for ((out, &source), carry) in res.iter_mut().zip(a).zip(carry.iter_mut()) {
        let (low, high) = normalization_floor(base2k, lsh, source, if CARRY_IN { *carry } else { 0 });
        let rounded = (low + step.half) >> padding;
        let digit = ((rounded << (64 - step.width)) as i64) >> (64 - step.width);
        *out = if PAD { digit << padding } else { digit };
        *carry = high + ((rounded as i128 - digit as i128) >> step.width);
    }
}

/// Extract a wide source digit and add its shifted value to a destination limb.
pub fn znx_extract_digit_addmul_i128_ref(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
    assert!(src.len() >= res.len());
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let digit = get_digit_i128(base2k, *s);
        *s = get_carry_i128(base2k, *s, digit);
        *r = r.wrapping_add((digit as i64).wrapping_shl(lsh as u32));
    }
}
