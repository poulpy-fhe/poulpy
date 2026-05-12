//! Extended-precision (`i128`) ring element vector operations for the NTT120 backend.
//!
//! This module provides standalone reference functions for [`VecZnxBig`] operations
//! when the backend's `ScalarBig` is `i128`.  Unlike the [`fft64`] backend — where
//! `ScalarBig = i64` and a `VecZnxBig` can be reinterpreted as a `VecZnx` — the
//! NTT120 backend stores `i128` values, so every operation must be implemented
//! directly on `i128` slices.
//!
//! # Layout
//!
//! A `VecZnxBig<_, NTT120Ref>` with `cols` columns, `size` limbs, and ring degree `n`
//! stores `cols × size × n` `i128` values in limb-major, column-minor order:
//!
//! ```text
//! at(col, limb) → &[i128] of length n, starting at n*(limb*cols + col)*16 bytes
//! ```
//!
//! # Functions
//!
//! - **Element-wise arithmetic**: [`ntt120_vec_znx_big_add_into`], [`ntt120_vec_znx_big_sub`],
//!   [`ntt120_vec_znx_big_negate`] and their inplace / mixed-precision variants.
//! - **Copy from small**: [`ntt120_vec_znx_big_from_small`] — sign-extend `i64` → `i128`.
//! - **Normalization**: [`ntt120_vec_znx_big_normalize`] — extract base-2k digits from
//!   `i128` limbs into `i64` `VecZnx` output.  Uses an `i128` carry buffer.
//! - **Automorphism**: [`ntt120_vec_znx_big_automorphism`] /
//!   [`ntt120_vec_znx_big_automorphism_assign`] — apply `X → X^p` on `i128` coefficients.
//! - **Gaussian noise**: [`ntt120_vec_znx_big_add_normal_ref`] — add rounded Gaussian
//!   noise into a specified limb of a `VecZnxBig`.
//!
//! [`fft64`]: crate::reference::fft64

use itertools::izip;
use rand_distr::{Distribution, Normal};

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, NoiseInfos, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{get_carry_i128, get_digit_i128},
    source::Source,
};

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers: i128-typed analogues of the znx normalize primitives
// ──────────────────────────────────────────────────────────────────────────────

/// Zero an `i128` slice.
#[inline(always)]
fn nfc_zero(x: &mut [i128]) {
    x.iter_mut().for_each(|v| *v = 0);
}

/// Add two `i128` slices in-place: `res += a`.
#[inline(always)]
fn nfc_add_assign(res: &mut [i128], a: &[i128]) {
    res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_add(ai));
}

#[inline(always)]
fn nfc_acc_assign<O: AssignOp>(res: &mut [i128], a: &[i128]) {
    if O::SUB {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_sub(ai));
    } else {
        nfc_add_assign(res, a);
    }
}

/// Multiply an `i128` slice by `2^power` in-place (positive = left shift, negative = right shift).
#[inline(always)]
fn nfc_mul_pow2_assign(power: i64, x: &mut [i128]) {
    if power > 0 {
        x.iter_mut().for_each(|xi| *xi <<= power as u32);
    } else if power < 0 {
        x.iter_mut().for_each(|xi| *xi >>= (-power) as u32);
    }
}

/// First carry-only step from the MSB `i128` limb of `a` (no prior carry).
///
/// Analogous to `znx_normalize_first_step_carry_only_ref` but for `i128`.
#[inline(always)]
fn nfc_first_carry_only(base2k: usize, lsh: usize, a: &[i128], carry: &mut [i128]) {
    debug_assert!(a.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        a.iter().zip(carry.iter_mut()).for_each(|(&ai, c)| {
            *c = get_carry_i128(base2k, ai, get_digit_i128(base2k, ai));
        });
    } else {
        let base2k_lsh = base2k - lsh;
        a.iter().zip(carry.iter_mut()).for_each(|(&ai, c)| {
            *c = get_carry_i128(base2k_lsh, ai, get_digit_i128(base2k_lsh, ai));
        });
    }
}

/// Middle carry-only step from an inner `i128` limb of `a` (adds previous carry).
///
/// Analogous to `znx_normalize_middle_step_carry_only_ref` but for `i128`.
#[inline(always)]
fn nfc_middle_carry_only(base2k: usize, lsh: usize, a: &[i128], carry: &mut [i128]) {
    debug_assert!(a.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        a.iter().zip(carry.iter_mut()).for_each(|(&ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            *c = co + get_carry_i128(base2k, d_plus_c, get_digit_i128(base2k, d_plus_c));
        });
    } else {
        let base2k_lsh = base2k - lsh;
        a.iter().zip(carry.iter_mut()).for_each(|(&ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            *c = co + get_carry_i128(base2k, d_plus_c, get_digit_i128(base2k, d_plus_c));
        });
    }
}

/// Middle normalization: convert `i128` input `a` + `i128` carry into `i64` output `res`,
/// updating carry in place.
///
/// Analogous to `znx_normalize_middle_step_ref` but with `i128` input and carry.
#[inline(always)]
#[allow(dead_code)]
fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert!(res.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

/// Compile-time selector for fused `±= normalize(a)` operations.
///
/// `AddOp` and `SubOp` are zero-sized tag types that pick `wrapping_add` vs
/// `wrapping_sub` at every leaf of the normalization pipeline.  Generic over
/// `O: AssignOp`, the cross/inter outer loops and the `I128NormalizeOps` hot
/// kernels share a single implementation for both directions; monomorphization
/// produces two specialized copies with the right sign at codegen time.
pub trait AssignOp {
    /// `true` for `SubOp`, `false` for `AddOp`. Const-foldable per monomorphization.
    const SUB: bool;
    /// `r ± x` (wrapping).
    fn apply_i64(r: i64, x: i64) -> i64;
}

/// Tag selecting `res += normalize(a)` semantics.
pub struct AddOp;
/// Tag selecting `res -= normalize(a)` semantics.
pub struct SubOp;

impl AssignOp for AddOp {
    const SUB: bool = false;
    #[inline(always)]
    fn apply_i64(r: i64, x: i64) -> i64 {
        r.wrapping_add(x)
    }
}

impl AssignOp for SubOp {
    const SUB: bool = true;
    #[inline(always)]
    fn apply_i64(r: i64, x: i64) -> i64 {
        r.wrapping_sub(x)
    }
}

#[inline(always)]
fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert!(res.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = O::apply_i64(*r, out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = O::apply_i64(*r, out as i64);
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

#[inline(always)]
fn nfc_middle_carry_assign<O: AssignOp>(base2k: usize, res: &mut [i64], carry: &mut [i128]) {
    debug_assert!(res.len() <= carry.len());
    res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
        let out = get_digit_i128(base2k, *c);
        *r = O::apply_i64(*r, out as i64);
        *c = get_carry_i128(base2k, *c, out);
    });
}

/// Middle in-place: update an existing `i64` `res` limb using `i128` carry.
///
/// Analogous to `znx_normalize_middle_step_assign_ref` but with `i128` carry.
#[inline(always)]
#[allow(dead_code)]
fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    debug_assert!(res.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            let digit = get_digit_i128(base2k, ri);
            let co = get_carry_i128(base2k, ri, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            let digit = get_digit_i128(base2k_lsh, ri);
            let co = get_carry_i128(base2k_lsh, ri, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *r = out as i64;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

/// Final in-place step: flush `i128` carry into the last `i64` res limb.
///
/// Analogous to `znx_normalize_final_step_assign_ref` but with `i128` carry.
#[inline(always)]
#[allow(dead_code)]
fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    debug_assert!(res.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            *r = get_digit_i128(base2k, get_digit_i128(base2k, ri) + *c) as i64;
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let ri = *r as i128;
            *r = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, ri) << lsh) + *c) as i64;
        });
    }
}

#[inline(always)]
fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    debug_assert!(res.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, get_digit_i128(base2k, *r as i128) + *c);
            *r = O::apply_i64(*r, out as i64);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
            let out = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, *r as i128) << lsh) + *c);
            *r = O::apply_i64(*r, out as i64);
        });
    }
}

#[inline(always)]
fn nfc_final_carry_assign<O: AssignOp>(base2k: usize, res: &mut [i64], carry: &mut [i128]) {
    debug_assert!(res.len() <= carry.len());
    res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
        *r = O::apply_i64(*r, get_digit_i128(base2k, *c) as i64);
    });
}

/// Normalize an `i128` limb into an `i128` intermediate (`a_norm`), with `i128` carry.
///
/// Used in the cross-base2k path to convert the `a` limb to the standard representation
/// before bit-extraction. Analogous to `znx_normalize_middle_step_ref` but fully `i128`.
#[inline(always)]
fn nfc_middle_step_i128(base2k: usize, lsh: usize, a_norm: &mut [i128], a: &[i128], carry: &mut [i128]) {
    debug_assert_eq!(a_norm.len(), a.len());
    debug_assert!(a.len() <= carry.len());
    debug_assert!(lsh < base2k);

    if lsh == 0 {
        izip!(a_norm.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, &ai, c)| {
            let digit = get_digit_i128(base2k, ai);
            let co = get_carry_i128(base2k, ai, digit);
            let d_plus_c = digit + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *x = out;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    } else {
        let base2k_lsh = base2k - lsh;
        izip!(a_norm.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, &ai, c)| {
            let digit = get_digit_i128(base2k_lsh, ai);
            let co = get_carry_i128(base2k_lsh, ai, digit);
            let d_plus_c = (digit << lsh) + *c;
            let out = get_digit_i128(base2k, d_plus_c);
            *x = out;
            *c = co + get_carry_i128(base2k, d_plus_c, out);
        });
    }
}

/// Extract `base2k` bits from `src` (i128) shifted by `scale`, accumulate into `res` (i64).
/// Updates `src` to hold the remaining carry.
///
/// Analogous to `znx_extract_digit_addmul_ref` but with `i128` src.
#[inline(always)]
fn nfc_extract_digit_addmul(base2k: usize, scale: usize, res: &mut [i64], src: &mut [i128]) {
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let digit = get_digit_i128(base2k, *s);
        *s = get_carry_i128(base2k, *s, digit);
        *r = r.wrapping_add((digit as i64).wrapping_shl(scale as u32));
    }
}

#[inline(always)]
fn nfc_extract_digit_assignmul<O: AssignOp>(base2k: usize, scale: usize, res: &mut [i64], src: &mut [i128]) {
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let digit = get_digit_i128(base2k, *s);
        *s = get_carry_i128(base2k, *s, digit);
        *r = O::apply_i64(*r, (digit as i64).wrapping_shl(scale as u32));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Normalization internals
// ──────────────────────────────────────────────────────────────────────────────

/// Inter-base2k normalization: same base for input (`a`) and output (`res`).
///
/// Structurally identical to `vec_znx_normalize_inter_base2k` but with `i128` input
/// and `i128` carry (the output `res` is `i64`).
#[allow(clippy::too_many_arguments)]
fn ntt120_vec_znx_big_normalize_inter<R, A, BE>(
    base2k: usize,
    res: &mut R,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_col: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();

    let (carry, _) = carry.split_at_mut(n);

    let mut lsh: i64 = res_offset % base2k as i64;
    let mut limbs_offset: i64 = res_offset / base2k as i64;

    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + base2k as i64) % (base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end: usize = (-limbs_offset).clamp(0, res_size as i64) as usize;
    let res_start: usize = (a_size as i64 - limbs_offset).clamp(0, res_size as i64) as usize;
    let a_end: usize = limbs_offset.clamp(0, a_size as i64) as usize;
    let a_start: usize = (res_size as i64 + limbs_offset).clamp(0, a_size as i64) as usize;

    let a_out_range: usize = a_size.saturating_sub(a_start);

    // Compute carry over discarded a limbs (those beyond res capacity).
    for j in 0..a_out_range {
        if j == 0 {
            nfc_first_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        } else {
            nfc_middle_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        }
    }

    if a_out_range == 0 {
        nfc_zero(carry);
    }

    // Zero bottom res limbs that will not receive a value.
    for j in res_start..res_size {
        res.at_mut(res_col, j).fill(0);
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    // Normalize overlapping a→res limbs.
    for j in 0..mid_range {
        BE::nfc_middle_step(
            base2k,
            lsh_pos,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    // Propagate carry to remaining lower res limbs (which were zeroed above).
    for j in 0..res_end {
        res.at_mut(res_col, res_end - j - 1).fill(0);
        if j == res_end - 1 {
            BE::nfc_final_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            BE::nfc_middle_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ntt120_vec_znx_big_normalize_inter_assign<O, R, A, BE>(
    base2k: usize,
    res: &mut R,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_col: usize,
    carry: &mut [i128],
) where
    O: AssignOp,
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();

    let (carry, _) = carry.split_at_mut(n);

    let mut lsh: i64 = res_offset % base2k as i64;
    let mut limbs_offset: i64 = res_offset / base2k as i64;

    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + base2k as i64) % (base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end: usize = (-limbs_offset).clamp(0, res_size as i64) as usize;
    let res_start: usize = (a_size as i64 - limbs_offset).clamp(0, res_size as i64) as usize;
    let a_end: usize = limbs_offset.clamp(0, a_size as i64) as usize;
    let a_start: usize = (res_size as i64 + limbs_offset).clamp(0, a_size as i64) as usize;

    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            nfc_first_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        } else {
            nfc_middle_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        }
    }

    if a_out_range == 0 {
        nfc_zero(carry);
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    for j in 0..mid_range {
        BE::nfc_middle_step_into::<O>(
            base2k,
            lsh_pos,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    for j in 0..res_end {
        if j == res_end - 1 {
            nfc_final_carry_assign::<O>(base2k, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            nfc_middle_carry_assign::<O>(base2k, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

/// Cross-base2k normalization: `a_base2k ≠ res_base2k`.
///
/// Structurally identical to `vec_znx_normalize_cross_base2k` but with `i128` input
/// limbs, `i128` carry buffers, and `i64` output.
#[allow(clippy::too_many_arguments)]
fn ntt120_vec_znx_big_normalize_cross<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128], // 3 * n elements
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();

    // Partition the scratch: [a_norm | res_carry | a_carry]
    let (a_norm, carry) = carry.split_at_mut(n);
    let (res_carry, a_carry) = carry[..2 * n].split_at_mut(n);
    nfc_zero(res_carry);

    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_size * res_base2k;

    let mut lsh: i64 = res_offset % a_base2k as i64;
    let mut limbs_offset: i64 = res_offset / a_base2k as i64;

    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + a_base2k as i64) % (a_base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end_bit: usize = (-limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let res_start_bit: usize = (a_tot_bits as i64 - limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let a_end_bit: usize = (limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;
    let a_start_bit: usize = (res_tot_bits as i64 + limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;

    let res_end: usize = res_end_bit / res_base2k;
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let a_end: usize = a_end_bit / a_base2k;
    let a_start: usize = a_start_bit.div_ceil(a_base2k);

    // Zero all res limbs.
    for j in 0..res_size {
        res.at_mut(res_col, j).fill(0);
    }

    if res_start == 0 {
        return;
    }

    // Compute carry over the a limbs that lie beyond the res precision range.
    let a_out_range: usize = a_size.saturating_sub(a_start);
    for j in 0..a_out_range {
        if j == 0 {
            nfc_first_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        } else {
            nfc_middle_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        }
    }
    if a_out_range == 0 {
        nfc_zero(a_carry);
    }

    let mut res_acc_left: usize = res_base2k;
    let mut res_limb: usize = res_start - 1;

    let mid_range: usize = a_start.saturating_sub(a_end);

    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;
        let a_slice: &[i128] = a.at(a_col, a_limb);

        let mut a_take_left: usize = a_base2k;

        // Normalize the j-th limb of a into a_norm (i128→i128, with i128 carry).
        nfc_middle_step_i128(a_base2k, lsh_pos, a_norm, a_slice, a_carry);

        if j == 0 {
            if !(a_tot_bits - a_start_bit).is_multiple_of(a_base2k) {
                let take: usize = (a_tot_bits - a_start_bit) % a_base2k;
                nfc_mul_pow2_assign(-(take as i64), a_norm);
                a_take_left -= take;
            } else if !(res_tot_bits - res_start_bit).is_multiple_of(res_base2k) {
                res_acc_left -= (res_tot_bits - res_start_bit) % res_base2k;
            }
        }

        'inner: loop {
            let res_slice: &mut [i64] = res.at_mut(res_col, res_limb);
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                let scale: usize = res_base2k - res_acc_left;
                nfc_extract_digit_addmul(a_take, scale, res_slice, a_norm);
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            if res_acc_left == 0 || a_limb == 0 {
                if a_limb == 0 && a_take_left == 0 {
                    nfc_add_assign(a_carry, a_norm);
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        nfc_extract_digit_addmul(res_acc_left, scale, res_slice, a_carry);
                    }
                    BE::nfc_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                    nfc_add_assign(res_carry, a_carry);
                    break 'outer;
                }

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
            }

            if a_take_left == 0 {
                nfc_add_assign(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    // Propagate carry into the lower (already-zero) res limbs.
    if res_end != 0 {
        let carry_to_use = if a_start == a_end { a_carry } else { res_carry };
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::nfc_final_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            } else {
                BE::nfc_middle_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ntt120_vec_znx_big_normalize_cross_assign<O, R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    O: AssignOp,
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();

    let (a_norm, carry) = carry.split_at_mut(n);
    let (res_carry, a_carry) = carry[..2 * n].split_at_mut(n);
    nfc_zero(res_carry);

    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_size * res_base2k;

    let mut lsh: i64 = res_offset % a_base2k as i64;
    let mut limbs_offset: i64 = res_offset / a_base2k as i64;

    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + a_base2k as i64) % (a_base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end_bit: usize = (-limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let res_start_bit: usize = (a_tot_bits as i64 - limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let a_end_bit: usize = (limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;
    let a_start_bit: usize = (res_tot_bits as i64 + limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;

    let res_end: usize = res_end_bit / res_base2k;
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let a_end: usize = a_end_bit / a_base2k;
    let a_start: usize = a_start_bit.div_ceil(a_base2k);

    if res_start == 0 {
        return;
    }

    let a_out_range: usize = a_size.saturating_sub(a_start);
    for j in 0..a_out_range {
        if j == 0 {
            nfc_first_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        } else {
            nfc_middle_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        }
    }
    if a_out_range == 0 {
        nfc_zero(a_carry);
    }

    let mut res_acc_left: usize = res_base2k;
    let mut res_limb: usize = res_start - 1;
    let mid_range: usize = a_start.saturating_sub(a_end);

    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;
        let a_slice: &[i128] = a.at(a_col, a_limb);

        let mut a_take_left: usize = a_base2k;
        nfc_middle_step_i128(a_base2k, lsh_pos, a_norm, a_slice, a_carry);

        if j == 0 {
            if !(a_tot_bits - a_start_bit).is_multiple_of(a_base2k) {
                let take: usize = (a_tot_bits - a_start_bit) % a_base2k;
                nfc_mul_pow2_assign(-(take as i64), a_norm);
                a_take_left -= take;
            } else if !(res_tot_bits - res_start_bit).is_multiple_of(res_base2k) {
                res_acc_left -= (res_tot_bits - res_start_bit) % res_base2k;
            }
        }

        'inner: loop {
            let res_slice: &mut [i64] = res.at_mut(res_col, res_limb);
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                let scale: usize = res_base2k - res_acc_left;
                nfc_extract_digit_assignmul::<O>(a_take, scale, res_slice, a_norm);
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            if res_acc_left == 0 || a_limb == 0 {
                if a_limb == 0 && a_take_left == 0 {
                    nfc_add_assign(a_carry, a_norm);
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        nfc_extract_digit_assignmul::<O>(res_acc_left, scale, res_slice, a_carry);
                    }
                    BE::nfc_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                    nfc_acc_assign::<O>(res_carry, a_carry);
                    break 'outer;
                }

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
            }

            if a_take_left == 0 {
                nfc_add_assign(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    if res_end != 0 {
        let carry_to_use = if a_start == a_end { a_carry } else { res_carry };
        for j in 0..res_end {
            if j == res_end - 1 {
                nfc_final_carry_assign::<O>(res_base2k, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            } else {
                nfc_middle_carry_assign::<O>(res_base2k, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice i128 arithmetic dispatch trait
// ──────────────────────────────────────────────────────────────────────────────

/// Per-slice `i128` arithmetic kernels, dispatched via the backend type parameter.
///
/// All methods have scalar default implementations.  `NTT120Ref` and other
/// scalar backends implement this trait with an empty body to use the defaults.
/// `NTT120Avx` (or any future SIMD backend) overrides the methods it can
/// accelerate; the outer loop logic in `ntt120_vec_znx_big_*` is unaffected.
///
/// This is the `i128`-equivalent of the `NttAdd` / `NttSub` / … dispatch traits
/// for q120b (NTT-domain) element operations.
pub trait I128BigOps {
    /// `res[i] = a[i].wrapping_add(b[i])` for each `i`.
    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_add(bi));
    }
    /// `res[i] = res[i].wrapping_add(a[i])` for each `i`.
    #[inline(always)]
    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_add(ai));
    }
    /// `res[i] = a[i].wrapping_add(b[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_add(bi as i128));
    }
    /// `res[i] = res[i].wrapping_add(a[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai as i128));
    }
    /// `res[i] = a[i].wrapping_sub(b[i])` for each `i`.
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_sub(bi));
    }
    /// `res[i] = res[i].wrapping_sub(a[i])` for each `i`.
    #[inline(always)]
    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_sub(ai));
    }
    /// `res[i] = a[i].wrapping_sub(res[i])` for each `i`.
    #[inline(always)]
    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai.wrapping_sub(*r));
    }
    /// `res[i] = (a[i] as i128).wrapping_sub(b[i])` for each `i`.
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = (ai as i128).wrapping_sub(bi));
    }
    /// `res[i] = a[i].wrapping_sub(b[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_sub(bi as i128));
    }
    /// `res[i] = res[i].wrapping_sub(a[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai as i128));
    }
    /// `res[i] = (a[i] as i128).wrapping_sub(res[i])` for each `i`.
    #[inline(always)]
    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = (ai as i128).wrapping_sub(*r));
    }
    /// `res[i] = a[i].wrapping_neg()` for each `i`.
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai.wrapping_neg());
    }
    /// `res[i] = res[i].wrapping_neg()` for each `i`.
    #[inline(always)]
    fn i128_negate_assign(res: &mut [i128]) {
        res.iter_mut().for_each(|r| *r = r.wrapping_neg());
    }
    /// `res[i] = -(a[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = -(ai as i128));
    }
    /// `res[i] = a[i] as i128` for each `i`.
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai as i128);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Normalization hot-path dispatch trait
// ──────────────────────────────────────────────────────────────────────────────

/// Per-slice `i128→i64` normalization kernels, dispatched via the backend type parameter.
///
/// The three hot-path helpers used inside [`ntt120_vec_znx_big_normalize`] are expressed
/// as trait methods so that SIMD backends can override them without duplicating the outer
/// loop logic.  All methods have scalar default implementations.
///
/// This is the normalization-specific counterpart of [`I128BigOps`].
pub trait I128NormalizeOps {
    /// Convert `i128` input + carry into `i64` output, updating carry in place.
    ///
    /// Equivalent to the private `nfc_middle_step` helper.
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if lsh == 0 {
            izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
                let digit = get_digit_i128(base2k, ai);
                let co = get_carry_i128(base2k, ai, digit);
                let d_plus_c = digit + *c;
                let out = get_digit_i128(base2k, d_plus_c);
                *r = out as i64;
                *c = co + get_carry_i128(base2k, d_plus_c, out);
            });
        } else {
            let base2k_lsh = base2k - lsh;
            izip!(res.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(r, &ai, c)| {
                let digit = get_digit_i128(base2k_lsh, ai);
                let co = get_carry_i128(base2k_lsh, ai, digit);
                let d_plus_c = (digit << lsh) + *c;
                let out = get_digit_i128(base2k, d_plus_c);
                *r = out as i64;
                *c = co + get_carry_i128(base2k, d_plus_c, out);
            });
        }
    }

    /// Fused middle step for `res ±= normalize(a)`.  `O = AddOp` adds; `O = SubOp` subtracts.
    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        nfc_middle_step_into::<O>(base2k, lsh, res, a, carry);
    }

    /// Update an existing `i64` res limb using `i128` carry, updating carry in place.
    ///
    /// Equivalent to the private `nfc_middle_step_assign` helper.
    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if lsh == 0 {
            res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
                let ri = *r as i128;
                let digit = get_digit_i128(base2k, ri);
                let co = get_carry_i128(base2k, ri, digit);
                let d_plus_c = digit + *c;
                let out = get_digit_i128(base2k, d_plus_c);
                *r = out as i64;
                *c = co + get_carry_i128(base2k, d_plus_c, out);
            });
        } else {
            let base2k_lsh = base2k - lsh;
            res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
                let ri = *r as i128;
                let digit = get_digit_i128(base2k_lsh, ri);
                let co = get_carry_i128(base2k_lsh, ri, digit);
                let d_plus_c = (digit << lsh) + *c;
                let out = get_digit_i128(base2k, d_plus_c);
                *r = out as i64;
                *c = co + get_carry_i128(base2k, d_plus_c, out);
            });
        }
    }

    /// Flush `i128` carry into the last `i64` res limb.
    ///
    /// Equivalent to the private `nfc_final_step_assign` helper.
    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if lsh == 0 {
            res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
                let ri = *r as i128;
                *r = get_digit_i128(base2k, get_digit_i128(base2k, ri) + *c) as i64;
            });
        } else {
            let base2k_lsh = base2k - lsh;
            res.iter_mut().zip(carry.iter_mut()).for_each(|(r, c)| {
                let ri = *r as i128;
                *r = get_digit_i128(base2k, (get_digit_i128(base2k_lsh, ri) << lsh) + *c) as i64;
            });
        }
    }

    /// Fused final step for `res ±= normalize(a)`.  `O = AddOp` adds; `O = SubOp` subtracts.
    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_final_step_into::<O>(base2k, lsh, res, carry);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the scratch size (in bytes) required by [`ntt120_vec_znx_big_normalize`].
pub fn ntt120_vec_znx_big_normalize_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<i128>()
}

/// Returns the scratch size (in bytes) required by
/// [`ntt120_vec_znx_big_automorphism_assign`].
pub fn ntt120_vec_znx_big_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i128>()
}

/// Element-wise addition: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Limbs present in both `a` and `b` are summed; limbs present in only one are copied;
/// extra res limbs beyond both are zeroed.
pub fn ntt120_vec_znx_big_add_into<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);

    for j in 0..sum_size {
        BE::i128_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }

    if a_size <= b_size {
        let b_cpy = b_size.min(res_size);
        for j in sum_size..b_cpy {
            let bj = b.at(b_col, j);
            res.at_mut(res_col, j).copy_from_slice(bj);
        }
        for j in b_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    } else {
        let a_cpy = a_size.min(res_size);
        for j in sum_size..a_cpy {
            let aj = a.at(a_col, j);
            res.at_mut(res_col, j).copy_from_slice(aj);
        }
        for j in a_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }
}

/// In-place addition: `res[res_col] += a[a_col]` over the first `min(res.size(), a.size())`
/// limbs.
pub fn ntt120_vec_znx_big_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Add a small (`i64`) polynomial `b` to a big (`i128`) polynomial `a`:
/// `res[res_col] = a[a_col] + b[b_col]`.
pub fn ntt120_vec_znx_big_add_small_into<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);
    let a_cpy = a_size.min(res_size);
    let b_cpy = b_size.min(res_size);

    for j in 0..sum_size {
        BE::i128_add_small(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }
    for j in sum_size..a_cpy {
        res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
    }
    for j in a_cpy..b_cpy {
        BE::i128_from_small(res.at_mut(res_col, j), b.at(b_col, j));
    }
    for j in a_cpy.max(b_cpy)..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// In-place: `res[res_col] += a[a_col]` where `a` is a `VecZnx` (i64 limbs).
pub fn ntt120_vec_znx_big_add_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Subtraction: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt120_vec_znx_big_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);

    for j in 0..sum_size {
        BE::i128_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }

    if a_size >= b_size {
        let a_cpy = a_size.min(res_size);
        for j in sum_size..a_cpy {
            res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
        }
        for j in a_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    } else {
        let b_cpy = b_size.min(res_size);
        for j in sum_size..b_cpy {
            BE::i128_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }
        for j in b_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }
}

/// In-place subtraction: `res[res_col] -= a[a_col]`.
pub fn ntt120_vec_znx_big_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Swap-subtract in-place: `res[res_col] = a[a_col] - res[res_col]`.
pub fn ntt120_vec_znx_big_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let sum_size = res_size.min(a.size());

    for j in 0..sum_size {
        BE::i128_sub_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in a.size()..res_size {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

/// `res = a - b` where `a` is `VecZnx` (i64) and `b` is `VecZnxBig` (i128).
pub fn ntt120_vec_znx_big_sub_small_a<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    B: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);
    let a_cpy = a_size.min(res_size);
    let b_cpy = b_size.min(res_size);

    for j in 0..sum_size {
        BE::i128_sub_small_a(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }
    for j in sum_size..a_cpy {
        BE::i128_from_small(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in sum_size..b_cpy {
        if j >= a_cpy {
            BE::i128_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }
    }
    for j in a_cpy.max(b_cpy)..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// `res = a - b` where `a` is `VecZnxBig` (i128) and `b` is `VecZnx` (i64).
pub fn ntt120_vec_znx_big_sub_small_b<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);
    let a_cpy = a_size.min(res_size);
    let b_cpy = b_size.min(res_size);

    for j in 0..sum_size {
        BE::i128_sub_small_b(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }
    for j in sum_size..a_cpy {
        res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
    }
    for j in a_cpy..b_cpy {
        BE::i128_neg_from_small(res.at_mut(res_col, j), b.at(b_col, j));
    }
    for j in a_cpy.max(b_cpy)..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// In-place: `res[res_col] -= a[a_col]` where `a` is a `VecZnx` (i64).
pub fn ntt120_vec_znx_big_sub_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// In-place: `res[res_col] = a[a_col] - res[res_col]` where `a` is a `VecZnx` (i64).
pub fn ntt120_vec_znx_big_sub_small_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let sum_size = res_size.min(a.size());

    for j in 0..sum_size {
        BE::i128_sub_small_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in a.size()..res_size {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

/// Negate: `res[res_col] = -a[a_col]`.
pub fn ntt120_vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let res_size = res.size();
    let cpy_size = a.size().min(res_size);

    for j in 0..cpy_size {
        BE::i128_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in cpy_size..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// In-place negation: `res[res_col] = -res[res_col]`.
pub fn ntt120_vec_znx_big_negate_assign<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    for j in 0..res.size() {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

/// Sign-extend `i64` coefficients from `a[a_col]` into `i128` limbs of `res[res_col]`.
///
/// Limbs beyond `a.size()` are zeroed.
pub fn ntt120_vec_znx_big_from_small<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for j in 0..min_size {
        BE::i128_from_small(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// Normalize `a[a_col]` (i128 `VecZnxBig`) into `res[res_col]` (i64 `VecZnx`).
///
/// Extracts `res.size()` signed base-2^`res_base2k` digits from the extended-precision
/// accumulator in `a`, applying a bit-level `res_offset` shift before decomposition.
///
/// `carry` must have at least `ntt120_vec_znx_big_normalize_tmp_bytes(n) / size_of::<i128>()`
/// elements (i.e., `3 * n` `i128` values).
#[allow(clippy::too_many_arguments)]
pub fn ntt120_vec_znx_big_normalize<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    if res_base2k == a_base2k {
        ntt120_vec_znx_big_normalize_inter(res_base2k, res, res_offset, res_col, a, a_col, carry);
    } else {
        ntt120_vec_znx_big_normalize_cross(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ntt120_vec_znx_big_normalize_assign<O, R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    O: AssignOp,
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    if res_base2k == a_base2k {
        ntt120_vec_znx_big_normalize_inter_assign::<O, _, _, _>(res_base2k, res, res_offset, res_col, a, a_col, carry);
    } else {
        ntt120_vec_znx_big_normalize_cross_assign::<O, _, _, _>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ntt120_vec_znx_big_normalize_add_assign<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    ntt120_vec_znx_big_normalize_assign::<AddOp, _, _, _>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
}

#[allow(clippy::too_many_arguments)]
pub fn ntt120_vec_znx_big_normalize_sub_assign<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    ntt120_vec_znx_big_normalize_assign::<SubOp, _, _, _>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
}

/// Apply the Galois automorphism `X → X^p` to `a[a_col]`, writing to `res[res_col]`.
///
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt120_vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i128>,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let n = res.n();
    let size = res.size().min(a.size());
    let mask = 2 * n - 1;
    let p_2n = (p & mask as i64) as usize;

    for limb in 0..size {
        let rj = res.at_mut(res_col, limb);
        let aj = a.at(a_col, limb);
        rj[0] = aj[0];
        let mut k: usize = 0;
        for &ai in &aj[1..] {
            k = (k + p_2n) & mask;
            if k < n {
                rj[k] = ai;
            } else {
                rj[k - n] = ai.wrapping_neg();
            }
        }
    }

    for limb in size..res.size() {
        res.at_mut(res_col, limb).iter_mut().for_each(|r| *r = 0);
    }
}

/// Apply `X → X^p` in-place to `res[res_col]`.
///
/// `tmp` must have at least `ntt120_vec_znx_big_automorphism_assign_tmp_bytes(n) / 16`
/// elements (i.e., `n` `i128` values).
pub fn ntt120_vec_znx_big_automorphism_assign<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i128])
where
    BE: Backend<ScalarBig = i128>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let n = res.n();
    let size = res.size();
    let mask = 2 * n - 1;
    let p_2n = (p & mask as i64) as usize;

    for limb in 0..size {
        let rj = res.at_mut(res_col, limb);
        tmp[..n].copy_from_slice(rj);
        rj[0] = tmp[0];
        let mut k: usize = 0;
        for &ti in &tmp[1..n] {
            k = (k + p_2n) & mask;
            if k < n {
                rj[k] = ti;
            } else {
                rj[k - n] = ti.wrapping_neg();
            }
        }
    }
}

/// Add rounded Gaussian noise `N(0, σ²)` into the limb of `res[res_col]` that
/// holds the precision bits around level `k` in base `2^base2k`.
///
/// # Panics
///
/// Panics if `ceil(log2(bound)) >= 64`.
pub fn ntt120_vec_znx_big_add_normal_ref<R, BE>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    noise_infos: NoiseInfos,
    source: &mut Source,
) where
    BE: Backend<ScalarBig = i128>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        noise_infos.bound.log2().ceil() as i64
    );

    let (limb, scale) = noise_infos.target_limb_and_scale(base2k);
    let scaled_sigma = noise_infos.sigma * scale;
    let scaled_bound = noise_infos.bound * scale;

    let normal: Normal<f64> = Normal::new(0.0, scaled_sigma).unwrap();
    let rj: &mut [i128] = res.at_mut(res_col, limb);

    rj.iter_mut().for_each(|r| {
        let mut s: f64 = normal.sample(source);
        while s.abs() > scaled_bound {
            s = normal.sample(source);
        }
        *r = r.wrapping_add(s.round() as i64 as i128);
    });
}
