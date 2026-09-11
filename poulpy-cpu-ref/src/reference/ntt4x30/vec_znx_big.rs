//! Extended-precision (`i128`) ring element vector operations for the NTT4x30 backend.
//!
//! This module provides standalone reference functions for [`VecZnxBig`] operations
//! when the backend's `BigWord` is `i128`.  Unlike the [`fft64`] backend — where
//! `BigWord = i64` and a `VecZnxBig` can be reinterpreted as a `VecZnx` — the
//! NTT4x30 backend stores `i128` values, so every operation must be implemented
//! directly on `i128` slices.
//!
//! # Layout
//!
//! A `VecZnxBig<_, NTT4x30Ref>` with `cols` columns, `size` limbs, and ring degree `n`
//! stores `cols × size × n` `i128` values in limb-major, column-minor order:
//!
//! ```text
//! at(col, limb) → &[i128] of length n, starting at n*(limb*cols + col)*16 bytes
//! ```
//!
//! # Functions
//!
//! - **Element-wise arithmetic**: [`ntt4x30_vec_znx_big_add`], [`ntt4x30_vec_znx_big_sub`],
//!   [`ntt4x30_vec_znx_big_negate`] and their inplace / mixed-precision variants.
//! - **Copy from small**: [`ntt4x30_vec_znx_big_from_small`] — sign-extend `i64` → `i128`.
//! - **Normalization**: [`ntt4x30_vec_znx_big_normalize`] — extract base-2k digits from
//!   `i128` limbs into `i64` `VecZnx` output.  Uses an `i128` carry buffer.
//! - **Automorphism**: [`ntt4x30_vec_znx_big_automorphism`] /
//!   [`ntt4x30_vec_znx_big_automorphism_assign`] — apply `X → X^p` on `i128` coefficients.
//! - **Gaussian noise**: [`ntt4x30_vec_znx_big_add_normal_ref`] — add rounded Gaussian
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
    reference::{
        normalization::I64NormalizeOps,
        vec_znx::VecZnxRangeMut,
        znx::{
            ZnxNormalizeMiddleStepAssign, get_carry_i128, get_digit_i128, znx_extract_digit_addmul_normalize_i128_ref,
            znx_extract_digit_mul_i128_ref,
        },
    },
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

/// Middle normalization: convert `i128` input `a` + `i128` carry into `i64` output `res`,
/// updating carry in place.
///
/// Analogous to `znx_normalize_middle_step_ref` but with `i128` input and carry.
#[inline(always)]
#[allow(dead_code)]
fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    assert_eq!(res.len(), a.len());
    assert!(res.len() <= carry.len());
    assert!(lsh < base2k);

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
    assert_eq!(res.len(), a.len());
    assert!(res.len() <= carry.len());
    assert!(lsh < base2k);

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

/// Middle in-place: update an existing `i64` `res` limb using `i128` carry.
///
/// Analogous to `znx_normalize_middle_step_assign_ref` but with `i128` carry.
#[inline(always)]
#[allow(dead_code)]
fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    assert!(res.len() <= carry.len());
    assert!(lsh < base2k);

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
    assert!(res.len() <= carry.len());
    assert!(lsh < base2k);

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
    assert!(res.len() <= carry.len());
    assert!(lsh < base2k);

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

// ──────────────────────────────────────────────────────────────────────────────
// Normalization internals
// ──────────────────────────────────────────────────────────────────────────────

/// Inter-base2k normalization: same base for input (`a`) and output (`res`).
///
/// Structurally identical to `vec_znx_normalize_inter_base2k` but with `i128` input
/// and `i128` carry (the output `res` is `i64`).
#[allow(clippy::too_many_arguments)]
fn ntt4x30_vec_znx_big_normalize_inter<A, BE>(
    base2k: usize,
    res: &mut VecZnxRangeMut<'_>,
    res_size: usize,
    res_offset: i64,
    a: &A,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i128],
) where
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let a = a.to_backend_ref();

    let (lo, hi) = (coeff_start, coeff_start + coeff_len);
    let a_size = a.size();

    let (carry, _) = carry.split_at_mut(coeff_len);

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

    // Floor discarded limbs and retain the rounding bit only at the cutoff.
    for j in 0..a_out_range {
        let source = &a.at(a_col, a_size - j - 1)[lo..hi];
        match (j == 0, j + 1 == a_out_range) {
            (true, false) => BE::nfc_normalize_floor::<false, false>(base2k, lsh_pos, source, carry),
            (true, true) => BE::nfc_normalize_floor::<false, true>(base2k, lsh_pos, source, carry),
            (false, false) => BE::nfc_normalize_floor::<true, false>(base2k, lsh_pos, source, carry),
            (false, true) => BE::nfc_normalize_floor::<true, true>(base2k, lsh_pos, source, carry),
        }
    }

    if a_out_range == 0 {
        nfc_zero(carry);
    }

    // Zero bottom res limbs that will not receive a value.
    for j in res_start..res_size {
        res.at_mut(j).fill(0);
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    // Normalize overlapping a→res limbs.
    for j in 0..mid_range {
        let res_limb = res_start - j - 1;
        BE::nfc_middle_step(
            base2k,
            lsh_pos,
            res.at_mut(res_limb),
            &a.at(a_col, a_start - j - 1)[lo..hi],
            carry,
        );
    }

    for j in (0..res_end).rev() {
        BE::znx_extract_digit_mul_i128(base2k, 0, res.at_mut(j), carry);
    }
}

/// Cross-base2k normalization: `a_base2k ≠ res_base2k`.
///
/// Structurally identical to `vec_znx_normalize_cross_base2k` but with `i128` input
/// limbs and source carry, and `i64` normalized digits, output carry and output.
#[allow(clippy::too_many_arguments)]
fn ntt4x30_vec_znx_big_normalize_cross<A, BE>(
    res: &mut VecZnxRangeMut<'_>,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i128], // 3 * coeff_len elements
) where
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let a = a.to_backend_ref();

    let (lo, hi) = (coeff_start, coeff_start + coeff_len);
    let a_size = a.size();

    // Normalized source digits and output carries fit in i64; source carries stay wide.
    let (small, carry) = carry.split_at_mut(coeff_len);
    let (a_norm, res_carry) = bytemuck::cast_slice_mut::<i128, i64>(small).split_at_mut(coeff_len);
    let a_carry = &mut carry[..coeff_len];
    res_carry.fill(0);

    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits = res_k;
    let drops_precision = a_tot_bits as i128 > res_k as i128 + res_offset as i128;

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

    for j in res_start..res_size {
        res.at_mut(j).fill(0);
    }

    if res_start == 0 {
        return;
    }

    // Floor the discarded limbs; only the boundary contributes rounding.
    let a_out_range: usize = a_size.saturating_sub(a_start);
    let take = (a_tot_bits - a_start_bit) % a_base2k;
    if a_out_range != 0 {
        for j in (a_start..a_size).rev() {
            let source = &a.at(a_col, j)[lo..hi];
            match (j + 1 == a_size, take == 0 && j == a_start) {
                (true, false) => BE::nfc_normalize_floor::<false, false>(a_base2k, lsh_pos, source, a_carry),
                (true, true) => BE::nfc_normalize_floor::<false, true>(a_base2k, lsh_pos, source, a_carry),
                (false, false) => BE::nfc_normalize_floor::<true, false>(a_base2k, lsh_pos, source, a_carry),
                (false, true) => BE::nfc_normalize_floor::<true, true>(a_base2k, lsh_pos, source, a_carry),
            }
        }
    } else if !drops_precision && take == 0 {
        nfc_zero(a_carry);
    }

    let mut res_acc_left = res_start_bit - (res_start - 1) * res_base2k;
    let mut res_limb: usize = res_start - 1;
    let mut initialized = false;

    let mid_range: usize = a_start.saturating_sub(a_end);

    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;
        let a_slice: &[i128] = &a.at(a_col, a_limb)[lo..hi];

        let mut a_take_left: usize = a_base2k;

        if j == 0 && (drops_precision || take != 0) {
            if a_out_range != 0 {
                BE::nfc_normalize_round::<true, false>(a_base2k, lsh_pos, take, a_norm, a_slice, a_carry);
            } else {
                BE::nfc_normalize_round::<false, false>(a_base2k, lsh_pos, take, a_norm, a_slice, a_carry);
            }
            a_take_left -= take;
        } else {
            BE::nfc_middle_step(a_base2k, lsh_pos, a_norm, a_slice, a_carry);
        }

        'inner: loop {
            let res_slice = res.at_mut(res_limb);
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            let flushed = a_take != 0 && a_take == res_acc_left;
            if a_take != 0 {
                let scale: usize = res_base2k - res_acc_left;
                if flushed {
                    if initialized {
                        BE::znx_extract_digit_addmul_normalize::<false>(a_take, scale, res_base2k, res_slice, a_norm, res_carry);
                    } else {
                        BE::znx_extract_digit_addmul_normalize::<true>(a_take, scale, res_base2k, res_slice, a_norm, res_carry);
                    }
                } else if initialized {
                    BE::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);
                } else {
                    BE::znx_extract_digit_mul(a_take, scale, res_slice, a_norm);
                }
                initialized = true;
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            if res_acc_left == 0 || a_limb == 0 {
                if a_limb == 0 && a_take_left == 0 {
                    BE::nfc_add_small_carry(a_carry, a_norm);
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        BE::znx_extract_digit_addmul_i128(res_acc_left, scale, res_slice, a_carry);
                    }
                    if !flushed {
                        BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                    }
                    BE::nfc_add_small_carry(a_carry, res_carry);

                    break 'outer;
                }

                if !flushed {
                    BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                }

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
                initialized = false;
            }

            if a_take_left == 0 {
                BE::nfc_add_small_carry(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    // Write the remaining more significant digits.
    if res_end != 0 {
        for j in (0..res_end).rev() {
            BE::znx_extract_digit_mul_i128(res_base2k, 0, res.at_mut(j), a_carry);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice i128 arithmetic dispatch trait
// ──────────────────────────────────────────────────────────────────────────────

/// Per-slice `i128` arithmetic kernels, dispatched via the backend type parameter.
///
/// All methods have scalar default implementations.  `NTT4x30Ref` and other
/// scalar backends implement this trait with an empty body to use the defaults.
/// `NTT4x30Avx` (or any future SIMD backend) overrides the methods it can
/// accelerate; the outer loop logic in `ntt4x30_vec_znx_big_*` is unaffected.
///
/// This is the `i128`-equivalent of the `NttAdd` / `NttSub` / … dispatch traits
/// for q120b (NTT-domain) element operations.
pub trait I128BigOps {
    /// `res[i] = (a[i] as i128).wrapping_mul(b[i] as i128)` for each `i`.
    #[inline(always)]
    fn i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = (ai as i128).wrapping_mul(bi as i128));
    }

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
/// The hot-path helpers used inside [`ntt4x30_vec_znx_big_normalize`] are expressed
/// as trait methods so that SIMD backends can override them without duplicating the outer
/// loop logic.  All methods have scalar default implementations.
///
/// This is the normalization-specific counterpart of [`I128BigOps`].
/// Input limbs and incoming carries must have magnitude at most `2^126`;
/// shifted digits and destination sums must be representable. These bounds
/// include the centered NTT4x30 IDFT output, whose magnitude is below `2^119`.
pub trait I128NormalizeOps: I64NormalizeOps + ZnxNormalizeMiddleStepAssign {
    #[inline(always)]
    fn nfc_add_small_carry(carry: &mut [i128], a: &[i64]) {
        assert!(a.len() >= carry.len());
        for (carry, &digit) in carry.iter_mut().zip(a) {
            *carry += digit as i128;
        }
    }

    #[inline(always)]
    fn znx_extract_digit_addmul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
        crate::reference::normalization::znx_extract_digit_addmul_i128_ref(base2k, lsh, res, src);
    }

    /// Floor-carry and boundary operations retain the reference kernels' magnitude bounds.
    #[inline(always)]
    fn nfc_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i128], carry: &mut [i128]) {
        crate::reference::normalization::nfc_normalize_floor_ref::<CARRY_IN, ROUND>(base2k, lsh, a, carry);
    }

    #[inline(always)]
    fn nfc_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i128],
        carry: &mut [i128],
    ) {
        crate::reference::normalization::nfc_normalize_round_ref::<CARRY_IN, PAD>(base2k, lsh, padding, res, a, carry);
    }

    /// Retained for compatibility; cross-base extraction now uses the i64 kernels.
    const FUSE_NORMALIZE: bool = true;

    /// Requires `1 <= base2k`, `base2k + lsh <= 63` and `src.len() >= res.len()`.
    fn znx_extract_digit_mul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
        znx_extract_digit_mul_i128_ref(base2k, lsh, res, src);
    }

    /// Also requires `1 <= res_base2k <= 63` and a representable i64 destination sum.
    /// Short source or carry slices panic before writing; unused tails are unchanged.
    fn znx_extract_digit_addmul_normalize_i128<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i128],
        carry: &mut [i128],
    ) {
        znx_extract_digit_addmul_normalize_i128_ref::<OVERWRITE>(base2k, lsh, res_base2k, res, src, carry);
    }

    /// Convert `i128` input + carry into `i64` output, updating carry in place.
    ///
    /// Equivalent to the private `nfc_middle_step` helper.
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        assert!(a.len() >= res.len() && carry.len() >= res.len());
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
        assert!(a.len() >= res.len() && carry.len() >= res.len());
        nfc_middle_step_into::<O>(base2k, lsh, res, a, carry);
    }

    /// Update an existing `i64` res limb using `i128` carry, updating carry in place.
    ///
    /// Equivalent to the private `nfc_middle_step_assign` helper.
    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        assert!(carry.len() >= res.len());
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
        assert!(carry.len() >= res.len());
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
        assert!(carry.len() >= res.len());
        nfc_final_step_into::<O>(base2k, lsh, res, carry);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the scratch size (in bytes) required by [`ntt4x30_vec_znx_big_normalize`].
pub fn ntt4x30_vec_znx_big_normalize_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<i128>()
}

/// Returns the scratch size (in bytes) required by
/// [`ntt4x30_vec_znx_big_automorphism_assign`].
pub fn ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i128>()
}

/// Element-wise addition: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Limbs present in both `a` and `b` are summed; limbs present in only one are copied;
/// extra res limbs beyond both are zeroed.
pub fn ntt4x30_vec_znx_big_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_add_small<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_add_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_small_a<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_small_b<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_sub_small_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_negate_assign<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
pub fn ntt4x30_vec_znx_big_from_small<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
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
/// Extracts `res_k` bits in base `2^res_base2k`, applying `res_offset` before
/// decomposition.
/// Source coefficients must have magnitude at most `2^126`, including after additions.
/// The centered NTT4x30 IDFT output satisfies the stronger bound `|a| < 2^119`.
///
/// `carry` must have at least `ntt4x30_vec_znx_big_normalize_tmp_bytes(n) / size_of::<i128>()`
/// elements (i.e., `3 * n` `i128` values).
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_vec_znx_big_normalize<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let (n, res_size) = {
        let res_view = res.to_backend_mut();
        poulpy_hal::layouts::assert_dense(&res_view, "ntt4x30_vec_znx_big_normalize");
        (res_view.n(), res_view.size())
    };
    poulpy_hal::layouts::assert_dense(&a.to_backend_ref(), "ntt4x30_vec_znx_big_normalize");
    assert!(res_k <= res_size * res_base2k);
    ntt4x30_vec_znx_big_normalize_range(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, 0, n, carry);
}

/// [`ntt4x30_vec_znx_big_normalize`] restricted to `[coeff_start, coeff_start + coeff_len)`;
/// `carry` needs `3 * coeff_len` elements private to the range.
#[allow(clippy::too_many_arguments)]
fn ntt4x30_vec_znx_big_normalize_range<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i128],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    {
        assert!(carry.len() >= 3 * coeff_len);
    }
    let mut res = res.to_backend_mut();
    let (n, cols, size) = (res.n(), res.cols(), res.size());
    let ptr = res.data_mut().as_mut().as_mut_ptr().cast::<i64>();
    unsafe {
        ntt4x30_vec_znx_big_normalize_range_raw::<A, BE>(
            ptr,
            n,
            cols,
            size,
            res_base2k,
            res_k,
            res_offset,
            res_col,
            a,
            a_base2k,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        )
    }
}

/// Normalizes one coefficient range into a raw host output.
///
/// # Safety
///
/// `res_ptr` must be non-null, aligned for `i64`, and address at least
/// `n * cols * size` initialized `i64` values; layout arithmetic must not overflow.
/// The source must have valid initialized storage, degree `n`, and column `a_col`.
/// Require `res_col < cols`, `coeff_start <= n`, `coeff_len <= n - coeff_start`,
/// and at least `3 * coeff_len` private scratch words in `carry`.
///
/// For every destination limb, the selected coefficient range must be exclusively
/// writable for this call and must not overlap source or scratch storage. Concurrent
/// calls may share the destination allocation only with disjoint coefficient ranges.
/// Source reads must remain immutable, and scratch must not overlap the source.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn ntt4x30_vec_znx_big_normalize_range_raw<A, BE>(
    res_ptr: *mut i64,
    n: usize,
    cols: usize,
    size: usize,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i128],
) where
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    {
        assert_eq!(n, a.to_backend_ref().n());
        assert!(res_col < cols);
        assert!(coeff_start + coeff_len <= n);
        assert!(carry.len() >= 3 * coeff_len);
        assert!(res_k <= size * res_base2k);
    }
    let mut res = unsafe { VecZnxRangeMut::new(res_ptr, n, cols, res_col, coeff_start, coeff_len) };
    if res_k == 0 {
        for limb in 0..size {
            res.at_mut(limb).fill(0);
        }
        return;
    }
    let active_size = res_k.div_ceil(res_base2k);
    let padding = (res_base2k - res_k % res_base2k) % res_base2k;
    let input = a.to_backend_ref();
    let partial = res_k != size * res_base2k;
    if res_base2k == a_base2k
        && res_base2k <= 63
        && (partial
            || crate::reference::vec_znx::normalize_needs_exact(input.size(), a_base2k, active_size, res_base2k, res_offset))
    {
        let limb_offset = res_offset.div_euclid(res_base2k as i64);
        let boundary = (active_size as i64 - 1).saturating_add(limb_offset);
        if (0..input.size() as i64).contains(&boundary) {
            let lsh = res_offset.rem_euclid(res_base2k as i64) as usize;
            let carry = &mut carry[..coeff_len];
            for j in (boundary as usize + 1..input.size()).rev() {
                let source = &input.at(a_col, j)[coeff_start..coeff_start + coeff_len];
                match (j + 1 == input.size(), padding == 0 && j == boundary as usize + 1) {
                    (true, false) => BE::nfc_normalize_floor::<false, false>(res_base2k, lsh, source, carry),
                    (true, true) => BE::nfc_normalize_floor::<false, true>(res_base2k, lsh, source, carry),
                    (false, false) => BE::nfc_normalize_floor::<true, false>(res_base2k, lsh, source, carry),
                    (false, true) => BE::nfc_normalize_floor::<true, true>(res_base2k, lsh, source, carry),
                }
            }
            let source = &input.at(a_col, boundary as usize)[coeff_start..coeff_start + coeff_len];
            if boundary as usize + 1 < input.size() {
                BE::nfc_normalize_round::<true, true>(res_base2k, lsh, padding, res.at_mut(active_size - 1), source, carry);
            } else {
                BE::nfc_normalize_round::<false, true>(res_base2k, lsh, padding, res.at_mut(active_size - 1), source, carry);
            }
            for j in (0..active_size - 1).rev() {
                let source = j as i64 + limb_offset;
                if source >= 0 {
                    BE::nfc_middle_step(
                        res_base2k,
                        lsh,
                        res.at_mut(j),
                        &input.at(a_col, source as usize)[coeff_start..coeff_start + coeff_len],
                        carry,
                    );
                } else {
                    BE::znx_extract_digit_mul_i128(res_base2k, 0, res.at_mut(j), carry);
                }
            }
            for j in active_size..size {
                res.at_mut(j).fill(0);
            }
            return;
        }
    }
    let needs_exact = if a_base2k == res_base2k {
        crate::reference::vec_znx::normalize_needs_exact(input.size(), a_base2k, active_size, res_base2k, res_offset)
            || (partial && input.size() as i128 * a_base2k as i128 > res_k as i128 + res_offset as i128)
    } else {
        res_base2k > 62
            || crate::reference::vec_znx::normalize_cross_needs_exact(input.size(), a_base2k, res_base2k, res_k, res_offset)
    };
    if needs_exact {
        for i in 0..coeff_len {
            crate::reference::vec_znx::normalize_exact::<false, _, _>(
                |j| input.at(a_col, j)[coeff_start + i],
                input.size(),
                a_base2k,
                size,
                res_base2k,
                res_k,
                res_offset,
                |j, digit| res.at_mut(j)[i] = digit,
            );
        }
        return;
    }

    if res_base2k == a_base2k {
        ntt4x30_vec_znx_big_normalize_inter(
            res_base2k,
            &mut res,
            active_size,
            res_offset,
            a,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        );
    } else {
        ntt4x30_vec_znx_big_normalize_cross(
            &mut res,
            active_size,
            res_base2k,
            res_k,
            res_offset,
            a,
            a_base2k,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        );
    }
    for j in active_size..size {
        res.at_mut(j).fill(0);
    }
}

/// Adds or subtracts normalized `a` under the source bounds of
/// [`ntt4x30_vec_znx_big_normalize`].
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_vec_znx_big_normalize_assign<O, R, A, BE>(
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
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let input = a.to_backend_ref();
    let mut output = res.to_backend_mut();
    let output_size = output.size();
    assert!(carry.len() >= 3 * output.n());
    for i in 0..output.n() {
        let mut extra = 0i128;
        crate::reference::vec_znx::normalize_exact::<false, _, _>(
            |j| input.at(a_col, j)[i],
            input.size(),
            a_base2k,
            output_size,
            res_base2k,
            output_size * res_base2k,
            res_offset,
            |j, digit| {
                let value = &mut output.at_mut(res_col, j)[i];
                let signed = if O::SUB { -(digit as i128) } else { digit as i128 };
                let total = *value as i128 + signed + extra;
                *value = total as i64;
                extra = (total - *value as i128) >> res_base2k;
            },
        );
    }
}

/// Adds normalized `a` under the source bounds of [`ntt4x30_vec_znx_big_normalize`].
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_vec_znx_big_normalize_add_assign<R, A, BE>(
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
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    ntt4x30_vec_znx_big_normalize_assign::<AddOp, _, _, _>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
}

/// Subtracts normalized `a` under the source bounds of [`ntt4x30_vec_znx_big_normalize`].
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_vec_znx_big_normalize_sub_assign<R, A, BE>(
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
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    ntt4x30_vec_znx_big_normalize_assign::<SubOp, _, _, _>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
}

/// Apply the Galois automorphism `X → X^p` to `a[a_col]`, writing to `res[res_col]`.
///
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt4x30_vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism");
    poulpy_hal::layouts::assert_dense(&a, "ntt4x30_vec_znx_big_automorphism");

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
/// `tmp` must have at least `ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(n) / 16`
/// elements (i.e., `n` `i128` values).
pub fn ntt4x30_vec_znx_big_automorphism_assign<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i128])
where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism_assign");
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
pub fn ntt4x30_vec_znx_big_add_normal_ref<R, BE>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    noise_infos: NoiseInfos,
    source: &mut Source,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        noise_infos.bound.log2().ceil() as i64
    );

    let (limb, shift) = noise_infos.target_limb_and_shift(base2k);
    let normal: Normal<f64> = Normal::new(0.0, noise_infos.sigma).unwrap();
    let rj: &mut [i128] = res.at_mut(res_col, limb);

    rj.iter_mut().for_each(|r| {
        let mut s: f64 = normal.sample(source);
        while s.abs() > noise_infos.bound {
            s = normal.sample(source);
        }
        *r = r.wrapping_add((s.round() as i64 as i128) << shift);
    });
}
