//! Extended-precision (`i128`) ring element vector operations for the NTT4x30 backend.
//!
//! This module provides standalone portable functions for [`VecZnxBig`](poulpy_hal::layouts::VecZnxBig) operations
//! when the backend's `BigWord` is `i128`.  Unlike the [`fft64`] backend — where
//! `BigWord = i64` and a `VecZnxBig` can be reinterpreted as a `VecZnx` — the
//! NTT4x30 backend stores `i128` values, so every operation must be implemented
//! directly on `i128` slices.
//!
//! # Layout
//!
//! A `VecZnxBig<_, NTT4x30Portable>` with `cols` columns, `size` limbs, and ring degree `n`
//! stores `cols × size × n` `i128` values in limb-major, column-minor order:
//!
//! ```text
//! at(col, limb) → &[i128] of length n, starting at n*(limb*cols + col)*16 bytes
//! ```
//!
//! # Functions
//!
//! - **Element-wise arithmetic**: [`ntt4x30_vec_znx_big_add_portable`], [`ntt4x30_vec_znx_big_sub_portable`],
//!   [`ntt4x30_vec_znx_big_negate_portable`] and their inplace / mixed-precision variants.
//! - **Copy from small**: [`ntt4x30_vec_znx_big_from_small_portable`] — sign-extend `i64` → `i128`.
//! - **Automorphism**: [`ntt4x30_vec_znx_big_automorphism_portable`] /
//!   [`ntt4x30_vec_znx_big_automorphism_assign_portable`] — apply `X → X^p` on `i128` coefficients.
//! - **Noise**: [`ntt4x30_vec_znx_big_add_noise_portable`] adds integer samples
//!   at precision `k` across `ceil(k / base2k)` limbs of a `VecZnxBig`.
//!
//! [`fft64`]: crate::kernels::fft64

use poulpy_core::Noise;

use crate::{
    kernels::{
        noise::add_noise_portable,
        vec_znx::{
            vec_znx_add_assign_mixed_portable, vec_znx_add_mixed_portable, vec_znx_from_small_mixed_portable,
            vec_znx_sub_assign_mixed_portable, vec_znx_sub_mixed_portable, vec_znx_sub_negate_assign_mixed_portable,
        },
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    source::Source,
};

// ──────────────────────────────────────────────────────────────────────────────
// Per-slice i128 arithmetic dispatch trait
// ──────────────────────────────────────────────────────────────────────────────

/// Per-slice `i128` arithmetic kernels, dispatched via the backend type parameter.
///
/// All methods have scalar default implementations.  `NTT4x30Portable` and other
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
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the scratch size (in bytes) required by
/// [`ntt4x30_vec_znx_big_automorphism_assign_portable`].
pub fn ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes_portable(n: usize) -> usize {
    n * size_of::<i128>()
}

/// Element-wise addition: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Limbs present in both `a` and `b` are summed; limbs present in only one are copied;
/// extra res limbs beyond both are zeroed.
pub fn ntt4x30_vec_znx_big_add_portable<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
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

    if a.n() != res.n() || b.n() != res.n() {
        return vec_znx_add_mixed_portable(&mut res, res_col, &a, a_col, &b, b_col);
    }

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
pub fn ntt4x30_vec_znx_big_add_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_add_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Add a small (`i64`) polynomial `b` to a big (`i128`) polynomial `a`:
/// `res[res_col] = a[a_col] + b[b_col]`.
pub fn ntt4x30_vec_znx_big_add_small_portable<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
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

    if a.n() != res.n() || b.n() != res.n() {
        return vec_znx_add_mixed_portable(&mut res, res_col, &a, a_col, &b, b_col);
    }

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
pub fn ntt4x30_vec_znx_big_add_small_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_add_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Subtraction: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt4x30_vec_znx_big_sub_portable<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
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

    if a.n() != res.n() || b.n() != res.n() {
        return vec_znx_sub_mixed_portable(&mut res, res_col, &a, a_col, &b, b_col);
    }

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
pub fn ntt4x30_vec_znx_big_sub_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_sub_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// Swap-subtract in-place: `res[res_col] = a[a_col] - res[res_col]`.
pub fn ntt4x30_vec_znx_big_sub_negate_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_sub_negate_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

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
pub fn ntt4x30_vec_znx_big_sub_small_a_portable<R, A, B, BE>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
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

    if a.n() != res.n() || b.n() != res.n() {
        return vec_znx_sub_mixed_portable(&mut res, res_col, &a, a_col, &b, b_col);
    }

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
pub fn ntt4x30_vec_znx_big_sub_small_b_portable<R, A, B, BE>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
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

    if a.n() != res.n() || b.n() != res.n() {
        return vec_znx_sub_mixed_portable(&mut res, res_col, &a, a_col, &b, b_col);
    }

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
pub fn ntt4x30_vec_znx_big_sub_small_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_sub_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// In-place: `res[res_col] = a[a_col] - res[res_col]` where `a` is a `VecZnx` (i64).
pub fn ntt4x30_vec_znx_big_sub_small_negate_assign_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_sub_negate_assign_mixed_portable(&mut res, res_col, &a, a_col);
    }

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
pub fn ntt4x30_vec_znx_big_negate_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
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
pub fn ntt4x30_vec_znx_big_negate_assign_portable<R, BE>(res: &mut R, res_col: usize)
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
pub fn ntt4x30_vec_znx_big_from_small_portable<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    if a.n() != res.n() {
        return vec_znx_from_small_mixed_portable(&mut res, res_col, &a, a_col);
    }

    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for j in 0..min_size {
        BE::i128_from_small(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

/// Apply the Galois automorphism `X → X^p` to `a[a_col]`, writing to `res[res_col]`.
///
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt4x30_vec_znx_big_automorphism_portable<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + crate::kernels::znx::ZnxAutomorphism,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism");
    poulpy_hal::layouts::assert_dense(&a, "ntt4x30_vec_znx_big_automorphism");

    let size = res.size().min(a.size());
    for limb in 0..size {
        BE::znx_automorphism_i128(p, res.at_mut(res_col, limb), a.at(a_col, limb));
    }

    for limb in size..res.size() {
        res.at_mut(res_col, limb).iter_mut().for_each(|r| *r = 0);
    }
}

/// Apply `X → X^p` in-place to `res[res_col]`.
///
/// `tmp` must have at least `ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(n) / 16`
/// elements (i.e., `n` `i128` values).
pub fn ntt4x30_vec_znx_big_automorphism_assign_portable<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i128])
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + crate::kernels::znx::ZnxAutomorphism,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism_assign");
    let n = res.n();
    for limb in 0..res.size() {
        let rj = res.at_mut(res_col, limb);
        tmp[..n].copy_from_slice(rj);
        BE::znx_automorphism_i128(p, rj, &tmp[..n]);
    }
}

/// Add full-width integer noise at precision `k`, in radix `2^base2k`.
pub fn ntt4x30_vec_znx_big_add_noise_portable<R, BE>(
    base2k: usize,
    k: usize,
    res: &mut R,
    res_col: usize,
    noise: Noise,
    source: &mut Source,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE>,
{
    add_noise_portable(base2k, k, &mut res.to_backend_mut(), res_col, noise, source);
}
