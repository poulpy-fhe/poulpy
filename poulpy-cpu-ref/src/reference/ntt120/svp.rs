//! SVP (scalar-vector product) operations for the NTT120 backend.
//!
//! This module provides the NTT-domain SVP primitives used by
//! `poulpy-cpu-ref`.  The workflow is:
//!
//! 1. **Prepare** — encode a `ScalarZnx` (i64 coefficients) into the
//!    [`SvpPPol`] prepared format (q120c, NTT domain) via
//!    [`ntt120_svp_prepare`].
//! 2. **Apply** — multiply a [`VecZnxDft`] (q120b) by a prepared
//!    [`SvpPPol`] (q120c) to obtain a new [`VecZnxDft`] (q120b) via
//!    one of the `ntt120_svp_apply_dft_to_dft*` functions.
//!
//! # Storage formats
//!
//! | Layout | Scalar type | u64/u32 view | Bytes/coeff |
//! |--------|-------------|--------------|-------------|
//! | `VecZnxDft` (q120b) | `Q120bScalar` | 4 u64 | 32 |
//! | `SvpPPol` (q120c)   | `Q120bScalar` | 8 u32 | 32 |
//!
//! Both layouts share the same [`Q120bScalar`] element type but differ in
//! their arithmetic interpretation.  Use [`bytemuck::cast_slice`] /
//! [`bytemuck::cast_slice_mut`] to obtain the appropriate `&[u32]` or
//! `&[u64]` view.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero, ntt::NttTable, primes::Primes30, types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};

// ──────────────────────────────────────────────────────────────────────────────
// Prepare
// ──────────────────────────────────────────────────────────────────────────────

/// Encode a scalar polynomial into the q120c NTT-domain prepared format.
///
/// Steps:
/// 1. Map i64 coefficients of `a` to q120b (via [`NttFromZnx64`]).
/// 2. Apply the forward NTT (via [`NttDFTExecute`]).
/// 3. Convert q120b → q120c (via [`NttCFromB`]) and store in `res`.
///
/// `res` must be a [`SvpPPol`] with `ScalarPrep = Q120bScalar`.
/// A temporary heap buffer of `4 * n` u64 values is allocated internally
/// (this is a setup/key-preparation function, not a hot path).
pub fn ntt120_svp_prepare<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut SvpPPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let n = res.n();

    // Temporary q120b working buffer (heap-allocated; prepare is not hot).
    let mut tmp = vec![0u64; 4 * n];
    BE::ntt_from_znx64(&mut tmp, a.at(a_col, 0));
    BE::ntt_dft_execute(module.get_ntt_table(), &mut tmp);

    // Write q120c into the SvpPPol buffer.
    let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(res_col, 0));
    BE::ntt_c_from_b(n, res_u32, &tmp);
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply: overwrite
// ──────────────────────────────────────────────────────────────────────────────

/// Pointwise DFT-domain multiply: `res = a ⊙ b`.
///
/// For each active limb `j` and each NTT coefficient index `n_i`:
/// ```text
/// res[res_col, j, n_i]  =  a[a_col, n_i]  ×  b[b_col, j, n_i]   (mod Q)
/// ```
/// Limbs of `res` beyond `b.size()` are zeroed.
///
/// `a`: prepared [`SvpPPol`] in q120c format.
/// `b`: input [`VecZnxDft`] in q120b format.
/// `res`: output [`VecZnxDft`] in q120b format.
pub fn ntt120_svp_apply_dft_to_dft<'r, 'a, 'b, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc + NttZero,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    // q120c view of the prepared polynomial (constant across all limbs).
    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    // Active limbs: pointwise multiply (overwrite).
    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u32: &[u32] = cast_slice(b.at(b_col, j));
        for n_i in 0..n {
            BE::ntt_mul_bbc(
                meta,
                1,
                &mut res_u64[4 * n_i..4 * n_i + 4],
                &b_u32[8 * n_i..8 * n_i + 8],
                &a_u32[8 * n_i..8 * n_i + 8],
            );
        }
    }

    // Remaining limbs: zero.
    for j in min_size..res_size {
        BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply: in-place overwrite
// ──────────────────────────────────────────────────────────────────────────────

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
///
/// For each active limb `j` and each NTT coefficient index `n_i`:
/// ```text
/// res[res_col, j, n_i]  =  a[a_col, n_i]  ×  res[res_col, j, n_i]   (mod Q)
/// ```
///
/// Processes each q120b coefficient by copying it (since [`Q120bScalar`] is
/// `Copy`) before overwriting to avoid aliasing conflicts.
pub fn ntt120_svp_apply_dft_to_dft_assign<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();

    // Borrow a's q120c data once; it is valid for the entire loop.
    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..res_size {
        let res_slice: &mut [Q120bScalar] = res.at_mut(res_col, j);
        let mut product = [0u64; 4];
        for n_i in 0..n {
            // Copy the coefficient (Q120bScalar is Copy) so we can reborrow res_slice.
            let x_elem: Q120bScalar = res_slice[n_i];
            let x_u32: &[u32] = cast_slice(std::slice::from_ref(&x_elem));
            BE::ntt_mul_bbc(meta, 1, &mut product, x_u32, &a_u32[8 * n_i..8 * n_i + 8]);
            res_slice[n_i] = Q120bScalar(product);
        }
    }
}
