//! SVP (scalar-vector product) operations for the NTT4x30 backend.
//!
//! Prepare encodes a `ScalarZnx` (i64) into the prepared q120c NTT-domain
//! format; apply multiplies a `VecZnxDft` (q120b) by it back into q120b.
//!
//! | Layout | Scalar type | u64/u32 view | Bytes/coeff |
//! |--------|-------------|--------------|-------------|
//! | `VecZnxDft` (q120b) | `Q120bScalar` | 4 u64 | 32 |
//! | prepared (q120c)    | `Q120bScalar` | 8 u32 | 32 |
//!
//! Every kernel takes its prepared operand as the concrete layout type. Both
//! tiers share the q120c encoding on this backend, so each pair forwards to one
//! private inner routine.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpTPolBackendMut,
        SvpTPolBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::ntt4x30::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero, ntt::NttTable, primes::Primes30, types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};

/// Maps the i64 coefficients to q120b, applies the forward NTT, then converts
/// q120b to q120c. Allocates a `4 * n` u64 scratch: this is a key-preparation
/// routine, not a hot path.
fn prepare_inner<BE>(module: &impl NttModuleHandle, n: usize, res: &mut [Q120bScalar], a: &[i64])
where
    BE: NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
{
    let mut tmp = vec![0u64; 4 * n];
    BE::ntt_from_znx64(&mut tmp, a);
    BE::ntt_dft_execute(module.get_ntt_table(), &mut tmp);
    BE::ntt_c_from_b(n, cast_slice_mut(res), &tmp);
}

fn dft_to_dft_inner<'r, 'b, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    pol: &[Q120bScalar],
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'b>: HostDataRef,
{
    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(b.size());

    let a_u32: &[u32] = cast_slice(pol);

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

    for j in min_size..res_size {
        BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
}

fn dft_to_dft_assign_inner<'r, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    pol: &[Q120bScalar],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc,
    BE::BufMut<'r>: HostDataMut,
{
    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();

    let a_u32: &[u32] = cast_slice(pol);

    for j in 0..res_size {
        let res_slice: &mut [Q120bScalar] = res.at_mut(res_col, j);
        let mut product = [0u64; 4];
        for n_i in 0..n {
            // Copy the coefficient (Q120bScalar is Copy) so we can reborrow res_slice.
            let x_elem: Q120bScalar = res_slice[n_i];
            let x_u32: &[u32] = cast_slice(std::slice::from_ref(&x_elem));
            BE::ntt_mul_bbc(meta, 1, &mut product, x_u32, &a_u32[8 * n_i..8 * n_i + 8]);
            res_slice[n_i] = crate::reference::ntt4x30::types::CrtWord(product);
        }
    }
}

/// Encodes a scalar polynomial into an [`SvpPPol`](crate::layouts::SvpPPol).
pub fn ntt4x30_svp_prepare_ppol<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut SvpPPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let n = res.n();
    prepare_inner::<BE>(module, n, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// Encodes a scalar polynomial into an [`SvpTPol`](crate::layouts::SvpTPol).
pub fn ntt4x30_svp_prepare_tpol<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut SvpTPolBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let n = res.n();
    prepare_inner::<BE>(module, n, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// Pointwise DFT-domain multiply `res = a * b`, zeroing limbs past `b.size()`.
pub fn ntt4x30_svp_apply_ppol_dft_to_dft<'r, 'a, 'b, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    dft_to_dft_inner::<BE>(module, res, res_col, a.at(a_col, 0), b, b_col);
}

/// Pointwise DFT-domain multiply `res = a * b`, zeroing limbs past `b.size()`.
pub fn ntt4x30_svp_apply_tpol_dft_to_dft<'r, 'a, 'b, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpTPolBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'b, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
    BE::BufMut<'r>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    dft_to_dft_inner::<BE>(module, res, res_col, a.at(a_col, 0), b, b_col);
}

/// Pointwise DFT-domain multiply in place: `res = a * res`.
pub fn ntt4x30_svp_apply_ppol_dft_to_dft_assign<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    dft_to_dft_assign_inner::<BE>(module, res, res_col, a.at(a_col, 0));
}

/// Pointwise DFT-domain multiply in place: `res = a * res`.
pub fn ntt4x30_svp_apply_tpol_dft_to_dft_assign<'r, 'a, BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxDftBackendMut<'r, BE>,
    res_col: usize,
    a: &SvpTPolBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    dft_to_dft_assign_inner::<BE>(module, res, res_col, a.at(a_col, 0));
}
