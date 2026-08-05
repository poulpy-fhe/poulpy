//! Backend extension points for scalar-vector product (SVP) operations
//! on [`SvpPPol`](poulpy_hal::layouts::SvpPPol).

use bytemuck::{cast_slice, cast_slice_mut};

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        svp::{
            svp_apply_dft as fft64_svp_apply_dft, svp_apply_dft_to_dft as fft64_svp_apply_dft_to_dft,
            svp_apply_dft_to_dft_assign as fft64_svp_apply_dft_to_dft_assign, svp_prepare as fft64_svp_prepare,
        },
    },
    ntt4x30::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero,
        ntt::NttTable,
        primes::Primes30,
        svp::{ntt4x30_svp_apply_dft_to_dft_assign, ntt4x30_svp_prepare},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::VecZnxDftApply,
    layouts::{
        Backend, HostDataRef, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolToBackendMut,
        SvpPPolToBackendRef, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
};

#[doc(hidden)]
pub trait FFT64SvpDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<BE>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, BE>, a_col: usize)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: SvpPPolToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        fft64_svp_prepare::<BE>(module.get_fft_table(), &mut res_ref, res_col, a, a_col);
    }

    fn svp_ppol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_apply_dft_default<'b, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxBackendRef<'b, BE>,
        b_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        fft64_svp_apply_dft::<BE>(module.get_fft_table(), res, res_col, &a_ref, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_default<'b, A>(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'b, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = f64> + ReimArith,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        fft64_svp_apply_dft_to_dft::<BE>(res, res_col, &a_ref, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<A>(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64> + ReimArith,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        fft64_svp_apply_dft_to_dft_assign::<BE>(res, res_col, &a_ref, a_col);
    }
}

impl<BE: Backend> FFT64SvpDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30SvpDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<BE>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, BE>, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: SvpPPolToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        ntt4x30_svp_prepare::<BE>(module, &mut res_ref, res_col, a, a_col);
    }

    fn svp_ppol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_apply_dft_default<'b, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxBackendRef<'b, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<DftWord = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttMulBbc + NttZero,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        let b_size = b.size();
        let mut b_dft = poulpy_hal::layouts::VecZnxDftOwned::<BE>::alloc(module.n(), 1, b_size);
        let mut b_dft_ref = b_dft.to_backend_mut::<BE>();

        module.vec_znx_dft_apply(1, 0, &mut b_dft_ref, 0, b, b_col);

        let meta = module.get_bbc_meta();
        let n = res.n();
        let min_size = res.size().min(b_dft_ref.size());
        let a_u32: &[u32] = cast_slice(a_ref.at(a_col, 0));

        for j in 0..min_size {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
            let b_u32: &[u32] = cast_slice(b_dft_ref.at(0, j));
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

        for j in min_size..res.size() {
            BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
        }
    }

    fn svp_apply_dft_to_dft_default<'b, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'b, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar> + NttMulBbc + NttZero,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        let meta = module.get_bbc_meta();
        let n = res.n();
        let min_size = res.size().min(b.size());
        let a_u32: &[u32] = cast_slice(a_ref.at(a_col, 0));

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

        for j in min_size..res.size() {
            BE::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
        }
    }

    fn svp_apply_dft_to_dft_assign_default<A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar> + NttMulBbc,
        for<'x> BE::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<BE>,
    {
        let a_ref = a.to_backend_ref();
        ntt4x30_svp_apply_dft_to_dft_assign::<BE>(module, res, res_col, &a_ref, a_col);
    }
}

impl<BE: Backend> NTT4x30SvpDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
