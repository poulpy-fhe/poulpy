//! Backend extension points for scalar-vector product (SVP) operations
//! on [`SvpPPol`](poulpy_hal::layouts::SvpPPol).

use bytemuck::{cast_slice, cast_slice_mut};

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        svp::{
            svp_apply_dft_to_dft as fft64_svp_apply_dft_to_dft, svp_apply_dft_to_dft_assign as fft64_svp_apply_dft_to_dft_assign,
            svp_prepare as fft64_svp_prepare,
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
use poulpy_hal::layouts::{
    Backend, HostDataRef, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolToBackendMut,
    SvpPPolToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, ZnxViewMut, check_degree,
};

#[doc(hidden)]
pub trait FFT64SvpDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<Self>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, Self>, a_col: usize)
    where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: SvpPPolToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let n: usize = res_ref.n();
        check_degree::<Self>(module.n(), n);
        assert!(a.n() == n, "svp_prepare: a.n() != res.n()");
        fft64_svp_prepare::<Self>(module.get_fft_table_for(n), &mut res_ref, res_col, a, a_col);
    }

    fn svp_ppol_copy_default(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) where
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
    {
        assert_eq!(res.n(), a.n(), "svp_ppol_copy: res.n() {} != a.n() {}", res.n(), a.n());
        assert_eq!(
            res.hint(),
            a.hint(),
            "svp_ppol_copy: res and a must carry the same PrepareHint ({:?} != {:?})",
            res.hint(),
            a.hint()
        );
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_apply_dft_to_dft_default<'b, A>(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'b, Self>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<Self>,
    {
        let a_ref = a.to_backend_ref();
        fft64_svp_apply_dft_to_dft::<Self>(res, res_col, &a_ref, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<A>(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<Self>,
    {
        let a_ref = a.to_backend_ref();
        fft64_svp_apply_dft_to_dft_assign::<Self>(res, res_col, &a_ref, a_col);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64SvpDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30SvpDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<Self>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, Self>, a_col: usize)
    where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: SvpPPolToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        ntt4x30_svp_prepare::<Self>(module, &mut res_ref, res_col, a, a_col);
    }

    fn svp_ppol_copy_default(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) where
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
    {
        assert_eq!(res.n(), a.n(), "svp_ppol_copy: res.n() {} != a.n() {}", res.n(), a.n());
        assert_eq!(
            res.hint(),
            a.hint(),
            "svp_ppol_copy: res and a must carry the same PrepareHint ({:?} != {:?})",
            res.hint(),
            a.hint()
        );
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_apply_dft_to_dft_default<'b, A>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &'b A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'b, Self>,
        b_col: usize,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc + NttZero,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<Self>,
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
                Self::ntt_mul_bbc(
                    meta,
                    1,
                    &mut res_u64[4 * n_i..4 * n_i + 4],
                    &b_u32[8 * n_i..8 * n_i + 8],
                    &a_u32[8 * n_i..8 * n_i + 8],
                );
            }
        }

        for j in min_size..res.size() {
            Self::ntt_zero(cast_slice_mut(res.at_mut(res_col, j)));
        }
    }

    fn svp_apply_dft_to_dft_assign_default<A>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttMulBbc,
        for<'x> Self::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        A: SvpPPolToBackendRef<Self>,
    {
        let a_ref = a.to_backend_ref();
        ntt4x30_svp_apply_dft_to_dft_assign::<Self>(module, res, res_col, &a_ref, a_col);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30SvpDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
