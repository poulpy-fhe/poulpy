//! Backend extension points for DFT-domain [`poulpy_hal::layouts::VecZnxDft`] operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        vec_znx_dft::{
            vec_znx_dft_add_assign as fft64_vec_znx_dft_add_assign, vec_znx_dft_add_into as fft64_vec_znx_dft_add_into,
            vec_znx_dft_add_scaled_assign as fft64_vec_znx_dft_add_scaled_assign, vec_znx_dft_apply as fft64_vec_znx_dft_apply,
            vec_znx_dft_copy as fft64_vec_znx_dft_copy, vec_znx_dft_sub as fft64_vec_znx_dft_sub,
            vec_znx_dft_sub_assign as fft64_vec_znx_dft_sub_assign,
            vec_znx_dft_sub_negate_assign as fft64_vec_znx_dft_sub_negate_assign, vec_znx_dft_zero as fft64_vec_znx_dft_zero,
            vec_znx_idft_apply as fft64_vec_znx_idft_apply, vec_znx_idft_apply_tmpa as fft64_vec_znx_idft_apply_tmpa,
        },
    },
    ntt120::{
        NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
        NttSubNegateAssign, NttToZnx128, NttZero,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::{
            NttModuleHandle, ntt120_vec_znx_dft_add_assign as ntt120_default_vec_znx_dft_add_assign,
            ntt120_vec_znx_dft_add_into as ntt120_default_vec_znx_dft_add_into,
            ntt120_vec_znx_dft_add_scaled_assign as ntt120_default_vec_znx_dft_add_scaled_assign,
            ntt120_vec_znx_dft_apply as ntt120_default_vec_znx_dft_apply,
            ntt120_vec_znx_dft_copy as ntt120_default_vec_znx_dft_copy, ntt120_vec_znx_dft_sub as ntt120_default_vec_znx_dft_sub,
            ntt120_vec_znx_dft_sub_assign as ntt120_default_vec_znx_dft_sub_assign,
            ntt120_vec_znx_dft_sub_negate_assign as ntt120_default_vec_znx_dft_sub_negate_assign,
            ntt120_vec_znx_dft_zero as ntt120_default_vec_znx_dft_zero,
            ntt120_vec_znx_idft_apply as ntt120_default_vec_znx_idft_apply,
            ntt120_vec_znx_idft_apply_tmp_bytes as ntt120_default_vec_znx_idft_apply_tmp_bytes,
            ntt120_vec_znx_idft_apply_tmpa as ntt120_default_vec_znx_idft_apply_tmpa,
        },
    },
    znx::ZnxZero,
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
        VecZnxDftBackendRef,
    },
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

#[doc(hidden)]
pub trait FFT64VecZnxDftDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn vec_znx_dft_apply_default(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    {
        fft64_vec_znx_dft_apply(module.get_fft_table(), step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(_module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        0
    }

    fn vec_znx_idft_apply_default<'s>(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        let _ = scratch;
        fft64_vec_znx_idft_apply(module.get_ifft_table(), res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_idft_apply_tmpa(module.get_ifft_table(), res, res_col, a, a_col);
    }

    fn vec_znx_dft_add_into_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<BE>, res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_dft_zero(res, res_col);
    }
}

impl<BE: Backend> FFT64VecZnxDftDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT120VecZnxDftDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn vec_znx_dft_apply_default(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    {
        ntt120_default_vec_znx_dft_apply::<BE>(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default<'s>(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128 + NttCopy,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, u64>(
            scratch.borrow(),
            ntt120_default_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>(),
        );
        ntt120_default_vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt120_default_vec_znx_idft_apply_tmpa(module, res, res_col, a, a_col);
    }

    fn vec_znx_dft_add_into_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAdd + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSub + NttNegate + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSubAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSubNegateAssign + NttNegateAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt120_default_vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<BE>, res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt120_default_vec_znx_dft_zero(res, res_col);
    }
}

impl<BE: Backend> NTT120VecZnxDftDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
