use std::mem::size_of;

use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_cpu_portable::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault,
};
use poulpy_cpu_portable::kernels::{
    fft64::ring_arith::Fft64RingArith,
    ntt4x30::{
        NttDFTExecute,
        convolution::{ntt4x30_cnv_by_const_apply_add_portable, ntt4x30_cnv_by_const_apply_portable},
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        vec_znx_dft::NttMetaHandle,
    },
    znx::ZnxAutomorphism,
};
use poulpy_hal::layouts::Ring;
use poulpy_hal::{
    execution::SerialTaskExecutor,
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef,
        VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    },
    oep::{HalCnvImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

pub(crate) use poulpy_cpu_portable::hal_defaults::take_host_typed;

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64Avx512 {
    poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64Avx512 {
    poulpy_cpu_portable::hal_impl_vec_znx_ci!();
}

unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for FFT64Avx512 {}

unsafe impl<Rg: Ring> poulpy_hal::oep::HalScalarZnxImpl<Rg> for FFT64Avx512 where Self: HalVecZnxImpl<Rg> {}
unsafe impl<Rg: Ring> poulpy_hal::oep::HalMatZnxImpl<Rg> for FFT64Avx512 where Self: HalVecZnxImpl<Rg> {}

unsafe impl<Rg: Ring> HalVecZnxImpl<Rg> for FFT64Avx512
where
    Self: ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!(Rg);
    poulpy_cpu_portable::hal_impl_vec_znx_normalize!(Rg);
}

unsafe impl HalModuleImpl for FFT64Avx512 {
    poulpy_cpu_portable::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl<Rg: Ring> HalVmpImpl<Rg> for FFT64Avx512
where
    Self: Fft64RingArith<Rg> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vmp!(FFT64VmpDefault, Rg);
}

unsafe impl<Rg: Ring> HalCnvImpl<Rg> for FFT64Avx512
where
    Self: Fft64RingArith<Rg> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_cnv!(FFT64ConvolutionDefault, Rg);
}

unsafe impl<Rg: Ring> HalVecZnxBigImpl<Rg> for FFT64Avx512
where
    Self: ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault, Rg);
}

unsafe impl<Rg: Ring> HalSvpImpl<Rg> for FFT64Avx512
where
    Self: Fft64RingArith<Rg> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_svp!(FFT64SvpDefault, Rg);
}

unsafe impl<Rg: Ring> HalVecZnxDftImpl<Rg> for FFT64Avx512
where
    Self: Fft64RingArith<Rg> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault, Rg);
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30Avx512 {
    poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30Avx512 {
    poulpy_cpu_portable::hal_impl_vec_znx_ci!();
}

unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for NTT4x30Avx512 {}

unsafe impl<Rg: Ring> poulpy_hal::oep::HalScalarZnxImpl<Rg> for NTT4x30Avx512 where Self: HalVecZnxImpl<Rg> {}
unsafe impl<Rg: Ring> poulpy_hal::oep::HalMatZnxImpl<Rg> for NTT4x30Avx512 where Self: HalVecZnxImpl<Rg> {}

unsafe impl<Rg: Ring> HalVecZnxImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!(Rg);
    poulpy_cpu_portable::hal_impl_vec_znx_normalize!(Rg);
}

unsafe impl HalModuleImpl for NTT4x30Avx512 {
    poulpy_cpu_portable::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl<Rg: Ring> HalVmpImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    fn vmp_pmat_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_pmat_prepare_tmp_bytes_avx(module.n())
    }

    fn vmp_pmat_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self, Rg>,
        a: &MatZnxBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_pmat_prepare_tmp_bytes_avx(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_pmat_prepare_avx_pm(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        b: &VmpPMatBackendRef<'_, Self, Rg>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_avx::<_, poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
    }

    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        b: &VmpPMatBackendRef<'_, Self, Rg>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_add_avx::<_, poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self, Rg>,
        a: &VmpPMatBackendRef<'_, Self, Rg>,
        first_row: usize,
        row_step: usize,
    ) {
        crate::ntt4x30_avx512::vmp::vmp_extract_selected_rows_avx512_pm(res, a, first_row, row_step)
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self, Rg>) {
        res.data_mut().fill(Default::default());
    }
}

unsafe impl<Rg: Ring> HalCnvImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    fn cnv_pvec_left_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(module.n())
    }

    fn cnv_pvec_left_prepare(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self, Rg>,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::convolution::cnv_pvec_left_prepare::<_, SerialTaskExecutor, Rg>(
            module,
            res,
            a,
            tmp,
            |n, dst, src| crate::ntt4x30_avx512::vec_znx_dft::dft_limb_wide(module, n, dst, src),
        );
    }

    fn cnv_pvec_right_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(module.n())
    }

    fn cnv_pvec_right_prepare(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self, Rg>,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::convolution::cnv_pvec_right_prepare::<_, SerialTaskExecutor, Rg>(
            module,
            res,
            a,
            tmp,
            |n, dst, src| crate::ntt4x30_avx512::vec_znx_dft::dft_limb_wide(module, n, dst, src),
        );
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (module, cnv_offset);
        crate::ntt4x30_avx512::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self, Rg>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = (module, scratch);
        ntt4x30_cnv_by_const_apply_portable::<Self, SerialTaskExecutor, Rg>(
            cnv_offset,
            res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            &mut [],
        );
    }

    fn cnv_by_const_apply_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self, Rg>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = (module, scratch);
        ntt4x30_cnv_by_const_apply_add_portable::<Self, SerialTaskExecutor, Rg>(
            cnv_offset,
            res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            &mut [],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
        b_col: usize,
        b_galois: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_apply_dft::<_, SerialTaskExecutor, Rg>(
                module,
                module.get_bbc_meta(),
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_galois,
            )
        };
    }

    fn cnv_apply_dft_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
        b_col: usize,
        b_galois: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_apply_dft_add::<_, SerialTaskExecutor, Rg>(
                module,
                module.get_bbc_meta(),
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_galois,
            )
        };
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_pairwise_apply_dft::<_, SerialTaskExecutor, Rg>(
                module,
                module.get_bbc_meta(),
                cnv_offset,
                res,
                res_col,
                a,
                b,
                i,
                j,
            )
        };
    }

    fn cnv_pvec_self_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(module.n())
    }

    fn cnv_pvec_self_prepare(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self, Rg>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self, Rg>,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::convolution::cnv_pvec_prepare_tmp_bytes(left.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::convolution::cnv_pvec_self_prepare::<_, SerialTaskExecutor, Rg>(
            module,
            left,
            right,
            a,
            tmp,
            |n, dst, src| crate::ntt4x30_avx512::vec_znx_dft::dft_limb_wide(module, n, dst, src),
        );
    }
}

unsafe impl<Rg: Ring> HalVecZnxBigImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    poulpy_cpu_portable::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault, Rg);
}

unsafe impl<Rg: Ring> HalSvpImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    fn svp_ppol_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        crate::ntt4x30_avx512::svp::svp_ppol_prepare(module, res, res_col, a, a_col);
    }

    fn svp_ppol_copy(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        crate::ntt4x30_avx512::svp::svp_ppol_copy(res, res_col, a, a_col);
    }

    fn svp_apply_dft_tmp_bytes(module: &Module<Self>, _b_size: usize) -> usize {
        crate::ntt4x30_avx512::svp::svp_apply_dft_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self, Rg>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::svp::svp_apply_dft_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::svp::svp_apply_dft(module, res, res_col, a, a_col, b, b_col, tmp);
    }

    fn svp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) {
        crate::ntt4x30_avx512::svp::svp_apply_dft_to_dft(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        crate::ntt4x30_avx512::svp::svp_apply_dft_to_dft_assign(module, res, res_col, a, a_col);
    }
}

unsafe impl<Rg: Ring> HalVecZnxDftImpl<Rg> for NTT4x30Avx512
where
    Self: NttDFTExecute<NttTable<Primes30, Rg>> + NttDFTExecute<NttTableInv<Primes30, Rg>> + ZnxAutomorphism<Rg>,
{
    fn vec_znx_idft_normalize_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        4 * module.n() * size_of::<u64>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self, Rg>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n = a.n();
        poulpy_hal::layouts::check_degree::<Self>(module.n(), n);
        assert_eq!(res.n(), n, "vec_znx_idft_normalize: res.n():{} != a.n():{n}", res.n());
        let arena = scratch.borrow();
        let (tmp, _) = take_host_typed::<Self, u64>(arena, 4 * n);
        crate::ntt4x30_avx512::vec_znx_dft::idft_compact_in_place(module, a, a_col, tmp);
        let a_shape = a.shape();
        if let Some((add, add_col)) = addend {
            let mut big: poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self, Rg> =
                poulpy_hal::layouts::VecZnxBig::from_shape(&mut **a.data_mut(), a_shape);
            let mut big_ref = &mut big;
            poulpy_cpu_portable::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_small_assign_portable::<_, _, Self, Rg>(
                &mut big_ref,
                a_col,
                &add,
                add_col,
            );
        }
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self, Rg> =
            poulpy_hal::layouts::VecZnxBig::from_shape(&**a.data(), a_shape);
        poulpy_cpu_portable::kernels::vec_znx_big_normalize::vec_znx_big_normalize_portable::<Self, Rg>(
            res, res_base2k, res_k, res_offset, res_col, &big_ref, a_base2k, a_col,
        );
    }
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_add::<_, SerialTaskExecutor>(res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_add_assign::<_, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_sub::<_, SerialTaskExecutor>(res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_sub_assign::<_, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_sub_negate_assign::<_, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_copy(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_copy::<_, SerialTaskExecutor>(step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self, Rg>, res_col: usize) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_zero(res, res_col)
    }

    type AutomorphismPlan = poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan<Rg>;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
        let _ = module;
        <Self as NttDFTExecute<NttTable<Primes30, Rg>>>::ntt_automorphism_plan(n, p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) {
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_automorphism(plan, res, res_col, a, a_col)
    }

    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        let _ = module;
        crate::ntt4x30_avx512::vec_znx_dft::vec_znx_dft_automorphism_add::<_, SerialTaskExecutor>(plan, res, res_col, a, a_col)
    }
}

#[cfg(feature = "enable-ifma")]
mod ifma_impl {
    use super::{Ring, ScratchArena, ZnxAutomorphism, take_host_typed};
    use crate::NTT3x42Ifma;
    use crate::ntt3x42_ifma::{
        primes::Primes42,
        tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
        traits::Ntt3x42IfmaDFTExecute,
    };
    use poulpy_cpu_portable::hal_defaults::HalVecZnxDefault;

    use poulpy_hal::{
        execution::SerialTaskExecutor,
        layouts::{
            CnvDftAccTerm, MatZnxBackendRef, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendMut,
            VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
            VmpPMatBackendRef,
        },
        oep::{HalCnvImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
    };
    use std::mem::size_of;

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT3x42Ifma {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT3x42Ifma {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for NTT3x42Ifma {}

    unsafe impl<Rg: Ring> poulpy_hal::oep::HalScalarZnxImpl<Rg> for NTT3x42Ifma where Self: HalVecZnxImpl<Rg> {}
    unsafe impl<Rg: Ring> poulpy_hal::oep::HalMatZnxImpl<Rg> for NTT3x42Ifma where Self: HalVecZnxImpl<Rg> {}

    unsafe impl<Rg: Ring> HalVecZnxImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!(Rg);
        poulpy_cpu_portable::hal_impl_vec_znx_normalize!(Rg);
    }

    unsafe impl HalModuleImpl for NTT3x42Ifma {
        fn new(n: u64) -> Module<Self> {
            crate::ntt3x42_ifma::module::module_new(n)
        }
    }

    unsafe impl<Rg: Ring> HalVmpImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        fn vmp_pmat_prepare_tmp_bytes(
            module: &Module<Self>,
            _rows: usize,
            _cols_in: usize,
            _cols_out: usize,
            _size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::vmp::vmp_pmat_prepare_tmp_bytes_ifma(module.n())
        }

        fn vmp_pmat_prepare(
            module: &Module<Self>,
            res: &mut VmpPMatBackendMut<'_, Self, Rg>,
            a: &MatZnxBackendRef<'_, Self, Rg>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vmp::vmp_pmat_prepare_tmp_bytes_ifma(res.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_pmat_prepare_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(module, res, a, tmp);
        }

        fn vmp_apply_dft_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            _b_cols_out: usize,
            _b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            b: &VmpPMatBackendRef<'_, Self, Rg>,
            limb_offset: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module,
                res,
                a,
                b,
                limb_offset,
                tmp,
            );
        }

        fn vmp_apply_dft_to_dft_add_tmp_bytes(
            _module: &Module<Self>,
            _res_size: usize,
            a_size: usize,
            b_rows: usize,
            b_cols_in: usize,
            _b_cols_out: usize,
            _b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft_add(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            b: &VmpPMatBackendRef<'_, Self, Rg>,
            limb_offset: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_add_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module,
                res,
                a,
                b,
                limb_offset,
                tmp,
            );
        }

        fn vmp_extract_selected_rows(
            _module: &Module<Self>,
            res: &mut VmpPMatBackendMut<'_, Self, Rg>,
            a: &VmpPMatBackendRef<'_, Self, Rg>,
            first_row: usize,
            row_step: usize,
        ) {
            crate::ntt3x42_ifma::vmp::vmp_extract_selected_rows_ifma(res, a, first_row, row_step)
        }

        fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self, Rg>) {
            crate::ntt3x42_ifma::vmp::vmp_zero(res);
        }
    }

    use poulpy_cpu_portable::hal_defaults::NTT4x30VecZnxBigDefault;

    unsafe impl<Rg: Ring> HalVecZnxBigImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        poulpy_cpu_portable::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault, Rg);
    }

    unsafe impl<Rg: Ring> HalSvpImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        fn svp_ppol_prepare(
            module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &ScalarZnxBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::svp::svp_ppol_prepare(module, res, res_col, a, a_col);
        }

        fn svp_ppol_copy(
            _module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::svp::svp_ppol_copy(res, res_col, a, a_col);
        }

        fn svp_apply_dft_tmp_bytes(module: &Module<Self>, _b_size: usize) -> usize {
            crate::ntt3x42_ifma::svp::svp_apply_dft_tmp_bytes(module.n())
        }

        #[allow(clippy::too_many_arguments)]
        fn svp_apply_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self, Rg>,
            b_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::svp::svp_apply_dft_tmp_bytes(res.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::svp::svp_apply_dft::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, res_col, a, a_col, b, b_col, tmp,
            );
        }

        fn svp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self, Rg>,
            b_col: usize,
        ) {
            crate::ntt3x42_ifma::svp::svp_apply_dft_to_dft::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, res_col, a, a_col, b, b_col,
            );
        }

        fn svp_apply_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::svp::svp_apply_dft_to_dft_assign::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, res_col, a, a_col,
            );
        }
    }

    unsafe impl<Rg: Ring> HalVecZnxDftImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        fn vec_znx_idft_normalize_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            3 * module.n() * size_of::<u64>()
        }

        #[allow(clippy::too_many_arguments)]
        fn vec_znx_idft_normalize(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self, Rg>,
            res_base2k: usize,
            res_k: usize,
            res_offset: i64,
            res_col: usize,
            a: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            a_col: usize,
            a_base2k: usize,
            addend: Option<(&poulpy_hal::layouts::VecZnxBackendRef<'_, Self, Rg>, usize)>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let n = a.n();
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), 3 * n);
            crate::ntt3x42_ifma::vec_znx_dft::idft_normalize_ifma::<_, poulpy_hal::execution::SerialTaskExecutor>(
                module, res, res_base2k, res_k, res_offset, res_col, a, a_col, a_base2k, addend, tmp,
            );
        }
        fn vec_znx_dft_apply(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col);
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
        }

        fn vec_znx_idft_apply(
            module: &Module<Self>,
            res: &mut VecZnxBigBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(res.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
        }

        fn vec_znx_dft_add(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self, Rg>,
            b_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_add::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, res_col, a, a_col, b, b_col,
            );
        }

        fn vec_znx_dft_add_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_add_assign::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, res_col, a, a_col,
            );
        }

        fn vec_znx_dft_sub(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self, Rg>,
            b_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, res_col, a, a_col, b, b_col,
            );
        }

        fn vec_znx_dft_sub_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub_assign::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, res_col, a, a_col,
            );
        }

        fn vec_znx_dft_sub_negate_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub_negate_assign::<_, poulpy_hal::execution::SerialTaskExecutor>(
                res, res_col, a, a_col,
            );
        }

        fn vec_znx_dft_copy(
            _module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_copy::<_, poulpy_hal::execution::SerialTaskExecutor>(
                step, offset, res, res_col, a, a_col,
            );
        }

        fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self, Rg>, res_col: usize) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_zero::<_, poulpy_hal::execution::SerialTaskExecutor>(res, res_col);
        }

        type AutomorphismPlan = poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan<Rg>;

        fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
            let _ = module;
            <Self as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>>::ntt_automorphism_plan(n, p)
        }

        fn vec_znx_dft_automorphism_with_plan(
            _module: &Module<Self>,
            plan: &Self::AutomorphismPlan,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_automorphism::<_, poulpy_hal::execution::SerialTaskExecutor>(
                plan, res, res_col, a, a_col,
            );
        }

        fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            0
        }

        #[allow(clippy::too_many_arguments)]
        fn vec_znx_dft_automorphism_add_with_plan(
            _module: &Module<Self>,
            plan: &Self::AutomorphismPlan,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self, Rg>,
            a_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let _ = scratch;
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_automorphism_add::<_, poulpy_hal::execution::SerialTaskExecutor>(
                plan, res, res_col, a, a_col,
            );
        }
    }

    unsafe impl<Rg: Ring> HalCnvImpl<Rg> for NTT3x42Ifma
    where
        Self: Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTable<Primes42, Rg>>
            + Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42, Rg>>
            + ZnxAutomorphism<Rg>,
    {
        fn cnv_pvec_left_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pvec_left_prepare_tmp_bytes(module.n())
        }

        fn cnv_pvec_left_prepare(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self, Rg>,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_pvec_left_prepare_tmp_bytes(res.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_pvec_left_prepare::<_, SerialTaskExecutor>(module, res, a, tmp);
        }

        fn cnv_pvec_right_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pvec_right_prepare_tmp_bytes(module.n())
        }

        fn cnv_pvec_right_prepare(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self, Rg>,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_pvec_right_prepare_tmp_bytes(res.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::convolution::cnv_pvec_right_prepare::<_, SerialTaskExecutor>(module, res, a, tmp);
        }

        fn cnv_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        fn cnv_by_const_apply_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxBigBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self, Rg>,
            b_col: usize,
            b_coeff: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<
                Self,
                poulpy_hal::execution::SerialTaskExecutor,
                Rg,
            >(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, tmp);
        }

        fn cnv_by_const_apply_add_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            Self::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply_add(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxBigBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self, Rg>,
            b_col: usize,
            b_coeff: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<
                Self,
                poulpy_hal::execution::SerialTaskExecutor,
                Rg,
            >(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, tmp);
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
            b_col: usize,
            b_galois: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_dft_ifma::<_, SerialTaskExecutor>(
                    module, res, cnv_offset, res_col, a, a_col, b, b_col, b_galois, tmp,
                );
            }
        }

        fn cnv_apply_dft_add_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            Self::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft_add(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
            b_col: usize,
            b_galois: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_dft_add_ifma::<_, SerialTaskExecutor>(
                    module, res, cnv_offset, res_col, a, a_col, b, b_col, b_galois, tmp,
                );
            }
        }

        fn cnv_apply_dft_sum_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_dft_sum_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        fn cnv_apply_dft_sum(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            terms: &[CnvDftAccTerm<'_, Self, Rg>],
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let a_size = terms.iter().map(|term| term.a.size()).max().unwrap_or(0);
            let b_size = terms.iter().map(|term| term.b.size()).max().unwrap_or(0);
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_dft_sum_ifma_tmp_bytes(res.size(), a_size, b_size);
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_dft_sum_ifma::<_, SerialTaskExecutor>(
                    module, res, cnv_offset, res_col, terms, tmp,
                );
            }
        }

        fn cnv_pairwise_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self, Rg>,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self, Rg>,
            i: usize,
            j: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_dft_ifma::<_, SerialTaskExecutor>(
                    module, res, cnv_offset, res_col, a, b, i, j, tmp,
                );
            }
        }

        fn cnv_pvec_self_prepare_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pvec_self_prepare_tmp_bytes(module.n())
        }

        fn cnv_pvec_self_prepare(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self, Rg>,
            right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self, Rg>,
            a: &VecZnxBackendRef<'_, Self, Rg>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_pvec_self_prepare_tmp_bytes(left.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_pvec_self_prepare::<_, SerialTaskExecutor>(module, left, right, a, tmp);
        }
    }
}
