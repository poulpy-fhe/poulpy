//! HAL extension points of [`NTT4x30Portable`] on the packed transform domain.

use std::mem::size_of;

use poulpy_hal::{
    api::HostBufMut,
    execution::SerialTaskExecutor,
    layouts::{
        Backend, DataView, DataViewMut, MatZnxBackendRef, Module, Ring, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut,
        SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
        VmpPMatBackendRef,
    },
    oep::{HalConvolutionImpl, HalSvpImpl, HalVecZnxDftImpl, HalVmpImpl},
};

use super::NTT4x30Portable;
use crate::kernels::{
    ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
    },
    znx::ZnxAutomorphism,
};

#[inline]
pub(super) fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    assert!(BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()));
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!((bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()));
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

unsafe impl<R: Ring> HalVmpImpl for NTT4x30Portable<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        super::vmp::vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vmp::vmp_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_prepare(module, res, a, tmp);
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
        super::vmp::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vmp::vmp_apply_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft::<_, SerialTaskExecutor>(res, a, b, limb_offset, tmp);
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
        super::vmp::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vmp::vmp_apply_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_add::<_, SerialTaskExecutor>(res, a, b, limb_offset, tmp);
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        super::vmp::vmp_extract_selected_rows(res, a, first_row, row_step)
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        res.data_mut().fill(Default::default());
    }
}

unsafe impl<R: Ring> HalConvolutionImpl for NTT4x30Portable<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::convolution::cnv_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_left::<_, SerialTaskExecutor>(module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::convolution::cnv_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_right::<_, SerialTaskExecutor>(module, res, a, tmp);
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        super::convolution::cnv_apply_dft_tmp_bytes(res_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        _res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        let _ = (module, cnv_offset);
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = (module, scratch);
        crate::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<Self, SerialTaskExecutor>(
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
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = (module, scratch);
        crate::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<Self, SerialTaskExecutor>(
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
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (tmp, _) = take_host_typed::<Self, u32>(scratch.borrow(), super::convolution::apply_tmp_words(res.size()));
        super::convolution::cnv_apply_dft::<_, SerialTaskExecutor>(module, cnv_offset, res, res_col, a, a_col, b, b_col, tmp);
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
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (tmp, _) = take_host_typed::<Self, u32>(scratch.borrow(), super::convolution::apply_tmp_words(res.size()));
        super::convolution::cnv_apply_dft_add::<_, SerialTaskExecutor>(module, cnv_offset, res, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_apply_dft_sum_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        super::convolution::cnv_apply_dft_tmp_bytes(res_size)
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (tmp, _) = take_host_typed::<Self, u32>(scratch.borrow(), super::convolution::apply_tmp_words(res.size()));
        super::convolution::cnv_apply_dft_sum::<_, SerialTaskExecutor>(module, cnv_offset, res, res_col, terms, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        super::convolution::cnv_apply_dft_tmp_bytes(res_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (tmp, _) = take_host_typed::<Self, u32>(scratch.borrow(), super::convolution::apply_tmp_words(res.size()));
        super::convolution::cnv_pairwise_apply_dft::<_, SerialTaskExecutor>(module, cnv_offset, res, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::convolution::cnv_prepare_tmp_bytes(left.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_self::<_, SerialTaskExecutor>(module, left, right, a, tmp);
    }
}

unsafe impl<R: Ring> HalSvpImpl for NTT4x30Portable<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::svp::svp_prepare(module, res, res_col, a, a_col);
    }

    fn svp_ppol_copy(
        _module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::svp::svp_ppol_copy(res, res_col, a, a_col);
    }

    fn svp_apply_dft_tmp_bytes(_module: &Module<Self>, _b_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        super::svp::svp_apply_dft(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        super::svp::svp_apply_dft_to_dft(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::svp::svp_apply_dft_to_dft_assign(module, res, res_col, a, a_col);
    }
}

unsafe impl<R: Ring> HalVecZnxDftImpl for NTT4x30Portable<R>
where
    Self: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        super::vec_znx_dft::idft_tmp_words(module.n()) * size_of::<u64>() + 3 * module.n() * size_of::<i128>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n = a.n();
        poulpy_hal::layouts::check_degree::<Self>(module.n(), n);
        assert_eq!(res.n(), n, "vec_znx_idft_normalize_consume: res.n():{} != a.n():{n}", res.n());
        let arena = scratch.borrow();
        let (tmp, arena) = take_host_typed::<Self, u64>(arena, super::vec_znx_dft::idft_tmp_words(n));
        let (carry, _) = take_host_typed::<Self, i128>(arena, 3 * n);
        super::vec_znx_dft::idft_compact_in_place(module, a, a_col, tmp);
        let a_shape = a.shape();
        if let Some((add, add_col)) = addend {
            let mut big: poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self> =
                poulpy_hal::layouts::VecZnxBig::from_shape(&mut **a.data_mut(), a_shape);
            let mut big_ref = &mut big;
            crate::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_small_assign_portable::<_, _, Self>(
                &mut big_ref,
                a_col,
                &add,
                add_col,
            );
        }
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self> =
            poulpy_hal::layouts::VecZnxBig::from_shape(&**a.data(), a_shape);
        let mut res_ref = &mut *res;
        crate::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_normalize_portable::<_, _, Self>(
            &mut res_ref,
            res_base2k,
            res_k,
            res_offset,
            res_col,
            &&big_ref,
            a_base2k,
            a_col,
            carry,
        );
    }

    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        super::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        super::vec_znx_dft::vec_znx_idft_apply_tmpa(module, res, res_col, a, a_col);
    }

    fn vec_znx_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_add::<R, SerialTaskExecutor>(res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_add_assign::<R, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_sub::<R, SerialTaskExecutor>(res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_sub_assign::<R, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_sub_negate_assign::<R, SerialTaskExecutor>(res, res_col, a, a_col)
    }

    fn vec_znx_dft_copy(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_copy::<R, SerialTaskExecutor>(step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_zero(res, res_col)
    }

    type AutomorphismPlan = crate::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
        let _ = module;
        <Self as NttDFTExecute<NttTable<Primes30, R>>>::ntt_automorphism_plan(n, p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_automorphism(plan, res, res_col, a, a_col)
    }

    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        let _ = module;
        super::vec_znx_dft::vec_znx_dft_automorphism_add::<_, SerialTaskExecutor>(plan, res, res_col, a, a_col)
    }
}
