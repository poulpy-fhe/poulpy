//! `Module` delegates for the `tmat` tier of the VMP API.
use super::*;

impl<B: Backend> VmpTMatAlloc<B> for Module<B> {
    fn vmp_tmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpTMatOwned<B> {
        VmpTMatOwned::<B>::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl_vmp_delegate!(
    VmpPrepareTMatTmpBytes,
    fn vmp_prepare_tmat_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_prepare_tmat_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
);

impl_vmp_delegate!(
    VmpPrepareTMat<B>,
    fn vmp_prepare_tmat(
        &self,
        res: &mut VmpTMatBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_prepare_tmat(self, res, a, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_small_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_small_to_dft(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_dft_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_dft_to_dft(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_small_to_dft_accumulate(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_dft_to_dft_accumulate(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_small_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_small_to_big(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_dft_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_dft_to_big(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_small_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_small_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_tmat_dft_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyTMatDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpTMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_tmat_dft_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, limb_offset, scratch);
    }
);
