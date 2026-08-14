//! `Module` delegates for the `pmat` tier of the VMP API.
use super::*;

impl<B: Backend> VmpPMatAlloc<B> for Module<B> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        VmpPMatOwned::<B>::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl_vmp_delegate!(
    VmpPreparePMatTmpBytes,
    fn vmp_prepare_pmat_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_prepare_pmat_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
);

impl_vmp_delegate!(
    VmpPreparePMat<B>,
    fn vmp_prepare_pmat(
        &self,
        res: &mut VmpPMatBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_prepare_pmat(self, res, a, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_small_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_small_to_dft(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_dft_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_dft_to_dft(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_small_to_dft_accumulate(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_dft_to_dft_accumulate(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_small_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_small_to_big(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_dft_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_dft_to_big(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_small_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_small_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_pmat_dft_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyPMatDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &VmpPMatBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_pmat_dft_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, limb_offset, scratch);
    }
);
