use crate::{
    api::*,
    layouts::{
        Backend, MatZnxBackendRef, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned, VmpTMatBackendMut,
        VmpTMatBackendRef, VmpTMatOwned,
    },
    oep::HalVmpImpl,
};

macro_rules! impl_vmp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVmpImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> VmpPMatBytesOf for Module<B> {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::bytes_of_vmp_pmat(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpTMatBytesOf for Module<B> {
    fn bytes_of_vmp_tmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::bytes_of_vmp_tmat(self.n(), rows, cols_in, cols_out, size)
    }
}

impl_vmp_delegate!(
    VmpApplySmallSmallToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_small_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_small_to_dft(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToDftTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_dft_to_dft_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_dft_to_dft(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_small_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_small_to_dft_accumulate(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToDftAccumulateTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToDftAccumulate<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_dft_to_dft_accumulate(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_small_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_small_to_big(self, res, a, b, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToBigTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_dft_to_big_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_dft_to_big(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_small_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_small_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToSmallTmpBytes,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small_tmp_bytes(
        &self,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_small_dft_to_small_tmp_bytes(self, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplySmallDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        a: &MatZnxBackendRef<'_, B>,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_small_dft_to_small(self, res, res_base2k, res_offset, a, b, b_base2k, limb_offset, scratch);
    }
);

mod pmat;
mod tmat;
