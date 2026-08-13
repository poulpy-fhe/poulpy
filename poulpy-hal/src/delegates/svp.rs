use crate::{
    api::*,
    layouts::{
        Backend, Module, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned,
        SvpTPolBackendMut, SvpTPolBackendRef, SvpTPolOwned, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxDftBackendMut, VecZnxDftBackendRef,
    },
    oep::HalSvpImpl,
};

macro_rules! impl_svp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalSvpImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        SvpPPolOwned::<B>::alloc(self.n(), cols)
    }
}

impl<B: Backend> SvpPPolBytesOf for Module<B> {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize {
        B::bytes_of_svp_ppol(self.n(), cols)
    }
}

impl<B: Backend> SvpTPolAlloc<B> for Module<B> {
    fn svp_tpol_alloc(&self, cols: usize) -> SvpTPolOwned<B> {
        SvpTPolOwned::<B>::alloc(self.n(), cols)
    }
}

impl<B: Backend> SvpTPolBytesOf for Module<B> {
    fn bytes_of_svp_tpol(&self, cols: usize) -> usize {
        B::bytes_of_svp_tpol(self.n(), cols)
    }
}

impl_svp_delegate!(
    SvpPreparePPol<B>,
    fn svp_prepare_ppol(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare_ppol(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpPrepareTPol<B>,
    fn svp_prepare_tpol(&self, res: &mut SvpTPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare_tpol(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpPPolCopyBackend<B>,
    fn svp_ppol_copy_backend(
        &self,
        res: &mut SvpPPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_ppol_copy_backend(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpTPolCopyBackend<B>,
    fn svp_tpol_copy_backend(
        &self,
        res: &mut SvpTPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_tpol_copy_backend(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyToBigTmpBytes,
    fn svp_apply_to_big_tmp_bytes(&self, res_size: usize) -> usize {
        B::svp_apply_to_big_tmp_bytes(self, res_size)
    }
);

impl_svp_delegate!(
    SvpApplyToSmallTmpBytes,
    fn svp_apply_to_small_tmp_bytes(&self, b_size: usize) -> usize {
        B::svp_apply_to_small_tmp_bytes(self, b_size)
    }
);

impl_svp_delegate!(
    SvpApplySmallSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_small_small_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplySmallDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_small_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyTPolSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_tpol_small_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyTPolDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_tpol_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyPPolSmallToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_ppol_small_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyPPolDftToDft<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_ppol_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplySmallSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_small_small_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplySmallDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_small_dft_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplyTPolSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_tpol_small_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplyTPolDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_tpol_dft_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplyPPolSmallToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_ppol_small_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplyPPolDftToBig<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_ppol_dft_to_big(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplySmallSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_small_small_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplySmallDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_small_dft_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplyTPolSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_tpol_small_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplyTPolDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_tpol_dft_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplyPPolSmallToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_ppol_small_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplyPPolDftToSmall<B>,
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_small(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_ppol_dft_to_small(
            self, res, res_base2k, res_offset, res_col, a, a_col, b, b_base2k, b_col, scratch,
        );
    }
);

impl_svp_delegate!(
    SvpApplySmallDftToDftAssign<B>,
    fn svp_apply_small_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_apply_small_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyTPolDftToDftAssign<B>,
    fn svp_apply_tpol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_apply_tpol_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyPPolDftToDftAssign<B>,
    fn svp_apply_ppol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_apply_ppol_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
);
