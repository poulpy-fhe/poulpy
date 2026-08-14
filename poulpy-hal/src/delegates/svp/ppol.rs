//! `Module` delegates for the `ppol` tier of the SVP API.
use super::*;

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        SvpPPolOwned::<B>::alloc(self.n(), cols)
    }
}

impl_svp_delegate!(
    SvpPreparePPol<B>,
    fn svp_prepare_ppol(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare_ppol(self, res, res_col, a, a_col);
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
