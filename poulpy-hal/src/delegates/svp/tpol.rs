//! `Module` delegates for the `tpol` tier of the SVP API.
use super::*;

impl<B: Backend> SvpTPolAlloc<B> for Module<B> {
    fn svp_tpol_alloc(&self, cols: usize) -> SvpTPolOwned<B> {
        SvpTPolOwned::<B>::alloc(self.n(), cols)
    }
}

impl_svp_delegate!(
    SvpPrepareTPol<B>,
    fn svp_prepare_tpol(&self, res: &mut SvpTPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare_tpol(self, res, res_col, a, a_col);
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
