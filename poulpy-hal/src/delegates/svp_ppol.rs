use crate::{
    api::{
        SvpApplyDft, SvpApplyDftTmpBytes, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolBytesOf, SvpPPolCopy,
        SvpPrepare,
    },
    layouts::{
        Backend, Module, PrepareHint, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned,
        VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
    },
    oep::HalSvpImpl,
};

macro_rules! impl_svp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalSvpImpl,
        {
            $($body)+
        }
    };
}

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc(&self, n: usize, cols: usize, hint: PrepareHint) -> SvpPPolOwned<B> {
        SvpPPolOwned::<B>::alloc(n, cols, hint)
    }
}

impl<B: Backend> SvpPPolBytesOf for Module<B> {
    fn bytes_of_svp_ppol(&self, n: usize, cols: usize, hint: PrepareHint) -> usize {
        B::bytes_of_svp_ppol(n, cols, hint)
    }
}

impl_svp_delegate!(
    SvpPrepare<B>,
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpPPolCopy<B>,
    fn svp_ppol_copy(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &SvpPPolBackendRef<'_, B>, a_col: usize) {
        B::svp_ppol_copy(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyDft<B>,
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_dft(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    SvpApplyDftTmpBytes,
    fn svp_apply_dft_tmp_bytes(&self, b_size: usize) -> usize {
        B::svp_apply_dft_tmp_bytes(self, b_size)
    }
);

impl_svp_delegate!(
    SvpApplyDftToDft<B>,
    fn svp_apply_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyDftToDftAssign<B>,
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::svp_apply_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
);
