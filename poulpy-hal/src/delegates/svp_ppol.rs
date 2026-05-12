use crate::{
    api::{SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolBytesOf, SvpPPolCopyBackend, SvpPrepare},
    layouts::{
        Backend, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef,
    },
    oep::HalSvpImpl,
};

macro_rules! impl_svp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalSvpImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B> {
        SvpPPolOwned::alloc(self.n(), cols)
    }
}

impl<B: Backend> SvpPPolBytesOf for Module<B> {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize {
        B::bytes_of_svp_ppol(self.n(), cols)
    }
}

impl_svp_delegate!(
    SvpPrepare<B>,
    fn svp_prepare(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::svp_prepare(self, res, res_col, a, a_col);
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
    SvpApplyDft<B>,
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::svp_apply_dft(self, res, res_col, a, a_col, b, b_col);
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
