use crate::{
    api::{
        SvpApplyDft, SvpApplyDftTmpBytes, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolBytesOf, SvpPPolCopy,
        SvpPPolPrepare,
    },
    layouts::{
        Backend, Module, PrepareHint, Ring, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef,
        SvpPPolOwned, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
    },
    oep::HalSvpImpl,
};

macro_rules! impl_svp_delegate {
    (free $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalSvpImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B, Rg: Ring> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalSvpImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> SvpPPolAlloc<B> for Module<B> {
    fn svp_ppol_alloc<Rg: Ring>(&self, n: usize, cols: usize, hint: PrepareHint) -> SvpPPolOwned<B, Rg> {
        SvpPPolOwned::<B, Rg>::alloc(n, cols, hint)
    }
}

impl<B: Backend> SvpPPolBytesOf for Module<B> {
    fn svp_ppol_bytes_of(&self, n: usize, cols: usize, hint: PrepareHint) -> usize {
        B::svp_ppol_bytes_of(n, cols, hint)
    }
}

impl_svp_delegate!(
    SvpPPolPrepare<B, Rg>,
    fn svp_ppol_prepare(
        &self,
        res: &mut SvpPPolBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::svp_ppol_prepare(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpPPolCopy<B, Rg>,
    fn svp_ppol_copy(
        &self,
        res: &mut SvpPPolBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::svp_ppol_copy(self, res, res_col, a, a_col);
    }
);

impl_svp_delegate!(
    SvpApplyDft<B, Rg>,
    fn svp_apply_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B, Rg>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::svp_apply_dft(self, res, res_col, a, a_col, b, b_col, scratch);
    }
);

impl_svp_delegate!(
    free SvpApplyDftTmpBytes,
    fn svp_apply_dft_tmp_bytes(&self, b_size: usize) -> usize {
        B::svp_apply_dft_tmp_bytes(self, b_size)
    }
);

impl_svp_delegate!(
    SvpApplyDftToDft<B, Rg>,
    fn svp_apply_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B, Rg>,
        b_col: usize,
    ) {
        B::svp_apply_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_svp_delegate!(
    SvpApplyDftToDftAssign<B, Rg>,
    fn svp_apply_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::svp_apply_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
);
