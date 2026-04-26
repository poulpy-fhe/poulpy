use crate::{
    api::{SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPPolBytesOf, SvpPrepare},
    layouts::{
        Backend, Module, ScalarZnxToRef, SvpPPolOwned, SvpPPolToMut, SvpPPolToRef, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
    },
    oep::HalImpl,
};

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

impl<B> SvpPrepare<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: SvpPPolToMut<B>,
        A: ScalarZnxToRef,
    {
        <B as HalImpl<B>>::svp_prepare(self, res, res_col, a, a_col);
    }
}

impl<B> SvpApplyDft<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn svp_apply_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxToRef,
    {
        <B as HalImpl<B>>::svp_apply_dft(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyDftToDft<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn svp_apply_dft_to_dft<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
        C: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::svp_apply_dft_to_dft(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> SvpApplyDftToDftAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn svp_apply_dft_to_dft_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: SvpPPolToRef<B>,
    {
        <B as HalImpl<B>>::svp_apply_dft_to_dft_assign(self, res, res_col, a, a_col);
    }
}
