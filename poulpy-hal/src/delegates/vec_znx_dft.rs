use crate::{
    api::{
        VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAddScaledAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxDftCopy, VecZnxDftFromBytes, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxDftZero,
        VecZnxIdftApply, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Data, DeviceBuf, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToRef,
    },
    oep::HalImpl,
};

impl<B: Backend> VecZnxDftFromBytes<B> for Module<B> {
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B> {
        VecZnxDft::<DeviceBuf<B>, B>::from_bytes(self.n(), cols, size, bytes)
    }
}

impl<B: Backend> VecZnxDftBytesOf for Module<B> {
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx_dft(self.n(), cols, size)
    }
}

impl<B: Backend> VecZnxDftAlloc<B> for Module<B> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        VecZnxDftOwned::alloc(self.n(), cols, size)
    }
}

impl<B> VecZnxIdftApplyTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize {
        <B as HalImpl<B>>::vec_znx_idft_apply_tmp_bytes(self)
    }
}

impl<B> VecZnxIdftApply<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_idft_apply<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_idft_apply(self, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxIdftApplyTmpA<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_idft_apply_tmpa<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>,
    {
        <B as HalImpl<B>>::vec_znx_idft_apply_tmpa(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxIdftApplyConsume<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_idft_apply_consume<D: Data>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>,
    {
        <B as HalImpl<B>>::vec_znx_idft_apply_consume(self, a)
    }
}

impl<B> VecZnxDftApply<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_apply<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_dft_apply(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftAddInto<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_add_into<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_add_into(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxDftAddAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_add_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftAddScaledAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_add_scaled_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_add_scaled_assign(self, res, res_col, a, a_col, a_scale);
    }
}

impl<B> VecZnxDftSub<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_sub(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxDftSubAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_sub_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftSubNegateAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_sub_negate_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftCopy<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_copy<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_copy(self, step, offset, res, res_col, a, a_col);
    }
}

impl<B> VecZnxDftZero<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_dft_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<B>,
    {
        <B as HalImpl<B>>::vec_znx_dft_zero(self, res, res_col);
    }
}
