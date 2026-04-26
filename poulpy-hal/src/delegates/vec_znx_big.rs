use crate::{
    api::{
        VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddNormal, VecZnxBigAddSmallAssign, VecZnxBigAddSmallInto, VecZnxBigAlloc,
        VecZnxBigAutomorphism, VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf,
        VecZnxBigFromBytes, VecZnxBigFromSmall, VecZnxBigNegate, VecZnxBigNegateAssign, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxBigSubSmallA,
        VecZnxBigSubSmallAssign, VecZnxBigSubSmallB, VecZnxBigSubSmallNegateAssign,
    },
    layouts::{
        Backend, DeviceBuf, Module, NoiseInfos, Scratch, VecZnxBig, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut,
        VecZnxToRef,
    },
    oep::HalImpl,
    source::Source,
};

impl<B> VecZnxBigFromSmall<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_from_small<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_from_small(res, res_col, a, a_col);
    }
}

impl<B: Backend> VecZnxBigAlloc<B> for Module<B> {
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B> {
        VecZnxBigOwned::alloc(self.n(), cols, size)
    }
}

impl<B: Backend> VecZnxBigFromBytes<B> for Module<B> {
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B> {
        VecZnxBig::<DeviceBuf<B>, B>::from_bytes(self.n(), cols, size, bytes)
    }
}

impl<B: Backend> VecZnxBigBytesOf for Module<B> {
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx_big(self.n(), cols, size)
    }
}

impl<B> VecZnxBigAddNormal<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_add_normal<R: VecZnxBigToMut<B>>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) {
        <B as HalImpl<B>>::vec_znx_big_add_normal(self, base2k, res, res_col, noise_infos, source);
    }
}

impl<B> VecZnxBigAddInto<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_add_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_add_into(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigAddAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_add_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigAddSmallInto<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_add_small_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_add_small_into(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigAddSmallAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_add_small_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_add_small_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSub<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_sub(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubNegateAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_negate_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubSmallA<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_small_a<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_small_a(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubSmallAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_small_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_small_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigSubSmallB<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_small_b<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_small_b(self, res, res_col, a, a_col, b, b_col);
    }
}

impl<B> VecZnxBigSubSmallNegateAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_sub_small_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
    {
        <B as HalImpl<B>>::vec_znx_big_sub_small_negate_assign(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigNegate<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_negate(self, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigNegateAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_negate_assign<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxBigToMut<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_negate_assign(self, a, a_col);
    }
}

impl<B> VecZnxBigNormalizeTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        <B as HalImpl<B>>::vec_znx_big_normalize_tmp_bytes(self)
    }
}

impl<B> VecZnxBigNormalize<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_normalize<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_normalize(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }

    fn vec_znx_big_normalize_add_assign<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_normalize_add_assign(
            self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
        );
    }

    fn vec_znx_big_normalize_sub_assign<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_normalize_sub_assign(
            self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
        );
    }
}

impl<B> VecZnxBigAutomorphism<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_automorphism(self, k, res, res_col, a, a_col);
    }
}

impl<B> VecZnxBigAutomorphismAssignTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize {
        <B as HalImpl<B>>::vec_znx_big_automorphism_assign_tmp_bytes(self)
    }
}

impl<B> VecZnxBigAutomorphismAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_big_automorphism_assign<A>(&self, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxBigToMut<B>,
    {
        <B as HalImpl<B>>::vec_znx_big_automorphism_assign(self, k, a, a_col, scratch);
    }
}
