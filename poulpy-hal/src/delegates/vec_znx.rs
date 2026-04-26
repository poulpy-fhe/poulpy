use crate::{
    api::{
        VecZnxAddAssign, VecZnxAddInto, VecZnxAddNormal, VecZnxAddScalarAssign, VecZnxAddScalarInto, VecZnxAutomorphism,
        VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxCopy, VecZnxFillNormal, VecZnxFillUniform, VecZnxLsh,
        VecZnxLshAddInto, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMergeRings, VecZnxMergeRingsTmpBytes,
        VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign, VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNegate, VecZnxNegateAssign,
        VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign,
        VecZnxRotateAssignTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshAssign, VecZnxRshSub, VecZnxRshTmpBytes,
        VecZnxSplitRing, VecZnxSplitRingTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxSubScalar,
        VecZnxSubScalarAssign, VecZnxSwitchRing, VecZnxZero,
    },
    layouts::{Backend, Module, NoiseInfos, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    oep::HalImpl,
    source::Source,
};

impl<B> VecZnxZero for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_zero(self, res, res_col);
    }
}

impl<B> VecZnxNormalizeTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_normalize_tmp_bytes(self)
    }
}

impl<B> VecZnxNormalize<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize<R, A>(
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
        A: VecZnxToRef,
    {
        B::vec_znx_normalize(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch)
    }
}

impl<B> VecZnxNormalizeAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_normalize_assign<A>(&self, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_normalize_assign(self, base2k, a, a_col, scratch)
    }
}

impl<B> VecZnxAddInto for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_add_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_add_into(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B> VecZnxAddAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_add_assign(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxAddScalarInto for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_add_scalar_into<R, A, D>(
        &self,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_add_scalar_into(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
}

impl<B> VecZnxAddScalarAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_add_scalar_assign<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_add_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B> VecZnxSub for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        B::vec_znx_sub(self, res, res_col, a, a_col, b, b_col)
    }
}

impl<B> VecZnxSubAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_assign(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSubNegateAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_sub_negate_assign(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxSubScalar for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_sub_scalar<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize, b_limb: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        D: VecZnxToRef,
    {
        B::vec_znx_sub_scalar(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
}

impl<B> VecZnxSubScalarAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_sub_scalar_assign<R, A>(&self, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        B::vec_znx_sub_scalar_assign(self, res, res_col, res_limb, a, a_col)
    }
}

impl<B> VecZnxNegate for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_negate(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxNegateAssign for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_negate_assign<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_negate_assign(self, a, a_col)
    }
}

impl<B> VecZnxRshTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rsh_tmp_bytes(&self) -> usize {
        B::vec_znx_rsh_tmp_bytes(self)
    }
}

impl<B> VecZnxLshTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_lsh_tmp_bytes(&self) -> usize {
        B::vec_znx_lsh_tmp_bytes(self)
    }
}

impl<B> VecZnxLsh<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_lsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxLshAddInto<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_lsh_add_into<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh_add_into(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxRsh<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rsh<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxRshAddInto<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rsh_add_into<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh_add_into(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxLshSub<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_lsh_sub<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_lsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxRshSub<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rsh_sub<R, A>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
}

impl<B> VecZnxLshAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_lsh_assign<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_lsh_assign(self, base2k, k, a, a_col, scratch)
    }
}

impl<B> VecZnxRshAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rsh_assign<A>(&self, base2k: usize, k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rsh_assign(self, base2k, k, a, a_col, scratch)
    }
}

impl<B> VecZnxRotate for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_rotate(self, k, res, res_col, a, a_col)
    }
}

impl<B> VecZnxRotateAssignTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_assign_tmp_bytes(self)
    }
}

impl<B> VecZnxRotateAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_rotate_assign<A>(&self, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        B::vec_znx_rotate_assign(self, k, a, a_col, scratch)
    }
}

impl<B> VecZnxAutomorphism for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_automorphism(self, k, res, res_col, a, a_col)
    }
}

impl<B> VecZnxAutomorphismAssignTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_automorphism_assign_tmp_bytes(self)
    }
}

impl<B> VecZnxAutomorphismAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_automorphism_assign<R>(&self, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_automorphism_assign(self, k, res, res_col, scratch)
    }
}

impl<B> VecZnxMulXpMinusOne for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_mul_xp_minus_one<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_mul_xp_minus_one(self, p, res, res_col, a, a_col);
    }
}

impl<B> VecZnxMulXpMinusOneAssignTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_mul_xp_minus_one_assign_tmp_bytes(self)
    }
}

impl<B> VecZnxMulXpMinusOneAssign<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_mul_xp_minus_one_assign<R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_mul_xp_minus_one_assign(self, p, res, res_col, scratch);
    }
}

impl<B> VecZnxSplitRingTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize {
        B::vec_znx_split_ring_tmp_bytes(self)
    }
}

impl<B> VecZnxSplitRing<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_split_ring<R, A>(&self, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_split_ring(self, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxMergeRingsTmpBytes for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize {
        B::vec_znx_merge_rings_tmp_bytes(self)
    }
}

impl<B> VecZnxMergeRings<B> for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_merge_rings<R, A>(&self, res: &mut R, res_col: usize, a: &[A], a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_merge_rings(self, res, res_col, a, a_col, scratch)
    }
}

impl<B> VecZnxSwitchRing for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_switch_ring<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_switch_ring(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxCopy for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_copy<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        B::vec_znx_copy(self, res, res_col, a, a_col)
    }
}

impl<B> VecZnxFillUniform for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_fill_uniform<R>(&self, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_uniform(self, base2k, res, res_col, source);
    }
}

impl<B> VecZnxFillNormal for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_fill_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_fill_normal(self, base2k, res, res_col, noise_infos, source_xe);
    }
}

impl<B> VecZnxAddNormal for Module<B>
where
    B: Backend + HalImpl<B>,
{
    fn vec_znx_add_normal<R>(&self, base2k: usize, res: &mut R, res_col: usize, noise_infos: NoiseInfos, source_xe: &mut Source)
    where
        R: VecZnxToMut,
    {
        B::vec_znx_add_normal(self, base2k, res, res_col, noise_infos, source_xe);
    }
}
