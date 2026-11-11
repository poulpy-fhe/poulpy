use crate::{
    api::{
        VecZnxAdd, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxAutomorphism, VecZnxAutomorphismAssign,
        VecZnxAutomorphismAssignTmpBytes, VecZnxCIEmbed, VecZnxCITrace, VecZnxCopy, VecZnxFillUniformSource,
        VecZnxFillUniformSourceAll, VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes, VecZnxLsh, VecZnxLshAdd,
        VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNegate, VecZnxNegateAssign, VecZnxNormalize, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign, VecZnxRotateAssignTmpBytes, VecZnxRsh, VecZnxRshAdd,
        VecZnxRshAssign, VecZnxRshSub, VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxSwitchRing,
        VecZnxZero,
    },
    layouts::{
        Backend, ConjugateInvariant, Module, Ring, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToBackendMut,
    },
    oep::{HalVecZnxCIImpl, HalVecZnxImpl, HalVecZnxMonomialImpl},
    source::Source,
};

// Every arm fences `ZnxWord = i64`: the coefficient-domain ops delegated here are i64-only.
macro_rules! impl_vec_znx_delegate {
    (ci $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxCIImpl,
        {
            $($body)+
        }
    };
    (monomial $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxMonomialImpl,
        {
            $($body)+
        }
    };
    (free $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B, Rg: Ring> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_delegate!(
    VecZnxZero<B, Rg>,
    fn vec_znx_zero(&self, res: &mut VecZnxBackendMut<'_, B, Rg>, res_col: usize) {
        B::vec_znx_zero(self, res, res_col);
    }
);

impl_vec_znx_delegate!(
    free VecZnxNormalizeTmpBytes,
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_normalize_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalize<B, Rg>,
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_normalize(self, res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    free VecZnxGadgetDecomposeTmpBytes,
    fn vec_znx_gadget_decompose_tmp_bytes(&self) -> usize {
        B::vec_znx_gadget_decompose_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxGadgetDecompose<B, Rg>,
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_gadget_decompose(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        base2k: usize,
        k_digit: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_gadget_decompose(self, res, res_col, a, a_col, base2k, k_digit, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeAssign<B, Rg>,
    fn vec_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_normalize_assign(self, base2k, k, a_offset, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxAdd<B, Rg>,
    fn vec_znx_add(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B, Rg>,
        b_col: usize,
    ) {
        B::vec_znx_add(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddAssign<B, Rg>,
    fn vec_znx_add_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_add_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssign<B, Rg>,
    fn vec_znx_add_scalar_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_add_scalar_assign(self, res, res_col, res_limb, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSub<B, Rg>,
    fn vec_znx_sub(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B, Rg>,
        b_col: usize,
    ) {
        B::vec_znx_sub(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSubAssign<B, Rg>,
    fn vec_znx_sub_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_sub_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSubNegateAssign<B, Rg>,
    fn vec_znx_sub_negate_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_sub_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNegate<B, Rg>,
    fn vec_znx_negate(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_negate(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateAssign<B, Rg>,
    fn vec_znx_negate_assign(&self, a: &mut VecZnxBackendMut<'_, B, Rg>, a_col: usize) {
        B::vec_znx_negate_assign(self, a, a_col);
    }
);

impl_vec_znx_delegate!(
    free VecZnxRshTmpBytes,
    fn vec_znx_rsh_tmp_bytes(&self, res_size: usize) -> usize {
        B::vec_znx_rsh_tmp_bytes(self, res_size)
    }
);

impl_vec_znx_delegate!(
    free VecZnxLshTmpBytes,
    fn vec_znx_lsh_tmp_bytes(&self, res_size: usize) -> usize {
        B::vec_znx_lsh_tmp_bytes(self, res_size)
    }
);

impl_vec_znx_delegate!(
    VecZnxLsh<B, Rg>,
    fn vec_znx_lsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAdd<B, Rg>,
    fn vec_znx_lsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_add(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRsh<B, Rg>,
    fn vec_znx_rsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAdd<B, Rg>,
    fn vec_znx_rsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_add(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshSub<B, Rg>,
    fn vec_znx_lsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshSub<B, Rg>,
    fn vec_znx_rsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAssign<B, Rg>,
    fn vec_znx_lsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_assign(self, base2k, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAssign<B, Rg>,
    fn vec_znx_rsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_assign(self, base2k, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxRotate<B>,
    fn vec_znx_rotate(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_rotate(self, k, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxRotateAssignTmpBytes,
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxRotateAssign<B>,
    fn vec_znx_rotate_assign(&self, k: i64, a: &mut VecZnxBackendMut<'_, B>, a_col: usize, scratch: &mut ScratchArena<'_, B>) {
        B::vec_znx_rotate_assign(self, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphism<B, Rg>,
    fn vec_znx_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_automorphism(self, k, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    free VecZnxAutomorphismAssignTmpBytes,
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_automorphism_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismAssign<B, Rg>,
    fn vec_znx_automorphism_assign(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_automorphism_assign(self, k, res, res_col, scratch);
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxMulXpMinusOne<B>,
    fn vec_znx_mul_xp_minus_one(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_mul_xp_minus_one(self, p, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxMulXpMinusOneAssignTmpBytes,
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self, size: usize) -> usize {
        B::vec_znx_mul_xp_minus_one_assign_tmp_bytes(self, size)
    }
);

impl_vec_znx_delegate!(
    monomial VecZnxMulXpMinusOneAssign<B>,
    fn vec_znx_mul_xp_minus_one_assign(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_mul_xp_minus_one_assign(self, p, res, res_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxSwitchRing<B, Rg>,
    fn vec_znx_switch_ring(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_switch_ring(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    ci VecZnxCIEmbed<B>,
    fn vec_znx_ci_embed(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, ConjugateInvariant>,
        a_col: usize,
    ) {
        B::vec_znx_ci_embed(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    ci VecZnxCITrace<B>,
    fn vec_znx_ci_trace(
        &self,
        res: &mut VecZnxBackendMut<'_, B, ConjugateInvariant>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_ci_trace(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxCopy<B, Rg>,
    fn vec_znx_copy(&self, res: &mut VecZnxBackendMut<'_, B, Rg>, res_col: usize, a: &VecZnxBackendRef<'_, B, Rg>, a_col: usize) {
        B::vec_znx_copy(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformSource<B, Rg>,
    fn vec_znx_fill_uniform_source(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        source: &mut Source,
    ) {
        B::vec_znx_fill_uniform_source(self, base2k, k, res, res_col, source);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformSourceAll<B, Rg>,
    fn vec_znx_fill_uniform_source_all<R>(&self, base2k: usize, k: usize, res: &mut R, source: &mut Source)
    where
        R: VecZnxToBackendMut<B, Rg>,
    {
        B::vec_znx_fill_uniform_source_all(self, base2k, k, res, source);
    }
);
