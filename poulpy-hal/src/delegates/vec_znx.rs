use crate::{
    api::{
        ScalarZnxAutomorphism, ScalarZnxFillBinaryBlockSource, ScalarZnxFillBinaryHwSource, ScalarZnxFillBinaryProbSource,
        ScalarZnxFillTernaryHwSource, ScalarZnxFillTernaryProbSource, VecZnxAdd, VecZnxAddAssign, VecZnxAddNormalSource,
        VecZnxAddScalarAssign, VecZnxAutomorphism, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxCopy,
        VecZnxFillUniformSource, VecZnxLsh, VecZnxLshAdd, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneAssign, VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNegate, VecZnxNegateAssign, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign, VecZnxRotateAssignTmpBytes, VecZnxRsh,
        VecZnxRshAdd, VecZnxRshAssign, VecZnxRshSub, VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign,
        VecZnxSwitchRing, VecZnxZero,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    oep::HalVecZnxImpl,
    source::Source,
};

macro_rules! impl_vec_znx_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            // `ZnxWord = i64` makes `Backend::ZnxWord` load-bearing: the coefficient-domain ops
            // delegated here are i64-only, so a backend declaring any other word must not receive
            // them. Interim fence until the coefficient aliases flip to `B::ZnxWord` (i32 plumbing),
            // which replaces these bounds. Same bound on all six delegate family macros.
            B: Backend<ZnxWord = i64> + HalVecZnxImpl<B>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_delegate!(
    VecZnxZero<B>,
    fn vec_znx_zero(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize) {
        B::vec_znx_zero(self, res, res_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeTmpBytes,
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_normalize_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalize<B>,
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_normalize(self, res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeAssign<B>,
    fn vec_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_normalize_assign(self, base2k, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxAdd<B>,
    fn vec_znx_add(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_add(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddAssign<B>,
    fn vec_znx_add_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_add_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssign<B>,
    fn vec_znx_add_scalar_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_add_scalar_assign(self, res, res_col, res_limb, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSub<B>,
    fn vec_znx_sub(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_sub(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSubAssign<B>,
    fn vec_znx_sub_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_sub_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxSubNegateAssign<B>,
    fn vec_znx_sub_negate_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNegate<B>,
    fn vec_znx_negate(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_negate(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateAssign<B>,
    fn vec_znx_negate_assign(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize) {
        B::vec_znx_negate_assign(self, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshTmpBytes,
    fn vec_znx_rsh_tmp_bytes(&self) -> usize {
        B::vec_znx_rsh_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshTmpBytes,
    fn vec_znx_lsh_tmp_bytes(&self) -> usize {
        B::vec_znx_lsh_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxLsh<B>,
    fn vec_znx_lsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAdd<B>,
    fn vec_znx_lsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_add(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRsh<B>,
    fn vec_znx_rsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAdd<B>,
    fn vec_znx_rsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_add(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshSub<B>,
    fn vec_znx_lsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshSub<B>,
    fn vec_znx_rsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_sub(self, base2k, k, res, res_col, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAssign<B>,
    fn vec_znx_lsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_lsh_assign(self, base2k, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAssign<B>,
    fn vec_znx_rsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_rsh_assign(self, base2k, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxRotate<B>,
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
    VecZnxRotateAssignTmpBytes,
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateAssign<B>,
    fn vec_znx_rotate_assign(&self, k: i64, a: &mut VecZnxBackendMut<'_, B>, a_col: usize, scratch: &mut ScratchArena<'_, B>) {
        B::vec_znx_rotate_assign(self, k, a, a_col, scratch);
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphism<B>,
    fn vec_znx_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_automorphism(self, k, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismAssignTmpBytes,
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_automorphism_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismAssign<B>,
    fn vec_znx_automorphism_assign(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_automorphism_assign(self, k, res, res_col, scratch);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxAutomorphism<B>,
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        let mut res_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<B>(res);
        let a_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<B>(a);
        B::vec_znx_automorphism(self, k, &mut res_vec, res_col, &a_vec, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOne<B>,
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
    VecZnxMulXpMinusOneAssignTmpBytes,
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_mul_xp_minus_one_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneAssign<B>,
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
    VecZnxSwitchRing<B>,
    fn vec_znx_switch_ring(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_switch_ring(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxCopy<B>,
    fn vec_znx_copy(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_copy(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryHwSource<B>,
    fn scalar_znx_fill_ternary_hw_source(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_ternary_hw(self, res, res_col, hw, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryProbSource<B>,
    fn scalar_znx_fill_ternary_prob_source(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_ternary_prob(self, res, res_col, prob, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryHwSource<B>,
    fn scalar_znx_fill_binary_hw_source(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_hw(self, res, res_col, hw, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryProbSource<B>,
    fn scalar_znx_fill_binary_prob_source(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_prob(self, res, res_col, prob, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryBlockSource<B>,
    fn scalar_znx_fill_binary_block_source(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        block_size: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_block(self, res, res_col, block_size, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformSource<B>,
    fn vec_znx_fill_uniform_source(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    ) {
        B::vec_znx_fill_uniform(self, base2k, k, res, res_col, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormalSource<B>,
    fn vec_znx_add_normal_source(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source_xe: &mut Source,
    ) {
        B::vec_znx_add_normal(self, base2k, res, res_col, noise_infos, source_xe.new_seed());
    }
);
