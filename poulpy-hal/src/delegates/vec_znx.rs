use crate::{
    api::{
        ScalarZnxFillBinaryBlockBackend, ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwBackend,
        ScalarZnxFillBinaryHwSourceBackend, ScalarZnxFillBinaryProbBackend, ScalarZnxFillBinaryProbSourceBackend,
        ScalarZnxFillTernaryHwBackend, ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbBackend,
        ScalarZnxFillTernaryProbSourceBackend, VecZnxAddAssignBackend, VecZnxAddConstAssignBackend, VecZnxAddConstIntoBackend,
        VecZnxAddIntoBackend, VecZnxAddNormalBackend, VecZnxAddNormalSourceBackend, VecZnxAddScalarAssignBackend,
        VecZnxAddScalarIntoBackend, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxAutomorphismBackend,
        VecZnxCopyBackend, VecZnxCopyRangeBackend, VecZnxExtractCoeffBackend, VecZnxFillNormalBackend,
        VecZnxFillNormalSourceBackend, VecZnxFillUniformBackend, VecZnxFillUniformSourceBackend, VecZnxLshAddCoeffIntoBackend,
        VecZnxLshAddIntoBackend, VecZnxLshAssignBackend, VecZnxLshBackend, VecZnxLshCoeffBackend, VecZnxLshSubBackend,
        VecZnxLshTmpBytes, VecZnxMergeRingsBackend, VecZnxMergeRingsTmpBytes, VecZnxMulXpMinusOneAssignBackend,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxMulXpMinusOneBackend, VecZnxNegateAssignBackend, VecZnxNegateBackend,
        VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeCoeffAssignBackend, VecZnxNormalizeCoeffBackend,
        VecZnxNormalizeTmpBytes, VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes, VecZnxRotateBackend,
        VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshAssignBackend, VecZnxRshBackend, VecZnxRshCoeffBackend,
        VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes, VecZnxSplitRingBackend, VecZnxSplitRingTmpBytes,
        VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubInnerProductAssignBackend, VecZnxSubNegateAssignBackend,
        VecZnxSubScalarAssignBackend, VecZnxSubScalarBackend, VecZnxSwitchRingBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
    },
    oep::HalVecZnxImpl,
    source::Source,
};

macro_rules! impl_vec_znx_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalVecZnxImpl<B>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_delegate!(
    VecZnxZeroBackend<B>,
    fn vec_znx_zero_backend<'r>(&self, res: &mut VecZnxBackendMut<'r, B>, res_col: usize) {
        B::vec_znx_zero_backend(self, res, res_col);
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
    fn vec_znx_normalize<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeAssignBackend<B>,
    fn vec_znx_normalize_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize_assign_backend(self, base2k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeCoeffAssignBackend<B>,
    fn vec_znx_normalize_coeff_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize_coeff_assign_backend(self, base2k, a, a_col, a_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxNormalizeCoeffBackend<B>,
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_coeff_backend<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_normalize_coeff_backend(
            self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, a_coeff, scratch,
        )
    }
);

impl_vec_znx_delegate!(
    VecZnxAddIntoBackend<B>,
    fn vec_znx_add_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    ) {
        B::vec_znx_add_into_backend(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddAssignBackend<B>,
    fn vec_znx_add_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_add_assign_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxCopyRangeBackend<B>,
    fn vec_znx_copy_range_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    ) {
        B::vec_znx_copy_range_backend(self, res, res_col, res_limb, res_offset, a, a_col, a_limb, a_offset, len)
    }
);

impl_vec_znx_delegate!(
    VecZnxExtractCoeffBackend<B>,
    fn vec_znx_extract_coeff_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
    ) {
        B::vec_znx_extract_coeff_backend(self, res, res_col, a, a_col, a_coeff)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddConstIntoBackend<B>,
    fn vec_znx_add_const_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        cnst: &VecZnxBackendRef<'a, B>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    ) {
        B::vec_znx_add_const_into_backend(self, res, res_col, a, a_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddConstAssignBackend<B>,
    fn vec_znx_add_const_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        cnst: &VecZnxBackendRef<'a, B>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    ) {
        B::vec_znx_add_const_assign_backend(self, res, res_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubInnerProductAssignBackend<B>,
    fn vec_znx_sub_inner_product_assign_backend<'r, 'a, 'b>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        b: &ScalarZnxBackendRef<'b, B>,
        b_col: usize,
        b_offset: usize,
        len: usize,
    ) {
        B::vec_znx_sub_inner_product_assign_backend(
            self, res, res_col, res_limb, res_offset, a, a_col, a_limb, a_offset, b, b_col, b_offset, len,
        )
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarIntoBackend<B>,
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    ) {
        B::vec_znx_add_scalar_into_backend(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxAddScalarAssignBackend<B>,
    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_add_scalar_assign_backend(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubBackend<B>,
    fn vec_znx_sub_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    ) {
        B::vec_znx_sub_backend(self, res, res_col, a, a_col, b, b_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubAssignBackend<B>,
    fn vec_znx_sub_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_assign_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubNegateAssignBackend<B>,
    fn vec_znx_sub_negate_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_negate_assign_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarBackend<B>,
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    ) {
        B::vec_znx_sub_scalar_backend(self, res, res_col, a, a_col, b, b_col, b_limb)
    }
);

impl_vec_znx_delegate!(
    VecZnxSubScalarAssignBackend<B>,
    fn vec_znx_sub_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_sub_scalar_assign_backend(self, res, res_col, res_limb, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateBackend<B>,
    fn vec_znx_negate_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_negate_backend(self, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxNegateAssignBackend<B>,
    fn vec_znx_negate_assign_backend(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize) {
        B::vec_znx_negate_assign_backend(self, a, a_col)
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
    VecZnxLshBackend<B>,
    fn vec_znx_lsh_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshCoeffBackend<B>,
    fn vec_znx_lsh_coeff_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_coeff_backend(self, base2k, k, res, res_col, a, a_col, a_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAddIntoBackend<B>,
    fn vec_znx_lsh_add_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_add_into_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAddCoeffIntoBackend<B>,
    fn vec_znx_lsh_add_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_add_coeff_into_backend(self, base2k, k, res, res_col, a, a_col, a_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshBackend<B>,
    fn vec_znx_rsh_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshCoeffBackend<B>,
    fn vec_znx_rsh_coeff_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_coeff_backend(self, base2k, k, res, res_col, a, a_col, a_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAddIntoBackend<B>,
    fn vec_znx_rsh_add_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_add_into_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAddCoeffIntoBackend<B>,
    fn vec_znx_rsh_add_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_add_coeff_into_backend(self, base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshSubCoeffIntoBackend<B>,
    fn vec_znx_rsh_sub_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_sub_coeff_into_backend(self, base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshSubBackend<B>,
    fn vec_znx_lsh_sub_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_sub_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshSubBackend<B>,
    fn vec_znx_rsh_sub_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_sub_backend(self, base2k, k, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxLshAssignBackend<B>,
    fn vec_znx_lsh_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_lsh_assign_backend(self, base2k, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRshAssignBackend<B>,
    fn vec_znx_rsh_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rsh_assign_backend(self, base2k, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateBackend<B>,
    fn vec_znx_rotate_backend<'r, 'a>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_rotate_backend(self, k, res, res_col, a, a_col)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateAssignTmpBytes,
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_rotate_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxRotateAssignBackend<B>,
    fn vec_znx_rotate_assign_backend<'s, 'r>(
        &self,
        k: i64,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_rotate_assign_backend(self, k, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxAutomorphismBackend<B>,
    fn vec_znx_automorphism_backend<'r, 'a>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    ) {
        B::vec_znx_automorphism_backend(self, k, res, res_col, a, a_col)
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
    fn vec_znx_automorphism_assign<'s, 'r>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_automorphism_assign(self, k, res, res_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneBackend<B>,
    fn vec_znx_mul_xp_minus_one_backend(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_mul_xp_minus_one_backend(self, p, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneAssignTmpBytes,
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_mul_xp_minus_one_assign_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxMulXpMinusOneAssignBackend<B>,
    fn vec_znx_mul_xp_minus_one_assign_backend<'s>(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_mul_xp_minus_one_assign_backend(self, p, res, res_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxSplitRingTmpBytes,
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize {
        B::vec_znx_split_ring_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxSplitRingBackend<B>,
    fn vec_znx_split_ring_backend<'s>(
        &self,
        res: &mut [VecZnxBackendMut<'_, B>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_split_ring_backend(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxMergeRingsTmpBytes,
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize {
        B::vec_znx_merge_rings_tmp_bytes(self)
    }
);

impl_vec_znx_delegate!(
    VecZnxMergeRingsBackend<B>,
    fn vec_znx_merge_rings_backend<'s>(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, B>],
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    ) {
        B::vec_znx_merge_rings_backend(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_delegate!(
    VecZnxSwitchRingBackend<B>,
    fn vec_znx_switch_ring_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_switch_ring_backend(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    VecZnxCopyBackend<B>,
    fn vec_znx_copy_backend(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize) {
        B::vec_znx_copy_backend(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryHwSourceBackend<B>,
    fn scalar_znx_fill_ternary_hw_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_ternary_hw_backend(self, res, res_col, hw, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryHwBackend<B>,
    fn scalar_znx_fill_ternary_hw_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    ) {
        B::scalar_znx_fill_ternary_hw_backend(self, res, res_col, hw, seed);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryProbSourceBackend<B>,
    fn scalar_znx_fill_ternary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_ternary_prob_backend(self, res, res_col, prob, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillTernaryProbBackend<B>,
    fn scalar_znx_fill_ternary_prob_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) {
        B::scalar_znx_fill_ternary_prob_backend(self, res, res_col, prob, seed);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryHwSourceBackend<B>,
    fn scalar_znx_fill_binary_hw_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_hw_backend(self, res, res_col, hw, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryHwBackend<B>,
    fn scalar_znx_fill_binary_hw_backend(&self, res: &mut ScalarZnxBackendMut<'_, B>, res_col: usize, hw: usize, seed: [u8; 32]) {
        B::scalar_znx_fill_binary_hw_backend(self, res, res_col, hw, seed);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryProbSourceBackend<B>,
    fn scalar_znx_fill_binary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_prob_backend(self, res, res_col, prob, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryProbBackend<B>,
    fn scalar_znx_fill_binary_prob_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) {
        B::scalar_znx_fill_binary_prob_backend(self, res, res_col, prob, seed);
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryBlockSourceBackend<B>,
    fn scalar_znx_fill_binary_block_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        block_size: usize,
        source: &mut Source,
    ) {
        B::scalar_znx_fill_binary_block_backend(self, res, res_col, block_size, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    ScalarZnxFillBinaryBlockBackend<B>,
    fn scalar_znx_fill_binary_block_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        block_size: usize,
        seed: [u8; 32],
    ) {
        B::scalar_znx_fill_binary_block_backend(self, res, res_col, block_size, seed);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformSourceBackend<B>,
    fn vec_znx_fill_uniform_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    ) {
        B::vec_znx_fill_uniform_backend(self, base2k, res, res_col, source.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxFillUniformBackend<B>,
    fn vec_znx_fill_uniform_backend(&self, base2k: usize, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, seed: [u8; 32]) {
        B::vec_znx_fill_uniform_backend(self, base2k, res, res_col, seed);
    }
);

impl_vec_znx_delegate!(
    VecZnxFillNormalSourceBackend<B>,
    fn vec_znx_fill_normal_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source_xe: &mut Source,
    ) {
        B::vec_znx_fill_normal_backend(self, base2k, res, res_col, noise_infos, source_xe.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxFillNormalBackend<B>,
    fn vec_znx_fill_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        B::vec_znx_fill_normal_backend(self, base2k, res, res_col, noise_infos, seed);
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormalSourceBackend<B>,
    fn vec_znx_add_normal_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source_xe: &mut Source,
    ) {
        B::vec_znx_add_normal_backend(self, base2k, res, res_col, noise_infos, source_xe.new_seed());
    }
);

impl_vec_znx_delegate!(
    VecZnxAddNormalBackend<B>,
    fn vec_znx_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        B::vec_znx_add_normal_backend(self, base2k, res, res_col, noise_infos, seed);
    }
);
