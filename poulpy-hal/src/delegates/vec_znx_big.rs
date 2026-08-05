use crate::{
    api::{
        VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddNormal, VecZnxBigAddNormalBackend, VecZnxBigAddSmallAssign,
        VecZnxBigAddSmallIntoBackend, VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismAssign,
        VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigColWeightedSum, VecZnxBigFromBytes,
        VecZnxBigFromSmallBackend, VecZnxBigInnerSumBackend, VecZnxBigNegate, VecZnxBigNegateAssign, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxBigSubSmallABackend,
        VecZnxBigSubSmallAssign, VecZnxBigSubSmallBBackend, VecZnxBigSubSmallNegateAssign, VecZnxScalarProduct,
    },
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig,
        VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxBigOwned,
    },
    oep::HalVecZnxBigImpl,
    source::Source,
};

macro_rules! impl_vec_znx_big_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxBigImpl<B>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_big_delegate!(
    VecZnxBigFromSmallBackend<B>,
    fn vec_znx_big_from_small_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_from_small_backend(res, res_col, a, a_col);
    }
);

impl<B: Backend> VecZnxBigAlloc<B> for Module<B> {
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B> {
        self.vec_znx_big_alloc_n(self.n(), cols, size)
    }

    fn vec_znx_big_alloc_n(&self, n: usize, cols: usize, size: usize) -> VecZnxBigOwned<B> {
        VecZnxBig::alloc::<B>(n, cols, size)
    }
}

impl<B: Backend> VecZnxBigFromBytes<B> for Module<B> {
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B> {
        self.vec_znx_big_from_bytes_n(self.n(), cols, size, bytes)
    }

    fn vec_znx_big_from_bytes_n(&self, n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B> {
        VecZnxBig::from_bytes::<B>(n, cols, size, bytes)
    }
}

impl<B: Backend> VecZnxBigBytesOf for Module<B> {
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        self.bytes_of_vec_znx_big_n(self.n(), cols, size)
    }

    fn bytes_of_vec_znx_big_n(&self, n: usize, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx_big(n, cols, size)
    }
}

impl_vec_znx_big_delegate!(
    VecZnxBigAddNormal<B>,
    fn vec_znx_big_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) {
        B::vec_znx_big_add_normal_backend(self, base2k, res, res_col, noise_infos, source.new_seed());
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAddNormalBackend<B>,
    fn vec_znx_big_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        B::vec_znx_big_add_normal_backend(self, base2k, res, res_col, noise_infos, seed);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAddInto<B>,
    fn vec_znx_big_add_into(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_big_add_into(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAddAssign<B>,
    fn vec_znx_big_add_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_add_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAddSmallIntoBackend<B>,
    fn vec_znx_big_add_small_into_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_big_add_small_into_backend(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAddSmallAssign<B>,
    fn vec_znx_big_add_small_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_add_small_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSub<B>,
    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_big_sub(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubAssign<B>,
    fn vec_znx_big_sub_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_sub_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubNegateAssign<B>,
    fn vec_znx_big_sub_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_sub_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubSmallABackend<B>,
    fn vec_znx_big_sub_small_a_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_big_sub_small_a_backend(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubSmallAssign<B>,
    fn vec_znx_big_sub_small_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_sub_small_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubSmallBBackend<B>,
    fn vec_znx_big_sub_small_b_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_big_sub_small_b_backend(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigSubSmallNegateAssign<B>,
    fn vec_znx_big_sub_small_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_sub_small_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigInnerSumBackend<B>,
    fn vec_znx_big_inner_sum_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_inner_sum_backend(self, res, res_col, res_coeff, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigColWeightedSum<B>,
    fn vec_znx_big_col_weighted_sum(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        weights: &ScalarZnxBackendRef<'_, B>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) {
        B::vec_znx_big_col_weighted_sum(self, res, res_col, a, weights, weights_col, cols, coeffs);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxScalarProduct<B>,
    fn vec_znx_scalar_product(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_scalar_product(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigNegate<B>,
    fn vec_znx_big_negate(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_negate(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigNegateAssign<B>,
    fn vec_znx_big_negate_assign(&self, a: &mut VecZnxBigBackendMut<'_, B>, a_col: usize) {
        B::vec_znx_big_negate_assign(self, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigNormalizeTmpBytes,
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize {
        B::vec_znx_big_normalize_tmp_bytes(self)
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigNormalize<B>,
    fn vec_znx_big_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_big_normalize(self, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch)
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAutomorphism<B>,
    fn vec_znx_big_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_big_automorphism(self, k, res, res_col, a, a_col);
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAutomorphismAssignTmpBytes,
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize {
        B::vec_znx_big_automorphism_assign_tmp_bytes(self)
    }
);

impl_vec_znx_big_delegate!(
    VecZnxBigAutomorphismAssign<B>,
    fn vec_znx_big_automorphism_assign(
        &self,
        k: i64,
        a: &mut VecZnxBigBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_big_automorphism_assign(self, k, a, a_col, scratch)
    }
);
