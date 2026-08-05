use crate::{
    api::{
        VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAddScaledAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftFromBytes, VecZnxDftSub, VecZnxDftSubAssign,
        VecZnxDftSubNegateAssign, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftOwned,
    },
    oep::HalVecZnxDftImpl,
};

macro_rules! impl_vec_znx_dft_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalVecZnxDftImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> VecZnxDftFromBytes<B> for Module<B> {
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B> {
        VecZnxDft::from_bytes::<B>(self.n(), cols, size, bytes)
    }
}

impl<B: Backend> VecZnxDftBytesOf for Module<B> {
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx_dft(self.n(), cols, size)
    }
}

impl<B: Backend> VecZnxDftAlloc<B> for Module<B> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        VecZnxDft::alloc::<B>(self.n(), cols, size)
    }
}

impl_vec_znx_dft_delegate!(
    VecZnxIdftApplyTmpBytes,
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize {
        B::vec_znx_idft_apply_tmp_bytes(self)
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxIdftApply<B>,
    fn vec_znx_idft_apply(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_idft_apply(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxIdftApplyTmpA<B>,
    fn vec_znx_idft_apply_tmpa(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_idft_apply_tmpa(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftApply<B>,
    fn vec_znx_dft_apply(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_apply(self, step, offset, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftAddInto<B>,
    fn vec_znx_dft_add_into(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_dft_add_into(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftAddAssign<B>,
    fn vec_znx_dft_add_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_add_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftAddScaledAssign<B>,
    fn vec_znx_dft_add_scaled_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        a_scale: i64,
    ) {
        B::vec_znx_dft_add_scaled_assign(self, res, res_col, a, a_col, a_scale);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSub<B>,
    fn vec_znx_dft_sub(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    ) {
        B::vec_znx_dft_sub(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSubAssign<B>,
    fn vec_znx_dft_sub_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_sub_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSubNegateAssign<B>,
    fn vec_znx_dft_sub_negate_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_sub_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftCopy<B>,
    fn vec_znx_dft_copy(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_copy(self, step, offset, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftZero<B>,
    fn vec_znx_dft_zero(&self, res: &mut VecZnxDftBackendMut<'_, B>, res_col: usize) {
        B::vec_znx_dft_zero(self, res, res_col);
    }
);

impl<B> VecZnxDftAutomorphismPlan<B> for Module<B>
where
    B: Backend + HalVecZnxDftImpl<B>,
{
    type Plan = <B as HalVecZnxDftImpl<B>>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(&self, p: i64) -> Self::Plan {
        B::vec_znx_dft_automorphism_plan(self, p)
    }
}

impl<B> VecZnxDftAutomorphism<B> for Module<B>
where
    B: Backend + HalVecZnxDftImpl<B>,
{
    fn vec_znx_dft_automorphism_with_plan(
        &self,
        plan: &Self::Plan,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::vec_znx_dft_automorphism_with_plan(self, plan, res, res_col, a, a_col);
    }
}
