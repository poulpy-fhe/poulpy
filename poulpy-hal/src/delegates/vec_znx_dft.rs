use crate::{
    api::{
        VecZnxDftAdd, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftSub,
        VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{
        Backend, Module, Ring, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftOwned,
    },
    oep::HalVecZnxDftImpl,
};

macro_rules! impl_vec_znx_dft_delegate {
    (free $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxDftImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B, Rg: Ring> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVecZnxDftImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> VecZnxDftBytesOf for Module<B> {
    fn vec_znx_dft_bytes_of(&self, n: usize, cols: usize, size: usize) -> usize {
        B::vec_znx_dft_bytes_of(n, cols, size)
    }
}

impl<B: Backend> VecZnxDftAlloc<B> for Module<B> {
    fn vec_znx_dft_alloc<Rg: Ring>(&self, n: usize, cols: usize, size: usize) -> VecZnxDftOwned<B, Rg> {
        VecZnxDftOwned::<B, Rg>::alloc(n, cols, size)
    }
}

impl_vec_znx_dft_delegate!(
    free VecZnxIdftApplyTmpBytes,
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize {
        B::vec_znx_idft_apply_tmp_bytes(self)
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxIdftApply<B, Rg>,
    fn vec_znx_idft_apply(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_idft_apply(self, res, res_col, a, a_col, scratch)
    }
);

impl_vec_znx_dft_delegate!(
    free VecZnxIdftNormalizeTmpBytes,
    fn vec_znx_idft_normalize_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        B::vec_znx_idft_normalize_tmp_bytes(self, res_size, a_size)
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxIdftNormalize<B, Rg>,
    fn vec_znx_idft_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, B, Rg>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, B, Rg>, usize)>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_idft_normalize(
            self, res, res_base2k, res_k, res_offset, res_col, a, a_col, a_base2k, addend, scratch,
        );
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftApply<B, Rg>,
    fn vec_znx_dft_apply(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_apply(self, step, offset, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftAdd<B, Rg>,
    fn vec_znx_dft_add(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B, Rg>,
        b_col: usize,
    ) {
        B::vec_znx_dft_add(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftAddAssign<B, Rg>,
    fn vec_znx_dft_add_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_add_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSub<B, Rg>,
    fn vec_znx_dft_sub(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B, Rg>,
        b_col: usize,
    ) {
        B::vec_znx_dft_sub(self, res, res_col, a, a_col, b, b_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSubAssign<B, Rg>,
    fn vec_znx_dft_sub_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_sub_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftSubNegateAssign<B, Rg>,
    fn vec_znx_dft_sub_negate_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_sub_negate_assign(self, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftCopy<B, Rg>,
    fn vec_znx_dft_copy(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_copy(self, step, offset, res, res_col, a, a_col);
    }
);

impl_vec_znx_dft_delegate!(
    VecZnxDftZero<B, Rg>,
    fn vec_znx_dft_zero(&self, res: &mut VecZnxDftBackendMut<'_, B, Rg>, res_col: usize) {
        B::vec_znx_dft_zero(self, res, res_col);
    }
);

impl<B, Rg: Ring> VecZnxDftAutomorphismPlan<B, Rg> for Module<B>
where
    B: Backend + HalVecZnxDftImpl<Rg>,
{
    type Plan = B::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(&self, n: usize, p: i64) -> Self::Plan {
        B::vec_znx_dft_automorphism_plan(self, n, p)
    }
}

impl<B, Rg: Ring> VecZnxDftAutomorphism<B, Rg> for Module<B>
where
    B: Backend<ZnxWord = i64> + HalVecZnxDftImpl<Rg>,
{
    fn vec_znx_dft_automorphism_with_plan(
        &self,
        plan: &Self::Plan,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_automorphism_with_plan(self, plan, res, res_col, a, a_col);
    }
    fn vec_znx_dft_automorphism(
        &self,
        p: i64,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::vec_znx_dft_automorphism(self, p, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan(
        &self,
        plan: &Self::Plan,
        res: &mut VecZnxDftBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_dft_automorphism_add_with_plan(self, plan, res, res_col, a, a_col, scratch);
    }
}

impl_vec_znx_dft_delegate!(
    free VecZnxDftAutomorphismAddWithPlanTmpBytes,
    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize {
        B::vec_znx_dft_automorphism_add_with_plan_tmp_bytes(self, res_size, a_size)
    }
);
