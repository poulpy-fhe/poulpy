use crate::{
    api::{MatZnxAddAssign, MatZnxCopy, MatZnxNormalize, MatZnxNormalizeAssign, MatZnxNormalizeTmpBytes},
    layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef, Module, Ring, ScratchArena},
    oep::HalMatZnxImpl,
};

macro_rules! impl_mat_znx_delegate {
    (free $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalMatZnxImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B, Rg: Ring> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalMatZnxImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_mat_znx_delegate!(
    MatZnxAddAssign<B, Rg>,
    fn mat_znx_add_assign(&self, res: &mut MatZnxBackendMut<'_, B, Rg>, a: &MatZnxBackendRef<'_, B, Rg>) {
        B::mat_znx_add_assign(self, res, a);
    }
);

impl_mat_znx_delegate!(
    MatZnxCopy<B, Rg>,
    fn mat_znx_copy(&self, res: &mut MatZnxBackendMut<'_, B, Rg>, res_col: usize, a: &MatZnxBackendRef<'_, B, Rg>, a_col: usize) {
        B::mat_znx_copy(self, res, res_col, a, a_col);
    }
);

impl_mat_znx_delegate!(
    free MatZnxNormalizeTmpBytes,
    fn mat_znx_normalize_tmp_bytes(&self) -> usize {
        B::mat_znx_normalize_tmp_bytes(self)
    }
);

impl_mat_znx_delegate!(
    MatZnxNormalize<B, Rg>,
    #[allow(clippy::too_many_arguments)]
    fn mat_znx_normalize(
        &self,
        res: &mut MatZnxBackendMut<'_, B, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &MatZnxBackendRef<'_, B, Rg>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::mat_znx_normalize(self, res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }
);

impl_mat_znx_delegate!(
    MatZnxNormalizeAssign<B, Rg>,
    fn mat_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut MatZnxBackendMut<'_, B, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::mat_znx_normalize_assign(self, base2k, k, a_offset, a, a_col, scratch);
    }
);
