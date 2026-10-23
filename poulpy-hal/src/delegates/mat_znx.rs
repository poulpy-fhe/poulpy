use crate::{
    api::{MatZnxAddAssign, MatZnxCopy, MatZnxNormalize, MatZnxNormalizeAssign, MatZnxNormalizeTmpBytes},
    layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef, Module, ScratchArena},
    oep::HalMatZnxImpl,
};

macro_rules! impl_mat_znx_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalMatZnxImpl,
        {
            $($body)+
        }
    };
}

impl_mat_znx_delegate!(
    MatZnxAddAssign<B>,
    fn mat_znx_add_assign(&self, res: &mut MatZnxBackendMut<'_, B>, a: &MatZnxBackendRef<'_, B>) {
        B::mat_znx_add_assign(self, res, a);
    }
);

impl_mat_znx_delegate!(
    MatZnxCopy<B>,
    fn mat_znx_copy(&self, res: &mut MatZnxBackendMut<'_, B>, res_col: usize, a: &MatZnxBackendRef<'_, B>, a_col: usize) {
        B::mat_znx_copy(self, res, res_col, a, a_col);
    }
);

impl_mat_znx_delegate!(
    MatZnxNormalizeTmpBytes,
    fn mat_znx_normalize_tmp_bytes(&self) -> usize {
        B::mat_znx_normalize_tmp_bytes(self)
    }
);

impl_mat_znx_delegate!(
    MatZnxNormalize<B>,
    #[allow(clippy::too_many_arguments)]
    fn mat_znx_normalize(
        &self,
        res: &mut MatZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &MatZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::mat_znx_normalize(self, res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }
);

impl_mat_znx_delegate!(
    MatZnxNormalizeAssign<B>,
    fn mat_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut MatZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::mat_znx_normalize_assign(self, base2k, k, a_offset, a, a_col, scratch);
    }
);
