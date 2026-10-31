use crate::{
    api::{ScalarZnxAutomorphism, ScalarZnxCIEmbed},
    layouts::{Backend, Module, ScalarZnxBackendMut, ScalarZnxBackendRef},
    oep::{HalScalarZnxImpl, HalVecZnxCIImpl},
};

macro_rules! impl_scalar_znx_delegate {
    (ci $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalScalarZnxImpl + HalVecZnxCIImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalScalarZnxImpl,
        {
            $($body)+
        }
    };
}

impl_scalar_znx_delegate!(
    ScalarZnxAutomorphism<B>,
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    ) {
        B::scalar_znx_automorphism(self, k, res, res_col, a, a_col);
    }
);

impl_scalar_znx_delegate!(
    ci ScalarZnxCIEmbed<B>,
    fn scalar_znx_ci_embed(&self, res: &mut ScalarZnxBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize) {
        B::scalar_znx_ci_embed(self, res, res_col, a, a_col);
    }
);
