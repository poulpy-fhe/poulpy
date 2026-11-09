use crate::{
    api::{ScalarZnxAutomorphism, ScalarZnxCIEmbed},
    layouts::{Backend, ConjugateInvariant, Module, Ring, ScalarZnxBackendMut, ScalarZnxBackendRef},
    oep::{HalScalarZnxCIImpl, HalScalarZnxImpl},
};

macro_rules! impl_scalar_znx_delegate {
    (ci $trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalScalarZnxCIImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item)+) => {
        impl<B, Rg: Ring> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalScalarZnxImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_scalar_znx_delegate!(
    ScalarZnxAutomorphism<B, Rg>,
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B, Rg>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B, Rg>,
        a_col: usize,
    ) {
        B::scalar_znx_automorphism(self, k, res, res_col, a, a_col);
    }
);

impl_scalar_znx_delegate!(
    ci ScalarZnxCIEmbed<B>,
    fn scalar_znx_ci_embed(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B, ConjugateInvariant>,
        a_col: usize,
    ) {
        B::scalar_znx_ci_embed(self, res, res_col, a, a_col);
    }
);
