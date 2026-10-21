use crate::{
    api::MatZnxAddAssign,
    layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef, Module},
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
