use crate::{
    api::MatZnxAddAssign,
    layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef, Module},
    oep::HalVecZnxImpl,
};

impl<B> MatZnxAddAssign<B> for Module<B>
where
    B: Backend<ZnxWord = i64> + HalVecZnxImpl,
{
    fn mat_znx_add_assign(&self, res: &mut MatZnxBackendMut<'_, B>, a: &MatZnxBackendRef<'_, B>) {
        B::mat_znx_add_assign(self, res, a);
    }
}
