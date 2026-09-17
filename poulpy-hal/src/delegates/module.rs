use crate::{
    api::{MatZnxAlloc, ModuleN, ModuleNew, ScalarZnxAlloc, VecZnxAlloc},
    layouts::{Backend, MatZnx, Module, ScalarZnx, VecZnx},
    oep::HalModuleImpl,
};

impl<B> ModuleNew<B> for Module<B>
where
    B: Backend + HalModuleImpl,
{
    fn new(n: u64) -> Self {
        B::new(n)
    }
}

impl<B> ModuleN for Module<B>
where
    B: Backend,
{
    fn n(&self) -> usize {
        self.n()
    }
}

impl<B: Backend> ScalarZnxAlloc<B> for Module<B> {
    fn scalar_znx_alloc(&self, n: usize, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord> {
        Module::<B>::scalar_znx_alloc(self, n, cols)
    }
}

impl<B: Backend> VecZnxAlloc<B> for Module<B> {
    fn vec_znx_alloc(&self, n: usize, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord> {
        Module::<B>::vec_znx_alloc(self, n, cols, size)
    }
}

impl<B: Backend> MatZnxAlloc<B> for Module<B> {
    fn mat_znx_alloc(
        &self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> MatZnx<B::OwnedBuf, B::ZnxWord> {
        Module::<B>::mat_znx_alloc(self, n, rows, cols_in, cols_out, size)
    }
}
