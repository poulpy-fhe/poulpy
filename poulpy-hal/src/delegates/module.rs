use crate::{
    api::{MatZnxAlloc, ModuleN, ModuleNew, ScalarZnxAlloc, VecZnxAlloc},
    layouts::{Backend, MatZnx, Module, ScalarZnx, VecZnx},
    oep::HalModuleImpl,
};

impl<B> ModuleNew<B> for Module<B>
where
    B: Backend + HalModuleImpl<B>,
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
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnx<B::OwnedBuf> {
        Module::<B>::scalar_znx_alloc(self, cols)
    }
}

impl<B: Backend> VecZnxAlloc<B> for Module<B> {
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnx<B::OwnedBuf> {
        Module::<B>::vec_znx_alloc(self, cols, size)
    }

    fn vec_znx_alloc_with_max_size(&self, cols: usize, size: usize, max_size: usize) -> VecZnx<B::OwnedBuf> {
        Module::<B>::vec_znx_alloc_with_max_size(self, cols, size, max_size)
    }
}

impl<B: Backend> MatZnxAlloc<B> for Module<B> {
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<B::OwnedBuf> {
        Module::<B>::mat_znx_alloc(self, rows, cols_in, cols_out, size)
    }
}
