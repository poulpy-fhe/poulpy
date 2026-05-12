use crate::layouts::{Backend, MatZnx, ScalarZnx, VecZnx};

/// Instantiate a new [crate::layouts::Module].
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}

/// Query the ring degree `N` of a [`Module`](crate::layouts::Module).
pub trait ModuleN {
    fn n(&self) -> usize;
}

/// Query `log2(N)` with a default implementation derived from [`ModuleN::n`].
pub trait ModuleLogN
where
    Self: ModuleN,
{
    fn log_n(&self) -> usize {
        (u64::BITS - (self.n() as u64 - 1).leading_zeros()) as usize
    }
}

/// Allocates backend-owned [`ScalarZnx`](crate::layouts::ScalarZnx) layouts.
pub trait ScalarZnxAlloc<B: Backend>: ModuleN {
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnx<B::OwnedBuf>;
}

/// Allocates backend-owned [`VecZnx`](crate::layouts::VecZnx) layouts.
pub trait VecZnxAlloc<B: Backend>: ModuleN {
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnx<B::OwnedBuf>;
    fn vec_znx_alloc_with_max_size(&self, cols: usize, size: usize, max_size: usize) -> VecZnx<B::OwnedBuf>;
}

/// Allocates backend-owned [`MatZnx`](crate::layouts::MatZnx) layouts.
pub trait MatZnxAlloc<B: Backend>: ModuleN {
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<B::OwnedBuf>;
}
