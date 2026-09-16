use crate::layouts::{Backend, MatZnx, ScalarZnx, VecZnx};

/// Instantiate a new [crate::layouts::Module].
///
/// ```text
/// op         Module::<BE>::new(n)
/// class      support
/// mutation   none
/// domain     n: a power of two, the ring degree N
/// ensures    returns the backend's module for R_N = Z[X]/(X^N + 1), carrying its precomputed transform tables
/// test       none
/// ```
pub trait ModuleNew<B: Backend> {
    fn new(n: u64) -> Self;
}

/// Query the maximum ring degree `N` of a [`Module`](crate::layouts::Module).
///
/// ```text
/// op         n()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the ring degree N the module was built for
/// test       none
/// ```
pub trait ModuleN {
    fn n(&self) -> usize;
}

/// Query `log2(N)` with a default implementation derived from [`ModuleN::n`].
///
/// ```text
/// op         log_n()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns log2(N)
/// test       none
/// ```
pub trait ModuleLogN
where
    Self: ModuleN,
{
    fn log_n(&self) -> usize {
        (u64::BITS - (self.n() as u64 - 1).leading_zeros()) as usize
    }
}

/// Allocates backend-owned [`ScalarZnx`](crate::layouts::ScalarZnx) layouts.
///
/// ```text
/// op         scalar_znx_alloc(cols)
/// class      support
/// mutation   none
/// domain     cols >= 1
/// ensures    returns an owned degree-N ScalarZnx of `cols` columns in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait ScalarZnxAlloc<B: Backend>: ModuleN {
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates backend-owned [`VecZnx`](crate::layouts::VecZnx) layouts.
///
/// ```text
/// op         vec_znx_alloc(cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns an owned degree-N VecZnx of `cols` columns and `size` limbs in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait VecZnxAlloc<B: Backend>: ModuleN {
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates backend-owned [`MatZnx`](crate::layouts::MatZnx) layouts.
///
/// ```text
/// op         mat_znx_alloc(rows, cols_in, cols_out, size)
/// class      support
/// mutation   none
/// domain     every dimension >= 1
/// ensures    returns an owned degree-N MatZnx of those dimensions in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait MatZnxAlloc<B: Backend>: ModuleN {
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<B::OwnedBuf, B::ZnxWord>;
}
