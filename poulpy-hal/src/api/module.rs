use crate::layouts::{Backend, MatZnx, ScalarZnx, VecZnx};

/// Instantiates a [`Module`](crate::layouts::Module).
///
/// ```text
/// op         Module::<BE>::new(n)
/// class      support
/// mutation   none
/// domain     n: a power of two, the ring degree N
/// ensures    returns the backend's module for R_N and every R_n, n a power of two with MIN_DEGREE <= n <= N, carrying the transform tables of each
/// test       none
/// ```
pub trait ModuleNew<B: Backend> {
    /// Returns the module of ring degree `n`, carrying its precomputed transform tables.
    fn new(n: u64) -> Self;
}

/// Queries the ring degree `N` of a [`Module`](crate::layouts::Module).
///
/// ```text
/// op         n()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the largest ring degree N the module serves
/// test       none
/// ```
pub trait ModuleN {
    /// Returns the ring degree `N` the module was built for.
    fn n(&self) -> usize;
}

/// Queries `log2(N)` of a [`Module`](crate::layouts::Module).
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
    /// Returns `log2(N)`.
    fn log_n(&self) -> usize {
        (u64::BITS - (self.n() as u64 - 1).leading_zeros()) as usize
    }
}

/// Allocates a [`ScalarZnx`](crate::layouts::ScalarZnx).
///
/// ```text
/// op         scalar_znx_alloc(n, cols)
/// class      support
/// mutation   none
/// domain     n: a power of two, MIN_DEGREE <= n <= the module's degree; cols >= 1
/// ensures    returns an owned degree-n ScalarZnx of `cols` columns in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait ScalarZnxAlloc<B: Backend>: ModuleN {
    /// Returns an owned degree-`n` [`ScalarZnx`](crate::layouts::ScalarZnx) of `cols` columns.
    fn scalar_znx_alloc(&self, n: usize, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates a [`VecZnx`](crate::layouts::VecZnx).
///
/// ```text
/// op         vec_znx_alloc(n, cols, size)
/// class      support
/// mutation   none
/// domain     n: a power of two, MIN_DEGREE <= n <= the module's degree; cols >= 1, size >= 1
/// ensures    returns an owned degree-n VecZnx of `cols` columns and `size` limbs in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait VecZnxAlloc<B: Backend>: ModuleN {
    /// Returns an owned degree-`n` [`VecZnx`](crate::layouts::VecZnx) of `cols` columns and `size` limbs.
    fn vec_znx_alloc(&self, n: usize, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates a [`MatZnx`](crate::layouts::MatZnx).
///
/// ```text
/// op         mat_znx_alloc(n, rows, cols_in, cols_out, size)
/// class      support
/// mutation   none
/// domain     n: a power of two, MIN_DEGREE <= n <= the module's degree; every other dimension >= 1
/// ensures    returns an owned degree-n MatZnx of those dimensions in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait MatZnxAlloc<B: Backend>: ModuleN {
    /// Returns an owned degree-`n` [`MatZnx`](crate::layouts::MatZnx) of the given dimensions.
    fn mat_znx_alloc(
        &self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> MatZnx<B::OwnedBuf, B::ZnxWord>;
}
