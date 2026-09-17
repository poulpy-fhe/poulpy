use crate::layouts::{Backend, MatZnx, ScalarZnx, VecZnx};

/// Instantiates a [`Module`](crate::layouts::Module).
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
/// ensures    returns the ring degree N the module was built for
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
/// op         scalar_znx_alloc(cols)
/// class      support
/// mutation   none
/// domain     cols >= 1
/// ensures    returns an owned degree-N ScalarZnx of `cols` columns in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait ScalarZnxAlloc<B: Backend>: ModuleN {
    /// Returns an owned [`ScalarZnx`](crate::layouts::ScalarZnx) of `cols` columns.
    fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates a [`VecZnx`](crate::layouts::VecZnx).
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
    /// Returns an owned [`VecZnx`](crate::layouts::VecZnx) of `cols` columns and `size` limbs.
    fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord>;
}

/// Allocates a [`MatZnx`](crate::layouts::MatZnx).
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
    /// Returns an owned [`MatZnx`](crate::layouts::MatZnx) of the given dimensions.
    fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<B::OwnedBuf, B::ZnxWord>;
}
