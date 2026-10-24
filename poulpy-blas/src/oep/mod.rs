//! Backend extension points. A backend selects the reference with
//! [`impl_blas_reference_full!`](crate::impl_blas_reference_full) or overrides a family, and an override reproduces the
//! reference bit for bit within the scratch its `_tmp_bytes` reports. As the reference does, an implementation first runs its
//! operation's validator, the public `validate_*` of [`crate::layouts`], before any write, returns its errors with
//! `res` and its batch record untouched, and on `Ok` stamps `res` with the batch the validator returns
//! ([`crate::layouts::BlasGridMut::set_mat_slots`]); the delegates dispatch without a check.
//!
//! The references run serially, block after block: concurrency stays inside the backend's kernels, within one
//! ciphertext operation. An override may choose its own level instead, for example the blocks of an operation on a
//! backend whose kernels are serial, but never both; the traits' `Send` and `Sync` bounds leave that choice open.
//!
//! BLAS spawns no thread and holds no shared mutable state, so calls with distinct `res` and `scratch` may run
//! concurrently over a shared module, context, public operands, prepared operands and key stores. An override keeps no
//! state across calls, so disjoint concurrent calls stay valid.

pub(crate) mod derived;
mod inter;
mod intra;
mod jkls;
mod matmul;

pub use inter::BlasMatInterImpl;
pub use intra::BlasMatIntraImpl;
pub use jkls::BlasJklsImpl;
pub use matmul::BlasMatMulImpl;

/// Selects every reference implementation for `$be`.
#[macro_export]
macro_rules! impl_blas_reference_full {
    ($be:ty) => {
        $crate::impl_blas_mat_inter_reference_full!($be);
        $crate::impl_blas_mat_intra_reference_full!($be);
        $crate::impl_blas_mat_mul_reference_full!($be);
        $crate::impl_blas_jkls_reference_full!($be);
    };
}

/// Every blas extension point; a test bound, never implemented directly.
///
/// # Safety
///
/// Implemented only through the blanket impl.
pub unsafe trait BlasImpl: BlasMatInterImpl + BlasMatIntraImpl + BlasMatMulImpl + BlasJklsImpl {}

unsafe impl<BE: BlasMatInterImpl + BlasMatIntraImpl + BlasMatMulImpl + BlasJklsImpl> BlasImpl for BE {}
