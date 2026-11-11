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

use poulpy_hal::layouts::{Ring, Standard};
pub(crate) mod derived;
mod encoding;
mod inter;
mod intra;
mod jkls;
mod matmul;
mod matvec;

pub use encoding::BlasEncodingImpl;
pub use inter::BlasMatInterImpl;
pub use intra::BlasMatIntraImpl;
pub use jkls::BlasJklsImpl;
pub use matmul::BlasMatMulImpl;
pub use matvec::BlasMatVecImpl;

/// Selects every reference implementation for `$be` on both rings, or on `$ring` alone through `($be, $ring)`.
#[macro_export]
macro_rules! impl_blas_reference_full {
    ($be:ty) => {
        $crate::impl_blas_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_blas_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        $crate::impl_blas_mat_inter_reference_full!($be, $ring);
        $crate::impl_blas_mat_intra_reference_full!($be, $ring);
        $crate::impl_blas_mat_mul_reference_full!($be, $ring);
        $crate::impl_blas_mat_vec_reference_full!($be, $ring);
        $crate::impl_blas_jkls_reference_full!($be, $ring);
        $crate::impl_blas_encoding_reference_full!($be, $ring);
    };
}

/// Every blas extension point; a test bound, never implemented directly.
///
/// # Safety
///
/// Implemented only through the blanket impl.
pub unsafe trait BlasImpl<Rg: Ring = Standard>
where
    Self: BlasMatInterImpl<Rg>
        + BlasMatIntraImpl<Rg>
        + BlasMatMulImpl<Rg>
        + BlasMatVecImpl<Rg>
        + BlasJklsImpl<Rg>
        + BlasEncodingImpl<Rg>,
{
}

unsafe impl<BE, Rg: Ring> BlasImpl<Rg> for BE where
    BE: BlasMatInterImpl<Rg>
        + BlasMatIntraImpl<Rg>
        + BlasMatMulImpl<Rg>
        + BlasMatVecImpl<Rg>
        + BlasJklsImpl<Rg>
        + BlasEncodingImpl<Rg>
{
}
