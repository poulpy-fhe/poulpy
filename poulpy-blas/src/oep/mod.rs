//! Backend extension points. A backend selects the reference with
//! [`impl_blas_reference_full!`](crate::impl_blas_reference_full) or overrides a family; an
//! override reproduces the reference bit for bit within the scratch its `_tmp_bytes_impl` reports.

mod inter;
mod intra;
mod matmul;

pub use inter::BlasMatInterImpl;
pub use intra::BlasMatIntraImpl;
pub use matmul::BlasMatMulImpl;

/// Selects every reference implementation for `$be`.
#[macro_export]
macro_rules! impl_blas_reference_full {
    ($be:ty) => {
        $crate::impl_blas_inter_reference!($be);
        $crate::impl_blas_intra_reference!($be);
        $crate::impl_blas_matmul_reference!($be);
    };
}

/// Every blas extension point; a test bound, never implemented directly.
///
/// # Safety
///
/// Implemented only through the blanket impl.
pub unsafe trait BlasImpl: BlasMatInterImpl + BlasMatIntraImpl + BlasMatMulImpl {}

unsafe impl<BE: BlasMatInterImpl + BlasMatIntraImpl + BlasMatMulImpl> BlasImpl for BE {}
