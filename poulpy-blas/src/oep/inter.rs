use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatInterOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// `reference::BlasMatInterReference` bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
///
/// A method first runs [`crate::layouts::validate_blas_rotate`] or [`crate::layouts::validate_blas_sum`], before any
/// write, and returns its errors with `res` and its batch record untouched. On inputs it accepted, it returns `Ok`,
/// after stamping `res` with the batch the validator returns ([`crate::layouts::BlasGridMut::set_mat_slots`]), or a
/// budget error: that error is the reference's, the one poulpy-ckks raises, returned before any write to `res`. Any
/// other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
pub unsafe trait BlasMatInterImpl: Backend {
    fn blas_rotate_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        r: i64,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_sum_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_sum_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;
}
