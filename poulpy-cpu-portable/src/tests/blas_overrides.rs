//! Downstream BLAS hooks: counting map and JKLS hooks that forward to the reference, under the provided
//! compositions, are entered by both preparations and by both product entry points.
use crate::hal_impl::delegating_backend::DifferentSamplingFFT64Portable as OverrideBackend;
use poulpy_blas::{
    oep::{BlasJklsImpl, BlasMatIntraImpl},
    reference::{BlasMatIntraReference, BlasMatMulReference},
};
use poulpy_ckks::{CKKSCtBounds, CKKSResult, SetCKKSInfos, api::LinearTransformationPrepared};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{HostBytesBackend, Module, ScratchArena};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

// Atomics, so that hooks entered from the workers of a parallel executor count too.
static MAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static JKLS_CALLS: AtomicUsize = AtomicUsize::new(0);

poulpy_blas::impl_blas_inter_reference!(OverrideBackend);
poulpy_blas::impl_blas_matmul_reference!(OverrideBackend);

unsafe impl BlasMatIntraImpl for OverrideBackend {
    fn blas_map_blocks_tmp_bytes_impl<D, S, K>(
        module: &Module<Self>,
        dst: &D,
        src: &S,
        sources: usize,
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
    ) -> usize
    where
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        module.blas_map_blocks_tmp_bytes_reference(dst, src, sources, lts, atk)
    }

    fn blas_map_blocks_impl<D, S, K>(
        module: &Module<Self>,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + Sync,
    {
        MAP_CALLS.fetch_add(1, Relaxed);
        module.blas_map_blocks_reference(dst, src, lts, atk, scratch)
    }
}

unsafe impl BlasJklsImpl for OverrideBackend {
    fn blas_jkls_block_tmp_bytes_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &R,
        lhs: &L,
        rhs: &Rh,
        terms: usize,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        module.blas_jkls_block_tmp_bytes_reference(res, lhs, rhs, terms, tsk)
    }

    fn blas_jkls_block_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &[&L],
        rhs: &[&Rh],
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        L: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        Rh: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        T: GetTensorKey<Self> + Sync,
    {
        JKLS_CALLS.fetch_add(1, Relaxed);
        module.blas_jkls_block_reference(res, lhs, rhs, tsk, scratch)
    }
}

/// The functional product test on the counting hooks: a `2 x 3` grid times a `3 x 2` grid by `blas_matmul`, then by
/// both preparations and `blas_matmul_prepared`. Each preparation makes two map-hook calls (sigma, then every shift),
/// and each product one JKLS call per block of the `2 x 2` result.
#[test]
fn blas_products_enter_the_map_and_jkls_hooks() {
    let params = poulpy_ckks::test_suite::BASE19_PARAMS_F64;
    let module = Module::<OverrideBackend>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    poulpy_blas::test_suite::matmul::test_blas_matmul::<OverrideBackend, f64, crate::ckks_encoding::EncodingFFTTable<f64>>(
        params,
        &module,
        &host_module,
        2,
    );
    assert_eq!(MAP_CALLS.load(Relaxed), 2 * 2 * 2, "map hook calls");
    assert_eq!(JKLS_CALLS.load(Relaxed), 2 * 4, "JKLS kernel calls");
}
