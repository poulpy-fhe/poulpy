//! Backend-parametric test functions.
//!
//! Provides fully generic test functions that can be instantiated for any
//! backend via the [`backend_test_suite!`](crate::backend_test_suite) and
//! [`cross_backend_test_suite!`](crate::cross_backend_test_suite) macros.
//! Tests validate correctness against the reference implementation in
//! [`poulpy-cpu-ref`](https://docs.rs/poulpy-cpu-ref).

use crate::layouts::{
    Backend, DataView, HostBytesBackend, HostDataRef, MatZnx, NormalizationState, ScalarZnx, ScalarZnxBackendMut,
    ScalarZnxBackendRef, ScalarZnxToBackendMut, ScalarZnxToBackendRef, VecZnx, VecZnxBackendMut, VecZnxOwned,
};

pub mod convolution;
pub mod serialization;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;
pub mod word_compat;

/// Parameters passed to every test function in a
/// [`backend_test_suite!`](crate::backend_test_suite) or
/// [`cross_backend_test_suite!`](crate::cross_backend_test_suite).
///
/// Centralising these values at the macro call-site makes it possible to
/// instantiate the same test suite with backend-appropriate parameters
/// (e.g. different `base2k` for FFT64 vs NTT4x30).
#[derive(Clone, Copy, Debug)]
pub struct TestParams {
    /// Ring degree N (polynomial degree).
    pub size: usize,
    /// Primary decomposition base (limbs are base-2^`base2k`).
    ///
    /// Secondary base values used inside individual tests are derived from
    /// this value via fixed offsets that preserve the original relative
    /// relationships between bases.
    pub base2k: usize,
}

/// Backend bound used by the generic test suites.
///
/// Tests upload only coefficient-domain host layouts (`ScalarZnx`, `VecZnx`,
/// `MatZnx`) and keep all intermediate layouts backend-local.
///
/// Pinned to `ZnxWord = i64`: the suites drive `encode_*`/`decode_*` and
/// `FillUniform`, which are i64-only. A backend with a narrower coefficient
/// word needs its own suites.
pub trait TestBackend: Backend<ZnxWord = i64> {}

impl<BE: Backend<ZnxWord = i64>> TestBackend for BE {}

pub use crate::layouts::{vec_znx_backend_mut, vec_znx_backend_ref};

/// Mutable backend view narrowed to `size` limbs, leaving the allocation intact.
///
/// Lets a test hand a kernel fewer limbs than were allocated, so that writes
/// past `size` show up in the full-buffer comparison.
pub fn vec_znx_backend_mut_sized<'a, BE: Backend, S: NormalizationState>(
    vec: &'a mut VecZnx<BE::OwnedBuf, BE::ZnxWord, S>,
    size: usize,
) -> VecZnxBackendMut<'a, BE, S> {
    crate::layouts::vec_znx_backend_mut_with_size::<BE, _>(vec_znx_backend_mut::<BE, _>(vec), size)
}

pub fn scalar_znx_backend_ref<'a, BE: Backend>(scalar: &'a ScalarZnx<BE::OwnedBuf, BE::ZnxWord>) -> ScalarZnxBackendRef<'a, BE> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(scalar)
}

pub fn scalar_znx_backend_mut<'a, BE: Backend>(
    scalar: &'a mut ScalarZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> ScalarZnxBackendMut<'a, BE> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendMut<BE>>::to_backend_mut(scalar)
}

/// Zeroed host template sized for `BE`'s coefficient word.
///
/// Sizing has to use `BE::ZnxWord` and not the host's own word: the result is
/// uploaded to `BE`, and a backend with a narrower word needs a proportionally
/// smaller buffer for the same shape.
pub fn alloc_host_vec_znx<BE: Backend>(n: usize, cols: usize, size: usize) -> VecZnxOwned<BE::ZnxWord> {
    VecZnx::from_data_with_state(
        crate::alloc_aligned::<u8>(VecZnxOwned::<BE::ZnxWord>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

pub fn upload_scalar_znx<BE: Backend>(host: &ScalarZnx<impl HostDataRef, BE::ZnxWord>) -> ScalarZnx<BE::OwnedBuf, BE::ZnxWord> {
    let shape = host.shape();
    ScalarZnx::from_data(BE::from_host_bytes(host.data.as_ref()), shape.n(), shape.cols())
}

pub fn download_scalar_znx<BE: Backend>(backend: &ScalarZnx<BE::OwnedBuf, BE::ZnxWord>) -> ScalarZnx<Vec<u8>, BE::ZnxWord> {
    let shape = backend.shape();
    let host_bytes = BE::to_host_bytes(&backend.data);
    ScalarZnx::from_data(HostBytesBackend::from_host_bytes(&host_bytes), shape.n(), shape.cols())
}

pub fn upload_vec_znx<BE: Backend, S: NormalizationState>(
    host: &VecZnx<impl HostDataRef, BE::ZnxWord, S>,
) -> VecZnx<BE::OwnedBuf, BE::ZnxWord, S> {
    let shape = host.shape();
    VecZnx::from_data_with_state(BE::from_host_bytes(host.data.as_ref()), shape.n(), shape.cols(), shape.size())
}

pub fn download_vec_znx<BE: Backend, S: NormalizationState>(
    backend: &VecZnx<BE::OwnedBuf, BE::ZnxWord, S>,
) -> VecZnx<Vec<u8>, BE::ZnxWord, S> {
    let shape = backend.shape();
    let host_bytes = BE::to_host_bytes(&backend.data);
    VecZnx::from_data_with_state(
        HostBytesBackend::from_host_bytes(&host_bytes),
        shape.n(),
        shape.cols(),
        shape.size(),
    )
}

pub fn upload_mat_znx<BE: Backend>(host: &MatZnx<impl HostDataRef, BE::ZnxWord>) -> MatZnx<BE::OwnedBuf, BE::ZnxWord> {
    let shape = host.shape();
    MatZnx::from_data(
        BE::from_host_bytes(host.data().as_ref()),
        shape.n(),
        shape.rows(),
        shape.cols_in(),
        shape.cols_out(),
        shape.size(),
    )
}

pub fn download_mat_znx<BE: Backend>(backend: &MatZnx<BE::OwnedBuf, BE::ZnxWord>) -> MatZnx<Vec<u8>, BE::ZnxWord> {
    let shape = backend.shape();
    let host_bytes = BE::to_host_bytes(backend.data());
    MatZnx::from_data(
        HostBytesBackend::from_host_bytes(&host_bytes),
        shape.n(),
        shape.rows(),
        shape.cols_in(),
        shape.cols_out(),
        shape.size(),
    )
}

#[macro_export]
macro_rules! backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE: Lazy<Module<$backend>> =
                Lazy::new(|| Module::<$backend>::new(PARAMS.size as u64));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*PARAMS, &*MODULE);
                }
            )+
        }
    };
}

#[macro_export]
macro_rules! cross_backend_test_suite {
    (
        mod $modname:ident,
        backend_ref = $backend_ref:ty,
        backend_test = $backend_test:ty,
        params = $params:expr,
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::{HostBytesBackend, Module}, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE_HOST: Lazy<Module<HostBytesBackend>> =
                Lazy::new(|| Module::<HostBytesBackend>::new(PARAMS.size as u64));
            static MODULE_REF: Lazy<Module<$backend_ref>> =
                Lazy::new(|| Module::<$backend_ref>::new(PARAMS.size as u64));
            static MODULE_TEST: Lazy<Module<$backend_test>> =
                Lazy::new(|| Module::<$backend_test>::new(PARAMS.size as u64));

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*PARAMS, &*MODULE_HOST, &*MODULE_REF, &*MODULE_TEST);
                }
            )+
        }
    };
}
