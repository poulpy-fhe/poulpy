//! Backend-parametric test functions.
//!
//! Provides fully generic test functions that can be instantiated for any
//! backend via the [`backend_test_suite!`](crate::backend_test_suite) and
//! [`cross_backend_test_suite!`](crate::cross_backend_test_suite) macros.
//! Tests validate correctness against the reference implementation in
//! [`poulpy-cpu-ref`](https://docs.rs/poulpy-cpu-ref).

use crate::layouts::{
    Backend, DataView, HostBytesBackend, HostDataRef, MatZnx, ScalarZnx, ScalarZnxBackendRef, ScalarZnxToBackendRef, VecZnx,
    VecZnxBackendMut, VecZnxBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
};

pub mod convolution;
pub mod serialization;
pub mod svp;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

/// Parameters passed to every test function in a
/// [`backend_test_suite!`](crate::backend_test_suite) or
/// [`cross_backend_test_suite!`](crate::cross_backend_test_suite).
///
/// Centralising these values at the macro call-site makes it possible to
/// instantiate the same test suite with backend-appropriate parameters
/// (e.g. different `base2k` for FFT64 vs NTT120).
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
pub trait TestBackend: Backend {}

impl<BE: Backend> TestBackend for BE {}

pub fn vec_znx_backend_ref<'a, BE: Backend>(vec: &'a VecZnx<BE::OwnedBuf>) -> VecZnxBackendRef<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(vec)
}

pub fn vec_znx_backend_mut<'a, BE: Backend>(vec: &'a mut VecZnx<BE::OwnedBuf>) -> VecZnxBackendMut<'a, BE> {
    <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(vec)
}

pub fn scalar_znx_backend_ref<'a, BE: Backend>(scalar: &'a ScalarZnx<BE::OwnedBuf>) -> ScalarZnxBackendRef<'a, BE> {
    <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(scalar)
}

pub fn upload_scalar_znx<BE: Backend>(host: &ScalarZnx<impl HostDataRef>) -> ScalarZnx<BE::OwnedBuf> {
    let shape = host.shape();
    ScalarZnx::from_data(BE::from_host_bytes(host.data.as_ref()), shape.n(), shape.cols())
}

pub fn download_scalar_znx<BE: Backend>(backend: &ScalarZnx<BE::OwnedBuf>) -> ScalarZnx<Vec<u8>> {
    let shape = backend.shape();
    let host_bytes = BE::to_host_bytes(&backend.data);
    ScalarZnx::from_data(HostBytesBackend::from_host_bytes(&host_bytes), shape.n(), shape.cols())
}

pub fn upload_vec_znx<BE: Backend>(host: &VecZnx<impl HostDataRef>) -> VecZnx<BE::OwnedBuf> {
    let shape = host.shape();
    VecZnx::from_data_with_max_size(
        BE::from_host_bytes(host.data.as_ref()),
        shape.n(),
        shape.cols(),
        shape.size(),
        shape.max_size(),
    )
}

pub fn download_vec_znx<BE: Backend>(backend: &VecZnx<BE::OwnedBuf>) -> VecZnx<Vec<u8>> {
    let shape = backend.shape();
    let host_bytes = BE::to_host_bytes(&backend.data);
    VecZnx::from_data_with_max_size(
        HostBytesBackend::from_host_bytes(&host_bytes),
        shape.n(),
        shape.cols(),
        shape.size(),
        shape.max_size(),
    )
}

pub fn upload_mat_znx<BE: Backend>(host: &MatZnx<impl HostDataRef>) -> MatZnx<BE::OwnedBuf> {
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

pub fn download_mat_znx<BE: Backend>(backend: &MatZnx<BE::OwnedBuf>) -> MatZnx<Vec<u8>> {
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
