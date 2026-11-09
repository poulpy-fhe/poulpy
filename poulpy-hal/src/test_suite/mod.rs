//! Backend-parametric test functions.
//!
//! Provides fully generic test functions that can be instantiated for any
//! backend via the [`backend_test_suite!`](crate::backend_test_suite) and
//! [`cross_backend_test_suite!`](crate::cross_backend_test_suite) macros.
//! Tests validate correctness against the reference implementation in
//! [`poulpy-cpu-portable`](https://docs.rs/poulpy-cpu-portable).

use crate::AlignedBuf;
use crate::layouts::{
    Backend, DataView, HostDataRef, MatZnx, Ring, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScalarZnxToBackendMut,
    ScalarZnxToBackendRef, Standard, VecZnx, VecZnxBackendMut, VecZnxOwned,
};
use crate::oep::HalVecZnxImpl;

pub mod convolution;
pub mod derived;
pub mod embed;
pub mod mat_znx;
pub mod reim;
pub mod serialization;
pub mod svp;
pub mod transfer;
pub mod vec_znx;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;
pub mod window;
pub mod word_compat;

/// Parameters passed to every test function in a
/// [`backend_test_suite!`](crate::backend_test_suite) or
/// [`cross_backend_test_suite!`](crate::cross_backend_test_suite).
///
/// Centralising these values at the macro call-site makes it possible to
/// instantiate the same test suite with backend-appropriate parameters
/// (e.g. different `base2k` for a floating-point FFT backend vs an NTT backend).
#[derive(Clone, Copy, Debug)]
pub struct TestParams {
    /// Ring degree N the module is built at.
    pub size: usize,
    /// Primary decomposition base (limbs are base-2^`base2k`).
    ///
    /// Secondary base values used inside individual tests are derived from
    /// this value via fixed offsets that preserve the original relative
    /// relationships between bases.
    pub base2k: usize,
    /// Operand degree `n`, a power of two at most `size`; every test
    /// allocates and computes at `n`. The suite macros run each test at `n`
    /// and, when it differs, at `size`.
    pub n: usize,
}

/// The degrees the suite macros run a test at: `params.n`, then `params.size`
/// when it differs.
pub fn sweep_degrees(params: &TestParams) -> Vec<usize> {
    if params.n == params.size {
        vec![params.size]
    } else {
        vec![params.n, params.size]
    }
}

/// Backend bound used by the generic test suites.
///
/// Tests upload only coefficient-domain host layouts (`ScalarZnx`, `VecZnx`,
/// `MatZnx`) and keep all intermediate layouts backend-local.
///
/// Pinned to `ZnxWord = i64`: the suites drive `encode_*`/`decode_*` and
/// `vec_znx_fill_uniform_source`, which are i64-only. A backend with a narrower
/// coefficient word needs its own suites. The `Standard` impl answers the ring-free queries.
pub trait TestBackend<Rg: Ring = Standard>: Backend<ZnxWord = i64> + HalVecZnxImpl<Rg> + HalVecZnxImpl<Standard> {}

impl<BE: Backend<ZnxWord = i64> + HalVecZnxImpl<Rg> + HalVecZnxImpl<Standard>, Rg: Ring> TestBackend<Rg> for BE {}

pub use crate::layouts::{vec_znx_backend_mut, vec_znx_backend_ref};

/// A scratch arena of exactly `bytes`, every byte filled with `0xA5` so that an operation reading scratch it did not
/// write, or needing more than its query reported, shows in the result.
pub(crate) fn poisoned_arena<BE: Backend>(bytes: usize) -> PoisonedArena<BE> {
    PoisonedArena {
        owned: crate::layouts::ScratchOwned {
            data: BE::from_host_bytes(&vec![0xA5; BE::scratch_aligned(bytes)]),
            _phantom: std::marker::PhantomData,
        },
        bytes,
    }
}

/// The owner of a [`poisoned_arena`].
pub(crate) struct PoisonedArena<BE: Backend> {
    owned: crate::layouts::ScratchOwned<BE>,
    bytes: usize,
}

impl<BE: Backend> PoisonedArena<BE> {
    /// The first `bytes` of the buffer: the allocation is padded to the backend's alignment, the arena is not.
    pub(crate) fn arena(&mut self) -> crate::layouts::ScratchArena<'_, BE> {
        self.owned.arena().split_at(self.bytes).0
    }

    /// Same as [`Self::arena`], under the name `ScratchOwnedBorrow` gives a `ScratchOwned`.
    pub(crate) fn borrow(&mut self) -> crate::layouts::ScratchArena<'_, BE> {
        self.arena()
    }
}

/// Mutable backend view narrowed to `size` limbs, leaving the allocation intact.
///
/// Lets a test hand a kernel fewer limbs than were allocated, so that writes
/// past `size` show up in the full-buffer comparison.
pub fn vec_znx_backend_mut_sized<'a, BE: Backend, Rg: Ring>(
    vec: &'a mut VecZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
    size: usize,
) -> VecZnxBackendMut<'a, BE, Rg> {
    crate::layouts::vec_znx_backend_mut_with_size::<BE, Rg>(vec_znx_backend_mut::<BE, Rg>(vec), size)
}

pub fn scalar_znx_backend_ref<'a, BE: Backend, Rg: Ring>(
    scalar: &'a ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> ScalarZnxBackendRef<'a, BE, Rg> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendRef<BE, Rg>>::to_backend_ref(scalar)
}

pub fn scalar_znx_backend_mut<'a, BE: Backend, Rg: Ring>(
    scalar: &'a mut ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> ScalarZnxBackendMut<'a, BE, Rg> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendMut<BE, Rg>>::to_backend_mut(scalar)
}

/// Zeroed host template sized for `BE`'s coefficient word.
///
/// Sizing has to use `BE::ZnxWord` and not the host's own word: the result is
/// uploaded to `BE`, and a backend with a narrower word needs a proportionally
/// smaller buffer for the same shape.
pub fn alloc_host_vec_znx<BE: Backend, Rg: Ring>(n: usize, cols: usize, size: usize) -> VecZnxOwned<BE::ZnxWord, Rg> {
    VecZnx::from_data(
        crate::alloc_aligned::<u8>(VecZnxOwned::<BE::ZnxWord, Rg>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

pub fn upload_scalar_znx<BE: Backend, Rg: Ring>(
    host: &ScalarZnx<impl HostDataRef, BE::ZnxWord, Rg>,
) -> ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let shape = host.shape();
    ScalarZnx::from_data(BE::from_host_bytes(host.data.as_ref()), shape.n(), shape.cols())
}

pub fn download_scalar_znx<BE: Backend, Rg: Ring>(
    backend: &ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> ScalarZnx<AlignedBuf, BE::ZnxWord, Rg> {
    let shape = backend.shape();
    ScalarZnx::from_data(AlignedBuf::from(BE::to_host_bytes(&backend.data)), shape.n(), shape.cols())
}

pub fn upload_vec_znx<BE: Backend, Rg: Ring>(
    host: &VecZnx<impl HostDataRef, BE::ZnxWord, Rg>,
) -> VecZnx<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let shape = host.shape();
    VecZnx::from_shape(BE::from_host_bytes(host.data().as_ref()), shape)
}

pub fn download_vec_znx<BE: Backend, Rg: Ring>(
    backend: &VecZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> VecZnx<AlignedBuf, BE::ZnxWord, Rg> {
    let shape = backend.shape();
    VecZnx::from_shape(AlignedBuf::from(BE::to_host_bytes(backend.data())), shape)
}

pub fn upload_mat_znx<BE: Backend, Rg: Ring>(
    host: &MatZnx<impl HostDataRef, BE::ZnxWord, Rg>,
) -> MatZnx<BE::OwnedBuf, BE::ZnxWord, Rg> {
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

pub fn download_mat_znx<BE: Backend, Rg: Ring>(
    backend: &MatZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> MatZnx<AlignedBuf, BE::ZnxWord, Rg> {
    let shape = backend.shape();
    MatZnx::from_data(
        AlignedBuf::from(BE::to_host_bytes(backend.data())),
        shape.n(),
        shape.rows(),
        shape.cols_in(),
        shape.cols_out(),
        shape.size(),
    )
}

/// Registers HAL suite tests on `backend`, each run at every degree of
/// [`sweep_degrees`](crate::test_suite::sweep_degrees) on a module of degree `params.size`.
///
/// An entry of `tests` is called as `path::<backend, ring>(&TestParams, &Module<backend>)`, the ring `Standard`
/// unless `ring = ..` names it. An entry of `ring_free_tests` takes no ring and is called as `path::<backend>(..)`:
/// a test of standard-only or cross-ring operations, whose operands have fixed rings, of operations serving both rings
/// that it fixes at `Standard`, or of no ringed layout.
#[macro_export]
macro_rules! backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr
        $(, tests = { $( $(#[$attr:meta])* $test_name:ident => $impl:path ),* $(,)? })?
        $(, ring_free_tests = { $( $(#[$fattr:meta])* $ftest_name:ident => $fimpl:path ),* $(,)? })?
        $(,)?
    ) => {
        $crate::backend_test_suite!(
            mod $modname,
            backend = $backend,
            ring = $crate::layouts::Standard,
            params = $params
            $(, tests = { $( $(#[$attr])* $test_name => $impl ),* })?
            $(, ring_free_tests = { $( $(#[$fattr])* $ftest_name => $fimpl ),* })?
        );
    };
    (
        mod $modname:ident,
        backend = $backend:ty,
        ring = $ring:ty,
        params = $params:expr
        $(, tests = { $( $(#[$attr:meta])* $test_name:ident => $impl:path ),* $(,)? })?
        $(, ring_free_tests = { $( $(#[$fattr:meta])* $ftest_name:ident => $fimpl:path ),* $(,)? })?
        $(,)?
    ) => {
        mod $modname {
            use $crate::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE: Lazy<Module<$backend>> =
                Lazy::new(|| Module::<$backend>::new(PARAMS.size as u64));

            $($(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    use $impl as helper;
                    for n in $crate::test_suite::sweep_degrees(&PARAMS) {
                        let params = TestParams { n, ..*PARAMS };
                        helper::<$backend, $ring>(&params, &*MODULE);
                    }
                }
            )*)?

            $($(
                $(#[$fattr])*
                #[test]
                fn $ftest_name() {
                    use $fimpl as helper;
                    for n in $crate::test_suite::sweep_degrees(&PARAMS) {
                        let params = TestParams { n, ..*PARAMS };
                        helper::<$backend>(&params, &*MODULE);
                    }
                }
            )*)?
        }
    };
}

/// Registers HAL suite tests comparing `backend_test` with `backend_ref`, each run at every degree of
/// [`sweep_degrees`](crate::test_suite::sweep_degrees) on modules of degree `params.size`.
///
/// An entry of `tests` is called as `path::<backend_ref, backend_test, ring>(&TestParams, &Module<backend_ref>,
/// &Module<backend_test>)`, the ring `Standard` unless `ring = ..` names it. An entry of `ring_free_tests` takes no
/// ring and is called as `path::<backend_ref, backend_test>(..)`: a test of standard-only or cross-ring operations,
/// whose operands have fixed rings, of operations serving both rings that it fixes at `Standard`, or of no ringed
/// layout.
#[macro_export]
macro_rules! cross_backend_test_suite {
    (
        mod $modname:ident,
        backend_ref = $backend_ref:ty,
        backend_test = $backend_test:ty,
        params = $params:expr
        $(, tests = { $( $(#[$attr:meta])* $test_name:ident => $impl:path ),* $(,)? })?
        $(, ring_free_tests = { $( $(#[$fattr:meta])* $ftest_name:ident => $fimpl:path ),* $(,)? })?
        $(,)?
    ) => {
        $crate::cross_backend_test_suite!(
            mod $modname,
            backend_ref = $backend_ref,
            backend_test = $backend_test,
            ring = $crate::layouts::Standard,
            params = $params
            $(, tests = { $( $(#[$attr])* $test_name => $impl ),* })?
            $(, ring_free_tests = { $( $(#[$fattr])* $ftest_name => $fimpl ),* })?
        );
    };
    (
        mod $modname:ident,
        backend_ref = $backend_ref:ty,
        backend_test = $backend_test:ty,
        ring = $ring:ty,
        params = $params:expr
        $(, tests = { $( $(#[$attr:meta])* $test_name:ident => $impl:path ),* $(,)? })?
        $(, ring_free_tests = { $( $(#[$fattr:meta])* $ftest_name:ident => $fimpl:path ),* $(,)? })?
        $(,)?
    ) => {
        mod $modname {
            use $crate::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static MODULE_REF: Lazy<Module<$backend_ref>> =
                Lazy::new(|| Module::<$backend_ref>::new(PARAMS.size as u64));
            static MODULE_TEST: Lazy<Module<$backend_test>> =
                Lazy::new(|| Module::<$backend_test>::new(PARAMS.size as u64));

            $($(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    use $impl as helper;
                    for n in $crate::test_suite::sweep_degrees(&PARAMS) {
                        let params = TestParams { n, ..*PARAMS };
                        helper::<$backend_ref, $backend_test, $ring>(&params, &*MODULE_REF, &*MODULE_TEST);
                    }
                }
            )*)?

            $($(
                $(#[$fattr])*
                #[test]
                fn $ftest_name() {
                    use $fimpl as helper;
                    for n in $crate::test_suite::sweep_degrees(&PARAMS) {
                        let params = TestParams { n, ..*PARAMS };
                        helper::<$backend_ref, $backend_test>(&params, &*MODULE_REF, &*MODULE_TEST);
                    }
                }
            )*)?
        }
    };
}

#[cfg(test)]
mod tests {
    use super::poisoned_arena;
    use crate::layouts::HostBytesBackend;

    #[test]
    fn poisoned_arena_is_exact() {
        let mut owner = poisoned_arena::<HostBytesBackend>(511);
        assert_eq!(owner.arena().available(), 511);
        let (region, _) = owner.arena().take_region(511);
        assert!(region.iter().all(|&b| b == 0xA5), "the arena is not poisoned");
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.arena().take_region(512);
        }))
        .is_err();
        assert!(refused, "a 511-byte arena accepted a 512-byte carve");
    }
}
