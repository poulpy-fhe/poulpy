//! Caller-selected paired CKKS conformance. A validated backend can bootstrap
//! another over the same supported operations and parameter ranges. Inputs are
//! transferred explicitly; prepared objects and scratch remain backend-owned.
mod bootstrapping;
mod dft;
mod fold;
pub mod helpers;
pub mod keys;
mod polynomial_evaluation;
mod replicated;
pub use bootstrapping::*;
pub use dft::*;
pub use fold::*;
pub use polynomial_evaluation::*;
pub use replicated::*;
mod arithmetic;
mod encryption;
mod plaintext;
pub use arithmetic::*;
pub use encryption::*;
pub use plaintext::*;

/// Registers paired tests for a caller-selected backend pair, scalar type and ring.
/// Each helper of `tests` is called as `helper::<BR, BT, F, Rg>(params, comparison_module, tested_module)`;
/// a test may carry attributes such as `#[ignore]`. The optional `ring = ..` argument
/// is `::poulpy_hal::layouts::Standard` when absent, and the conjugate-invariant lists pass
/// `ring = ::poulpy_hal::layouts::ConjugateInvariant`.
/// A helper of `ring_free_tests` takes no ring and is called as `helper::<BR, BT, F>(..)`:
/// a test of standard-only or cross-ring operations, or one fixing `Standard`.
#[macro_export]
macro_rules! ckks_parity_test_suite {
    (mod $name:ident, backend_ref = $reference:ty, backend_test = $tested:ty,
     scalar = $scalar:ty, params = $params:expr
     $(, tests = { $($(#[$attr:meta])* $test:ident => $helper:path),* $(,)? })?
     $(, ring_free_tests = { $($(#[$fattr:meta])* $ftest:ident => $fhelper:path),* $(,)? })? $(,)?) => {
        $crate::ckks_parity_test_suite!(
            mod $name,
            backend_ref = $reference,
            backend_test = $tested,
            ring = ::poulpy_hal::layouts::Standard,
            scalar = $scalar,
            params = $params
            $(, tests = { $($(#[$attr])* $test => $helper),* })?
            $(, ring_free_tests = { $($(#[$fattr])* $ftest => $fhelper),* })?
        );
    };
    (mod $name:ident, backend_ref = $reference:ty, backend_test = $tested:ty, ring = $ring:ty,
     scalar = $scalar:ty, params = $params:expr
     $(, tests = { $($(#[$attr:meta])* $test:ident => $helper:path),* $(,)? })?
     $(, ring_free_tests = { $($(#[$fattr:meta])* $ftest:ident => $fhelper:path),* $(,)? })? $(,)?) => {
        mod $name {
            $($($(#[$attr])*
            #[test]
            fn $test() {
                use $helper as helper;
                let params = $params;
                let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                let tested = ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64);
                helper::<$reference, $tested, $scalar, $ring>(params, &reference, &tested);
            })*)?

            $($($(#[$fattr])*
            #[test]
            fn $ftest() {
                use $fhelper as helper;
                let params = $params;
                let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                let tested = ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64);
                helper::<$reference, $tested, $scalar>(params, &reference, &tested);
            })*)?
        }
    };
}

mod encoding;
pub use encoding::*;
