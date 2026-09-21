//! Caller-selected paired CKKS conformance. A validated backend can bootstrap
//! another over the same supported operations and parameter ranges. Inputs are
//! transferred explicitly; prepared objects and scratch remain backend-owned.
mod bootstrapping;
mod dft;
pub(crate) mod helpers;
pub(crate) mod keys;
mod polynomial_evaluation;
pub use bootstrapping::*;
pub use dft::*;
pub use polynomial_evaluation::*;
mod arithmetic;
mod encryption;
mod plaintext;
pub use arithmetic::*;
pub use encryption::*;
pub use plaintext::*;

/// Registers paired tests for a caller-selected backend pair and scalar type.
/// Each helper receives `(params, comparison_module, tested_module)`.
#[macro_export]
macro_rules! ckks_parity_test_suite {
    (mod $name:ident, backend_ref = $reference:ty, backend_test = $tested:ty,
     scalar = $scalar:ty, params = $params:expr,
     tests = { $($test:ident => $helper:path),+ $(,)? } $(,)?) => {
        mod $name {
            $(#[test]
            fn $test() {
                use $helper as helper;
                let params = $params;
                let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                let tested = ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64);
                helper::<$reference, $tested, $scalar>(params, &reference, &tested);
            })+
        }
    };
}

mod encoding;
pub use encoding::*;
