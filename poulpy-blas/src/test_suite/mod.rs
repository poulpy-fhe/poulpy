//! Backend-generic poulpy-blas tests.
//!
//! Functional tests take `(params, module, host_module)` and are instantiated per backend by
//! [`blas_backend_test_suite!`](crate::blas_backend_test_suite); parity tests take
//! `(params, reference_module, tested_module)` and are instantiated with
//! `poulpy_ckks::ckks_parity_test_suite!`.

pub mod helpers;
pub mod inter;
pub mod intra;
pub mod keys;
pub mod matmul;
pub mod meta;
pub mod parity;

/// Registers the functional tests for one backend, scalar, encoder and parameter set.
#[macro_export]
macro_rules! blas_backend_test_suite {
    (mod $modname:ident, backend = $backend:ty, scalar = $scalar:ty, encoder = $encoder:ty, params = $params:expr $(,)?) => {
        mod $modname {
            use std::sync::LazyLock;

            use poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> = LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder>($params, &*MODULE, &*HOST_MODULE);
                    }
                };
            }

            run_test!(rotate, $crate::test_suite::inter::test_blas_rotate);
            run_test!(sum, $crate::test_suite::inter::test_blas_sum);
            run_test!(inter_guards, $crate::test_suite::inter::test_blas_inter_guards);
            run_test!(transpose, $crate::test_suite::intra::test_blas_transpose);
            run_test!(rotate_rows_cols, $crate::test_suite::intra::test_blas_rotate_rows_cols);
            run_test!(intra_guards, $crate::test_suite::intra::test_blas_intra_guards);
            run_test!(matmul, $crate::test_suite::matmul::test_blas_matmul);
            run_test!(matmul_pt, $crate::test_suite::matmul::test_blas_matmul_pt);
            run_test!(matmul_guards, $crate::test_suite::matmul::test_blas_matmul_guards);
            run_test!(matmul_pt_guards, $crate::test_suite::matmul::test_blas_matmul_pt_guards);
            run_test!(matmul_long_sums, $crate::test_suite::matmul::test_blas_matmul_long_sums);
            run_test!(key_widths, $crate::test_suite::keys::test_blas_key_widths);
            run_test!(result_meta, $crate::test_suite::meta::test_blas_result_meta);
        }
    };
}
