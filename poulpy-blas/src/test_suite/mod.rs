//! Backend-generic poulpy-blas tests.
//!
//! Functional tests take `(params, module, host_module)`, most of them then the block dimension's `log_d`, and are
//! instantiated per backend by [`blas_backend_test_suite!`](crate::blas_backend_test_suite), or on the
//! conjugate-invariant ring by [`blas_conjugate_invariant_test_suite!`](crate::blas_conjugate_invariant_test_suite);
//! parity tests take `(params, shapes, reference_module, tested_module)` and are instantiated by
//! [`blas_parity_test_suite!`](crate::blas_parity_test_suite). The backend guide, `docs/blas-contracts.md`, covers
//! wiring, registration and attestation.

pub mod conjugate_invariant;
pub mod context;
pub mod helpers;
pub mod inter;
pub mod intra;
pub mod keys;
pub mod matmul;
pub mod meta;
pub mod parity;

/// Registers the functional tests for one backend, scalar, encoder and parameter set, on blocks of dimension
/// `2^log_d`: 2 unless the optional `log_d` sets it. `test_blas_context` sweeps its own dimensions and
/// `test_blas_matmul_long_sums` runs on its own ring.
#[macro_export]
macro_rules! blas_backend_test_suite {
    (mod $modname:ident, backend = $backend:ty, scalar = $scalar:ty, encoder = $encoder:ty, params = $params:expr
     $(, log_d = $log_d:expr)? $(,)?) => {
        mod $modname {
            use std::sync::LazyLock;

            use ::poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> = LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            #[allow(unused_mut)]
            fn log_d() -> usize {
                let mut log_d = 2;
                $(log_d = $log_d;)?
                log_d
            }

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder>($params, &*MODULE, &*HOST_MODULE);
                    }
                };
            }

            macro_rules! run_test_with_arg {
                ($name:ident, $path:path, $arg:expr) => {
                    #[test]
                    fn $name() {
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder>($params, &*MODULE, &*HOST_MODULE, $arg);
                    }
                };
            }

            run_test_with_arg!(rotate, $crate::test_suite::inter::test_blas_rotate, log_d());
            run_test_with_arg!(sum, $crate::test_suite::inter::test_blas_sum, log_d());
            run_test_with_arg!(inter_guards, $crate::test_suite::inter::test_blas_inter_guards, log_d());
            run_test_with_arg!(transpose, $crate::test_suite::intra::test_blas_transpose, log_d());
            run_test_with_arg!(rotate_rows_cols, $crate::test_suite::intra::test_blas_rotate_rows_cols, log_d());
            run_test_with_arg!(intra_guards, $crate::test_suite::intra::test_blas_intra_guards, log_d());
            run_test_with_arg!(matmul, $crate::test_suite::matmul::test_blas_matmul, log_d());
            run_test_with_arg!(matmul_pt, $crate::test_suite::matmul::test_blas_matmul_pt, log_d());
            run_test_with_arg!(matmul_guards, $crate::test_suite::matmul::test_blas_matmul_guards, log_d());
            run_test_with_arg!(matmul_pt_guards, $crate::test_suite::matmul::test_blas_matmul_pt_guards, log_d());
            run_test_with_arg!(matmul_pt_left, $crate::test_suite::matmul::test_blas_matmul_pt_left, log_d());
            run_test_with_arg!(
                matmul_pt_left_guards,
                $crate::test_suite::matmul::test_blas_matmul_pt_left_guards,
                log_d()
            );
            run_test!(matmul_long_sums, $crate::test_suite::matmul::test_blas_matmul_long_sums);
            run_test_with_arg!(key_widths, $crate::test_suite::keys::test_blas_key_widths, log_d());
            run_test_with_arg!(result_meta, $crate::test_suite::meta::test_blas_result_meta, log_d());
            run_test_with_arg!(batch_record, $crate::test_suite::meta::test_blas_batch_record, log_d());
            run_test!(context, $crate::test_suite::context::test_blas_context);
        }
    };
}

/// Registers [`test_blas_conjugate_invariant`](crate::test_suite::conjugate_invariant::test_blas_conjugate_invariant),
/// every operation once on real slots, as the test `conjugate_invariant`, for one backend of the conjugate-invariant
/// ring and parameter set. The expansion names `::poulpy_hal`, so the invoking crate depends on it directly.
#[macro_export]
macro_rules! blas_conjugate_invariant_test_suite {
    (mod $modname:ident, backend = $backend:ty, params = $params:expr $(,)?) => {
        mod $modname {
            #[test]
            fn conjugate_invariant() {
                use ::poulpy_hal::layouts::{ConjugateInvariant, HostBytesBackend, Module};
                let params = $params;
                $crate::test_suite::conjugate_invariant::test_blas_conjugate_invariant(
                    params,
                    &Module::<$backend>::new(params.n as u64),
                    &Module::<HostBytesBackend<ConjugateInvariant>>::new(params.n as u64),
                );
            }
        }
    };
}

/// Registers blas parity of `backend_test` against `backend_ref` for one scalar and parameter set: the tests `inter`,
/// `intra` and `matmul`, each on its own pair of modules. Rank and `dsize` come from `params`; the optional `shapes`, a
/// [`BlasParityShapes`](crate::test_suite::parity::BlasParityShapes), narrows the `log_d` sweep. The expansion names
/// `::poulpy_hal`, so the invoking crate depends on it directly.
#[macro_export]
macro_rules! blas_parity_test_suite {
    (mod $modname:ident, backend_ref = $reference:ty, backend_test = $tested:ty, scalar = $scalar:ty, params = $params:expr
     $(, shapes = $shapes:expr)? $(,)?) => {
        mod $modname {
            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        use $path as __test_fn;
                        let params = $params;
                        #[allow(unused_mut)]
                        let mut shapes = $crate::test_suite::parity::BlasParityShapes::default();
                        $(shapes = $shapes;)?
                        let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                        let tested = ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64);
                        __test_fn::<$reference, $tested, $scalar>(params, &shapes, &reference, &tested);
                    }
                };
            }

            run_test!(inter, $crate::test_suite::parity::test_blas_inter_parity);
            run_test!(intra, $crate::test_suite::parity::test_blas_intra_parity);
            run_test!(matmul, $crate::test_suite::parity::test_blas_matmul_parity);
        }
    };
}
