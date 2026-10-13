//! Backend-generic poulpy-blas tests.
//!
//! Functional tests take `(params, module, host_module)`, most of them then the block dimension's `log_d`, and are
//! instantiated per backend by [`blas_backend_test_suite!`](crate::blas_backend_test_suite), or on the
//! conjugate-invariant ring by [`blas_conjugate_invariant_test_suite!`](crate::blas_conjugate_invariant_test_suite);
//! parity tests take `(params, shapes, reference_module, tested_module)` and are instantiated by
//! [`blas_parity_test_suite!`](crate::blas_parity_test_suite). The backend guide, `docs/blas-contracts.md`, covers
//! wiring, the override contract, registration and attestation.

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
/// `2^log_d`: 2 unless the optional `log_d` sets it. The ring degree `N` is free within two bounds: the blocks fit the
/// `N / 2` slots, `d^2 <= N / 2` (`N >= 32` at the default), and the guards encode on a module of degree `2 N`, which
/// the backend must support (`N <= 2^16` at most). `test_blas_context` sweeps `log_d` 1 to 3 and skips with a message
/// those whose blocks do not fit; `test_blas_batch_record`, below four matrix slots, and the operand at another
/// dimension in the guards of the products by public matrices, where only `2 x 2` blocks fit, are skipped with a
/// message too. Maps and public matrices run at the parameter set's scale and a quarter of it below and above
/// ([`other_scales`](crate::test_suite::helpers::other_scales)), so its `k` must hold two maps at the larger scale and
/// a product; through the latter, the products also multiply factors at two scales, in either order, one an eighth of
/// the scale lower and either as much narrower, so that their terms keep one budget, or as wide, so that its terms have
/// the larger budget (crossed scales). `test_blas_key_widths` runs with the maps and the public matrices below that
/// scale, where a key width taken from it in place of theirs, or capped by it, finds no key.
/// `test_blas_matmul_long_sums`, sums past the `2^(63 - base2k)` terms the JKLS kernel sums between normalizations, runs
/// on its own ring at the registered radix when `base2k >= 57`, where such sums are short, and every other suite skips
/// it with a message: an NTT-family backend registers a parameter set at `base2k >= 57`.
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
/// ring and parameter set. It runs on `4 x 4` blocks, which the `N` slots hold from `N = 16`; below, it is skipped with
/// a message. The expansion names `::poulpy_hal`, so the invoking crate depends on it directly.
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
///
/// `matmul` runs products of more terms than the `2^(63 - base2k)` the JKLS kernel sums between normalizations, `1 x K`
/// grids by `K x 1` grids: of distinct blocks, where some `K <= 32` of them pass that bound, and, from `base2k` 57 (58
/// at `log_d` 1), of copies of one block, which overflow an `i64` digit in a kernel that normalizes only at the end. An
/// NTT-family `backend_test`, on either ring, therefore registers a parameter set at `base2k >= 57` with a `log_d` of 2
/// or more swept: it pins that the kernel accepts such sums, normalizes as it sums them and returns the reference's
/// bits.
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
