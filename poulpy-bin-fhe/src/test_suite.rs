//! Instantiation of the backend-generic test bodies.
//!
//! The bodies live next to the code they exercise, in
//! [`blind_rotation::test_suite`](crate::blind_rotation::test_suite),
//! [`circuit_bootstrapping::test_suite`](crate::circuit_bootstrapping::test_suite)
//! and [`bdd_arithmetic::test_suite`](crate::bdd_arithmetic::test_suite). A
//! backend crate turns them into `#[test]` functions with
//! [`bin_fhe_backend_test_suite!`](crate::bin_fhe_backend_test_suite), so no
//! backend is ever named from this crate.

/// Instantiates the whole gate-level suite for one backend.
///
/// ```ignore
/// poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Ref);
/// ```
#[macro_export]
macro_rules! bin_fhe_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty $(,)?
    ) => {
        mod $modname {
            mod blind_rotation {
                use poulpy_hal::{api::ModuleNew, layouts::Module};

                use $crate::blind_rotation::{
                    CGGI,
                    test_suite::{
                        generic_blind_rotation::test_blind_rotation,
                        generic_lut::{test_lut_extended, test_lut_standard, test_lut_transfer_into_carries_scalars},
                    },
                };

                #[test]
                fn lut_standard() {
                    test_lut_standard(&Module::<$backend>::new(32));
                }

                #[test]
                fn lut_extended() {
                    test_lut_extended(&Module::<$backend>::new(32));
                }

                #[test]
                fn lut_transfer_into_carries_scalars() {
                    test_lut_transfer_into_carries_scalars(&Module::<$backend>::new(32));
                }

                #[test]
                fn standard() {
                    let module: Module<$backend> = Module::<$backend>::new(512);
                    test_blind_rotation::<CGGI, _, $backend>(&module, 224, 1, 1);
                }

                #[test]
                fn block_binary() {
                    let module: Module<$backend> = Module::<$backend>::new(512);
                    test_blind_rotation::<CGGI, _, $backend>(&module, 224, 7, 1);
                }

                #[test]
                fn block_binary_extended() {
                    let module: Module<$backend> = Module::<$backend>::new(512);
                    test_blind_rotation::<CGGI, _, $backend>(&module, 224, 7, 2);
                }
            }

            mod circuit_bootstrapping {
                use poulpy_hal::{api::ModuleNew, layouts::Module};

                use $crate::{
                    blind_rotation::CGGI,
                    circuit_bootstrapping::test_suite::{
                        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
                    },
                };

                #[test]
                fn to_constant_cggi() {
                    let module: Module<$backend> = Module::<$backend>::new(256);
                    test_circuit_bootstrapping_to_constant::<$backend, _, CGGI>(&module);
                }

                #[test]
                fn to_exponent_cggi() {
                    let module: Module<$backend> = Module::<$backend>::new(256);
                    test_circuit_bootstrapping_to_exponent::<$backend, _, CGGI>(&module);
                }
            }

            mod bdd_arithmetic {
                use std::sync::LazyLock;

                use $crate::{bdd_arithmetic::test_suite, blind_rotation::CGGI};

                static CTX: LazyLock<test_suite::TestContext<CGGI, $backend>> =
                    LazyLock::new(test_suite::TestContext::<CGGI, $backend>::new);

                #[test]
                fn glwe_blind_retriever() {
                    test_suite::test_glwe_blind_retriever(&CTX);
                }

                #[test]
                fn glwe_blind_retrieval_statefull() {
                    test_suite::test_glwe_blind_retrieval_statefull(&CTX);
                }

                #[test]
                fn glwe_blind_selection() {
                    test_suite::test_glwe_blind_selection(&CTX);
                }

                #[test]
                fn glwe_to_glwe_blind_rotation() {
                    test_suite::test_glwe_to_glwe_blind_rotation(&CTX);
                }

                #[test]
                fn scalar_to_ggsw_blind_rotation() {
                    test_suite::test_scalar_to_ggsw_blind_rotation(&CTX);
                }

                #[test]
                fn fhe_uint_swap() {
                    test_suite::test_fhe_uint_swap(&CTX);
                }

                #[test]
                fn fhe_uint_get_bit_glwe() {
                    test_suite::test_fhe_uint_get_bit_glwe(&CTX);
                }

                #[test]
                fn fhe_uint_sext() {
                    test_suite::test_fhe_uint_sext(&CTX);
                }

                #[test]
                fn fhe_uint_splice_u8() {
                    test_suite::test_fhe_uint_splice_u8(&CTX);
                }

                #[test]
                fn fhe_uint_splice_u16() {
                    test_suite::test_fhe_uint_splice_u16(&CTX);
                }

                #[test]
                fn cswap() {
                    test_suite::test_cswap_direct(&CTX);
                }

                #[test]
                fn cmux() {
                    test_suite::test_cmux_direct(&CTX);
                }

                #[test]
                fn bdd_prepare() {
                    test_suite::test_bdd_prepare(&CTX);
                }

                #[test]
                fn bdd_add() {
                    test_suite::test_bdd_add(&CTX);
                }

                #[test]
                fn bdd_sub() {
                    test_suite::test_bdd_sub(&CTX);
                }

                #[test]
                fn bdd_and() {
                    test_suite::test_bdd_and(&CTX);
                }

                #[test]
                fn bdd_or() {
                    test_suite::test_bdd_or(&CTX);
                }

                #[test]
                fn bdd_xor() {
                    test_suite::test_bdd_xor(&CTX);
                }

                #[test]
                fn bdd_sll() {
                    test_suite::test_bdd_sll(&CTX);
                }

                #[test]
                fn bdd_srl() {
                    test_suite::test_bdd_srl(&CTX);
                }

                #[test]
                fn bdd_sra() {
                    test_suite::test_bdd_sra(&CTX);
                }

                #[test]
                fn bdd_slt() {
                    test_suite::test_bdd_slt(&CTX);
                }

                #[test]
                fn bdd_sltu() {
                    test_suite::test_bdd_sltu(&CTX);
                }
            }
        }
    };
}
