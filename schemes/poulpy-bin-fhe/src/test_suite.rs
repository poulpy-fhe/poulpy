//! Instantiation of the backend-generic test bodies.
//!
//! The bodies live in [`blind_rotation`], [`circuit_bootstrapping`] and
//! [`bdd_arithmetic`]. A backend crate turns them into `#[test]` functions with
//! [`bin_fhe_backend_test_suite!`](crate::bin_fhe_backend_test_suite), so no
//! backend is ever named from this crate.

pub mod bdd_arithmetic;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
pub mod parity;

/// Instantiates the whole gate-level suite for one backend.
///
/// ```ignore
/// poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_correctness, backend = MyBackend);
/// ```
#[macro_export]
macro_rules! bin_fhe_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty $(,)?
    ) => {
        mod $modname {
            mod blind_rotation {
                use ::poulpy_hal::{api::ModuleNew, layouts::Module};

                use $crate::{
                    blind_rotation::CGGI,
                    test_suite::blind_rotation::{
                        generic_blind_rotation::test_blind_rotation,
                        generic_lut::{test_lut_extended, test_lut_standard, test_lut_transfer_into_carries_scalars},
                        key_encryption::{
                            test_blind_rotation_key_compressed_encrypt_sk_rejects_secret_shape,
                            test_blind_rotation_key_encrypt_sk_rejects_secret_shape,
                        },
                    },
                };

                #[test]
                fn key_encrypt_sk_rejects_secret_shape() {
                    test_blind_rotation_key_encrypt_sk_rejects_secret_shape(&Module::<$backend>::new(32));
                }

                #[test]
                fn key_compressed_encrypt_sk_rejects_secret_shape() {
                    test_blind_rotation_key_compressed_encrypt_sk_rejects_secret_shape(&Module::<$backend>::new(32));
                }

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
                use ::poulpy_hal::{api::ModuleNew, layouts::Module};

                use $crate::{
                    blind_rotation::CGGI,
                    test_suite::circuit_bootstrapping::{
                        test_circuit_bootstrapping_key_encrypt_sk_rejects_secret_shape, test_circuit_bootstrapping_to_constant,
                        test_circuit_bootstrapping_to_exponent,
                    },
                };

                #[test]
                fn key_encrypt_sk_rejects_secret_shape_cggi() {
                    let module: Module<$backend> = Module::<$backend>::new(64);
                    test_circuit_bootstrapping_key_encrypt_sk_rejects_secret_shape::<$backend, CGGI>(&module);
                }

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
                use ::std::sync::LazyLock;

                use $crate::{blind_rotation::CGGI, test_suite::bdd_arithmetic as test_suite};

                static CTX: LazyLock<test_suite::TestContext<CGGI, $backend>> =
                    LazyLock::new(test_suite::TestContext::<CGGI, $backend>::new);

                #[test]
                fn glwe_blind_retriever() {
                    test_suite::test_glwe_blind_retriever(&CTX);
                }

                #[test]
                fn glwe_blind_retrieval_stateful_assign() {
                    test_suite::test_glwe_blind_retrieval_stateful_assign(&CTX);
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
                fn ggsw_blind_rotation_from_scalar() {
                    test_suite::test_ggsw_blind_rotation_from_scalar(&CTX);
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
                fn fhe_uint_get_byte() {
                    test_suite::test_fhe_uint_get_byte(&CTX);
                }

                #[test]
                fn fhe_uint_get_bit_lwe() {
                    test_suite::test_fhe_uint_get_bit_lwe(&CTX);
                }

                #[test]
                fn fhe_uint_from_fhe_uint_prepared() {
                    test_suite::test_fhe_uint_from_fhe_uint_prepared(&CTX);
                }

                #[test]
                fn fhe_uint_zero_byte() {
                    test_suite::test_fhe_uint_zero_byte(&CTX);
                }

                #[test]
                fn fhe_uint_encrypt_sk_rejects_degree() {
                    test_suite::test_fhe_uint_encrypt_sk_rejects_degree(&CTX);
                }

                #[test]
                fn fhe_uint_decrypt_rejects_degree() {
                    test_suite::test_fhe_uint_decrypt_rejects_degree(&CTX);
                }

                #[test]
                fn fhe_uint_noise_stats_rejects_degree() {
                    test_suite::test_fhe_uint_noise_stats_rejects_degree(&CTX);
                }

                #[test]
                fn fhe_uint_prepared_encrypt_sk_rejects_degree() {
                    test_suite::test_fhe_uint_prepared_encrypt_sk_rejects_degree(&CTX);
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
                fn glwe_cswap_assign() {
                    test_suite::test_glwe_cswap_assign_direct(&CTX);
                }

                #[test]
                fn glwe_cmux() {
                    test_suite::test_glwe_cmux_direct(&CTX);
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

/// Registers the complete binary-FHE parity suite for a caller-selected pair.
/// Inputs and raw keys are shared in coefficient form and prepared independently.
/// Register this suite beside each backend's explicit scheme opt-in.
#[macro_export]
macro_rules! bin_fhe_parity_test_suite {
    (mod $modname:ident, backend_ref = $backend_ref:ty, backend_test = $backend_test:ty $(,)?) => {
        mod $modname {
            use ::poulpy_hal::layouts::Module;
            use $crate::test_suite::parity;
            #[test]
            fn lifecycle() {
                parity::lifecycle::test_lifecycle_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn glwe_cmux() {
                parity::bdd::test_glwe_cmux_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn glwe_cswap_assign() {
                parity::bdd::test_glwe_cswap_assign_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn glwe_blind_rotation() {
                parity::bdd::test_glwe_blind_rotation_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn glwe_blind_selection() {
                parity::bdd::test_glwe_blind_selection_parity(
                    &Module::<$backend_ref>::new(64),
                    &Module::<$backend_test>::new(64),
                );
            }
            #[test]
            fn glwe_blind_retrieval() {
                parity::bdd::test_glwe_blind_retrieval_parity(
                    &Module::<$backend_ref>::new(64),
                    &Module::<$backend_test>::new(64),
                );
            }
            #[test]
            fn glwe_blind_retriever() {
                parity::bdd::test_glwe_blind_retriever_parity(
                    &Module::<$backend_ref>::new(64),
                    &Module::<$backend_test>::new(64),
                );
            }
            #[test]
            fn ggsw_blind_rotation() {
                parity::bdd::test_ggsw_blind_rotation_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn bdd_circuit_execute() {
                parity::bdd::test_bdd_circuit_execute_parity(&Module::<$backend_ref>::new(64), &Module::<$backend_test>::new(64));
            }
            #[test]
            fn blind_rotation() {
                parity::blind_rotation::test_blind_rotation_parity(
                    &Module::<$backend_ref>::new(64),
                    &Module::<$backend_test>::new(64),
                );
            }
            #[test]
            fn circuit_bootstrapping() {
                parity::circuit_bootstrapping::test_circuit_bootstrapping_parity(
                    &Module::<$backend_ref>::new(64),
                    &Module::<$backend_test>::new(64),
                );
            }
        }
    };
}

/// Checks selected randomized key operations against the callable references on
/// one backend, using the same sampler on both paths. This suite additionally
/// requires the reference bodies' lower-layer capabilities; the paired suite
/// does not impose them on replacements.
#[macro_export]
macro_rules! bin_fhe_reference_test_suite {
    (mod $modname:ident, backend = $backend:ty $(,)?) => {
        mod $modname {
            use ::poulpy_hal::layouts::Module;
            #[test]
            fn blind_rotation_key_lifecycle() {
                $crate::test_suite::parity::blind_rotation::test_blind_rotation_key_lifecycle(&Module::<$backend>::new(64));
            }
            #[test]
            fn lifecycle_reference() {
                $crate::test_suite::parity::lifecycle::test_lifecycle_reference(&Module::<$backend>::new(64));
            }
        }
    };
}
