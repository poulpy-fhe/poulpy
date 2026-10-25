//! Backend-generic tests of `poulpy-mhe`, instantiated by backend crates
//! through [`mhe_backend_test_suite!`](crate::mhe_backend_test_suite) and,
//! against a comparison backend, [`mhe_parity_test_suite!`](crate::mhe_parity_test_suite).

pub mod evaluation_key;
pub(crate) mod fixtures;
pub mod ggsw;
pub mod keyswitch;
pub mod parity;
pub mod pat;
pub mod public_key;
pub mod sharing;
pub mod tensor_key;

/// Runs every `poulpy-mhe` test against `$backend`.
#[macro_export]
macro_rules! mhe_backend_test_suite {
    (mod $modname:ident, backend = $backend:ty $(,)?) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module};

            #[test]
            fn glwe_pat_compressed_ops() {
                $crate::test_suite::pat::test_glwe_pat_compressed_ops(&Module::<$backend>::new(256));
            }

            #[test]
            fn gglwe_pat_compressed_ops() {
                $crate::test_suite::pat::test_gglwe_pat_compressed_ops(&Module::<$backend>::new(256));
            }

            #[test]
            fn gglwe_pat_ops() {
                $crate::test_suite::pat::test_gglwe_pat_ops(&Module::<$backend>::new(256));
            }

            #[test]
            fn glwe_pat_compressed_aggregate_assign_rejects_seed_mismatch() {
                $crate::test_suite::pat::test_glwe_pat_compressed_aggregate_assign_rejects_seed_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn gglwe_pat_compressed_aggregate_assign_rejects_seed_mismatch() {
                $crate::test_suite::pat::test_gglwe_pat_compressed_aggregate_assign_rejects_seed_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_pat_compressed_aggregate_assign_rejects_layout_mismatch() {
                $crate::test_suite::pat::test_glwe_pat_compressed_aggregate_assign_rejects_layout_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_pat_compressed_finalize_rejects_layout_mismatch() {
                $crate::test_suite::pat::test_glwe_pat_compressed_finalize_rejects_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_public_key() {
                $crate::test_suite::public_key::test_glwe_public_key(&Module::<$backend>::new(256));
            }

            #[test]
            fn ciphertext_encryption_noise_tags() {
                $crate::test_suite::public_key::test_ciphertext_encryption_noise_tags(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_glwe_public_key_share_finalize_rejects_dist_none() {
                $crate::test_suite::public_key::test_mhe_glwe_public_key_share_finalize_rejects_dist_none(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_public_key_share_read_rank_mismatch() {
                $crate::test_suite::public_key::test_glwe_public_key_share_read_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_glwe_public_key_share_aggregate_rejects_rank_mismatch() {
                $crate::test_suite::public_key::test_mhe_glwe_public_key_share_aggregate_rejects_rank_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_public_key_share_aggregate_rejects_dist_mismatch() {
                $crate::test_suite::public_key::test_mhe_glwe_public_key_share_aggregate_rejects_dist_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_public_key_share_finalize_rejects_shared_seed() {
                $crate::test_suite::public_key::test_mhe_glwe_public_key_share_finalize_rejects_shared_seed(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_public_key_share_gen_rejects_secret_none() {
                $crate::test_suite::public_key::test_mhe_glwe_public_key_share_gen_rejects_secret_none(&Module::<$backend>::new(
                    64,
                ));
            }

            #[test]
            fn ggsw_share_read_rank_mismatch() {
                $crate::test_suite::ggsw::test_ggsw_share_read_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn ggsw_share() {
                $crate::test_suite::ggsw::test_ggsw_share(&Module::<$backend>::new(256));
            }

            #[test]
            fn ggsw_share_rank_one() {
                $crate::test_suite::ggsw::test_ggsw_share_rank_one(&Module::<$backend>::new(256));
            }

            #[test]
            fn ggsw_share_digit_widths() {
                $crate::test_suite::ggsw::test_ggsw_share_digit_widths(&Module::<$backend>::new(256));
            }

            #[test]
            fn ggsw_share_external_product() {
                $crate::test_suite::ggsw::test_ggsw_share_external_product(&Module::<$backend>::new(256));
            }

            #[test]
            fn mhe_ggsw_share_aggregate_rejects_seed_mismatch() {
                $crate::test_suite::ggsw::test_mhe_ggsw_share_aggregate_rejects_seed_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_ggsw_share_gen_rejects_ephemeral_rank_mismatch() {
                $crate::test_suite::ggsw::test_mhe_ggsw_share_gen_rejects_ephemeral_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_ggsw_share_finalize_rejects_key_rank_mismatch() {
                $crate::test_suite::ggsw::test_mhe_ggsw_share_finalize_rejects_key_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn ggsw_share_key_precision_mismatch() {
                $crate::test_suite::ggsw::test_ggsw_share_key_precision_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_switching_key() {
                $crate::test_suite::evaluation_key::test_glwe_switching_key(&Module::<$backend>::new(256));
            }

            #[test]
            fn glwe_automorphism_key() {
                $crate::test_suite::evaluation_key::test_glwe_automorphism_key(&Module::<$backend>::new(256));
            }

            #[test]
            fn mhe_glwe_switching_key_share_aggregate_rejects_degree_mismatch() {
                $crate::test_suite::evaluation_key::test_mhe_glwe_switching_key_share_aggregate_rejects_degree_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_switching_key_share_aggregate_rejects_out_degree_mismatch() {
                $crate::test_suite::evaluation_key::test_mhe_glwe_switching_key_share_aggregate_rejects_out_degree_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_automorphism_key_share_aggregate_rejects_p_mismatch() {
                $crate::test_suite::evaluation_key::test_mhe_glwe_automorphism_key_share_aggregate_rejects_p_mismatch(&Module::<
                    $backend,
                >::new(
                    64
                ));
            }

            #[test]
            fn glwe_private_keyswitch_flood_bound_guards() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_flood_bound_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_glwe_private_keyswitch_share_aggregate_rejects_layout_mismatch() {
                $crate::test_suite::keyswitch::test_mhe_glwe_private_keyswitch_share_aggregate_rejects_layout_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_private_keyswitch() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch(&Module::<$backend>::new(256));
            }

            #[test]
            fn glwe_public_keyswitch() {
                $crate::test_suite::keyswitch::test_glwe_public_keyswitch(&Module::<$backend>::new(256));
            }

            #[test]
            fn mhe_glwe_public_keyswitch_share_finalize_rejects_layout_mismatch() {
                $crate::test_suite::keyswitch::test_mhe_glwe_public_keyswitch_share_finalize_rejects_layout_mismatch(&Module::<
                    $backend,
                >::new(
                    64
                ));
            }

            #[test]
            fn mhe_glwe_public_keyswitch_share_gen_rejects_pk_precision() {
                $crate::test_suite::keyswitch::test_mhe_glwe_public_keyswitch_share_gen_rejects_pk_precision(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_public_key_gen_shape_guards() {
                $crate::test_suite::public_key::test_glwe_public_key_gen_shape_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_tensor_key_pk_shape_guards() {
                $crate::test_suite::tensor_key::test_glwe_tensor_key_pk_shape_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn ggsw_share_shape_guards() {
                $crate::test_suite::ggsw::test_ggsw_share_shape_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_evaluation_key_share_shape_guards() {
                $crate::test_suite::evaluation_key::test_glwe_evaluation_key_share_shape_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_private_keyswitch_share_layout_guards() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_share_layout_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_public_keyswitch_share_layout_guards() {
                $crate::test_suite::keyswitch::test_glwe_public_keyswitch_share_layout_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_private_keyswitch_rejects_flag_clear_mask() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_rejects_flag_clear_mask(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_public_keyswitch_rejects_flag_clear_mask() {
                $crate::test_suite::keyswitch::test_glwe_public_keyswitch_rejects_flag_clear_mask(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_enc_to_share_rejects_flag_clear_mask() {
                $crate::test_suite::sharing::test_glwe_enc_to_share_rejects_flag_clear_mask(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_private_keyswitch_finalize_layout_guards() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_finalize_layout_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_enc_to_share_flood_guards() {
                $crate::test_suite::sharing::test_glwe_enc_to_share_flood_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_sharing_layout_guards() {
                $crate::test_suite::sharing::test_glwe_sharing_layout_guards(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_enc_to_share() {
                $crate::test_suite::sharing::test_glwe_enc_to_share(&Module::<$backend>::new(256));
            }

            #[test]
            fn glwe_share_to_enc() {
                $crate::test_suite::sharing::test_glwe_share_to_enc(&Module::<$backend>::new(256));
            }

            #[test]
            fn mhe_glwe_share_to_enc_share_gen_rejects_precision() {
                $crate::test_suite::sharing::test_mhe_glwe_share_to_enc_share_gen_rejects_precision(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_glwe_enc_to_share_share_gen_rejects_precision() {
                $crate::test_suite::sharing::test_mhe_glwe_enc_to_share_share_gen_rejects_precision(&Module::<$backend>::new(64));
            }

            #[test]
            fn mhe_glwe_enc_to_share_share_finalize_rejects_layout_mismatch() {
                $crate::test_suite::sharing::test_mhe_glwe_enc_to_share_share_finalize_rejects_layout_mismatch(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn glwe_tensor_key() {
                $crate::test_suite::tensor_key::test_glwe_tensor_key(&Module::<$backend>::new(256));
            }

            #[test]
            fn mhe_glwe_tensor_key_share_gen_rejects_pk_precision() {
                $crate::test_suite::tensor_key::test_mhe_glwe_tensor_key_share_gen_rejects_pk_precision(
                    &Module::<$backend>::new(64),
                );
            }

            #[test]
            fn mhe_glwe_tensor_key_share_gen_rejects_secret_degree() {
                $crate::test_suite::tensor_key::test_mhe_glwe_tensor_key_share_gen_rejects_secret_degree(
                    &Module::<$backend>::new(64),
                );
            }
        }
    };
}
