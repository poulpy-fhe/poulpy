//! Backend-generic tests of `poulpy-mhe`, instantiated by backend crates
//! through [`mhe_backend_test_suite!`](crate::mhe_backend_test_suite).

pub mod evaluation_key;
pub(crate) mod fixtures;
pub mod ggsw;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub mod refresh;
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
            #[should_panic(expected = "seeds differ")]
            fn pat_aggregate_seed_mismatch() {
                $crate::test_suite::pat::test_pat_aggregate_seed_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "seeds differ")]
            fn gglwe_pat_compressed_aggregate_seed_mismatch() {
                $crate::test_suite::pat::test_gglwe_pat_compressed_aggregate_seed_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: layouts differ")]
            fn pat_aggregate_layout_mismatch() {
                $crate::test_suite::pat::test_pat_aggregate_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: layouts differ")]
            fn pat_finalize_layout_mismatch() {
                $crate::test_suite::pat::test_pat_finalize_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_public_key() {
                $crate::test_suite::public_key::test_glwe_public_key(&Module::<$backend>::new(256));
            }

            #[test]
            #[should_panic(expected = "samplable distribution")]
            fn glwe_public_key_finalize_dist_none() {
                $crate::test_suite::public_key::test_glwe_public_key_finalize_dist_none(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_public_key_share_read_rank_mismatch() {
                $crate::test_suite::public_key::test_glwe_public_key_share_read_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: layouts differ")]
            fn glwe_public_key_aggregate_rank_mismatch() {
                $crate::test_suite::public_key::test_glwe_public_key_aggregate_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: secret distributions differ")]
            fn glwe_public_key_aggregate_dist_mismatch() {
                $crate::test_suite::public_key::test_glwe_public_key_aggregate_dist_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: public key entries share a seed")]
            fn glwe_public_key_finalize_shared_seed() {
                $crate::test_suite::public_key::test_glwe_public_key_finalize_shared_seed(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid secret")]
            fn glwe_public_key_gen_secret_none() {
                $crate::test_suite::public_key::test_glwe_public_key_gen_secret_none(&Module::<$backend>::new(64));
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
            fn ggsw_share_external_product() {
                $crate::test_suite::ggsw::test_ggsw_share_external_product(&Module::<$backend>::new(256));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: seeds differ")]
            fn ggsw_share_seed_mismatch() {
                $crate::test_suite::ggsw::test_ggsw_share_seed_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid share: ephemeral secret rank differs from the secret's")]
            fn ggsw_share_ephemeral_rank_mismatch() {
                $crate::test_suite::ggsw::test_ggsw_share_ephemeral_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: key output rank differs from the GGSW's")]
            fn ggsw_share_key_rank_mismatch() {
                $crate::test_suite::ggsw::test_ggsw_share_key_rank_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: key does not cover the GGSW precision")]
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
            #[should_panic(expected = "invalid aggregation: degrees differ")]
            fn glwe_switching_key_degree_mismatch() {
                $crate::test_suite::evaluation_key::test_glwe_switching_key_degree_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: degrees differ")]
            fn glwe_switching_key_out_degree_mismatch() {
                $crate::test_suite::evaluation_key::test_glwe_switching_key_out_degree_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: Galois elements differ")]
            fn glwe_automorphism_key_p_mismatch() {
                $crate::test_suite::evaluation_key::test_glwe_automorphism_key_p_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_private_keyswitch_flood_bound_guards() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_flood_bound_guards(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid aggregation: layouts differ")]
            fn glwe_private_keyswitch_aggregate_layout_mismatch() {
                $crate::test_suite::keyswitch::test_glwe_private_keyswitch_aggregate_layout_mismatch(&Module::<$backend>::new(
                    64,
                ));
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
            #[should_panic(expected = "invalid finalization: ciphertext and output layouts differ")]
            fn glwe_public_keyswitch_finalize_layout_mismatch() {
                $crate::test_suite::keyswitch::test_glwe_public_keyswitch_finalize_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid share: public key less precise than the share")]
            fn glwe_public_keyswitch_pk_precision() {
                $crate::test_suite::keyswitch::test_glwe_public_keyswitch_pk_precision(&Module::<$backend>::new(64));
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
            #[should_panic(expected = "invalid share: secret share more precise than the output")]
            fn glwe_share_to_enc_precision() {
                $crate::test_suite::sharing::test_glwe_share_to_enc_precision(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid share: share and mask layouts differ")]
            fn glwe_enc_to_share_precision() {
                $crate::test_suite::sharing::test_glwe_enc_to_share_precision(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: ciphertext and share layouts differ")]
            fn glwe_enc_to_share_finalize_layout_mismatch() {
                $crate::test_suite::sharing::test_glwe_enc_to_share_finalize_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_refresh() {
                $crate::test_suite::refresh::test_glwe_refresh(&Module::<$backend>::new(256));
            }

            #[test]
            #[should_panic(expected = "invalid share: bound outside the ciphertext precision")]
            fn glwe_refresh_bound() {
                $crate::test_suite::refresh::test_glwe_refresh_bound(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: ciphertext, share and output layouts differ")]
            fn glwe_refresh_finalize_layout_mismatch() {
                $crate::test_suite::refresh::test_glwe_refresh_finalize_layout_mismatch(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid finalization: ciphertext more precise than the output")]
            fn glwe_refresh_finalize_precision() {
                $crate::test_suite::refresh::test_glwe_refresh_finalize_precision(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_tensor_key() {
                $crate::test_suite::tensor_key::test_glwe_tensor_key(&Module::<$backend>::new(256));
            }

            #[test]
            #[should_panic(expected = "invalid share: public key less precise than the share")]
            fn glwe_tensor_key_pk_precision() {
                $crate::test_suite::tensor_key::test_glwe_tensor_key_pk_precision(&Module::<$backend>::new(64));
            }

            #[test]
            #[should_panic(expected = "invalid share: secret degree differs from the key's")]
            fn glwe_tensor_key_secret_degree() {
                $crate::test_suite::tensor_key::test_glwe_tensor_key_secret_degree(&Module::<$backend>::new(64));
            }
        }
    };
}
