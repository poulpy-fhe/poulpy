//! Backend-generic tests of `poulpy-mhe`, instantiated by backend crates
//! through [`mhe_backend_test_suite!`](crate::mhe_backend_test_suite).

pub mod evaluation_key;
pub(crate) mod fixtures;
pub mod keyswitch;
pub mod layouts;
pub mod pat;
pub mod public_key;

/// Runs every `poulpy-mhe` test against `$backend`.
#[macro_export]
macro_rules! mhe_backend_test_suite {
    (mod $modname:ident, backend = $backend:ty $(,)?) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module};

            #[test]
            fn glwe_pat_compressed_layout() {
                $crate::test_suite::layouts::test_glwe_pat_compressed(&Module::<$backend>::new(64));
            }

            #[test]
            fn gglwe_pat_compressed_layout() {
                $crate::test_suite::layouts::test_gglwe_pat_compressed(&Module::<$backend>::new(64));
            }

            #[test]
            fn gglwe_pat_layout() {
                $crate::test_suite::layouts::test_gglwe_pat(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_switching_key_pat_compressed_layout() {
                $crate::test_suite::layouts::test_glwe_switching_key_pat_compressed(&Module::<$backend>::new(64));
            }

            #[test]
            fn glwe_automorphism_key_pat_compressed_layout() {
                $crate::test_suite::layouts::test_glwe_automorphism_key_pat_compressed(&Module::<$backend>::new(64));
            }

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
            #[should_panic(expected = "invalid secret")]
            fn glwe_public_key_share_secret_none() {
                $crate::test_suite::public_key::test_glwe_public_key_share_secret_none(&Module::<$backend>::new(64));
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
            fn glwe_keyswitch() {
                $crate::test_suite::keyswitch::test_glwe_keyswitch(&Module::<$backend>::new(256));
            }
        }
    };
}
