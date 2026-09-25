//! Backend-generic tests of `poulpy-mhe`, instantiated by backend crates
//! through [`mhe_backend_test_suite!`](crate::mhe_backend_test_suite).

pub(crate) mod fixtures;
pub mod layouts;
pub mod pat;

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
        }
    };
}
