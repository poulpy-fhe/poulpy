pub mod automorphism;
pub mod encryption;
pub mod external_product;
pub mod glwe_tensor;
pub mod keyswitch;

mod conversion;
mod glwe_packer;
mod glwe_packing;
mod trace;

pub use conversion::*;
pub use glwe_packer::*;
pub use glwe_packing::*;
pub use trace::*;

use crate::oep::CoreImpl;
use poulpy_hal::layouts::Backend;

pub trait TestBackend: Backend + CoreImpl<Self> {}

impl<BE> TestBackend for BE where BE: Backend + CoreImpl<BE> {}

#[macro_export]
macro_rules! core_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr $(,)?
    ) => {
        poulpy_hal::backend_test_suite!(
            mod $modname,
            backend = $backend,
            params = $params,
            tests = {
                glwe_encrypt_sk => $crate::test_suite::encryption::test_glwe_encrypt_sk,
                glwe_compressed_encrypt_sk => $crate::test_suite::encryption::test_glwe_compressed_encrypt_sk,
                glwe_encrypt_zero_sk => $crate::test_suite::encryption::test_glwe_encrypt_zero_sk,
                glwe_encrypt_pk => $crate::test_suite::encryption::test_glwe_encrypt_pk,
                glwe_base2k_conv => $crate::test_suite::test_glwe_base2k_conversion,
                test_glwe_tensoring => $crate::test_suite::glwe_tensor::test_glwe_tensoring,
                test_glwe_tensor_apply_add_assign => $crate::test_suite::glwe_tensor::test_glwe_tensor_apply_add_assign,
                test_glwe_tensor_square => $crate::test_suite::glwe_tensor::test_glwe_tensor_square,
                test_glwe_mul_plain => $crate::test_suite::glwe_tensor::test_glwe_mul_plain,
                test_glwe_mul_const => $crate::test_suite::glwe_tensor::test_glwe_mul_const,
                glwe_keyswitch => $crate::test_suite::keyswitch::test_glwe_keyswitch,
                glwe_keyswitch_assign => $crate::test_suite::keyswitch::test_glwe_keyswitch_assign,
                glwe_automorphism => $crate::test_suite::automorphism::test_glwe_automorphism,
                glwe_automorphism_assign => $crate::test_suite::automorphism::test_glwe_automorphism_assign,
                glwe_external_product => $crate::test_suite::external_product::test_glwe_external_product,
                glwe_external_product_assign => $crate::test_suite::external_product::test_glwe_external_product_assign,
                glwe_trace_assign => $crate::test_suite::test_glwe_trace_assign,
                glwe_packing => $crate::test_suite::test_glwe_packing,
                glwe_packer => $crate::test_suite::test_glwe_packer,
                gglwe_switching_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_switching_key_encrypt_sk,
                gglwe_switching_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_switching_key_compressed_encrypt_sk,
                gglwe_compressed_encrypt_sk => $crate::test_suite::encryption::test_gglwe_compressed_encrypt_sk,
                gglwe_automorphism_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_automorphism_key_encrypt_sk,
                gglwe_automorphism_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_automorphism_key_compressed_encrypt_sk,
                gglwe_tensor_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_tensor_key_encrypt_sk,
                gglwe_tensor_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_tensor_key_compressed_encrypt_sk,
                gglwe_to_ggsw_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_to_ggsw_key_encrypt_sk,
                gglwe_switching_key_keyswitch => $crate::test_suite::keyswitch::test_gglwe_switching_key_keyswitch,
                gglwe_switching_key_keyswitch_assign => $crate::test_suite::keyswitch::test_gglwe_switching_key_keyswitch_assign,
                gglwe_switching_key_external_product =>
                    $crate::test_suite::external_product::test_gglwe_switching_key_external_product,
                gglwe_switching_key_external_product_assign =>
                    $crate::test_suite::external_product::test_gglwe_switching_key_external_product_assign,
                gglwe_automorphism_key_automorphism =>
                    $crate::test_suite::automorphism::test_gglwe_automorphism_key_automorphism,
                gglwe_automorphism_key_automorphism_assign =>
                    $crate::test_suite::automorphism::test_gglwe_automorphism_key_automorphism_assign,
                ggsw_encrypt_sk => $crate::test_suite::encryption::test_ggsw_encrypt_sk,
                ggsw_compressed_encrypt_sk => $crate::test_suite::encryption::test_ggsw_compressed_encrypt_sk,
                ggsw_keyswitch => $crate::test_suite::keyswitch::test_ggsw_keyswitch,
                ggsw_keyswitch_assign => $crate::test_suite::keyswitch::test_ggsw_keyswitch_assign,
                ggsw_external_product => $crate::test_suite::external_product::test_ggsw_external_product,
                ggsw_external_product_assign => $crate::test_suite::external_product::test_ggsw_external_product_assign,
                ggsw_automorphism => $crate::test_suite::automorphism::test_ggsw_automorphism,
                ggsw_automorphism_assign => $crate::test_suite::automorphism::test_ggsw_automorphism_assign,
                lwe_keyswitch => $crate::test_suite::keyswitch::test_lwe_keyswitch,
                glwe_to_lwe => $crate::test_suite::test_glwe_to_lwe,
                lwe_to_glwe => $crate::test_suite::test_lwe_to_glwe,
            }
        );
    };
}

pub use crate::core_backend_test_suite;
