//! Backend-generic CKKS test suite.
//!
//! All test functions are generic over `BE: Backend` and take a
//! [`TestContext`](helpers::TestContext) that owns the module, secret key,
//! and optional evaluation keys.  The backend-specific test harnesses
//! (`ntt120_ref`, `ntt120_avx`) instantiate and invoke these functions.

use poulpy_core::{
    EncryptionLayout,
    layouts::{GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank},
};

use crate::CKKSMeta;

/// Shared CKKS parameter set for test instantiation.
#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub prec: CKKSMeta,
    pub hw: usize,
    pub dsize: usize,
}

impl CKKSTestParams {
    pub fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: self.k.into(),
            rank: Rank(1),
        })
        .unwrap()
    }

    pub fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }

    pub fn atk_layout(&self) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }
}

/// NTT120 parameter set.
pub const NTT120_PARAMS_F64: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 52,
    k: 8 * 40,
    prec: CKKSMeta {
        log_delta: 40,
        log_budget: 30,
    },
    hw: 192,
    dsize: 1,
};

/// FFT64 parameter set.
pub const FFT64_PARAMS_F64: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 19,
    k: 8 * 19,
    prec: CKKSMeta {
        log_delta: 30,
        log_budget: 10,
    },
    hw: 192,
    dsize: 1,
};

/// NTT120 parameter set.
pub const NTT120_PARAMS_F128: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 52,
    k: 8 * 80,
    prec: CKKSMeta {
        log_delta: 80,
        log_budget: 30,
    },
    hw: 192,
    dsize: 1,
};

#[macro_export]
macro_rules! ckks_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        scalar = $scalar:ty,
        params = $params:expr,
        rotations = $rotations:expr $(,)?
    ) => {
        mod $modname {
            use std::sync::LazyLock;

            use anyhow::Result;

            use $crate::leveled::tests::test_suite::helpers::TestContext;

            static CTX: LazyLock<TestContext<$backend, $scalar>> = LazyLock::new(|| TestContext::new($params, $rotations));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        $path(&CTX);
                    }
                };
            }

            macro_rules! run_test_with_arg {
                ($name:ident, $path:path, $arg:expr) => {
                    #[test]
                    fn $name() {
                        $path(&CTX, $arg);
                    }
                };
            }

            macro_rules! run_test_result {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() -> Result<()> {
                        $path(&CTX)
                    }
                };
            }

            run_test!(
                encrypt_decrypt,
                $crate::leveled::tests::test_suite::encryption::test_encrypt_decrypt
            );
            run_test!(
                decrypt_extract_same_meta,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_same_meta
            );
            run_test!(
                decrypt_extract_truncates_log_budget,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_truncates_log_budget
            );
            run_test!(
                decrypt_extract_rsh_for_smaller_log_delta,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_rsh_for_smaller_log_delta
            );
            run_test!(
                decrypt_extract_lsh_for_larger_log_delta,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_lsh_for_larger_log_delta
            );
            run_test!(
                decrypt_extract_output_hom_rem_too_large,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_output_hom_rem_too_large
            );
            run_test!(
                decrypt_extract_base2k_mismatch_error,
                $crate::leveled::tests::test_suite::encryption::test_decrypt_extract_base2k_mismatch_error
            );
            run_test!(
                reallocate_limbs_checked_error,
                $crate::leveled::tests::test_suite::errors::test_reallocate_limbs_checked_error
            );
            run_test!(
                compact_limbs_copy,
                $crate::leveled::tests::test_suite::errors::test_compact_limbs_copy
            );
            run_test!(
                add_pt_vec_znx_alignment_error,
                $crate::leveled::tests::test_suite::errors::test_add_pt_vec_znx_alignment_error
            );
            run_test!(
                add_ct_aligned,
                $crate::leveled::tests::test_suite::add::test_add_ct_aligned
            );
            run_test!(
                add_ct_delta_a_lt_b,
                $crate::leveled::tests::test_suite::add::test_add_ct_delta_a_lt_b
            );
            run_test!(
                add_ct_delta_a_gt_b,
                $crate::leveled::tests::test_suite::add::test_add_ct_delta_a_gt_b
            );
            run_test!(
                add_ct_delta_log_delta,
                $crate::leveled::tests::test_suite::add::test_add_ct_delta_log_delta
            );
            run_test!(
                add_ct_aligned_smaller_output,
                $crate::leveled::tests::test_suite::add::test_add_ct_aligned_smaller_output
            );
            run_test!(
                add_ct_assign_aligned,
                $crate::leveled::tests::test_suite::add::test_add_ct_assign_aligned
            );
            run_test!(
                add_ct_assign_self_lt,
                $crate::leveled::tests::test_suite::add::test_add_ct_assign_self_lt
            );
            run_test!(
                add_ct_assign_self_gt,
                $crate::leveled::tests::test_suite::add::test_add_ct_assign_self_gt
            );
            run_test!(
                add_pt_vec_znx_assign,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_znx_assign
            );
            run_test!(
                add_pt_vec_znx_into_aligned,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_znx_into_aligned
            );
            run_test!(
                add_pt_vec_znx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_znx_into_delta_log_delta
            );
            run_test!(
                add_pt_vec_rnx_assign,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_rnx_assign
            );
            run_test!(
                add_pt_vec_rnx_into_aligned,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_rnx_into_aligned
            );
            run_test!(
                add_pt_vec_rnx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_rnx_into_delta_log_delta
            );
            run_test!(
                add_const_into_aligned,
                $crate::leveled::tests::test_suite::add::test_add_const_rnx_into_aligned
            );
            run_test!(
                add_const_assign,
                $crate::leveled::tests::test_suite::add::test_add_const_rnx_assign
            );
            run_test!(
                add_const_into_delta_log_delta,
                $crate::leveled::tests::test_suite::add::test_add_const_rnx_into_delta_log_delta
            );
            run_test!(
                add_const_into_real_only,
                $crate::leveled::tests::test_suite::add::test_add_const_rnx_into_real_only
            );
            run_test!(
                add_const_znx_into_aligned,
                $crate::leveled::tests::test_suite::add::test_add_const_znx_into_aligned
            );
            run_test!(
                add_pt_vec_znx_into_smaller_output,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_znx_into_smaller_output
            );
            run_test!(
                add_pt_vec_znx_base2k_mismatch_error,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_znx_base2k_mismatch_error
            );
            run_test!(
                add_pt_vec_rnx_into_smaller_output,
                $crate::leveled::tests::test_suite::add::test_add_pt_vec_rnx_into_smaller_output
            );
            run_test!(
                add_ct_aligned_unsafe,
                $crate::leveled::tests::test_suite::add_unsafe::test_add_ct_aligned_unsafe
            );
            run_test!(
                add_ct_assign_aligned_unsafe,
                $crate::leveled::tests::test_suite::add_unsafe::test_add_ct_assign_aligned_unsafe
            );
            run_test!(
                add_pt_vec_znx_into_aligned_unsafe,
                $crate::leveled::tests::test_suite::add_unsafe::test_add_pt_vec_znx_into_aligned_unsafe
            );
            run_test!(
                add_pt_vec_rnx_into_aligned_unsafe,
                $crate::leveled::tests::test_suite::add_unsafe::test_add_pt_vec_rnx_into_aligned_unsafe
            );
            run_test!(
                add_const_znx_into_aligned_unsafe,
                $crate::leveled::tests::test_suite::add_unsafe::test_add_const_znx_into_aligned_unsafe
            );
            run_test!(
                sub_ct_aligned,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_aligned
            );
            run_test!(
                sub_ct_delta_a_lt_b,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_delta_a_lt_b
            );
            run_test!(
                sub_ct_delta_a_gt_b,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_delta_a_gt_b
            );
            run_test!(
                sub_ct_delta_log_delta,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_delta_log_delta
            );
            run_test!(
                sub_ct_smaller_output,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_smaller_output
            );
            run_test!(
                sub_ct_assign_aligned,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_assign_aligned
            );
            run_test!(
                sub_ct_assign_self_lt,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_assign_self_lt
            );
            run_test!(
                sub_ct_assign_self_gt,
                $crate::leveled::tests::test_suite::sub::test_sub_ct_assign_self_gt
            );
            run_test!(
                sub_pt_vec_znx_assign,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_znx_assign
            );
            run_test!(
                sub_pt_vec_znx_into,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_znx
            );
            run_test!(
                sub_pt_vec_znx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_znx_into_delta_log_delta
            );
            run_test!(
                sub_pt_vec_rnx_assign,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_rnx_assign
            );
            run_test!(
                sub_pt_vec_rnx_into,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_rnx
            );
            run_test!(
                sub_pt_vec_rnx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_rnx_into_delta_log_delta
            );
            run_test!(
                sub_pt_vec_znx_into_smaller_output,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_znx_into_smaller_output
            );
            run_test!(
                sub_pt_vec_rnx_into_smaller_output,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_vec_rnx_into_smaller_output
            );
            run_test!(
                sub_pt_const_znx_into_aligned,
                $crate::leveled::tests::test_suite::sub::test_sub_pt_const_znx_into_aligned
            );
            run_test!(
                sub_ct_aligned_unsafe,
                $crate::leveled::tests::test_suite::sub_unsafe::test_sub_ct_aligned_unsafe
            );
            run_test!(
                sub_ct_assign_aligned_unsafe,
                $crate::leveled::tests::test_suite::sub_unsafe::test_sub_ct_assign_aligned_unsafe
            );
            run_test!(
                sub_pt_vec_znx_into_unsafe,
                $crate::leveled::tests::test_suite::sub_unsafe::test_sub_pt_vec_znx_into_unsafe
            );
            run_test!(
                sub_pt_vec_rnx_into_unsafe,
                $crate::leveled::tests::test_suite::sub_unsafe::test_sub_pt_vec_rnx_into_unsafe
            );
            run_test!(
                sub_pt_const_znx_into_aligned_unsafe,
                $crate::leveled::tests::test_suite::sub_unsafe::test_sub_pt_const_znx_into_aligned_unsafe
            );
            run_test!(
                dot_product_overflow_guard,
                $crate::leveled::tests::test_suite::errors::test_dot_product_overflow_guard
            );
            run_test_result!(neg, $crate::leveled::tests::test_suite::neg::test_neg_aligned);
            run_test_result!(
                neg_smaller_output,
                $crate::leveled::tests::test_suite::neg::test_neg_smaller_output
            );
            run_test_result!(neg_assign, $crate::leveled::tests::test_suite::neg::test_neg_assign);
            run_test!(
                conjugate_aligned,
                $crate::leveled::tests::test_suite::conjugate::test_conjugate_aligned
            );
            run_test!(
                conjugate_smaller_output,
                $crate::leveled::tests::test_suite::conjugate::test_conjugate_smaller_output
            );
            run_test!(
                conjugate_assign,
                $crate::leveled::tests::test_suite::conjugate::test_conjugate_assign
            );
            run_test_with_arg!(
                rotate_aligned,
                $crate::leveled::tests::test_suite::rotate::test_rotate_aligned,
                $rotations
            );
            run_test_with_arg!(
                rotate_smaller_output,
                $crate::leveled::tests::test_suite::rotate::test_rotate_smaller_output,
                $rotations
            );
            run_test_with_arg!(
                rotate_assign,
                $crate::leveled::tests::test_suite::rotate::test_rotate_assign,
                $rotations
            );
            run_test!(
                rotate_assign_missing_key_error,
                $crate::leveled::tests::test_suite::rotate::test_rotate_assign_missing_key_error
            );
            run_test!(
                mul_ct_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_aligned
            );
            run_test!(
                mul_ct_delta_a_gt_b,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_delta_a_gt_b
            );
            run_test!(
                mul_ct_delta_a_lt_b,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_delta_a_lt_b
            );
            run_test!(
                mul_ct_delta_log_delta,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_delta_log_delta
            );
            run_test!(
                mul_ct_smaller_output,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_smaller_output
            );
            run_test!(
                mul_ct_assign_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_assign_aligned
            );
            run_test!(
                mul_ct_assign_self_lt,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_assign_self_lt
            );
            run_test!(
                mul_ct_assign_self_gt,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_assign_self_gt
            );
            run_test!(
                square_aligned,
                $crate::leveled::tests::test_suite::mul::test_square_aligned
            );
            run_test!(
                square_rescaled_input,
                $crate::leveled::tests::test_suite::mul::test_square_rescaled_input
            );
            run_test!(
                square_assign,
                $crate::leveled::tests::test_suite::mul::test_square_assign
            );
            run_test!(
                square_smaller_output,
                $crate::leveled::tests::test_suite::mul::test_square_smaller_output
            );
            run_test!(
                mul_pt_vec_znx_into_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_znx_into_aligned
            );
            run_test!(
                mul_pt_vec_znx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_znx_into_delta_log_delta
            );
            run_test!(
                mul_pt_vec_znx_into_smaller_output,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_znx_into_smaller_output
            );
            run_test!(
                mul_pt_vec_znx_assign,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_znx_assign
            );
            run_test!(
                mul_pt_vec_rnx_into_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_rnx_into_aligned
            );
            run_test!(
                mul_pt_vec_rnx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_rnx_into_delta_log_delta
            );
            run_test!(
                mul_pt_vec_rnx_into_smaller_output,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_rnx_into_smaller_output
            );
            run_test!(
                mul_pt_vec_rnx_assign,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_vec_rnx_assign
            );
            run_test!(
                mul_const_into_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_const_rnx_into_aligned
            );
            run_test!(
                mul_const_assign,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_const_assign
            );
            run_test!(
                mul_const_into_delta_log_delta,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_const_rnx_into_delta_log_delta
            );
            run_test!(
                mul_const_into_real_only,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_const_rnx_into_real_only
            );
            run_test!(
                mul_const_znx_into_aligned,
                $crate::leveled::tests::test_suite::mul::test_mul_pt_const_znx_into_aligned
            );
            run_test!(
                mul_ct_explicit_metadata_error,
                $crate::leveled::tests::test_suite::mul::test_mul_ct_explicit_metadata_error
            );
            run_test!(
                mul_pow2_aligned,
                $crate::leveled::tests::test_suite::mul_pow2::test_mul_pow2_aligned
            );
            run_test!(
                mul_pow2_smaller_output,
                $crate::leveled::tests::test_suite::mul_pow2::test_mul_pow2_smaller_output
            );
            run_test!(
                mul_pow2_assign,
                $crate::leveled::tests::test_suite::mul_pow2::test_mul_pow2_assign
            );
            run_test!(
                div_pow2_aligned,
                $crate::leveled::tests::test_suite::mul_pow2::test_div_pow2_aligned
            );
            run_test!(
                div_pow2_smaller_output,
                $crate::leveled::tests::test_suite::mul_pow2::test_div_pow2_smaller_output
            );
            run_test!(
                div_pow2_assign,
                $crate::leveled::tests::test_suite::mul_pow2::test_div_pow2_assign
            );
            run_test!(
                div_pow2_assign_explicit_error,
                $crate::leveled::tests::test_suite::mul_pow2::test_div_pow2_assign_explicit_error
            );
            run_test!(
                composition_linear_sum,
                $crate::leveled::tests::test_suite::composition::test_linear_sum
            );
            run_test!(
                composition_poly2_sum,
                $crate::leveled::tests::test_suite::composition::test_poly2_sum
            );
            run_test!(
                composition_poly2_sum_with_const,
                $crate::leveled::tests::test_suite::composition::test_poly2_sum_with_const
            );
            run_test!(
                composition_poly2_mul,
                $crate::leveled::tests::test_suite::composition::test_poly2_mul
            );
            run_test!(
                composition_repeated_square_exhausts_capacity,
                $crate::leveled::tests::test_suite::composition::test_repeated_square_exhausts_capacity
            );
            run_test!(
                add_many_aligned,
                $crate::leveled::tests::test_suite::add_many::test_add_many_aligned
            );
            run_test!(
                add_many_single_smaller_output,
                $crate::leveled::tests::test_suite::add_many::test_add_many_single_smaller_output
            );
            run_test!(
                add_many_unaligned_log_budget,
                $crate::leveled::tests::test_suite::add_many::test_add_many_unaligned_log_budget
            );
            run_test!(
                add_many_delta_log_delta,
                $crate::leveled::tests::test_suite::add_many::test_add_many_delta_log_delta
            );
            run_test!(
                add_many_smaller_output,
                $crate::leveled::tests::test_suite::add_many::test_add_many_smaller_output
            );
            run_test!(
                mul_many_aligned,
                $crate::leveled::tests::test_suite::mul_many::test_mul_many_aligned
            );
            run_test!(
                mul_many_single_smaller_output,
                $crate::leveled::tests::test_suite::mul_many::test_mul_many_single_smaller_output
            );
            run_test!(
                mul_many_odd_tree,
                $crate::leveled::tests::test_suite::mul_many::test_mul_many_odd_tree
            );
            run_test!(
                mul_many_unaligned_log_budget,
                $crate::leveled::tests::test_suite::mul_many::test_mul_many_unaligned_log_budget
            );
            run_test!(
                mul_many_smaller_output,
                $crate::leveled::tests::test_suite::mul_many::test_mul_many_smaller_output
            );
            run_test!(
                mul_add_ct_aligned,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_ct_aligned
            );
            run_test!(
                mul_add_ct_unaligned_dst,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_ct_unaligned_dst
            );
            run_test!(
                mul_add_pt_vec_znx_into_aligned,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_pt_vec_znx_into_aligned
            );
            run_test!(
                mul_add_pt_vec_znx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_pt_vec_znx_into_delta_log_delta
            );
            run_test!(
                mul_add_pt_vec_rnx_into_aligned,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_pt_vec_rnx_into_aligned
            );
            run_test!(
                mul_add_const_znx_into_aligned,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_const_znx_into_aligned
            );
            run_test!(
                mul_add_const_rnx_into_aligned,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_const_rnx_into_aligned
            );
            run_test!(
                mul_add_const_znx_zero_preserves_dst_meta,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_const_znx_zero_preserves_dst_meta
            );
            run_test!(
                mul_add_const_rnx_zero_preserves_dst_meta,
                $crate::leveled::tests::test_suite::mul_add::test_mul_add_const_rnx_zero_preserves_dst_meta
            );
            run_test!(
                mul_sub_ct_aligned,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_ct_aligned
            );
            run_test!(
                mul_sub_ct_unaligned_dst,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_ct_unaligned_dst
            );
            run_test!(
                mul_sub_pt_vec_znx_aligned,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_vec_znx_aligned
            );
            run_test!(
                mul_sub_pt_vec_znx_into_delta_log_delta,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_vec_znx_into_delta_log_delta
            );
            run_test!(
                mul_sub_pt_vec_rnx_aligned,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_vec_rnx_aligned
            );
            run_test!(
                mul_sub_pt_const_znx_into_aligned,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_const_znx_into_aligned
            );
            run_test!(
                mul_sub_pt_const_rnx_aligned,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_const_rnx_aligned
            );
            run_test!(
                mul_sub_pt_const_znx_zero_preserves_dst_meta,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_const_znx_zero_preserves_dst_meta
            );
            run_test!(
                mul_sub_pt_const_rnx_zero_preserves_dst_meta,
                $crate::leveled::tests::test_suite::mul_sub::test_mul_sub_pt_const_rnx_zero_preserves_dst_meta
            );
            run_test!(
                dot_product_ct_aligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_ct_aligned
            );
            run_test!(
                dot_product_ct_unaligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_ct_unaligned
            );
            run_test!(
                dot_product_ct_unaligned_b,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_ct_unaligned_b
            );
            run_test!(
                dot_product_ct_delta_log_delta,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_ct_delta_log_delta
            );
            run_test!(
                dot_product_ct_smaller_output,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_ct_smaller_output
            );
            run_test!(
                dot_product_pt_vec_znx_aligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_pt_vec_znx_aligned
            );
            run_test!(
                dot_product_pt_vec_rnx_aligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_pt_vec_rnx_aligned
            );
            run_test!(
                dot_product_const_znx_aligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_const_znx_aligned
            );
            run_test!(
                dot_product_const_rnx_aligned,
                $crate::leveled::tests::test_suite::dot_product::test_dot_product_const_rnx_aligned
            );
        }
    };
}

pub use crate::ckks_backend_test_suite;

pub mod add;
pub mod add_many;
pub mod add_unsafe;
pub mod composition;
pub mod conjugate;
pub mod dot_product;
pub mod encryption;
pub mod errors;
pub mod helpers;
pub mod mul;
pub mod mul_add;
pub mod mul_many;
pub mod mul_pow2;
pub mod mul_sub;
pub mod neg;
pub mod rotate;
pub mod sub;
pub mod sub_unsafe;
