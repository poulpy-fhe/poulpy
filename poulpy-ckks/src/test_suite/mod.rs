//! Backend-generic CKKS test suite.
//!
//! All test functions are generic over `BE: super::helpers::TestContextBackend` and take
//! `(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)`.
//! The backend-specific test harnesses (in downstream crates such as `poulpy-cpu-ref`)
//! instantiate and invoke these functions via the [`ckks_backend_test_suite!`] macro.

use poulpy_core::{
    EncryptionLayout,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKeyLayout, GLWELayout, GLWESwitchingKeyLayout, GLWETensorKeyLayout, Rank, TorusPrecision,
    },
};

use crate::SlotsKind;
use crate::{CKKSLayout, CKKSMeta};

/// Shared CKKS parameter set for test instantiation.
#[derive(Clone, Copy)]
pub struct CKKSTestParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    /// Plaintext precision metadata (`log_delta`, `log_sparsity`). The effective
    /// torus width is `log_delta + prec_log_budget`; see [`Self::prec`].
    pub prec_meta: CKKSMeta,
    /// Plaintext budget bits, i.e. `prec.k - prec_meta.log_delta`.
    pub prec_log_budget: usize,
    pub hw: usize,
    pub dsize: usize,
    /// GLWE rank of ciphertexts and keys (`1` for all standard parameter sets).
    pub rank: usize,
}

impl CKKSTestParams {
    /// Derives the full plaintext [`CKKSLayout`] from `prec_meta`/`prec_log_budget`,
    /// reusing the param set's `n`/`base2k` (rank-1). This is the single source of
    /// truth for the test plaintext precision; storing it would duplicate `n`,
    /// `base2k`, and the `log_delta + log_budget` sum.
    pub fn prec(&self) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(self.n as u32),
                base2k: Base2K(self.base2k as u32),
                k: TorusPrecision((self.prec_meta.log_delta + self.prec_log_budget) as u32),
                rank: Rank(self.rank as u32),
            },
            meta: self.prec_meta,
        }
    }

    /// `log2(n)` — the ring dimension in bits.
    pub fn log_n(&self) -> usize {
        self.n.ilog2() as usize
    }

    pub fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: self.k.into(),
            rank: Rank(self.rank as u32),
        })
        .unwrap()
    }

    pub fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let total = self.k + self.dsize * self.base2k;
        let dnum = total / (self.dsize * self.base2k);
        let k_aux = self.dsize * self.base2k + self.log_n();
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            rank: Rank(self.rank as u32),
            dsize: self.dsize.into(),
        })
        .unwrap()
    }

    pub fn atk_layout(&self) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
        let total = self.k + self.dsize * self.base2k;
        let dnum = total / (self.dsize * self.base2k);
        let k_aux = self.dsize * self.base2k + self.log_n();
        EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            rank: Rank(self.rank as u32),
            dsize: self.dsize.into(),
        })
        .unwrap()
    }

    /// Layout of a rank-1 GLWE key-switching key whose input ciphertext has
    /// modulus `k_in` bits (e.g. the encapsulation `denseToSparse` /
    /// `sparseToDense` keys, sized at the input level and at `k_boot`).
    pub fn ksk_layout(&self, k_in: usize) -> EncryptionLayout<GLWESwitchingKeyLayout> {
        let total = k_in + self.dsize * self.base2k;
        let dnum = total / (self.dsize * self.base2k);
        let k_aux = self.dsize * self.base2k + self.log_n();
        EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            rank_in: Rank(self.rank as u32),
            rank_out: Rank(self.rank as u32),
            dsize: self.dsize.into(),
        })
        .unwrap()
    }
}

/// NTT4x30 parameter set.
pub const NTT4X30_PARAMS_F64: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 52,
    k: 8 * 40,
    prec_meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: 40,
        slots: SlotsKind::Complex,
    },
    prec_log_budget: 30,
    hw: 192,
    dsize: 1,
    rank: 1,
};

/// FFT64 parameter set.
pub const FFT64_PARAMS_F64: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 19,
    k: 8 * 19,
    prec_meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: 30,
        slots: SlotsKind::Complex,
    },
    prec_log_budget: 10,
    hw: 192,
    dsize: 1,
    rank: 1,
};

/// NTT4x30 parameter set.
pub const NTT4X30_PARAMS_F128: CKKSTestParams = CKKSTestParams {
    n: 256,
    base2k: 52,
    k: 8 * 80,
    prec_meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: 80,
        slots: SlotsKind::Complex,
    },
    prec_log_budget: 30,
    hw: 192,
    dsize: 1,
    rank: 1,
};

/// Registers the **rank-generic arithmetic subset** of the CKKS suite for a
/// parameter set with `rank > 1`: encrypt/decrypt, add/sub (ct and pt), mul
/// (ct, square, prepared, pt), neg, copy, pow2, rotate, and conjugate.
///
/// The full [`ckks_backend_test_suite!`](crate::ckks_backend_test_suite) hardwires rank-1-only pipelines
/// (bootstrapping, EvalMod, DFT, PaCo — which reject `rank != 1` by
/// construction), so higher-rank coverage uses this subset instead.
/// [`NTT4X30_PARAMS_F64`] at GLWE rank 2, for the rank-generic arithmetic
/// subset ([`ckks_backend_rank2_test_suite!`](crate::ckks_backend_rank2_test_suite)).
pub const NTT4X30_PARAMS_F64_RANK2: CKKSTestParams = CKKSTestParams {
    rank: 2,
    ..NTT4X30_PARAMS_F64
};

#[macro_export]
macro_rules! ckks_backend_rank2_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        scalar = $scalar:ty,
        encoder = $encoder_ty:ty,
        params = $params:expr,
        rotations = $rotations:expr $(,)?
    ) => {
        mod $modname {
            use std::sync::LazyLock;

            use anyhow::Result;

            use poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> = LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE);
                    }
                };
            }

            macro_rules! run_test_with_arg {
                ($name:ident, $path:path, $arg:expr) => {
                    #[test]
                    fn $name() {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE, $arg);
                    }
                };
            }

            macro_rules! run_test_result {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() -> Result<()> {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE)
                    }
                };
            }

            run_test!(encrypt_decrypt, $crate::test_suite::encryption::test_encrypt_decrypt);
            run_test!(add_ct_aligned, $crate::test_suite::add::test_add_ct_aligned);
            run_test!(add_ct_delta_a_lt_b, $crate::test_suite::add::test_add_ct_delta_a_lt_b);
            run_test!(add_ct_delta_a_gt_b, $crate::test_suite::add::test_add_ct_delta_a_gt_b);
            run_test!(add_pt_vec_assign, $crate::test_suite::add::test_add_pt_vec_assign);
            run_test!(add_const_assign, $crate::test_suite::add::test_add_const_assign);
            run_test!(add_one_assign, $crate::test_suite::add::test_add_one_assign);
            run_test!(sub_ct_aligned, $crate::test_suite::sub::test_sub_ct_aligned);
            run_test!(sub_ct_delta_a_lt_b, $crate::test_suite::sub::test_sub_ct_delta_a_lt_b);
            run_test!(sub_ct_delta_a_gt_b, $crate::test_suite::sub::test_sub_ct_delta_a_gt_b);
            run_test!(sub_pt_vec_assign, $crate::test_suite::sub::test_sub_pt_vec_assign);
            run_test!(sub_one_assign, $crate::test_suite::sub::test_sub_one_assign);
            run_test_result!(copy_aligned, $crate::test_suite::copy::test_copy_aligned);
            run_test_result!(neg, $crate::test_suite::neg::test_neg_aligned);
            run_test_result!(neg_wider_output, $crate::test_suite::neg::test_neg_wider_output);
            run_test_result!(neg_assign, $crate::test_suite::neg::test_neg_assign);
            run_test!(mul_ct_aligned, $crate::test_suite::mul::test_mul_ct_aligned);
            run_test!(
                mul_ct_assign_aligned,
                $crate::test_suite::mul::test_mul_ct_assign_aligned
            );
            run_test!(
                mul_ct_smaller_output,
                $crate::test_suite::mul::test_mul_ct_smaller_output
            );
            run_test!(
                mul_ct_smaller_output_exact_scratch,
                $crate::test_suite::mul::test_mul_ct_smaller_output_exact_scratch
            );
            run_test!(square_aligned, $crate::test_suite::mul::test_square_aligned);
            run_test!(
                mul_pt_vec_into_aligned,
                $crate::test_suite::mul::test_mul_pt_vec_into_aligned
            );
            run_test!(mul_pt_vec_assign, $crate::test_suite::mul::test_mul_pt_vec_assign);
            run_test!(mul_pow2_aligned, $crate::test_suite::mul_pow2::test_mul_pow2_aligned);
            run_test!(div_pow2_aligned, $crate::test_suite::mul_pow2::test_div_pow2_aligned);
            run_test!(div_pow2_assign, $crate::test_suite::mul_pow2::test_div_pow2_assign);
            run_test_with_arg!(
                rotate_aligned,
                $crate::test_suite::rotate::test_rotate_aligned,
                $rotations
            );
            run_test!(conjugate_aligned, $crate::test_suite::conjugate::test_conjugate_aligned);
            run_test!(conjugate_assign, $crate::test_suite::conjugate::test_conjugate_assign);
        }
    };
}

#[macro_export]
macro_rules! ckks_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        scalar = $scalar:ty,
        encoder = $encoder_ty:ty,
        params = $params:expr,
        rotations = $rotations:expr $(,)?
    ) => {
        mod $modname {
            use std::sync::LazyLock;

            use anyhow::Result;

            use poulpy_hal::layouts::{HostBytesBackend, Module};

            static MODULE: LazyLock<Module<$backend>> = LazyLock::new(|| Module::<$backend>::new($params.n as u64));
            static HOST_MODULE: LazyLock<Module<HostBytesBackend>> =
                LazyLock::new(|| Module::<HostBytesBackend>::new($params.n as u64));

            macro_rules! run_test {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE);
                    }
                };
            }

            macro_rules! run_test_with_arg {
                ($name:ident, $path:path, $arg:expr) => {
                    #[test]
                    fn $name() {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE, $arg);
                    }
                };
            }

            macro_rules! run_test_ignored {
                ($name:ident, $path:path) => {
                    #[test]
                    #[ignore = "large parameters; run explicitly with --ignored --release"]
                    fn $name() {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE);
                    }
                };
            }

            macro_rules! run_test_result {
                ($name:ident, $path:path) => {
                    #[test]
                    fn $name() -> Result<()> {
                        #[allow(clippy::unsafe_removed_from_name)]
                        use $path as __test_fn;
                        __test_fn::<$backend, $scalar, $encoder_ty>($params, &*MODULE, &*HOST_MODULE)
                    }
                };
            }

            run_test!(
                encode_decode_reim_roundtrip,
                $crate::test_suite::encoding::test_encode_decode_reim_roundtrip
            );
            run_test!(
                slots_kind_composition,
                $crate::test_suite::slots_kind::test_slots_kind_composition
            );
            run_test!(encrypt_decrypt, $crate::test_suite::encryption::test_encrypt_decrypt);
            run_test!(
                decrypt_extract_same_meta,
                $crate::test_suite::encryption::test_decrypt_extract_same_meta
            );
            run_test!(
                decrypt_extract_truncates_log_budget,
                $crate::test_suite::encryption::test_decrypt_extract_truncates_log_budget
            );
            run_test!(
                decrypt_extract_rsh_for_smaller_log_delta,
                $crate::test_suite::encryption::test_decrypt_extract_rsh_for_smaller_log_delta
            );
            run_test!(
                decrypt_extract_lsh_for_larger_log_delta,
                $crate::test_suite::encryption::test_decrypt_extract_lsh_for_larger_log_delta
            );
            run_test!(
                decrypt_extract_output_hom_rem_too_large,
                $crate::test_suite::encryption::test_decrypt_extract_output_hom_rem_too_large
            );
            run_test!(
                decrypt_extract_base2k_mismatch_error,
                $crate::test_suite::encryption::test_decrypt_extract_base2k_mismatch_error
            );
            run_test!(
                add_pt_vec_alignment_error,
                $crate::test_suite::errors::test_add_pt_vec_alignment_error
            );
            run_test!(add_ct_aligned, $crate::test_suite::add::test_add_ct_aligned);
            run_test!(add_ct_delta_a_lt_b, $crate::test_suite::add::test_add_ct_delta_a_lt_b);
            run_test!(add_ct_delta_a_gt_b, $crate::test_suite::add::test_add_ct_delta_a_gt_b);
            run_test!(
                add_ct_delta_log_delta,
                $crate::test_suite::add::test_add_ct_delta_log_delta
            );
            run_test!(
                add_ct_aligned_smaller_output,
                $crate::test_suite::add::test_add_ct_aligned_smaller_output
            );
            run_test!(
                add_ct_assign_aligned,
                $crate::test_suite::add::test_add_ct_assign_aligned
            );
            run_test!(
                add_ct_assign_self_lt,
                $crate::test_suite::add::test_add_ct_assign_self_lt
            );
            run_test!(
                add_ct_assign_self_gt,
                $crate::test_suite::add::test_add_ct_assign_self_gt
            );
            run_test!(add_pt_vec_assign, $crate::test_suite::add::test_add_pt_vec_assign);
            run_test!(
                add_pt_vec_into_aligned,
                $crate::test_suite::add::test_add_pt_vec_into_aligned
            );
            run_test!(
                add_pt_vec_into_delta_log_delta,
                $crate::test_suite::add::test_add_pt_vec_into_delta_log_delta
            );
            run_test!(
                add_pt_vec_into_lsh_alignment,
                $crate::test_suite::add::test_add_pt_vec_into_lsh_alignment
            );
            run_test!(
                add_const_into_aligned,
                $crate::test_suite::add::test_add_const_into_aligned
            );
            run_test!(
                add_const_into_lsh_alignment,
                $crate::test_suite::add::test_add_const_into_lsh_alignment
            );
            run_test!(add_const_assign, $crate::test_suite::add::test_add_const_assign);
            run_test!(add_one_assign, $crate::test_suite::add::test_add_one_assign);
            run_test!(
                add_const_into_delta_log_delta,
                $crate::test_suite::add::test_add_const_into_delta_log_delta
            );
            run_test!(
                add_const_into_smaller_output,
                $crate::test_suite::add::test_add_const_into_smaller_output
            );
            run_test!(
                add_const_into_real_only,
                $crate::test_suite::add::test_add_const_into_real_only
            );
            run_test!(
                add_pt_vec_into_smaller_output,
                $crate::test_suite::add::test_add_pt_vec_into_smaller_output
            );
            run_test!(
                add_pt_vec_base2k_mismatch_error,
                $crate::test_suite::add::test_add_pt_vec_base2k_mismatch_error
            );
            run_test!(
                add_ct_aligned_unsafe,
                $crate::test_suite::add_unsafe::test_add_ct_aligned_unsafe
            );
            run_test!(
                add_ct_assign_aligned_unsafe,
                $crate::test_suite::add_unsafe::test_add_ct_assign_aligned_unsafe
            );
            run_test!(
                add_pt_vec_into_aligned_unsafe,
                $crate::test_suite::add_unsafe::test_add_pt_vec_into_aligned_unsafe
            );
            run_test!(
                add_const_into_aligned_unsafe,
                $crate::test_suite::add_unsafe::test_add_const_into_aligned_unsafe
            );
            run_test!(sub_ct_aligned, $crate::test_suite::sub::test_sub_ct_aligned);
            run_test!(sub_ct_delta_a_lt_b, $crate::test_suite::sub::test_sub_ct_delta_a_lt_b);
            run_test!(sub_ct_delta_a_gt_b, $crate::test_suite::sub::test_sub_ct_delta_a_gt_b);
            run_test!(
                sub_ct_delta_log_delta,
                $crate::test_suite::sub::test_sub_ct_delta_log_delta
            );
            run_test!(
                sub_ct_smaller_output,
                $crate::test_suite::sub::test_sub_ct_smaller_output
            );
            run_test!(
                sub_ct_assign_aligned,
                $crate::test_suite::sub::test_sub_ct_assign_aligned
            );
            run_test!(
                sub_ct_assign_self_lt,
                $crate::test_suite::sub::test_sub_ct_assign_self_lt
            );
            run_test!(
                sub_ct_assign_self_gt,
                $crate::test_suite::sub::test_sub_ct_assign_self_gt
            );
            run_test!(sub_pt_vec_assign, $crate::test_suite::sub::test_sub_pt_vec_assign);
            run_test!(sub_pt_vec_into, $crate::test_suite::sub::test_sub_pt_vec);
            run_test!(
                sub_pt_vec_into_delta_log_delta,
                $crate::test_suite::sub::test_sub_pt_vec_into_delta_log_delta
            );
            run_test!(
                sub_pt_vec_into_smaller_output,
                $crate::test_suite::sub::test_sub_pt_vec_into_smaller_output
            );
            run_test!(
                sub_pt_const_into_aligned,
                $crate::test_suite::sub::test_sub_pt_const_into_aligned
            );
            run_test!(
                sub_const_into_lsh_alignment,
                $crate::test_suite::sub::test_sub_const_into_lsh_alignment
            );
            run_test!(sub_one_assign, $crate::test_suite::sub::test_sub_one_assign);
            run_test!(
                sub_ct_aligned_unsafe,
                $crate::test_suite::sub_unsafe::test_sub_ct_aligned_unsafe
            );
            run_test!(
                sub_ct_assign_aligned_unsafe,
                $crate::test_suite::sub_unsafe::test_sub_ct_assign_aligned_unsafe
            );
            run_test!(
                sub_pt_vec_into_unsafe,
                $crate::test_suite::sub_unsafe::test_sub_pt_vec_into_unsafe
            );
            run_test!(
                sub_pt_const_into_aligned_unsafe,
                $crate::test_suite::sub_unsafe::test_sub_pt_const_into_aligned_unsafe
            );
            run_test!(
                dot_product_overflow_guard,
                $crate::test_suite::errors::test_dot_product_overflow_guard
            );
            run_test_result!(copy_aligned, $crate::test_suite::copy::test_copy_aligned);
            run_test_result!(copy_smaller_output, $crate::test_suite::copy::test_copy_smaller_output);
            run_test_result!(neg, $crate::test_suite::neg::test_neg_aligned);
            run_test_result!(neg_smaller_output, $crate::test_suite::neg::test_neg_smaller_output);
            run_test_result!(neg_wider_output, $crate::test_suite::neg::test_neg_wider_output);
            run_test_result!(neg_assign, $crate::test_suite::neg::test_neg_assign);
            run_test!(conjugate_aligned, $crate::test_suite::conjugate::test_conjugate_aligned);
            run_test!(
                conjugate_smaller_output,
                $crate::test_suite::conjugate::test_conjugate_smaller_output
            );
            run_test!(conjugate_assign, $crate::test_suite::conjugate::test_conjugate_assign);
            run_test_with_arg!(
                rotate_aligned,
                $crate::test_suite::rotate::test_rotate_aligned,
                $rotations
            );
            run_test_with_arg!(
                rotate_smaller_output,
                $crate::test_suite::rotate::test_rotate_smaller_output,
                $rotations
            );
            run_test_with_arg!(
                rotate_assign,
                $crate::test_suite::rotate::test_rotate_assign,
                $rotations
            );
            run_test!(
                rotate_assign_missing_key_error,
                $crate::test_suite::rotate::test_rotate_assign_missing_key_error
            );
            run_test!(
                linear_transformation,
                $crate::test_suite::linear_transformation::test_linear_transformation
            );
            run_test!(paco_partial_c2s, $crate::test_suite::paco_lt::test_paco_partial_c2s);
            run_test!(paco_packing, $crate::test_suite::paco_lt::test_paco_packing);
            run_test!(
                paco_cleartext_reference,
                $crate::test_suite::paco_reference::test_paco_cleartext_reference
            );
            run_test!(paco_stc, $crate::test_suite::paco_lt::test_paco_stc);
            run_test!(paco_slot_trace, $crate::test_suite::paco_ops::test_paco_slot_trace);
            run_test!(paco_slot_product, $crate::test_suite::paco_ops::test_paco_slot_product);
            run_test!(paco_conj_rotate, $crate::test_suite::paco_ops::test_paco_conj_rotate);
            run_test!(
                paco_partial_pipeline,
                $crate::test_suite::paco_keys::test_paco_partial_pipeline
            );
            run_test!(
                paco_seq_bootstrap,
                $crate::test_suite::paco_bootstrap::test_paco_seq_bootstrap
            );
            run_test!(
                paco_parallel_bootstrap,
                $crate::test_suite::paco_parallel::test_paco_parallel_bootstrap
            );
            run_test!(
                paco_encapsulated_bootstrap,
                $crate::test_suite::paco_parallel::test_paco_encapsulated_bootstrap
            );
            run_test_ignored!(
                paco_paper_scale,
                $crate::test_suite::paco_bootstrap::test_paco_paper_scale
            );
            run_test!(ship_host_replica, $crate::test_suite::ship::test_ship_host_replica);
            run_test!(ship_mux_rotate, $crate::test_suite::ship::test_ship_mux_rotate);
            run_test!(ship_bootstrap, $crate::test_suite::ship::test_ship_bootstrap);
            run_test!(
                ship_bootstrap_complex,
                $crate::test_suite::ship::test_ship_bootstrap_complex
            );
            run_test!(
                dft_coeffs_to_slots_standard,
                $crate::test_suite::dft::test_dft_coeffs_to_slots_standard
            );
            run_test!(
                dft_slots_to_coeffs_standard,
                $crate::test_suite::dft::test_dft_slots_to_coeffs_standard
            );
            run_test!(
                dft_coeffs_to_slots_split,
                $crate::test_suite::dft::test_dft_coeffs_to_slots_split
            );
            run_test!(
                dft_slots_to_coeffs_split,
                $crate::test_suite::dft::test_dft_slots_to_coeffs_split
            );
            run_test!(
                dft_coeffs_to_slots_repack_sparse,
                $crate::test_suite::dft::test_dft_coeffs_to_slots_repack_sparse
            );
            run_test!(
                dft_slots_to_coeffs_repack_sparse,
                $crate::test_suite::dft::test_dft_slots_to_coeffs_repack_sparse
            );
            run_test!(
                dft_plan_helpers_match_compiled,
                $crate::test_suite::dft::test_dft_plan_helpers_match_compiled
            );
            run_test!(
                eval_mod_pair_matches_singles,
                $crate::test_suite::eval_mod::test_eval_mod_pair_matches_singles
            );
            run_test!(
                backend_views_expose_logical_width,
                $crate::test_suite::logical_width::test_backend_views_expose_logical_width
            );
            run_test!(
                inactive_capacity_does_not_change_results,
                $crate::test_suite::logical_width::test_inactive_capacity_does_not_change_results
            );
            run_test!(
                dft_ping_pong_matches_assign_chain,
                $crate::test_suite::logical_width::test_dft_ping_pong_matches_assign_chain
            );
            run_test!(mul_ct_aligned, $crate::test_suite::mul::test_mul_ct_aligned);
            run_test!(mul_ct_delta_a_gt_b, $crate::test_suite::mul::test_mul_ct_delta_a_gt_b);
            run_test!(mul_ct_delta_a_lt_b, $crate::test_suite::mul::test_mul_ct_delta_a_lt_b);
            run_test!(
                mul_ct_delta_log_delta,
                $crate::test_suite::mul::test_mul_ct_delta_log_delta
            );
            run_test!(
                mul_ct_smaller_output,
                $crate::test_suite::mul::test_mul_ct_smaller_output
            );
            run_test!(
                mul_ct_smaller_output_exact_scratch,
                $crate::test_suite::mul::test_mul_ct_smaller_output_exact_scratch
            );
            run_test!(
                mul_ct_assign_aligned,
                $crate::test_suite::mul::test_mul_ct_assign_aligned
            );
            run_test!(
                mul_ct_assign_self_lt,
                $crate::test_suite::mul::test_mul_ct_assign_self_lt
            );
            run_test!(
                mul_ct_assign_self_gt,
                $crate::test_suite::mul::test_mul_ct_assign_self_gt
            );
            run_test!(square_aligned, $crate::test_suite::mul::test_square_aligned);
            run_test!(
                square_rescaled_input,
                $crate::test_suite::mul::test_square_rescaled_input
            );
            run_test!(square_assign, $crate::test_suite::mul::test_square_assign);
            run_test!(
                square_smaller_output,
                $crate::test_suite::mul::test_square_smaller_output
            );
            run_test!(
                mul_pt_vec_into_aligned,
                $crate::test_suite::mul::test_mul_pt_vec_into_aligned
            );
            run_test!(
                mul_pt_vec_into_delta_log_delta,
                $crate::test_suite::mul::test_mul_pt_vec_into_delta_log_delta
            );
            run_test!(
                mul_pt_vec_into_smaller_output,
                $crate::test_suite::mul::test_mul_pt_vec_into_smaller_output
            );
            run_test!(mul_pt_vec_assign, $crate::test_suite::mul::test_mul_pt_vec_assign);
            run_test!(
                mul_const_into_aligned,
                $crate::test_suite::mul::test_mul_pt_const_into_aligned
            );
            run_test!(mul_const_assign, $crate::test_suite::mul::test_mul_pt_const_assign);
            run_test!(
                mul_const_into_delta_log_delta,
                $crate::test_suite::mul::test_mul_pt_const_into_delta_log_delta
            );
            run_test!(
                mul_prepared_layout_mismatch_error,
                $crate::test_suite::mul::test_mul_prepared_layout_mismatch_error
            );
            run_test!(
                mul_ct_explicit_metadata_error,
                $crate::test_suite::mul::test_mul_ct_explicit_metadata_error
            );
            run_test!(mul_pow2_aligned, $crate::test_suite::mul_pow2::test_mul_pow2_aligned);
            run_test!(
                mul_pow2_smaller_output,
                $crate::test_suite::mul_pow2::test_mul_pow2_smaller_output
            );
            run_test!(mul_pow2_assign, $crate::test_suite::mul_pow2::test_mul_pow2_assign);
            run_test!(div_pow2_aligned, $crate::test_suite::mul_pow2::test_div_pow2_aligned);
            run_test!(
                div_pow2_smaller_output,
                $crate::test_suite::mul_pow2::test_div_pow2_smaller_output
            );
            run_test!(div_pow2_assign, $crate::test_suite::mul_pow2::test_div_pow2_assign);
            run_test!(
                div_pow2_assign_explicit_error,
                $crate::test_suite::mul_pow2::test_div_pow2_assign_explicit_error
            );
            run_test!(mul_i_aligned, $crate::test_suite::imag::test_mul_i_aligned);
            run_test!(
                mul_i_smaller_output,
                $crate::test_suite::imag::test_mul_i_smaller_output
            );
            run_test!(mul_i_assign, $crate::test_suite::imag::test_mul_i_assign);
            run_test!(div_i_aligned, $crate::test_suite::imag::test_div_i_aligned);
            run_test!(
                div_i_smaller_output,
                $crate::test_suite::imag::test_div_i_smaller_output
            );
            run_test!(div_i_assign, $crate::test_suite::imag::test_div_i_assign);
            run_test!(composition_linear_sum, $crate::test_suite::composition::test_linear_sum);
            run_test!(composition_poly2_sum, $crate::test_suite::composition::test_poly2_sum);
            run_test!(
                composition_poly2_sum_with_const,
                $crate::test_suite::composition::test_poly2_sum_with_const
            );
            run_test!(composition_poly2_mul, $crate::test_suite::composition::test_poly2_mul);
            run_test!(
                composition_repeated_square_exhausts_capacity,
                $crate::test_suite::composition::test_repeated_square_exhausts_capacity
            );
            run_test!(add_many_aligned, $crate::test_suite::add_many::test_add_many_aligned);
            run_test!(
                add_many_single_smaller_output,
                $crate::test_suite::add_many::test_add_many_single_smaller_output
            );
            run_test!(
                add_many_unaligned_log_budget,
                $crate::test_suite::add_many::test_add_many_unaligned_log_budget
            );
            run_test!(
                add_many_delta_log_delta,
                $crate::test_suite::add_many::test_add_many_delta_log_delta
            );
            run_test!(
                add_many_smaller_output,
                $crate::test_suite::add_many::test_add_many_smaller_output
            );
            run_test!(mul_add_ct_aligned, $crate::test_suite::mul_add::test_mul_add_ct_aligned);
            run_test!(
                mul_add_ct_unaligned_dst,
                $crate::test_suite::mul_add::test_mul_add_ct_unaligned_dst
            );
            run_test!(
                mul_add_pt_vec_into_aligned,
                $crate::test_suite::mul_add::test_mul_add_pt_vec_into_aligned
            );
            run_test!(
                mul_add_pt_vec_into_delta_log_delta,
                $crate::test_suite::mul_add::test_mul_add_pt_vec_into_delta_log_delta
            );
            run_test!(
                mul_add_const_into_aligned,
                $crate::test_suite::mul_add::test_mul_add_const_into_aligned
            );
            run_test!(
                power_basis_populate_degree7,
                $crate::test_suite::polynomial_evaluation::test_power_basis_populate_degree7
            );
            run_test!(
                power_basis_populate_chebyshev_degree7,
                $crate::test_suite::polynomial_evaluation::test_power_basis_populate_chebyshev_degree7
            );
            run_test!(
                chebyshev_interpolation_quadratic,
                $crate::test_suite::polynomial_evaluation::test_chebyshev_interpolation_quadratic
            );
            run_test!(
                encode_bsgs_preserves_chebyshev_eval,
                $crate::test_suite::polynomial_evaluation::test_encode_bsgs_preserves_chebyshev_eval
            );
            run_test!(
                eval_poly_const_coeffs_cubic,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_cubic
            );
            run_test!(
                eval_poly_rejects_power_basis_mismatch,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_rejects_power_basis_mismatch
            );
            run_test!(
                eval_poly_const_coeffs_exp7,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_exp7
            );
            run_test!(
                eval_poly_const_coeffs_even_monomial,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_even_monomial
            );
            run_test!(
                eval_poly_const_coeffs_odd_monomial,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_odd_monomial
            );
            run_test!(
                eval_poly_const_coeffs_chebyshev_degree31,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_chebyshev_degree31
            );
            run_test!(
                eval_poly_const_coeffs_chebyshev_degree31_min_mult,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_chebyshev_degree31_min_mult
            );
            run_test!(
                eval_poly_const_coeffs_parity_folds,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_parity_folds
            );
            run_test!(
                eval_poly_consumed_bits_sweep,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_consumed_bits_sweep
            );
            run_test!(
                eval_poly_const_coeffs_complex_cubic,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_complex_cubic
            );
            run_test!(
                eval_poly_const_coeffs_complex_chebyshev,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_complex_chebyshev
            );
            run_test!(
                eval_poly_const_coeffs_complex_even,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_complex_even
            );
            run_test!(
                eval_poly_const_coeffs_complex_odd,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_complex_odd
            );
            run_test!(
                eval_poly_const_coeffs_complex_fold,
                $crate::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_complex_fold
            );
            run_test!(
                eval_mod_sin_continuous_minimal,
                $crate::test_suite::eval_mod::test_eval_mod_sin_continuous_minimal
            );
            run_test!(
                eval_mod_sin_continuous_with_arcsine,
                $crate::test_suite::eval_mod::test_eval_mod_sin_continuous_with_arcsine
            );
            run_test!(
                eval_mod_cos_discrete,
                $crate::test_suite::eval_mod::test_eval_mod_cos_discrete
            );
            run_test!(
                eval_mod_cos_continuous,
                $crate::test_suite::eval_mod::test_eval_mod_cos_continuous
            );
            run_test!(eval_mod_exp, $crate::test_suite::eval_mod::test_eval_mod_exp);
            run_test!(
                bootstrapping_standard_e2e,
                $crate::test_suite::bootstrapping::test_bootstrapping_standard_e2e
            );
            run_test!(
                bootstrapping_evalround_e2e,
                $crate::test_suite::bootstrapping::test_bootstrapping_evalround_e2e
            );
            run_test!(
                bootstrapping_s2c_first_e2e,
                $crate::test_suite::bootstrapping::test_bootstrapping_s2c_first_e2e
            );
            run_test!(
                functional_bootstrapping_e2e,
                $crate::test_suite::functional_bootstrapping::test_functional_bootstrapping_e2e
            );
            run_test!(
                functional_bootstrapping_multi_e2e,
                $crate::test_suite::functional_bootstrapping::test_functional_bootstrapping_multi_e2e
            );
            run_test!(
                functional_bootstrapping_binary_e2e,
                $crate::test_suite::functional_bootstrapping::test_functional_bootstrapping_binary_e2e
            );
            run_test!(
                mul_add_const_zero_preserves_dst_meta,
                $crate::test_suite::mul_add::test_mul_add_const_zero_preserves_dst_meta
            );
            run_test!(
                affine_pt_const_into_aligned,
                $crate::test_suite::affine::test_affine_pt_const_into_aligned
            );
            run_test!(
                affine_pt_const_zero_bias_matches_mul,
                $crate::test_suite::affine::test_affine_pt_const_zero_bias_matches_mul
            );
            run_test!(
                affine_pt_const_assign_aligned,
                $crate::test_suite::affine::test_affine_pt_const_assign_aligned
            );
            run_test!(
                affine_pt_vec_into_aligned,
                $crate::test_suite::affine::test_affine_pt_vec_into_aligned
            );
            run_test!(
                affine_pt_vec_assign_aligned,
                $crate::test_suite::affine::test_affine_pt_vec_assign_aligned
            );
            run_test!(mul_sub_ct_aligned, $crate::test_suite::mul_sub::test_mul_sub_ct_aligned);
            run_test!(
                mul_sub_ct_unaligned_dst,
                $crate::test_suite::mul_sub::test_mul_sub_ct_unaligned_dst
            );
            run_test!(
                mul_sub_pt_vec_aligned,
                $crate::test_suite::mul_sub::test_mul_sub_pt_vec_aligned
            );
            run_test!(
                mul_sub_pt_vec_into_delta_log_delta,
                $crate::test_suite::mul_sub::test_mul_sub_pt_vec_into_delta_log_delta
            );
            run_test!(
                mul_sub_pt_const_into_aligned,
                $crate::test_suite::mul_sub::test_mul_sub_pt_const_into_aligned
            );
            run_test!(
                mul_sub_pt_const_zero_preserves_dst_meta,
                $crate::test_suite::mul_sub::test_mul_sub_pt_const_zero_preserves_dst_meta
            );
            run_test!(
                dot_product_ct_aligned,
                $crate::test_suite::dot_product::test_dot_product_ct_aligned
            );
            run_test!(
                dot_product_ct_unaligned,
                $crate::test_suite::dot_product::test_dot_product_ct_unaligned
            );
            run_test!(
                dot_product_ct_unaligned_b,
                $crate::test_suite::dot_product::test_dot_product_ct_unaligned_b
            );
            run_test!(
                dot_product_ct_delta_log_delta,
                $crate::test_suite::dot_product::test_dot_product_ct_delta_log_delta
            );
            run_test!(
                dot_product_ct_smaller_output,
                $crate::test_suite::dot_product::test_dot_product_ct_smaller_output
            );
            run_test!(
                dot_product_pt_vec_aligned,
                $crate::test_suite::dot_product::test_dot_product_pt_vec_aligned
            );
            run_test!(
                dot_product_const_aligned,
                $crate::test_suite::dot_product::test_dot_product_const_aligned
            );
        }
    };
}

pub use crate::ckks_backend_test_suite;

pub mod add;
pub mod add_many;
pub mod add_unsafe;
pub mod affine;
pub mod bootstrapping;
pub mod composition;
pub mod conjugate;
pub mod copy;
pub mod dft;
pub mod dot_product;
pub mod encoding;
pub mod encryption;
pub mod errors;
pub mod eval_mod;
pub mod functional_bootstrapping;
pub mod helpers;
pub mod imag;
pub mod linear_transformation;
pub mod logical_width;
pub mod mul;
pub mod mul_add;
pub mod mul_pow2;
pub mod mul_sub;
pub mod neg;
pub mod paco_bootstrap;
pub mod paco_keys;
pub mod paco_lt;
pub mod paco_ops;
pub mod paco_parallel;
pub mod paco_reference;
pub(crate) mod paco_reference_model;
pub mod polynomial_evaluation;
#[doc(hidden)]
pub mod reference_encoder;
pub mod rotate;
pub mod ship;
pub mod slots_kind;
pub mod sub;
pub mod sub_unsafe;
