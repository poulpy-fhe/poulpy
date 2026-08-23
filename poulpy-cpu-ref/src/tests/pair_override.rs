//! Acceptance coverage for a backend that overrides *only* the paired EvalMod.
//!
//! [`DelegatingFFT64Ref`] is wired through every core and CKKS default except
//! `impl_ckks_eval_mod_defaults!`: it writes `CKKSEvalModOpsDefault` itself and
//! substitutes the two paired methods, inheriting the single-EvalMod reference
//! pipeline. That is exactly the shape a device backend needs, and it pins:
//!
//! - the public pair sizing dispatches to the override;
//! - the bootstrap budget includes it;
//! - both bootstrap pipelines actually invoke the paired hook.

use std::sync::atomic::{AtomicUsize, Ordering};

use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps},
    default::eval_mod::CKKSEvalModOpsDefault,
    impl_ckks_add_defaults, impl_ckks_bootstrap_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults,
    impl_ckks_dft_defaults, impl_ckks_encapsulated_mod_up_default, impl_ckks_encryption_defaults, impl_ckks_imag_defaults,
    impl_ckks_mul_defaults, impl_ckks_neg_defaults, impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults,
    impl_ckks_rotate_defaults, impl_ckks_sub_defaults,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, eval_mod::EvalMod},
};
use poulpy_core::{
    GLWECopy, impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensoring_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned},
};

use crate::hal_impl::delegating_backend::DelegatingFFT64Ref;

type BE = DelegatingFFT64Ref;

// Everything except EvalMod comes from the reference defaults.
impl_glwe_tensoring_default!(BE);
impl_gglwe_product_digits_strided_default!(BE);
impl_glwe_automorphism_defaults_full!(BE);
impl_ggsw_automorphism_defaults_full!(BE);
impl_gglwe_automorphism_defaults_full!(BE);
impl_decryption_defaults_full!(BE);
impl_glwe_trace_defaults_full!(BE);
impl_glwe_packing_defaults_full!(BE);
impl_conversion_defaults_full!(BE);
impl_glwe_keyswitch_defaults_full!(BE);
impl_gglwe_keyswitch_defaults_full!(BE);
impl_ggsw_keyswitch_defaults_full!(BE);
impl_lwe_keyswitch_defaults_full!(BE);
impl_encryption_defaults_full!(BE);
impl_glwe_external_product_defaults_full!(BE);
impl_gglwe_external_product_defaults_full!(BE);
impl_ggsw_external_product_defaults_full!(BE);
impl_linear_transformation_defaults_full!(BE);

impl_ckks_encapsulated_mod_up_default!(BE);
impl_ckks_conjugate_defaults!(BE);
impl_ckks_copy_defaults!(BE);
impl_ckks_encryption_defaults!(BE);
impl_ckks_imag_defaults!(BE);
impl_ckks_mul_defaults!(BE);
impl_ckks_neg_defaults!(BE);
impl_ckks_pow2_defaults!(BE);
impl_ckks_rotate_defaults!(BE);
impl_ckks_add_defaults!(BE);
impl_ckks_sub_defaults!(BE);
impl_ckks_plaintext_defaults!(BE);
impl_ckks_dft_defaults!(BE);
impl_ckks_bootstrap_defaults!(BE);

impl<F> crate::ckks_encoding::CKKSEncodingTransform<F> for BE
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}
crate::impl_ckks_encoding!(BE);

// Deliberately NOT `impl_ckks_eval_mod_defaults!(BE)`: the marker is written by
// hand so the two paired methods can be substituted.

/// Scratch the fused pair claims. Far above every other bootstrap stage at
/// these parameters, so it is visible in the aggregate budget.
const PAIR_SENTINEL_BYTES: usize = 1 << 26;

/// Incremented by the overridden paired evaluation.
static PAIR_CALLS: AtomicUsize = AtomicUsize::new(0);

impl CKKSEvalModOpsDefault<BE> for Module<BE> {
    fn ckks_eval_mod_pair_tmp_bytes_default<R0, R1, C0, C1, P, F, T>(
        &self,
        _res_0: &R0,
        _res_1: &R1,
        _ct_0: &C0,
        _ct_1: &C1,
        _params: &EvalMod<F, P>,
        _tsk: &T,
    ) -> usize
    where
        Self: std::borrow::Borrow<Module<BE>>,
        Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        PAIR_SENTINEL_BYTES
    }

    fn ckks_eval_mod_pair_default<R0, R1, C0, C1, P, F>(
        &self,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<<BE as Backend>::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: std::borrow::Borrow<Module<BE>>,
        Module<BE>: CKKSPolynomialEvaluationOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSPow2Ops<BE>
            + GLWECopy<BE>,
        R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C0: GLWEToBackendRef<BE> + CKKSCtBounds,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        GLWETensorKeyPrepared<<BE as Backend>::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        PAIR_CALLS.fetch_add(1, Ordering::Relaxed);
        // Still correct: a real fused kernel would replace these two calls.
        self.ckks_eval_mod_default(res_0, ct_0, params, tsk, scratch)?;
        self.ckks_eval_mod_default(res_1, ct_1, params, tsk, scratch)
    }
}

/// A `Module` for the override backend at the parameter set the bootstrap
/// suite builds its own modules from; the suite functions ignore the module
/// they are handed and construct their own, so the size here is immaterial.
fn modules() -> (Module<BE>, Module<HostBytesBackend>) {
    (Module::new(64), Module::new(64))
}

/// The public sizing method reaches the backend's paired override rather than
/// the reference default.
#[test]
fn public_pair_sizing_dispatches_to_the_override() {
    let (module, _) = modules();
    let params = poulpy_ckks::test_suite::FFT64_PARAMS_F64;
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 22);
    let plan = poulpy_ckks::layouts::EvalModPlan {
        eval_mod_type: poulpy_ckks::layouts::EvalModType::SinCheby,
        log_msg_ratio: 8,
        f_mod_degree: 31,
        f_mod_interval: 14,
        f_mod_log_interval_reduction: 0,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: poulpy_ckks::polynomial::SplitStrategy::MinDepth,
        coeffs_meta: poulpy_ckks::CoeffsMeta::from_delta_budget(60, params.base2k),
        f_mod_log_delta: 60,
    };
    let compiled =
        poulpy_ckks::layouts::eval_mod::compile_eval_mod::<BE, f64>(params.base2k.into(), plan, &module, &mut scratch.borrow())
            .expect("compile_eval_mod");

    let ct = params.prec();
    assert_eq!(
        module.ckks_eval_mod_pair_tmp_bytes(&ct, &ct, &ct, &ct, &compiled, &params.tsk_layout()),
        PAIR_SENTINEL_BYTES,
        "public pair sizing did not dispatch to the backend override"
    );
}

/// Both bootstrap schedules invoke the paired hook, and both size their scratch
/// off it: the suite's own `ckks_bootstrap_tmp_bytes >= ckks_eval_mod_pair_tmp_bytes`
/// assertion runs against this backend's sentinel, which is larger than every
/// other stage.
///
/// Single test on purpose: `PAIR_CALLS` is process-global, so the two schedules
/// are separated by resetting it rather than by two parallel tests.
#[test]
fn both_bootstrap_pipelines_invoke_the_paired_hook() {
    let (module, host_module) = modules();
    let params = poulpy_ckks::test_suite::FFT64_PARAMS_F64;

    PAIR_CALLS.store(0, Ordering::Relaxed);
    poulpy_ckks::test_suite::bootstrapping::test_bootstrapping_standard_e2e::<BE, f64, crate::FFT64ReimTable<f64>>(
        params,
        &module,
        &host_module,
    );
    assert!(
        PAIR_CALLS.swap(0, Ordering::Relaxed) > 0,
        "the C2S-first pipeline never reached the paired EvalMod hook"
    );

    poulpy_ckks::test_suite::bootstrapping::test_bootstrapping_s2c_first_e2e::<BE, f64, crate::FFT64ReimTable<f64>>(
        params,
        &module,
        &host_module,
    );
    assert!(
        PAIR_CALLS.load(Ordering::Relaxed) > 0,
        "the S2C-first pipeline never reached the paired EvalMod hook"
    );
}
