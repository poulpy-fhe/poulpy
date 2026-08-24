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
    impl_ckks_neg_defaults, impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults, impl_ckks_rotate_defaults,
    impl_ckks_sub_defaults,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, eval_mod::EvalMod},
};
use poulpy_core::{
    GLWECopy,
    default::operations::GLWETensoringDefault,
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full,
    impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned},
};

use crate::hal_impl::delegating_backend::DelegatingFFT64Ref;

type BE = DelegatingFFT64Ref;

// Everything except EvalMod, the prepared-right tensor apply and the ordered
// baby batch comes from the reference defaults.
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

// ---------------------------------------------------------------------------
// Prepared-right tensor apply: hand-written `GLWETensoringImpl`, exactly what a
// device backend writes. Only `glwe_tensor_apply_prepared_right` is
// substituted; the required methods forward to the reference defaults.
// ---------------------------------------------------------------------------

static PREPARED_RIGHT_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl poulpy_core::oep::GLWETensoringImpl<BE> for BE {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        module.glwe_tensor_apply_tmp_bytes_default(res, a, b)
    }

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        module.glwe_tensor_square_apply_tmp_bytes_default(res, a)
    }

    fn glwe_tensor_apply<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_tensor_apply_default(cnv_offset, res, a, b, scratch)
    }

    fn glwe_tensor_square_apply<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_tensor_square_apply_default(cnv_offset, res, a, scratch)
    }

    fn glwe_tensor_relinearize<R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<BE>,
    {
        module.glwe_tensor_relinearize_default(res, a, tsk, scratch)
    }

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        module.glwe_tensor_relinearize_tmp_bytes_default(res, a, tsk)
    }

    fn glwe_tensor_apply_prepared_right<R, A, BP>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b_prep: &BP,
        b_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: GLWETensoringDefault<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        BP: poulpy_hal::layouts::CnvPVecRToBackendRef<BE>,
    {
        PREPARED_RIGHT_CALLS.fetch_add(1, Ordering::Relaxed);
        module.glwe_tensor_apply_prepared_right_default(cnv_offset, res, a, b_prep, b_size, scratch);
    }
}

// ---------------------------------------------------------------------------
// Ordered baby batch: `impl_ckks_mul_defaults!` replaced by an explicit
// `CKKSMulDefault`, overriding only the new provided method.
// ---------------------------------------------------------------------------

static BABY_BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static BABY_BATCH_TERMS: AtomicUsize = AtomicUsize::new(0);

impl poulpy_ckks::default::mul::CKKSMulDefault<BE> for Module<BE> {
    fn ckks_mul_add_pt_consts_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        terms: &[(&A, usize)],
        plans: &[poulpy_ckks::default::mul::CKKSMulAddPtConstPlan],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: poulpy_ckks::api::CKKSMulOps<BE> + poulpy_ckks::api::CKKSAddOps<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + poulpy_core::layouts::IntPolyInfos + CKKSCtBounds,
    {
        BABY_BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        BABY_BATCH_TERMS.fetch_add(terms.len(), Ordering::Relaxed);
        poulpy_ckks::default::mul::ckks_mul_add_pt_consts_into_ordered(self, dst, terms, plans, coeffs, scratch)
    }
}

/// Both new seams dispatch to a backend override: the BSGS baby loop reaches the
/// ordered batch hook, and the giant-step multiply reaches prepared-right tensor
/// apply instead of the free helper.
#[test]
fn new_seams_reach_the_backend_overrides() {
    // The polynomial-evaluation suite evaluates on the module it is handed.
    let params = poulpy_ckks::test_suite::FFT64_PARAMS_F64;
    let (module, host_module) = (
        Module::<BE>::new(params.n as u64),
        Module::<HostBytesBackend>::new(params.n as u64),
    );
    BABY_BATCH_CALLS.store(0, Ordering::Relaxed);
    BABY_BATCH_TERMS.store(0, Ordering::Relaxed);
    PREPARED_RIGHT_CALLS.store(0, Ordering::Relaxed);

    poulpy_ckks::test_suite::polynomial_evaluation::test_eval_poly_const_coeffs_chebyshev_degree31::<
        BE,
        f64,
        crate::FFT64ReimTable<f64>,
    >(params, &module, &host_module);

    assert!(
        BABY_BATCH_CALLS.load(Ordering::Relaxed) > 0,
        "the BSGS baby loop never reached ckks_mul_add_pt_consts_into"
    );
    assert!(
        BABY_BATCH_TERMS.load(Ordering::Relaxed) > BABY_BATCH_CALLS.load(Ordering::Relaxed),
        "at least one baby step should have batched several terms"
    );
    assert!(
        PREPARED_RIGHT_CALLS.load(Ordering::Relaxed) > 0,
        "the giant-step multiply never reached glwe_tensor_apply_prepared_right"
    );
}
