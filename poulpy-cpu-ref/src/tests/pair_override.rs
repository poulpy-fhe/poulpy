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

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    impl_gglwe_product_bound_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full,
    impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
    layouts::{
        BSGSMeta, GGLWEActiveUse, GGLWEInfos, GLWEInfos, GLWERelinearizationKeyHelper, GLWERelinearizationKeyLayoutHelper,
        GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, SetBSGSMeta,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedBound, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, ScratchArenaTakeBasic, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned},
};

use crate::hal_impl::delegating_backend::DelegatingFFT64Ref;

type BE = DelegatingFFT64Ref;

// Everything except EvalMod, the prepared-right tensor apply and the ordered
// baby batch comes from the reference defaults.
impl_gglwe_product_bound_default!(BE);
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
    fn ckks_eval_mod_pair_tmp_bytes_default<R0, R1, C0, C1, P, F, H>(
        &self,
        _res_0: &R0,
        _res_1: &R1,
        _ct_0: &C0,
        _ct_1: &C1,
        _params: &EvalMod<F, P>,
        _tsk: &H,
    ) -> usize
    where
        Self: std::borrow::Borrow<Module<BE>>,
        Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        PAIR_SENTINEL_BYTES
    }

    fn ckks_eval_mod_pair_default<R0, R1, C0, C1, P, F, H>(
        &self,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &H,
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
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
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
// substituted; the required methods forward to the reference defaults. It
// implements none of the fused `*_relinearize*` composites, so compiling it is
// the source-compatibility proof, and `PREPARED_RIGHT_CALLS` firing shows their
// defaults dispatch through `BE::glwe_tensor_*` rather than the suboperation
// defaults.
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
        SCALAR_TENSOR_CALLS.fetch_add(1, Ordering::Relaxed);
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
        SCALAR_TENSOR_CALLS.fetch_add(1, Ordering::Relaxed);
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
        SCALAR_TENSOR_CALLS.fetch_add(1, Ordering::Relaxed);
        module.glwe_tensor_apply_prepared_right_default(cnv_offset, res, a, b_prep, b_size, scratch);
    }
}

// ---------------------------------------------------------------------------
// Ordered baby batch: `impl_ckks_mul_defaults!` replaced by an explicit
// `CKKSMulDefault`, overriding only the new provided method.
// ---------------------------------------------------------------------------

/// Extra scratch this backend charges every frontier batch, in limbs, so its
/// batch requirement genuinely exceeds the scalar default.
const PAD_LIMBS: usize = 8192;

/// Off by default: the other suites in this file size their arenas from the
/// generic budgets, which do not know about the pad.
static BATCH_PAD: AtomicBool = AtomicBool::new(false);
type FrontierShape = (&'static str, Vec<(u32, usize, u32, usize)>);
static FRONTIERS: Mutex<Vec<FrontierShape>> = Mutex::new(Vec::new());
static SCALAR_TENSOR_CALLS: AtomicUsize = AtomicUsize::new(0);

fn pad_bytes(module: &Module<BE>) -> usize {
    if BATCH_PAD.load(Ordering::Relaxed) {
        <BE as Backend>::bytes_of_vec_znx(module.n(), 1, PAD_LIMBS)
    } else {
        0
    }
}

/// Records `(destination k, destination capacity, left operand k, right operand
/// limbs)` per item, matching what the lockstep scratch query priced.
fn record_frontier(kind: &'static str, shapes: Vec<(u32, usize, u32, usize)>) {
    if BATCH_PAD.load(Ordering::Relaxed) {
        FRONTIERS.lock().unwrap().push((kind, shapes));
    }
}

/// Runs `body` after reserving the pad, so the batch really consumes what
/// [`pad_bytes`] advertises.
fn with_pad<R>(module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>, body: impl FnOnce(&mut ScratchArena<'_, BE>) -> R) -> R {
    if BATCH_PAD.load(Ordering::Relaxed) {
        let arena = scratch.borrow();
        let (_pad, mut arena) = arena.take_vec_znx_scratch(module.n(), 1, PAD_LIMBS);
        body(&mut arena)
    } else {
        body(scratch)
    }
}

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

    // ── Frontier batches ────────────────────────────────────────────────────
    //
    // Each one records its `(kind, length)`, then takes `PAD_LIMBS` of scratch
    // *before* running the reference batch, so this backend genuinely needs
    // more than the scalar default. The matching `*_tmp_bytes` adds the same
    // amount, so a caller that sized from the batch query fits and a caller
    // that priced only the scalar path does not.

    fn ckks_mul_into_batch_default<Dst, A, B>(
        &self,
        items: &mut [poulpy_ckks::api::CKKSMulIntoItem<&mut Dst, &A, &B>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + poulpy_ckks::CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + poulpy_ckks::CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + poulpy_ckks::CKKSInfos + GLWEInfos,
    {
        record_frontier(
            "mul_into",
            items
                .iter()
                .map(|item| (item.dst.k().as_u32(), item.dst.max_size(), item.a.k().as_u32(), item.b.size()))
                .collect(),
        );
        with_pad(self, scratch, |s| {
            poulpy_ckks::default::mul::ckks_mul_into_batch_ordered(self, items, bounds, s)
        })
    }

    fn ckks_mul_into_batch_tmp_bytes_default<Dst, A, B>(
        &self,
        items: &[poulpy_ckks::api::CKKSMulIntoItem<&Dst, &A, &B>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        poulpy_ckks::default::mul::ckks_mul_into_batch_tmp_bytes_ordered(self, items, uses) + pad_bytes(self)
    }

    fn ckks_square_into_batch_default<Dst, A>(
        &self,
        items: &mut [poulpy_ckks::api::CKKSSquareIntoItem<&mut Dst, &A>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + poulpy_ckks::CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + poulpy_ckks::CKKSInfos + GLWEInfos,
    {
        record_frontier(
            "square_into",
            items
                .iter()
                .map(|item| (item.dst.k().as_u32(), item.dst.max_size(), item.a.k().as_u32(), item.a.size()))
                .collect(),
        );
        with_pad(self, scratch, |s| {
            poulpy_ckks::default::mul::ckks_square_into_batch_ordered(self, items, bounds, s)
        })
    }

    fn ckks_square_into_batch_tmp_bytes_default<Dst, A>(
        &self,
        items: &[poulpy_ckks::api::CKKSSquareIntoItem<&Dst, &A>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEInfos,
        A: GLWEInfos,
    {
        poulpy_ckks::default::mul::ckks_square_into_batch_tmp_bytes_ordered(self, items, uses) + pad_bytes(self)
    }

    fn ckks_square_assign_batch_default<Dst>(
        &self,
        items: &mut [poulpy_ckks::api::CKKSSquareAssignItem<&mut Dst>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + poulpy_ckks::CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        record_frontier(
            "square_assign",
            items
                .iter()
                .map(|item| {
                    (
                        item.dst.k().as_u32(),
                        item.dst.max_size(),
                        item.dst.k().as_u32(),
                        item.dst.size(),
                    )
                })
                .collect(),
        );
        with_pad(self, scratch, |s| {
            poulpy_ckks::default::mul::ckks_square_assign_batch_ordered(self, items, bounds, s)
        })
    }

    fn ckks_square_assign_batch_tmp_bytes_default<Dst>(
        &self,
        items: &[poulpy_ckks::api::CKKSSquareAssignItem<&Dst>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEInfos,
    {
        poulpy_ckks::default::mul::ckks_square_assign_batch_tmp_bytes_ordered(self, items, uses) + pad_bytes(self)
    }

    fn ckks_mul_prepared_assign_batch_default<Dst>(
        &self,
        items: &mut [poulpy_ckks::api::CKKSPreparedMulAssignItem<&mut Dst, &poulpy_ckks::layouts::CKKSPreparedRight<BE>>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + poulpy_ckks::CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        record_frontier(
            "prepared_assign",
            items
                .iter()
                .map(|item| {
                    (
                        item.dst.k().as_u32(),
                        item.dst.max_size(),
                        item.dst.k().as_u32(),
                        poulpy_ckks::layouts::CKKSPreparedRightInfos::prepared_size(item.prepared),
                    )
                })
                .collect(),
        );
        with_pad(self, scratch, |s| {
            poulpy_ckks::default::mul::ckks_mul_prepared_assign_batch_ordered(self, items, bounds, s)
        })
    }

    fn ckks_mul_prepared_assign_batch_tmp_bytes_default<Dst, PR>(
        &self,
        items: &[poulpy_ckks::api::CKKSPreparedMulAssignItem<&Dst, &PR>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Self: poulpy_core::GLWETensoring<BE> + Sized,
        Dst: GLWEInfos,
        PR: poulpy_ckks::layouts::CKKSPreparedRightInfos,
    {
        poulpy_ckks::default::mul::ckks_mul_prepared_assign_batch_tmp_bytes_ordered(self, items, uses) + pad_bytes(self)
    }
}

/// The lockstep EvalMod dispatches every tensor product as a priced frontier,
/// and runs inside exactly the bytes its scratch query advertises.
///
/// This backend charges every frontier batch `PAD_LIMBS` of scratch on top of
/// the scalar default and actually consumes it, so a query that priced only the
/// scalar path would hand the driver too small an arena.
#[test]
fn lockstep_batches_every_frontier_inside_the_advertised_scratch() {
    let params = poulpy_ckks::test_suite::FFT64_PARAMS_F64;
    let (module, host_module) = (
        Module::<BE>::new(params.n as u64),
        Module::<HostBytesBackend>::new(params.n as u64),
    );
    // The active bootstrap shape: `CosHKEven` folded through `T2`, three
    // range-extension squares, at the EvalMod digit size.
    let plan = poulpy_ckks::layouts::EvalModPlan {
        eval_mod_type: poulpy_ckks::layouts::EvalModType::CosHKEven,
        log_msg_ratio: 8,
        f_mod_degree: 30,
        f_mod_interval: 16,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: poulpy_ckks::polynomial::SplitStrategy::MinDepth,
        coeffs_meta: poulpy_ckks::CoeffsMeta::from_delta_budget(0, 0),
        f_mod_log_delta: 60,
    };

    let predicted = poulpy_ckks::test_suite::eval_mod::run_eval_mod_pair_case::<BE, f64, crate::FFT64ReimTable<f64>>(
        params,
        &module,
        &host_module,
        "coshk_even_dsize8",
        plan,
        8,
        None,
        // The pad is switched on only for the lockstep leg, so the two singles
        // and the sequential pair that precede it keep their own budgets.
        &|| {
            FRONTIERS.lock().unwrap().clear();
            SCALAR_TENSOR_CALLS.store(0, Ordering::Relaxed);
            BATCH_PAD.store(true, Ordering::Relaxed);
        },
    );
    BATCH_PAD.store(false, Ordering::Relaxed);

    let observed = FRONTIERS.lock().unwrap().clone();
    check(&observed, &predicted);

    // Again with a destination narrower than the evaluated result, so the final
    // copy charges a unary offset and every downstream layout shifts.
    let narrow = poulpy_ckks::test_suite::eval_mod::run_eval_mod_pair_case::<BE, f64, crate::FFT64ReimTable<f64>>(
        params,
        &module,
        &host_module,
        "coshk_even_dsize8_narrow_res",
        plan,
        8,
        Some(300),
        &|| {
            FRONTIERS.lock().unwrap().clear();
            SCALAR_TENSOR_CALLS.store(0, Ordering::Relaxed);
            BATCH_PAD.store(true, Ordering::Relaxed);
        },
    );
    BATCH_PAD.store(false, Ordering::Relaxed);
    check(&FRONTIERS.lock().unwrap().clone(), &narrow);
}

/// The observed frontiers must be exactly the priced ones, must include a `B=2`
/// and a `B=4`, and must account for every tensor product that ran.
fn check(observed: &[FrontierShape], predicted: &[FrontierShape]) {
    assert_eq!(
        observed, predicted,
        "the lockstep issued frontiers, or item layouts, the scratch query did not price"
    );
    assert!(
        observed.iter().any(|(_, items)| items.len() == 2),
        "no B=2 frontier was dispatched: {observed:?}"
    );
    assert!(
        observed.iter().any(|(_, items)| items.len() == 4),
        "no B=4 frontier was dispatched: {observed:?}"
    );
    let batched: usize = observed.iter().map(|(_, items)| items.len()).sum();
    assert_eq!(
        SCALAR_TENSOR_CALLS.load(Ordering::Relaxed),
        batched,
        "a tensor product ran outside a batch frontier (scalar fallback)"
    );
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
