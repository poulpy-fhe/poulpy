//! Exact scratch budgets.
//!
//! Every operation runs on exactly its query's bytes, in a poisoned arena whose guard bytes must stay intact, with `res`
//! at the width of its input, one limb and a bit narrower, and one limb wider. Each exact run must give the same digits
//! and metadata as the same call on a generous arena. The operands are random fixtures: scratch use does not depend on
//! the decrypted values, so no encryption is needed.

use anyhow::Result;
use poulpy_core::{
    GLWEFill,
    layouts::{GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared, LWEInfos},
};
use poulpy_hal::{
    AlignedBuf,
    api::{ReimNew, ScratchOwnedAlloc},
    layouts::{HostBytesBackend, Module, ScratchArena, ScratchOwned, Standard},
};

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSCopyOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, ckks_spec, gen_atk, gen_sk_with_raw, gen_tsk, upload_pt},
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
    },
};

use crate::{
    arithmetic::CKKSArithmeticOps,
    comparison::{CKKSComparisonOps, CKKSPredicateOps, CKKSSignOps, COEFFS_SIGN_X4_CHEBYSHEV, SignComposite},
    exp::{CKKSExpOps, Exp2Plan, Exp10Plan, ExpOptions, ExpPlan, Expm1Plan},
    log::{CKKSLogOps, LogOptions, LogPlan},
    pow::{CKKSPowOps, PowOptions, PowPlan},
    reduce::CKKSReductionOps,
    roots::{CKKSDivSqrtOps, CKKSInverseDomainOps, CKKSInverseOps, CKKSRootOps, CbrtPlan, HypotPlan, IntervalNorm, RootOptions},
    softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan},
    trig::{Atan2Options, Atan2Plan, CKKSAtan2Ops, CKKSTrigOps, CosPlan, SinPlan, TrigOptions},
};

use super::{helpers::params_for, reduce::rotation_keys};

type Pt<BE> = CKKSPlaintextOwned<BE>;
type HostPt = CKKSPlaintext<AlignedBuf, i64, Standard>;

/// Plans, keys and fixtures shared by the exact-budget tests.
pub(super) struct Setup<BE: TestContextBackend<Ring = Standard>> {
    pub(super) params: CKKSTestParams,
    pub(super) tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    pub(super) conj_key: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>,
    pub(super) rotation_keys: std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    pub(super) composite: SignComposite<f64, Pt<BE>>,
    pub(super) bounds: Pt<BE>,
    pub(super) coeff_meta: CKKSLayout,
}

impl<BE: TestContextBackend<Ring = Standard>> Setup<BE>
where
    Module<BE>: TestContextModule<BE> + CKKSReductionOps<BE> + GLWEFill<BE>,
{
    pub(super) fn new(
        base: CKKSTestParams,
        consumed: usize,
        module: &Module<BE>,
        host_module: &Module<HostBytesBackend>,
    ) -> Self {
        let log_delta = base.prec().log_delta();
        let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
        let composite = SignComposite::<f64, _>::from_coeffs(
            base.base2k.into(),
            &[COEFFS_SIGN_X4_CHEBYSHEV],
            coeff_meta,
            poulpy_ckks::polynomial::SplitStrategy::MinDepth,
            host_module,
        )
        .expect("SignComposite::from_coeffs");
        let params = params_for(&base, consumed + composite.consumed_bits(log_delta));
        let mut bounds = host_module.ckks_pt_coeffs_alloc(2, base.base2k.into(), coeff_meta.k());
        bounds.set_meta(coeff_meta.meta());
        bounds.encode_host_floats(&[-0.5f64, 0.5]).expect("bounds");
        let (sk_raw, _) = gen_sk_with_raw(&params, module, host_module, [0x5Bu8; 32]);
        let key_bytes = module.ckks_all_ops_tmp_bytes(&params.prec(), &coeff_meta, &params.tsk_layout());
        let mut scratch = ScratchOwned::<BE>::alloc(key_bytes);
        let tsk = gen_tsk(
            &params,
            module,
            &sk_raw,
            &mut poulpy_hal::api::ScratchOwnedBorrow::borrow(&mut scratch),
        );
        let conj_key = gen_atk(
            &params,
            module,
            -1,
            &sk_raw,
            &mut poulpy_hal::api::ScratchOwnedBorrow::borrow(&mut scratch),
        );
        let rotation_keys = rotation_keys(&params, module, &sk_raw, &mut scratch);
        Self {
            params,
            tsk,
            conj_key,
            rotation_keys,
            composite: composite.map_plaintexts(|pt| upload_pt(module, pt)),
            bounds: upload_pt(module, &bounds),
            coeff_meta,
        }
    }

    pub(super) fn layout(&self, k: usize) -> CKKSLayout {
        let mut layout = self.params.prec();
        layout.glwe_layout.k = k.into();
        layout
    }

    pub(super) fn fixture(&self, module: &Module<BE>, seed: u8) -> CKKSCiphertextOwned<BE> {
        fixture_ciphertext(module, &self.params.prec(), seed)
    }

    /// Output widths: the input's, one limb and a bit narrower, one limb wider.
    pub(super) fn widths(&self) -> [usize; 3] {
        let (k, base2k) = (self.params.k, self.params.base2k);
        [k, k - base2k - 1, k + base2k]
    }
}

/// Runs `run` on a fresh `res` of `layout`, first on exactly `query(&res)` bytes, then on a generous arena, and
/// compares the two outputs.
fn check_exact<BE>(
    label: &str,
    module: &Module<BE>,
    layout: &CKKSLayout,
    query: impl Fn(&CKKSCiphertextOwned<BE>) -> usize,
    run: impl Fn(&mut CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
{
    let mut exact = module.ckks_ciphertext_alloc_from_infos(layout);
    let bytes = query(&exact);
    with_scratch::<BE, _>(bytes, |scratch| run(&mut exact, scratch)).unwrap_or_else(|e| panic!("{label}: {e:#}"));
    let mut reference = module.ckks_ciphertext_alloc_from_infos(layout);
    with_scratch::<BE, _>(2 * bytes + (1 << 16), |scratch| run(&mut reference, scratch))
        .unwrap_or_else(|e| panic!("{label} (generous arena): {e:#}"));
    assert_eq!(
        snapshot::<BE, _>(&exact),
        snapshot::<BE, _>(&reference),
        "{label}: the exact-budget run differs"
    );
}

/// Runs an in-place `run` on a fresh copy of `fixture` at `layout`, as [`check_exact`] does.
fn check_exact_in_place<BE>(
    label: &str,
    module: &Module<BE>,
    make: impl Fn() -> CKKSCiphertextOwned<BE>,
    query: impl Fn(&CKKSCiphertextOwned<BE>) -> usize,
    run: impl Fn(&mut CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
{
    let _ = module;
    let mut exact = make();
    let bytes = query(&exact);
    with_scratch::<BE, _>(bytes, |scratch| run(&mut exact, scratch)).unwrap_or_else(|e| panic!("{label}: {e:#}"));
    let mut reference = make();
    with_scratch::<BE, _>(2 * bytes + (1 << 16), |scratch| run(&mut reference, scratch))
        .unwrap_or_else(|e| panic!("{label} (generous arena): {e:#}"));
    assert_eq!(
        snapshot::<BE, _>(&exact),
        snapshot::<BE, _>(&reference),
        "{label}: the exact-budget run differs"
    );
}

/// Exact budgets of the plan-driven operations: the `exp` family, a unary approximation, `cbrt`, `sincos`, `pow`,
/// `powi`, `hypot`, `fma` and `scalbn`.
pub fn test_exact_scratch_plans<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + GLWEFill<BE>
        + CKKSArithmeticOps<BE>
        + CKKSExpOps<BE>
        + CKKSLogOps<BE>
        + CKKSPowOps<BE>
        + CKKSReductionOps<BE>
        + CKKSRootOps<BE>
        + CKKSTrigOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let b2k = base.base2k.into();
    let exp = ExpPlan::from_precision(0.0f64, 3.0, b2k, coeff_meta, ExpOptions::default(), host_module).unwrap();
    let exp2 = Exp2Plan::from_precision(0.0f64, 3.0, b2k, coeff_meta, ExpOptions::default(), host_module).unwrap();
    let exp10 = Exp10Plan::from_precision(0.0f64, 1.0, b2k, coeff_meta, ExpOptions::default(), host_module).unwrap();
    let expm1 = Expm1Plan::from_precision(0.0f64, 1.0, b2k, coeff_meta, ExpOptions::default(), host_module).unwrap();
    let log = LogPlan::from_precision(0.5f64, 2.0, b2k, coeff_meta, LogOptions::default(), host_module).unwrap();
    let cbrt = CbrtPlan::from_precision(0.5f64, 2.0, b2k, coeff_meta, RootOptions::default(), host_module).unwrap();
    let sin = SinPlan::from_precision(-1.0f64, 1.0, b2k, coeff_meta, TrigOptions::default(), host_module).unwrap();
    let cos = CosPlan::from_precision(-1.0f64, 1.0, b2k, coeff_meta, TrigOptions::default(), host_module).unwrap();
    let pow = PowPlan::from_precision(0.5f64, 1.5, -1.0, 1.0, b2k, coeff_meta, PowOptions::default(), host_module).unwrap();
    let hypot = HypotPlan::from_precision(0.5f64, 1.0, 0.5, 1.0, b2k, coeff_meta, RootOptions::default(), host_module).unwrap();
    let consumed = [
        exp.consumed_bits(log_delta),
        exp2.consumed_bits(log_delta),
        expm1.consumed_bits(log_delta),
        pow.consumed_bits(log_delta, log_delta),
        hypot.consumed_bits(log_delta),
        5 * log_delta,
    ]
    .into_iter()
    .max()
    .unwrap();
    let setup = Setup::new(base, consumed, module, host_module);
    let up = |pt: &HostPt| upload_pt(module, pt);
    let (exp, exp2, exp10, expm1) = (
        exp.map_plaintexts(up),
        exp2.map_plaintexts(up),
        exp10.map_plaintexts(up),
        expm1.map_plaintexts(up),
    );
    let (log, cbrt, sin, cos) = (
        log.map_plaintexts(up),
        cbrt.map_plaintexts(up),
        sin.map_plaintexts(up),
        cos.map_plaintexts(up),
    );
    let (pow, hypot) = (pow.map_plaintexts(up), hypot.map_plaintexts(up));
    let (a, b, c) = (setup.fixture(module, 1), setup.fixture(module, 2), setup.fixture(module, 3));
    let tsk = &setup.tsk;
    let tl = setup.params.tsk_layout();

    for k in setup.widths() {
        let layout = setup.layout(k);
        let l = |op: &str| format!("{op} at res.k = {k}");
        check_exact(
            &l("ckks_exp"),
            module,
            &layout,
            |r| module.ckks_exp_tmp_bytes(r, &a, &tl, &exp),
            |r, s| module.ckks_exp(r, &a, &exp, tsk, s),
        );
        check_exact(
            &l("ckks_exp2"),
            module,
            &layout,
            |r| module.ckks_exp2_tmp_bytes(r, &a, &tl, &exp2),
            |r, s| module.ckks_exp2(r, &a, &exp2, tsk, s),
        );
        check_exact(
            &l("ckks_exp10"),
            module,
            &layout,
            |r| module.ckks_exp10_tmp_bytes(r, &a, &tl, &exp10),
            |r, s| module.ckks_exp10(r, &a, &exp10, tsk, s),
        );
        check_exact(
            &l("ckks_expm1"),
            module,
            &layout,
            |r| module.ckks_expm1_tmp_bytes(r, &a, &tl, &expm1),
            |r, s| module.ckks_expm1(r, &a, &expm1, tsk, s),
        );
        check_exact(
            &l("ckks_log"),
            module,
            &layout,
            |r| module.ckks_log_tmp_bytes(r, &a, &tl, &log),
            |r, s| module.ckks_log(r, &a, &log, tsk, s),
        );
        check_exact(
            &l("ckks_cbrt"),
            module,
            &layout,
            |r| module.ckks_cbrt_tmp_bytes(r, &a, &tl, &cbrt),
            |r, s| module.ckks_cbrt(r, &a, &cbrt, tsk, s),
        );
        check_exact(
            &l("ckks_sincos"),
            module,
            &layout,
            |r| module.ckks_sincos_tmp_bytes(r, r, &a, &tl, &sin, &cos),
            |r, s| {
                let mut cos_out = module.ckks_ciphertext_alloc_from_infos(&layout);
                module.ckks_sincos(r, &mut cos_out, &a, &sin, &cos, tsk, s)
            },
        );
        check_exact(
            &l("ckks_pow"),
            module,
            &layout,
            |r| module.ckks_pow_tmp_bytes(r, &a, &b, &tl, &pow),
            |r, s| module.ckks_pow(r, &a, &b, &pow, tsk, s),
        );
        check_exact(
            &l("ckks_powi"),
            module,
            &layout,
            |r| module.ckks_powi_tmp_bytes(r, &a, &tl),
            |r, s| module.ckks_powi(5, r, &a, tsk, s),
        );
        check_exact(
            &l("ckks_hypot"),
            module,
            &layout,
            |r| module.ckks_hypot_tmp_bytes(r, &a, &b, &tl, &hypot),
            |r, s| module.ckks_hypot(r, &a, &b, &hypot, tsk, s),
        );
        check_exact(
            &l("ckks_fma"),
            module,
            &layout,
            |r| module.ckks_fma_tmp_bytes(r, &a, &b, &c, &tl),
            |r, s| module.ckks_fma(r, &a, &b, &c, tsk, s),
        );
        check_exact(
            &l("ckks_scalbn"),
            module,
            &layout,
            |r| module.ckks_scalbn_tmp_bytes(r),
            |r, s| module.ckks_scalbn(-3, r, &a, s),
        );
    }
}

/// Exact budgets of the sign-based operations: `sign`, `step`, the comparisons, the predicates, `select`, `atan2`, the
/// slot reductions and `softmax`.
pub fn test_exact_scratch_sign<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + GLWEFill<BE>
        + CKKSSignOps<BE>
        + CKKSComparisonOps<BE>
        + CKKSPredicateOps<BE>
        + CKKSAtan2Ops<BE>
        + CKKSReductionOps<BE>
        + CKKSSoftmaxOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let b2k = base.base2k.into();
    let atan2_options = Atan2Options {
        input_bound: 4.0,
        ..Atan2Options::default()
    };
    let atan2 = Atan2Plan::<f64, _>::from_precision(0.9, b2k, coeff_meta, atan2_options, host_module).unwrap();
    let softmax = SoftmaxPlan::from_precision(-1.0f64, 1.0, b2k, coeff_meta, SoftmaxOptions::default(), host_module).unwrap();
    // Two signs and a product per step, then the atan2 and softmax stages.
    let consumed = (atan2.consumed_bits(log_delta) + 3 * log_delta).max(softmax.consumed_bits(log_delta) + 3 * log_delta);
    let setup = Setup::new(base, consumed, module, host_module);
    let atan2 = atan2.map_plaintexts(|pt| upload_pt(module, pt));
    let softmax = softmax.map_plaintexts(|pt| upload_pt(module, pt));
    let (a, b, m) = (setup.fixture(module, 4), setup.fixture(module, 5), setup.fixture(module, 6));
    let (tsk, conj, rot, comp, bounds) = (
        &setup.tsk,
        &setup.conj_key,
        &setup.rotation_keys,
        &setup.composite,
        &setup.bounds,
    );
    let tl = setup.params.tsk_layout();
    let al = setup.params.atk_layout();
    let cm = setup.coeff_meta;

    for k in setup.widths() {
        let layout = setup.layout(k);
        let l = |op: &str| format!("{op} at res.k = {k}");
        let sign_q = |r: &CKKSCiphertextOwned<BE>| module.ckks_sign_tmp_bytes(r, &a, &tl, &al, &cm);
        check_exact(&l("ckks_sign"), module, &layout, sign_q, |r, s| {
            module.ckks_sign(r, &a, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_step"), module, &layout, sign_q, |r, s| {
            module.ckks_step(r, &a, comp, tsk, conj, s)
        });
        let cmp_q = |r: &CKKSCiphertextOwned<BE>| module.ckks_comparison_tmp_bytes(r, &a, &b, &tl, &al, &cm);
        check_exact(&l("ckks_fmax"), module, &layout, cmp_q, |r, s| {
            module.ckks_fmax(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_fmin"), module, &layout, cmp_q, |r, s| {
            module.ckks_fmin(r, &a, &b, comp, tsk, conj, s)
        });
        let cst_q = |r: &CKKSCiphertextOwned<BE>| module.ckks_comparison_const_tmp_bytes(r, &a, &tl, &al, &cm);
        check_exact(&l("ckks_fmax_const"), module, &layout, cst_q, |r, s| {
            module.ckks_fmax_const(r, &a, bounds, 0, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_fmin_const"), module, &layout, cst_q, |r, s| {
            module.ckks_fmin_const(r, &a, bounds, 1, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_clamp"), module, &layout, cst_q, |r, s| {
            module.ckks_clamp(r, &a, bounds, 0, 1, comp, tsk, conj, s)
        });
        let pred_q = |r: &CKKSCiphertextOwned<BE>| module.ckks_predicate_tmp_bytes(r, &a, &b, &tl, &al, &cm);
        check_exact(&l("ckks_fabs"), module, &layout, pred_q, |r, s| {
            module.ckks_fabs(r, &a, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_cmp"), module, &layout, pred_q, |r, s| {
            module.ckks_cmp(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_gt"), module, &layout, pred_q, |r, s| {
            module.ckks_gt(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_lt"), module, &layout, pred_q, |r, s| {
            module.ckks_lt(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_indicator"), module, &layout, pred_q, |r, s| {
            module.ckks_indicator(r, &a, bounds, 0, 1, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_eq"), module, &layout, pred_q, |r, s| {
            module.ckks_eq(r, &a, &b, bounds, 1, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_fdim"), module, &layout, pred_q, |r, s| {
            module.ckks_fdim(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(&l("ckks_copysign"), module, &layout, pred_q, |r, s| {
            module.ckks_copysign(r, &a, &b, comp, tsk, conj, s)
        });
        check_exact(
            &l("ckks_select"),
            module,
            &layout,
            |r| module.ckks_select_tmp_bytes(r, &m, &a, &b, &tl),
            |r, s| module.ckks_select(r, &m, &a, &b, tsk, s),
        );
        check_exact(
            &l("ckks_atan2"),
            module,
            &layout,
            |r| module.ckks_atan2_tmp_bytes(r, &a, &b, &tl, &al, &atan2),
            |r, s| module.ckks_atan2(r, &a, &b, &atan2, tsk, conj, s),
        );
        check_exact(
            &l("ckks_sum_slots"),
            module,
            &layout,
            |r| module.ckks_sum_slots_tmp_bytes(r, &a, &al),
            |r, s| module.ckks_sum_slots(r, &a, rot, s),
        );
        let ext_q = |r: &CKKSCiphertextOwned<BE>| module.ckks_fmax_slots_tmp_bytes(r, &a, &tl, &al, &al, &cm);
        check_exact(&l("ckks_fmax_slots"), module, &layout, ext_q, |r, s| {
            module.ckks_fmax_slots(r, &a, comp, tsk, rot, conj, s)
        });
        check_exact(&l("ckks_fmin_slots"), module, &layout, ext_q, |r, s| {
            module.ckks_fmin_slots(r, &a, comp, tsk, rot, conj, s)
        });
        check_exact(
            &l("ckks_softmax"),
            module,
            &layout,
            |r| module.ckks_softmax_tmp_bytes(r, &a, &tl, &al, &softmax),
            |r, s| module.ckks_softmax(r, &a, &softmax, tsk, rot, s),
        );
    }
}

/// Exact budgets of the iterative kernels: Goldschmidt division, `rsqrt`, interval normalization, the inverse domains,
/// `div` and `sqrt`. The in-place forms run at the input's width and one limb and a bit narrower.
pub fn test_exact_scratch_iterative<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + GLWEFill<BE>
        + CKKSInverseOps<BE>
        + CKKSInverseDomainOps<BE>
        + CKKSDivSqrtOps<BE>
        + CKKSReductionOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let norm = IntervalNorm::from_max::<f64>(4.0, base.base2k.into(), coeff_meta, host_module).unwrap();
    let (iters, r) = (2, 2);
    let consumed = (norm.consumed_bits(log_delta))
        .max((iters + 3) * log_delta)
        .max(2 * r * log_delta + log_delta);
    let setup = Setup::new(base, consumed, module, host_module);
    let norm = norm.map_plaintexts(|pt| upload_pt(module, pt));
    let (a, b) = (setup.fixture(module, 7), setup.fixture(module, 8));
    let (tsk, conj, comp) = (&setup.tsk, &setup.conj_key, &setup.composite);
    let tl = setup.params.tsk_layout();
    let al = setup.params.atk_layout();
    let cm = setup.coeff_meta;

    for k in setup.widths() {
        let layout = setup.layout(k);
        let l = |op: &str| format!("{op} at res.k = {k}");
        check_exact(
            &l("ckks_div"),
            module,
            &layout,
            |r| module.ckks_div_tmp_bytes(r, &a, &b, &tl),
            |r, s| module.ckks_div(r, &a, &b, iters, tsk, s),
        );
        check_exact(
            &l("ckks_sqrt"),
            module,
            &layout,
            |r| module.ckks_sqrt_tmp_bytes(r, &a, &b, &tl),
            |r, s| module.ckks_sqrt(r, &a, &b, r_steps(r), tsk, s),
        );
        check_exact(
            &l("ckks_interval_normalization"),
            module,
            &layout,
            |r| module.ckks_interval_normalization_tmp_bytes(r, &a, &tl, &norm),
            |r, s| {
                let mut ct = a.clone();
                module.ckks_interval_normalization(r, &mut ct, &norm, tsk, s)
            },
        );
    }

    for k in [setup.params.k, setup.params.k - setup.params.base2k - 1] {
        let layout = setup.layout(k);
        let make = || {
            let mut ct = module.ckks_ciphertext_alloc_from_infos(&layout);
            let mut s = ScratchOwned::<BE>::alloc(module.ckks_copy_tmp_bytes(&ct, &a));
            module
                .ckks_copy(&mut ct, &a, &mut poulpy_hal::api::ScratchOwnedBorrow::borrow(&mut s))
                .unwrap();
            ct
        };
        let l = |op: &str| format!("{op} at ct.k = {k}");
        check_exact_in_place(
            &l("ckks_goldschmidt_division"),
            module,
            make,
            |c| module.ckks_goldschmidt_division_tmp_bytes(c, &tl),
            |c, s| module.ckks_goldschmidt_division(c, iters, tsk, s),
        );
        check_exact_in_place(
            &l("ckks_rsqrt"),
            module,
            make,
            |c| module.ckks_rsqrt_tmp_bytes(c, &b, &tl),
            |c, s| module.ckks_rsqrt(c, &b, r, tsk, s),
        );
        let dom_q = |c: &CKKSCiphertextOwned<BE>| module.ckks_inverse_domain_tmp_bytes(c, &tl, &al, &cm);
        check_exact_in_place(&l("ckks_inverse_positive_domain"), module, make, dom_q, |c, s| {
            module.ckks_inverse_positive_domain(c, iters, tsk, s)
        });
        check_exact_in_place(&l("ckks_inverse_negative_domain"), module, make, dom_q, |c, s| {
            module.ckks_inverse_negative_domain(c, iters, tsk, s)
        });
        check_exact_in_place(&l("ckks_inverse_full_domain"), module, make, dom_q, |c, s| {
            module.ckks_inverse_full_domain(c, iters, comp, tsk, conj, s)
        });
    }
}

fn r_steps<T>(_: &T) -> usize {
    2
}
