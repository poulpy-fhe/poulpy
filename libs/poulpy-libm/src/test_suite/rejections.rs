//! Evaluation-time rejections.
//!
//! Each case breaks one precondition of an operation (budget, iteration count, sparse packing, plan slot count) and
//! checks the error message and that `res` and every `&mut` operand keep their digits and metadata.

use poulpy_core::GLWEFill;
use poulpy_hal::{
    api::ReimNew,
    layouts::{HostBytesBackend, Module, Standard},
};

use poulpy_ckks::{
    CKKSError, CKKSInfos, SetGLWEPlaintextInfos,
    layouts::CKKSCiphertextOwned,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, ckks_spec, upload_pt},
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
    },
};

use crate::{
    comparison::CKKSSignOps,
    exp::{CKKSExpOps, Exp2Plan, Exp10Plan, ExpOptions, ExpPlan, Expm1Plan},
    log::{CKKSLogOps, LogOptions, LogPlan},
    pow::CKKSPowOps,
    reduce::CKKSReductionOps,
    roots::{CKKSInverseDomainOps, CKKSInverseOps, IntervalNorm},
    softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan},
};

use super::budgets::Setup;

/// A generous arena: rejections fire before any carve.
const ARENA: usize = 1 << 24;

/// Asserts that `run` fails with an error whose chain contains `expected` and leaves `targets` unchanged.
fn assert_rejects<T>(
    label: &str,
    expected: &str,
    targets: &mut T,
    snap: impl Fn(&T) -> Vec<poulpy_ckks::test_suite::parity::helpers::Snapshot>,
    run: impl FnOnce(&mut T) -> anyhow::Result<()>,
) -> anyhow::Error {
    let before = snap(targets);
    let err = match run(targets) {
        Ok(()) => panic!("{label}: expected an error containing `{expected}`"),
        Err(err) => err,
    };
    let message = format!("{err:#}");
    assert!(
        message.contains(expected),
        "{label}: error `{message}` does not contain `{expected}`"
    );
    assert!(before == snap(targets), "{label}: a refused call changed its outputs");
    err
}

/// Every evaluation-time rejection of libm, with `res` and `&mut` operands checked unchanged.
pub fn test_rejections<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + GLWEFill<BE>
        + CKKSExpOps<BE>
        + CKKSLogOps<BE>
        + CKKSPowOps<BE>
        + CKKSSignOps<BE>
        + CKKSInverseOps<BE>
        + CKKSInverseDomainOps<BE>
        + CKKSReductionOps<BE>
        + CKKSSoftmaxOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let b2k = base.base2k.into();
    let opts = ExpOptions::default();
    let exp = ExpPlan::from_precision(0.0f64, 3.0, b2k, coeff_meta, opts, host_module).unwrap();
    let exp2 = Exp2Plan::from_precision(0.0f64, 3.0, b2k, coeff_meta, opts, host_module).unwrap();
    let exp10 = Exp10Plan::from_precision(0.0f64, 1.0, b2k, coeff_meta, opts, host_module).unwrap();
    let expm1 = Expm1Plan::from_precision(0.0f64, 1.0, b2k, coeff_meta, opts, host_module).unwrap();
    let log = LogPlan::from_precision(0.5f64, 2.0, b2k, coeff_meta, LogOptions::default(), host_module).unwrap();
    let softmax = SoftmaxPlan::from_precision(-1.0f64, 1.0, b2k, coeff_meta, SoftmaxOptions::default(), host_module).unwrap();
    // A plan for twice as many slots as the module holds.
    let wide_meta = ckks_spec(2 * base.n, base.base2k, log_delta, base.base2k);
    let wide_softmax = SoftmaxPlan::from_precision(-1.0f64, 1.0, b2k, wide_meta, SoftmaxOptions::default(), host_module).unwrap();
    let norm = IntervalNorm::from_max::<f64>(4.0, b2k, coeff_meta, host_module).unwrap();
    let setup = Setup::new(base, softmax.consumed_bits(log_delta), module, host_module);
    let up = |pt: &_| upload_pt(module, pt);
    let (exp, exp2, exp10, expm1) = (
        exp.map_plaintexts(up),
        exp2.map_plaintexts(up),
        exp10.map_plaintexts(up),
        expm1.map_plaintexts(up),
    );
    let (log, softmax, wide_softmax, norm) = (
        log.map_plaintexts(up),
        softmax.map_plaintexts(up),
        wide_softmax.map_plaintexts(up),
        norm.map_plaintexts(up),
    );
    let (tsk, conj, rot, comp) = (&setup.tsk, &setup.conj_key, &setup.rotation_keys, &setup.composite);

    // An input with one limb of budget, and `res` holding other content.
    let low = fixture_ciphertext(module, &setup.layout(log_delta + setup.params.base2k), 9);
    let full = setup.fixture(module, 10);
    let mut sparse = setup.fixture(module, 11);
    sparse.set_log_sparsity(1);
    let fresh_res = || setup.fixture(module, 12);
    let one = |c: &CKKSCiphertextOwned<BE>| vec![snapshot::<BE, _>(c)];
    let two = |c: &(CKKSCiphertextOwned<BE>, CKKSCiphertextOwned<BE>)| vec![snapshot::<BE, _>(&c.0), snapshot::<BE, _>(&c.1)];

    macro_rules! rejects {
        ($label:expr, $expected:expr, |$res:ident, $s:ident| $call:expr) => {{
            let mut target = fresh_res();
            assert_rejects($label, $expected, &mut target, one, |$res| {
                with_scratch::<BE, _>(ARENA, |$s| $call)
            })
        }};
    }

    rejects!("ckks_exp budget", "ckks_exp: log_budget", |r, s| module
        .ckks_exp(r, &low, &exp, tsk, s));
    rejects!("ckks_exp2 budget", "ckks_exp2: log_budget", |r, s| module
        .ckks_exp2(r, &low, &exp2, tsk, s));
    rejects!("ckks_exp10 budget", "ckks_exp10: log_budget", |r, s| module
        .ckks_exp10(r, &low, &exp10, tsk, s));
    rejects!("ckks_expm1 budget", "ckks_expm1: log_budget", |r, s| module
        .ckks_expm1(r, &low, &expm1, tsk, s));
    rejects!("ckks_powi budget", "ckks_powi: log_budget", |r, s| module
        .ckks_powi(5, r, &low, tsk, s));
    rejects!("ckks_sign budget", "ckks_sign: log_budget", |r, s| module
        .ckks_sign(r, &low, comp, tsk, conj, s));
    rejects!("ckks_step budget", "ckks_step: log_budget", |r, s| module
        .ckks_step(r, &low, comp, tsk, conj, s));
    rejects!(
        "ckks_sum_slots sparse",
        "ckks_sum_slots: sparse packing is not supported",
        |r, s| module.ckks_sum_slots(r, &sparse, rot, s)
    );
    rejects!(
        "ckks_fmax_slots sparse",
        "ckks_fmax_slots: sparse packing is not supported",
        |r, s| module.ckks_fmax_slots(r, &sparse, comp, tsk, rot, conj, s)
    );
    rejects!(
        "ckks_fmin_slots sparse",
        "ckks_fmin_slots: sparse packing is not supported",
        |r, s| module.ckks_fmin_slots(r, &sparse, comp, tsk, rot, conj, s)
    );
    rejects!(
        "ckks_softmax sparse",
        "ckks_softmax: sparse packing is not supported",
        |r, s| module.ckks_softmax(r, &sparse, &softmax, tsk, rot, s)
    );
    rejects!("ckks_softmax slots", "ckks_softmax: plan has", |r, s| module.ckks_softmax(
        r,
        &full,
        &wide_softmax,
        tsk,
        rot,
        s
    ));

    // An inner poulpy-ckks refusal keeps its type through the libm context.
    let err = rejects!(
        "ckks_log budget",
        "ckks_polynomial_approximation_eval cannot consume",
        |r, s| module.ckks_log(r, &low, &log, tsk, s)
    );
    assert!(
        err.downcast_ref::<CKKSError>().is_some(),
        "ckks_log: the poulpy-ckks error lost its type"
    );

    // In-place operations keep their operand.
    macro_rules! rejects_in_place {
        ($label:expr, $expected:expr, $operand:expr, |$ct:ident, $s:ident| $call:expr) => {{
            let mut target = $operand.clone();
            assert_rejects($label, $expected, &mut target, one, |$ct| {
                with_scratch::<BE, _>(ARENA, |$s| $call)
            });
        }};
    }
    rejects_in_place!("ckks_goldschmidt_division_assign iters", "iters must be > 0", full, |c, s| {
        module.ckks_goldschmidt_division_assign(c, 0, tsk, s)
    });
    rejects_in_place!(
        "ckks_goldschmidt_division_assign budget",
        "ckks_goldschmidt_division_assign: log_budget",
        low,
        |c, s| module.ckks_goldschmidt_division_assign(c, 2, tsk, s)
    );
    rejects_in_place!("ckks_rsqrt_assign r", "ckks_rsqrt_assign: r must be > 0", full, |c, s| module
        .ckks_rsqrt_assign(c, &full, 0, tsk, s));
    rejects_in_place!("ckks_rsqrt_assign budget", "ckks_rsqrt_assign: log_budget", low, |c, s| {
        module.ckks_rsqrt_assign(c, &full, 2, tsk, s)
    });
    rejects_in_place!(
        "ckks_inverse_negative_domain_assign budget",
        "ckks_inverse_negative_domain_assign: log_budget",
        low,
        |c, s| { module.ckks_inverse_negative_domain_assign(c, 2, tsk, s) }
    );
    rejects_in_place!(
        "ckks_inverse_full_domain_assign iters",
        "ckks_inverse_full_domain_assign: iters must be > 0",
        full,
        |c, s| { module.ckks_inverse_full_domain_assign(c, 0, comp, tsk, conj, s) }
    );
    rejects_in_place!(
        "ckks_inverse_full_domain_assign budget",
        "ckks_inverse_full_domain_assign: log_budget",
        low,
        |c, s| { module.ckks_inverse_full_domain_assign(c, 2, comp, tsk, conj, s) }
    );

    let mut pair = (fresh_res(), low.clone());
    assert_rejects(
        "ckks_interval_normalization budget",
        "ckks_interval_normalization: log_budget",
        &mut pair,
        two,
        |(norm_out, ct)| with_scratch::<BE, _>(ARENA, |s| module.ckks_interval_normalization(norm_out, ct, &norm, tsk, s)),
    );
}
