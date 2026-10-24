use poulpy_hal::layouts::{HostBytesBackend, Module};
use poulpy_hal::source::Source;

use poulpy_ckks::{
    CKKSMeta,
    layouts::{CKKSPlaintext, CKKSScalar},
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, ckks_spec, upload_pt},
    },
};

use poulpy_ckks::test_suite::helpers::{
    assert_precision_for_log_delta, expected_log2_precision, precision_stats,
};

use crate::sign::SignComposite;

pub(super) fn assert_error<T>(result: anyhow::Result<T>, expected: &str) {
    let error = match result {
        Ok(_) => panic!("expected error: {expected}"),
        Err(error) => error,
    };
    assert_eq!(error.to_string(), expected);
}

pub(super) fn assert_error_starts_with<T>(result: anyhow::Result<T>, expected: &str) {
    let error = match result {
        Ok(_) => panic!("expected error starting with: {expected}"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.starts_with(expected),
        "error `{error}` does not start with `{expected}`"
    );
}

#[allow(clippy::type_complexity)]
pub(super) fn gen_composite<F, BE>(
    base: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> (
    SignComposite<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>>,
    SignComposite<F, CKKSPlaintext<Vec<u8>, i64>>,
    usize,
)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: CKKSScalar + TestScalar,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let build = || {
        SignComposite::<F, _>::from_minimax(
            F::from_f64(0.1).unwrap(),
            20.0,
            &[15],
            12,
            base.base2k.into(),
            coeff_meta,
            SplitStrategy::MinDepth,
            host_module,
        )
        .expect("from_minimax")
    };
    let host = build();
    let consumed = host.consumed_bits(log_delta);
    (
        host.map_plaintexts(|pt| upload_pt(module, pt)),
        build(),
        consumed,
    )
}

pub(super) fn params_for(base: &CKKSTestParams, consumed: usize) -> CKKSTestParams {
    params_for_with_headroom(base, consumed, 2 * base.prec_meta.log_delta)
}

pub(super) fn params_for_with_headroom(
    base: &CKKSTestParams,
    consumed: usize,
    headroom: usize,
) -> CKKSTestParams {
    let log_delta = base.prec_meta.log_delta;
    let k = (consumed + headroom + 2 * base.base2k).next_multiple_of(base.dsize * base.base2k);
    CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: Default::default(),
        },
        prec_log_budget: k - log_delta,
        hw: base.hw,
        dsize: base.dsize,
        rank: base.rank,
    }
}

pub(super) fn sample_uniform<F: TestScalar>(slots: usize, lo: f64, hi: f64, seed: u8) -> Vec<F> {
    let mut source = Source::new([seed; 32]);
    (0..slots)
        .map(|_| F::from_f64(source.next_f64(lo, hi)).unwrap())
        .collect()
}

pub(super) fn sample_interval<F: TestScalar>(slots: usize, lo: f64, hi: f64, seed: u8) -> Vec<F> {
    assert!(slots > 0, "sample_interval: empty slot set");
    let mut values = sample_uniform(slots, lo, hi, seed);
    values[0] = F::from_f64(lo).unwrap();
    if slots > 1 {
        values[1] = F::from_f64(hi).unwrap();
    }
    values
}

// Bits by which the worst slot may trail the mean-precision bar before failing.
const WORST_SLOT_MARGIN_BITS: f64 = 8.0;

pub(super) fn assert_precision_bits<F: TestScalar>(
    label: &str,
    got: &[F],
    want: &[F],
    target_bits: f64,
    degree: usize,
) {
    let mut log_delta = target_bits.ceil().max(0.0) as usize;
    while expected_log2_precision(log_delta, degree) < target_bits {
        log_delta += 1;
    }
    assert_precision_for_log_delta(label, got, want, log_delta, degree);
    let floor = (expected_log2_precision(log_delta, degree) - WORST_SLOT_MARGIN_BITS).max(0.0);
    let stats = precision_stats(got, want, log_delta);
    assert!(
        stats.min_log2_prec >= floor,
        "{label}: worst-slot precision {:.2} < {floor:.2} bits (mean {:.2})",
        stats.min_log2_prec,
        stats.avg_log2_prec,
    );
}
