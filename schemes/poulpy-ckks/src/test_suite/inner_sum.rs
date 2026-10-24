//! Tests for `CKKSInnerSumOps`: sums of rotations by multiples of a power-of-two batch, and their reverse.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_inner_sum_aligned`] | sums, means and replicates over batches of one, two and a quarter of the slots, and the keyless empty window, at the input's width |
//! | [`test_inner_sum_smaller_output`] | the same into a destination one limb and a bit narrower, every key looked up at its width |
//! | [`test_inner_sum_assign`] | the in-place forms against the out-of-place ones |
//! | [`test_inner_sum_window_error`] | a window past the slots, refused before the missing keys, and its Galois query |
//! | [`test_inner_sum_integer_error`] | a mean of `Integer` slots, refused before the missing keys |
//! | [`test_inner_sum_missing_key_error`] | an absent key, refused before the budget, and a key one bit narrower than the sum |
//! | [`test_inner_sum_budget_error`] | the budget of the copy and of the division |
//! | [`test_inner_sum_scratch_error`] | an arena one byte below the query |
//! | [`test_inner_sum_below_module_degree`] | every sum on a module of twice the ciphertext's degree, and a window past the ciphertext's slots |

use std::{
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
};

use poulpy_core::layouts::{
    BackendGLWESecret, Degree, GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEPlaintextInfos, GetAutomorphismKey, LWEInfos,
    SetGLWEPlaintextInfos, prepared::GLWESecretPrepared,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned, Standard},
};

use crate::{
    CKKSCompositionError, CKKSError, CKKSInfos, CKKSResult, SlotsKind,
    api::{CKKSCopyOps, CKKSInnerSumOps},
    layouts::CKKSCiphertextOwned,
    test_suite::{
        CKKSTestParams,
        parity::helpers::{snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};

use super::helpers::{
    QueryLog, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, ckks_encrypt, dirty_ct, gen_atk, gen_sk_with_raw, test_vector_1, want_rotate,
};

/// Every window sums four batches.
const LOG_COUNT: usize = 2;

/// The three sums of the trait.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Sum,
    Mean,
    Replicate,
}

const KINDS: [Kind; 3] = [Kind::Sum, Kind::Mean, Kind::Replicate];

impl Kind {
    fn op(self) -> &'static str {
        match self {
            Kind::Sum | Kind::Mean => "inner_sum",
            Kind::Replicate => "replicate",
        }
    }

    /// The bits the sum moves under `log_delta`.
    fn shift(self, log_count: usize) -> usize {
        if self == Kind::Mean { log_count } else { 0 }
    }

    fn galois_elements<BE: Backend>(self, module: &Module<BE>, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>
    where
        Module<BE>: CKKSInnerSumOps<BE>,
    {
        match self {
            Kind::Sum | Kind::Mean => module.ckks_inner_sum_galois_elements(n, log_batch, log_count),
            Kind::Replicate => module.ckks_replicate_galois_elements(n, log_batch, log_count),
        }
    }

    #[expect(clippy::too_many_arguments, reason = "the sum's operands, window, keys and scratch")]
    fn run<BE: TestContextBackend, H: GetAutomorphismKey<BE>>(
        self,
        module: &Module<BE>,
        res: &mut CKKSCiphertextOwned<BE>,
        a: &CKKSCiphertextOwned<BE>,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Module<BE>: CKKSInnerSumOps<BE>,
    {
        match self {
            Kind::Sum => module.ckks_inner_sum(res, a, log_batch, log_count, false, keys, scratch),
            Kind::Mean => module.ckks_inner_sum(res, a, log_batch, log_count, true, keys, scratch),
            Kind::Replicate => module.ckks_replicate(res, a, log_batch, log_count, keys, scratch),
        }
    }

    fn run_assign<BE: TestContextBackend, H: GetAutomorphismKey<BE>>(
        self,
        module: &Module<BE>,
        res: &mut CKKSCiphertextOwned<BE>,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Module<BE>: CKKSInnerSumOps<BE>,
    {
        match self {
            Kind::Sum => module.ckks_inner_sum_assign(res, log_batch, log_count, false, keys, scratch),
            Kind::Mean => module.ckks_inner_sum_assign(res, log_batch, log_count, true, keys, scratch),
            Kind::Replicate => module.ckks_replicate_assign(res, log_batch, log_count, keys, scratch),
        }
    }
}

/// `(log_batch, log_count)`: batches of one, two and a quarter of the `m` slots, and the empty window at the top.
fn windows(m: usize) -> [(usize, usize); 4] {
    let log_m = m.ilog2() as usize;
    [(0, LOG_COUNT), (1, LOG_COUNT), (log_m - LOG_COUNT, LOG_COUNT), (log_m, 0)]
}

type Keys<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;

/// An encryption of `test_vector_1` at `params.k`, with its secret and encoder.
struct Fixture<BE: Backend, F, E> {
    encoder: ReferenceEncoder<E>,
    re: Vec<F>,
    im: Vec<F>,
    sk_raw: BackendGLWESecret<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: ScratchOwned<BE>,
    ct: CKKSCiphertextOwned<BE>,
}

fn fixture<BE, F, E>(params: &CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>) -> Fixture<BE, F, E>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let ct = ckks_encrypt(
        params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    Fixture {
        encoder,
        re,
        im,
        sk_raw,
        sk,
        scratch,
        ct,
    }
}

/// The keys of every sum of [`KINDS`] over `windows`, at the layout of `params`.
fn gen_keys<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &BackendGLWESecret<BE>,
    windows: &[(usize, usize)],
    scratch: &mut ScratchArena<'_, BE>,
) -> Keys<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
{
    let mut keys = HashMap::new();
    for &(log_batch, log_count) in windows {
        for kind in KINDS {
            for p in kind.galois_elements(module, params.n.into(), log_batch, log_count) {
                keys.entry(p).or_insert_with(|| gen_atk(params, module, p, sk_raw, scratch));
            }
        }
    }
    keys
}

/// `sum_{j < 2^log_count} rot(v, j 2^log_batch)`, by `-j 2^log_batch` for a replicate, over `2^log_count` for a mean.
fn want<F: TestScalar>(re: &[F], im: &[F], kind: Kind, log_batch: usize, log_count: usize) -> (Vec<F>, Vec<F>) {
    let m = re.len();
    let sign = if kind == Kind::Replicate { -1 } else { 1 };
    let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
    for j in 0..1i64 << log_count {
        let (r, i) = want_rotate(re, im, sign * (j << log_batch), m);
        for s in 0..m {
            want_re[s] = want_re[s] + r[s];
            want_im[s] = want_im[s] + i[s];
        }
    }
    if kind == Kind::Mean {
        let scale = F::from_f64(1.0 / (1u64 << log_count) as f64).unwrap();
        want_re.iter_mut().chain(want_im.iter_mut()).for_each(|x| *x = *x * scale);
    }
    (want_re, want_im)
}

/// Every sum, mean and replicate decrypts to its cleartext model, untagged, at the metadata of its input.
pub fn test_inner_sum_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let windows = windows(params.n / 2);
    let keys = gen_keys(&params, module, &f.sk_raw, &windows, &mut f.scratch.borrow());
    let atk = params.atk_layout();
    for (log_batch, log_count) in windows {
        for kind in KINDS {
            let label = format!("{kind:?} log_batch={log_batch} log_count={log_count}");
            let mut res = dirty_ct(module, &f.ct, params.k);
            with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
                kind.run(module, &mut res, &f.ct, log_batch, log_count, &keys, s)
            })
            .unwrap();
            let shift = kind.shift(log_count);
            assert_ct_meta(&label, &res, f.ct.log_delta() + shift, f.ct.log_budget() - shift);
            assert!(res.noise().is_none(), "{label}: the sum kept a noise estimate");
            let (want_re, want_im) = want(&f.re, &f.im, kind, log_batch, log_count);
            assert_decrypt_precision(
                &label,
                &params,
                module,
                &f.encoder,
                &res,
                &f.sk,
                &want_re,
                &want_im,
                &mut f.scratch.borrow(),
            );
        }
    }
}

/// Into a destination one limb and a bit narrower than the input, the copy spends the difference, and every key is
/// looked up at the destination's width, exactly the keys of the Galois query.
pub fn test_inner_sum_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let windows = windows(params.n / 2);
    let keys = QueryLog::new(gen_keys(&params, module, &f.sk_raw, &windows, &mut f.scratch.borrow()));
    let atk = params.atk_layout();
    let narrow_k = params.k - params.base2k - 1;
    let offset = f.ct.k().as_usize() - narrow_k;
    for (log_batch, log_count) in windows {
        for kind in KINDS {
            let label = format!("{kind:?} log_batch={log_batch} log_count={log_count} narrower");
            let mut res = dirty_ct(module, &f.ct, narrow_k);
            with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
                kind.run(module, &mut res, &f.ct, log_batch, log_count, &keys, s)
            })
            .unwrap();
            let seen = keys.take();
            assert!(
                seen.iter().all(|&(_, k)| k.as_usize() == narrow_k),
                "{label}: a key was looked up at another width than {narrow_k}"
            );
            let looked_up: HashSet<i64> = seen.iter().map(|&(p, _)| p).collect();
            let listed: HashSet<i64> = kind
                .galois_elements(module, f.ct.n(), log_batch, log_count)
                .into_iter()
                .collect();
            assert_eq!(
                looked_up, listed,
                "{label}: the keys looked up are not those of the Galois query"
            );
            let shift = kind.shift(log_count);
            assert_ct_meta(&label, &res, f.ct.log_delta() + shift, f.ct.log_budget() - offset - shift);
            let (want_re, want_im) = want(&f.re, &f.im, kind, log_batch, log_count);
            assert_decrypt_precision(
                &label,
                &params,
                module,
                &f.encoder,
                &res,
                &f.sk,
                &want_re,
                &want_im,
                &mut f.scratch.borrow(),
            );
        }
    }
}

/// Each `_assign` form equals its out-of-place form, digits, layout, flag and noise tag, each on exactly its query.
pub fn test_inner_sum_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let windows = windows(params.n / 2);
    let keys = gen_keys(&params, module, &f.sk_raw, &windows, &mut f.scratch.borrow());
    let atk = params.atk_layout();
    for (log_batch, log_count) in windows {
        for kind in KINDS {
            let label = format!("{kind:?} log_batch={log_batch} log_count={log_count}");
            let mut res = dirty_ct(module, &f.ct, params.k);
            with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
                kind.run(module, &mut res, &f.ct, log_batch, log_count, &keys, s)
            })
            .unwrap();
            let mut res_assign = alloc_ct(&params, module, params.k);
            module.ckks_copy(&mut res_assign, &f.ct, &mut f.scratch.borrow()).unwrap();
            with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res_assign, &res_assign, &atk), |s| {
                kind.run_assign(module, &mut res_assign, log_batch, log_count, &keys, s)
            })
            .unwrap();
            assert!(
                snapshot::<BE, _>(&res_assign) == snapshot::<BE, _>(&res),
                "{label}: in place differs from out of place"
            );
        }
    }
}

/// Asserts `result` failed as `check` wants and `res` kept its `before` snapshot.
fn assert_refused<BE>(
    label: &str,
    result: CKKSResult<()>,
    res: &CKKSCiphertextOwned<BE>,
    before: &crate::test_suite::parity::helpers::Snapshot,
    check: impl FnOnce(&CKKSError) -> bool,
) where
    BE: TestContextBackend,
{
    let err = result.expect_err(label);
    assert!(check(&err), "{label}: unexpected error: {err}");
    assert!(snapshot::<BE, _>(res) == *before, "{label}: wrote the destination");
}

/// A window of `2^log_count` batches of `2^log_batch` slots past the slot count is refused with `InvalidOperand`,
/// before the absent keys, by every form on an empty arena, and its Galois query panics.
pub fn test_inner_sum_window_error<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let f = fixture::<BE, F, E>(&params, module, host_module);
    let log_batch = (params.n / 2).ilog2() as usize - 1;
    let no_keys: Keys<BE> = HashMap::new();
    for kind in KINDS {
        let label = format!("{kind:?} window past the slots");
        let invalid = |err: &CKKSError| matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand { op, .. }) if *op == kind.op());
        let mut res = dirty_ct(module, &f.ct, params.k);
        let before = snapshot::<BE, _>(&res);
        let result = with_scratch::<BE, _>(0, |s| kind.run(module, &mut res, &f.ct, log_batch, LOG_COUNT, &no_keys, s));
        assert_refused::<BE>(&label, result, &res, &before, invalid);
        let result = with_scratch::<BE, _>(0, |s| kind.run_assign(module, &mut res, log_batch, LOG_COUNT, &no_keys, s));
        assert_refused::<BE>(&format!("{label}, in place"), result, &res, &before, invalid);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            kind.galois_elements(module, f.ct.n(), log_batch, LOG_COUNT)
        }))
        .expect_err("the Galois query accepted a window past the slots");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
        let want = match kind {
            Kind::Sum | Kind::Mean => "ckks_inner_sum_galois_elements: the window exceeds the slot count",
            Kind::Replicate => "ckks_replicate_galois_elements: the window exceeds the slot count",
        };
        assert_eq!(message, Some(want), "{label}: Galois query panic");
    }
}

/// A mean of `Integer` slots is refused with `InvalidOperand`, before the absent keys, on an empty arena; a sum of
/// them is not refused for its slots, and neither is an empty mean, which divides by one.
pub fn test_inner_sum_integer_error<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    f.ct.set_slots(SlotsKind::Integer);
    let no_keys: Keys<BE> = HashMap::new();
    let integer = |err: &CKKSError| {
        matches!(
            err.composition(),
            Some(CKKSCompositionError::InvalidOperand { op: "inner_sum", .. })
        )
    };
    let mut res = dirty_ct(module, &f.ct, params.k);
    let before = snapshot::<BE, _>(&res);
    let result = with_scratch::<BE, _>(0, |s| Kind::Mean.run(module, &mut res, &f.ct, 0, LOG_COUNT, &no_keys, s));
    assert_refused::<BE>("mean of Integer slots", result, &res, &before, integer);
    let mut res_assign = dirty_ct(module, &f.ct, params.k);
    let before_assign = snapshot::<BE, _>(&res_assign);
    let result = with_scratch::<BE, _>(0, |s| {
        Kind::Mean.run_assign(module, &mut res_assign, 0, LOG_COUNT, &no_keys, s)
    });
    assert_refused::<BE>(
        "mean of Integer slots, in place",
        result,
        &res_assign,
        &before_assign,
        integer,
    );

    let result = with_scratch::<BE, _>(0, |s| Kind::Sum.run(module, &mut res, &f.ct, 0, LOG_COUNT, &no_keys, s));
    assert_refused::<BE>("sum of Integer slots", result, &res, &before, |err| {
        matches!(err.composition(), Some(CKKSCompositionError::MissingAutomorphismKey { .. }))
    });
    let atk = params.atk_layout();
    let log_top = (params.n / 2).ilog2() as usize;
    with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
        Kind::Mean.run(module, &mut res, &f.ct, log_top, 0, &no_keys, s)
    })
    .unwrap();
    assert_eq!(res.slots(), SlotsKind::Integer, "an empty mean relabelled Integer slots");
}

/// An absent key is refused with `MissingAutomorphismKey` at the sum's width, before the budget, on an empty arena,
/// and so is a key of `max_input_k` one bit below the width the sum switches at, which a destination at that
/// `max_input_k` accepts.
pub fn test_inner_sum_missing_key_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let no_keys: Keys<BE> = HashMap::new();
    for kind in KINDS {
        let label = format!("{kind:?} absent key");
        let first_rotation = if kind == Kind::Replicate { -2 } else { 1 };
        // A mean into one bit of budget would also fail its division.
        let k = f.ct.log_delta() + 1;
        let missing = |err: &CKKSError| {
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MissingAutomorphismKey { op, rotation, k: key_k, .. })
                    if *op == kind.op() && *rotation == first_rotation && *key_k as usize == k
            )
        };
        let mut res = dirty_ct(module, &f.ct, k);
        let before = snapshot::<BE, _>(&res);
        let result = with_scratch::<BE, _>(0, |s| kind.run(module, &mut res, &f.ct, 0, LOG_COUNT, &no_keys, s));
        assert_refused::<BE>(&label, result, &res, &before, missing);
        let result = with_scratch::<BE, _>(0, |s| kind.run_assign(module, &mut res, 0, LOG_COUNT, &no_keys, s));
        assert_refused::<BE>(&format!("{label}, in place"), result, &res, &before, missing);
    }

    // Keys whose gadget decomposes `max_input_k` bits, below the input's width.
    let k_digit = params.base2k * params.dsize;
    let max_input_k = (params.k - 1) / k_digit * k_digit;
    let key_params = CKKSTestParams {
        k: max_input_k,
        ..params
    };
    assert_eq!(key_params.atk_layout().gglwe_layout().max_input_k().as_usize(), max_input_k);
    let keys = gen_keys(&key_params, module, &f.sk_raw, &[(0, LOG_COUNT)], &mut f.scratch.borrow());
    let atk = key_params.atk_layout();
    for kind in KINDS {
        let label = format!("{kind:?} key one bit narrower");
        let mut res = dirty_ct(module, &f.ct, max_input_k + 1);
        let before = snapshot::<BE, _>(&res);
        let result = with_scratch::<BE, _>(0, |s| kind.run(module, &mut res, &f.ct, 0, LOG_COUNT, &keys, s));
        assert_refused::<BE>(&label, result, &res, &before, |err| {
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MissingAutomorphismKey { op, k, .. })
                    if *op == kind.op() && *k as usize == max_input_k + 1
            )
        });
        let mut res = dirty_ct(module, &f.ct, max_input_k);
        with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
            kind.run(module, &mut res, &f.ct, 0, LOG_COUNT, &keys, s)
        })
        .unwrap_or_else(|err| panic!("{kind:?} at the keys' max_input_k: {err}"));
    }
}

/// A copy into a destination below the input's `log_delta`, and a mean dividing by more than the budget the copy
/// leaves, are refused with `InsufficientHomomorphicCapacity`, on an empty arena.
pub fn test_inner_sum_budget_error<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let keys = gen_keys(&params, module, &f.sk_raw, &[(0, LOG_COUNT)], &mut f.scratch.borrow());
    let (ct_k, log_delta, log_budget) = (f.ct.k().as_usize(), f.ct.log_delta(), f.ct.log_budget());

    for kind in KINDS {
        let label = format!("{kind:?} copy below log_delta");
        let k = log_delta - 1;
        let mut res = dirty_ct(module, &f.ct, k);
        let before = snapshot::<BE, _>(&res);
        let result = with_scratch::<BE, _>(0, |s| kind.run(module, &mut res, &f.ct, 0, LOG_COUNT, &keys, s));
        let want = CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: kind.op(),
            available_log_budget: log_budget,
            required_bits: ct_k - k,
        };
        assert_refused::<BE>(&label, result, &res, &before, |err| {
            assert_ckks_error(&label, err, want);
            true
        });
    }

    let mut res = dirty_ct(module, &f.ct, log_delta + LOG_COUNT - 1);
    let before = snapshot::<BE, _>(&res);
    let result = with_scratch::<BE, _>(0, |s| Kind::Mean.run(module, &mut res, &f.ct, 0, LOG_COUNT, &keys, s));
    let want = CKKSCompositionError::InsufficientHomomorphicCapacity {
        op: "inner_sum",
        available_log_budget: LOG_COUNT - 1,
        required_bits: LOG_COUNT,
    };
    assert_refused::<BE>("mean past the budget", result, &res, &before, |err| {
        assert_ckks_error("mean past the budget", err, want.clone());
        true
    });
    let mut res_assign = dirty_ct(module, &f.ct, log_delta + LOG_COUNT - 1);
    let before_assign = snapshot::<BE, _>(&res_assign);
    let result = with_scratch::<BE, _>(0, |s| Kind::Mean.run_assign(module, &mut res_assign, 0, LOG_COUNT, &keys, s));
    assert_refused::<BE>("mean past the budget, in place", result, &res_assign, &before_assign, |err| {
        assert_ckks_error("mean past the budget, in place", err, want);
        true
    });
}

/// One byte below the query, every form returns `CKKSError::Internal` before any write.
pub fn test_inner_sum_scratch_error<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let keys = gen_keys(&params, module, &f.sk_raw, &[(0, LOG_COUNT)], &mut f.scratch.borrow());
    let atk = params.atk_layout();
    for kind in KINDS {
        let label = format!("{kind:?} one byte short");
        let internal = |err: &CKKSError| {
            matches!(err, CKKSError::Internal(_))
                && err.to_string().contains(&format!(
                    "{}: scratch below CKKSInnerSumOps::ckks_inner_sum_tmp_bytes",
                    kind.op()
                ))
        };
        let mut res = dirty_ct(module, &f.ct, params.k);
        let before = snapshot::<BE, _>(&res);
        let bytes = module.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk);
        let result = with_scratch::<BE, _>(bytes - 1, |s| kind.run(module, &mut res, &f.ct, 0, LOG_COUNT, &keys, s));
        assert_refused::<BE>(&label, result, &res, &before, internal);
        let bytes = module.ckks_inner_sum_tmp_bytes(&res, &res, &atk);
        let result = with_scratch::<BE, _>(bytes - 1, |s| kind.run_assign(module, &mut res, 0, LOG_COUNT, &keys, s));
        assert_refused::<BE>(&format!("{label}, in place"), result, &res, &before, internal);
    }
}

/// On a module of twice the ciphertext's degree, the Galois queries at the ciphertext's degree and every sum equal
/// those of a module of that degree, and a window past the ciphertext's slots, within the module's, is refused with
/// `InvalidOperand` before any write.
pub fn test_inner_sum_below_module_degree<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut f = fixture::<BE, F, E>(&params, module, host_module);
    let windows = windows(params.n / 2);
    let keys = gen_keys(&params, module, &f.sk_raw, &windows, &mut f.scratch.borrow());
    let wide = Module::<BE>::new((2 * params.n) as u64);
    let n = f.ct.n();
    let atk = params.atk_layout();
    for kind in KINDS {
        for &(log_batch, log_count) in &windows {
            let label = format!("{kind:?} log_batch={log_batch} log_count={log_count} below the module degree");
            assert_eq!(
                kind.galois_elements(&wide, n, log_batch, log_count),
                kind.galois_elements(module, n, log_batch, log_count),
                "{label}: Galois query"
            );
            let mut want = dirty_ct(module, &f.ct, params.k);
            with_scratch::<BE, _>(module.ckks_inner_sum_tmp_bytes(&want, &f.ct, &atk), |s| {
                kind.run(module, &mut want, &f.ct, log_batch, log_count, &keys, s)
            })
            .unwrap();
            let mut res = dirty_ct(module, &f.ct, params.k);
            with_scratch::<BE, _>(wide.ckks_inner_sum_tmp_bytes(&res, &f.ct, &atk), |s| {
                kind.run(&wide, &mut res, &f.ct, log_batch, log_count, &keys, s)
            })
            .unwrap_or_else(|err| panic!("{label}: {err}"));
            assert!(
                snapshot::<BE, _>(&res) == snapshot::<BE, _>(&want),
                "{label}: differs from the sum at the ciphertext's degree"
            );
        }

        let label = format!("{kind:?} window past the ciphertext's slots");
        let log_m = (params.n / 2).ilog2() as usize;
        let invalid = |err: &CKKSError| matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand { op, .. }) if *op == kind.op());
        let mut res = dirty_ct(module, &f.ct, params.k);
        let before = snapshot::<BE, _>(&res);
        let result = with_scratch::<BE, _>(0, |s| kind.run(&wide, &mut res, &f.ct, log_m, 1, &keys, s));
        assert_refused::<BE>(&label, result, &res, &before, invalid);
    }
}
