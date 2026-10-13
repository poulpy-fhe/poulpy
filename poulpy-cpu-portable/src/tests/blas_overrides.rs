//! Downstream BLAS hooks: counting map and JKLS hooks that forward to the reference, under the provided
//! compositions. The overriding backend matches the oracle bit for bit, every operation over a hook enters it, and no
//! call the API rejects enters either hook.
use crate::{ckks_encoding::EncodingFFTTable, hal_impl::delegating_backend::DifferentSamplingFFT64Portable as OverrideBackend};
use poulpy_blas::{
    api::{BlasMatIntraOps, BlasMatMulOps},
    oep::{BlasJklsImpl, BlasMatIntraImpl},
    reference::{BlasMatIntraReference, BlasMatMulReference},
    test_suite::{
        helpers::{
            Grid, TestKeys, alloc_grid, context_and_keys, encrypt_grid, foreign_contexts, foreign_layouts, rejected_grids,
        },
        parity::{BlasParityShapes, test_blas_inter_parity, test_blas_intra_parity, test_blas_matmul_parity},
    },
};
use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    api::LinearTransformationPrepared,
    test_suite::{
        BASE19_PARAMS_F64, CKKSTestParams,
        helpers::{alloc_scratch, gen_sk_with_raw, gen_tsk},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchArena, ScratchOwned},
};
use std::{
    collections::HashMap,
    sync::{
        Mutex, MutexGuard, PoisonError,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};

static MAP_CALLS: AtomicUsize = AtomicUsize::new(0);
static JKLS_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Held by every test that enters the hooks, which share the counters.
static HOOKS: Mutex<()> = Mutex::new(());

/// The counters, reset, for the holder of the guard alone.
fn counting() -> MutexGuard<'static, ()> {
    let guard = HOOKS.lock().unwrap_or_else(PoisonError::into_inner);
    reset();
    guard
}

fn reset() {
    MAP_CALLS.store(0, Relaxed);
    JKLS_CALLS.store(0, Relaxed);
}

/// `(map hook calls, JKLS kernel calls)` since the last reset.
fn calls() -> (usize, usize) {
    (MAP_CALLS.load(Relaxed), JKLS_CALLS.load(Relaxed))
}

poulpy_blas::impl_blas_inter_reference!(OverrideBackend);
poulpy_blas::impl_blas_matmul_reference!(OverrideBackend);

unsafe impl BlasMatIntraImpl for OverrideBackend {
    fn blas_map_blocks_tmp_bytes_impl<D, S, K>(
        module: &Module<Self>,
        dst: &D,
        src: &S,
        sources: usize,
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
    ) -> usize
    where
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        module.blas_map_blocks_tmp_bytes_reference(dst, src, sources, lts, atk)
    }

    fn blas_map_blocks_impl<D, S, K>(
        module: &Module<Self>,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + Sync,
    {
        MAP_CALLS.fetch_add(1, Relaxed);
        module.blas_map_blocks_reference(dst, src, lts, atk, scratch)
    }
}

unsafe impl BlasJklsImpl for OverrideBackend {
    fn blas_jkls_block_tmp_bytes_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &R,
        lhs: &L,
        rhs: &Rh,
        terms: usize,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        module.blas_jkls_block_tmp_bytes_reference(res, lhs, rhs, terms, tsk)
    }

    fn blas_jkls_block_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &[&L],
        rhs: &[&Rh],
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        L: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        Rh: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        T: GetTensorKey<Self> + Sync,
    {
        JKLS_CALLS.fetch_add(1, Relaxed);
        module.blas_jkls_block_reference(res, lhs, rhs, tsk, scratch)
    }
}

/// Parity of the overriding backend with the oracle, every family, through both hooks.
#[test]
fn blas_overrides_match_the_oracle() {
    let _counting = counting();
    let params = CKKSTestParams {
        n: 64,
        hw: 48,
        base2k: 12,
        ..BASE19_PARAMS_F64
    };
    let shapes = BlasParityShapes::default();
    let oracle = Module::<FFT64Oracle>::new(params.n as u64);
    let module = Module::<OverrideBackend>::new(params.n as u64);
    test_blas_inter_parity::<FFT64Oracle, OverrideBackend, f64>(params, &shapes, &oracle, &module);
    test_blas_intra_parity::<FFT64Oracle, OverrideBackend, f64>(params, &shapes, &oracle, &module);
    test_blas_matmul_parity::<FFT64Oracle, OverrideBackend, f64>(params, &shapes, &oracle, &module);
    let (map, jkls) = calls();
    assert!(
        map > 0 && jkls > 0,
        "parity entered the map hook {map} times and the JKLS kernel {jkls} times"
    );
}

/// Runs `call`, which must succeed, and asserts it entered the map hook and the JKLS kernel `want` times.
fn assert_enters<T>(op: &str, want: (usize, usize), call: impl FnOnce() -> CKKSResult<T>) -> T {
    let before = calls();
    let out = call().unwrap_or_else(|e| panic!("{op}: {e}"));
    let after = calls();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        want,
        "{op}: (map hook, JKLS kernel) calls"
    );
    out
}

/// Asserts `result` is an error, and that no hook was entered since the counters were reset.
fn assert_rejected<T>(label: &str, result: CKKSResult<T>) {
    assert!(result.is_err(), "{label}: accepted");
    assert_eq!(calls(), (0, 0), "{label}: a rejected call entered a hook");
}

/// On `2 x 2` grids of `4 x 4` blocks, every operation over a hook enters it as the reference composes it: the
/// transpose and both shifts across blocks one map-hook call, each preparation two (sigma or tau, then every shift),
/// `blas_matmul_prepared` one JKLS call per block of `res`, and `blas_matmul` both preparations' and the prepared
/// product's. Then no call the API rejects enters either hook: every grid [`rejected_grids`] and [`foreign_layouts`]
/// build, a grid of another shape and one at radix 63, as an input or as `res` of each of these operations (a lone
/// preparation input of another shape is valid; one of another rank than the keys is not), an inner-dimension mismatch,
/// foreign contexts, and key stores without rotation keys or without the tensor key.
#[test]
fn blas_hooks_run_on_accepted_calls_alone() {
    let _counting = counting();
    let (params, log_d) = (BASE19_PARAMS_F64, 2);
    let module = Module::<OverrideBackend>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<EncodingFFTTable<f64>>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);
    let (ctx, mut atks) = context_and_keys::<OverrideBackend, f64>(&params, &module, log_d, &sk_raw, &mut scratch.borrow());
    let foreign = foreign_contexts::<OverrideBackend, f64>(&params, &module, log_d, &sk_raw, &mut atks, &mut scratch.borrow());
    let keys = TestKeys {
        atks,
        tsk: Some(gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow())),
    };
    let a = encrypt_grid::<OverrideBackend, f64, EncodingFFTTable<f64>>(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(log_d, 2, 2, slots >> (2 * log_d), 7),
        &mut scratch.borrow(),
    );
    let alloc = |rows: usize, cols: usize| alloc_grid(&params, &module, log_d, rows, cols, params.k);
    let mut res = alloc(2, 2);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let bytes = module
        .blas_matmul_tmp_bytes(&res, &a, &a, &ctx, &tsk_layout, &atk_layout)
        .max(module.blas_transpose_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_rotate_rows_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_rotate_cols_tmp_bytes(&res, &a, &ctx, &atk_layout));
    let mut op_scratch = ScratchOwned::<OverrideBackend>::alloc(bytes);

    assert_enters("blas_transpose", (1, 0), || {
        module.blas_transpose(&mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_rotate_rows", (1, 0), || {
        module.blas_rotate_rows(&mut res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_rotate_cols", (1, 0), || {
        module.blas_rotate_cols(&mut res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    let left = assert_enters("blas_matmul_prepare_left", (2, 0), || {
        module.blas_matmul_prepare_left(&a, &ctx, &keys, &mut op_scratch.borrow())
    });
    let right = assert_enters("blas_matmul_prepare_right", (2, 0), || {
        module.blas_matmul_prepare_right(&a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_matmul_prepared", (0, 4), || {
        module.blas_matmul_prepared(&mut res, &left, &right, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_matmul", (4, 4), || {
        module.blas_matmul(&mut res, &a, &a, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });

    reset();
    let radix_63 = CKKSTestParams { base2k: 63, ..params };
    let wrong_res = [
        ("shape", alloc(1, 2)),
        ("radix 63", alloc_grid(&radix_63, &module, log_d, 2, 2, params.k)),
    ];
    let bad = rejected_grids(&module, &a)
        .into_iter()
        .chain(foreign_layouts(&module, &a))
        .chain(wrong_res);
    for (label, mut bad) in bad {
        let s = &mut op_scratch.borrow();
        let res = &mut alloc(2, 2);
        let rejected = |call: &str, result: CKKSResult<()>| assert_rejected(&format!("{label}: {call}"), result);
        rejected("transpose a", module.blas_transpose(res, &bad, &ctx, &keys, s));
        rejected("transpose res", module.blas_transpose(&mut bad, &a, &ctx, &keys, s));
        rejected("rotate_rows a", module.blas_rotate_rows(res, &bad, 1, &ctx, &keys, s));
        rejected("rotate_rows res", module.blas_rotate_rows(&mut bad, &a, 1, &ctx, &keys, s));
        rejected("rotate_cols a", module.blas_rotate_cols(res, &bad, 1, &ctx, &keys, s));
        rejected("rotate_cols res", module.blas_rotate_cols(&mut bad, &a, 1, &ctx, &keys, s));
        rejected("matmul a", module.blas_matmul(res, &bad, &a, &ctx, &keys, &keys, s));
        rejected("matmul b", module.blas_matmul(res, &a, &bad, &ctx, &keys, &keys, s));
        rejected("matmul res", module.blas_matmul(&mut bad, &a, &a, &ctx, &keys, &keys, s));
        rejected("prepared res", module.blas_matmul_prepared(&mut bad, &left, &right, &keys, s));
        if label != "shape" {
            assert_rejected(
                &format!("{label}: prepare_left"),
                module.blas_matmul_prepare_left(&bad, &ctx, &keys, s),
            );
            assert_rejected(
                &format!("{label}: prepare_right"),
                module.blas_matmul_prepare_right(&bad, &ctx, &keys, s),
            );
        }
    }
    let s = &mut op_scratch.borrow();
    let row = alloc(1, 2);
    assert_rejected("inner: matmul", module.blas_matmul(&mut res, &a, &row, &ctx, &keys, &keys, s));
    for (label, ctx) in &foreign {
        let rejected = |call: &str, result: CKKSResult<()>| assert_rejected(&format!("{label} context: {call}"), result);
        rejected("transpose", module.blas_transpose(&mut res, &a, ctx, &keys, s));
        rejected("rotate_rows", module.blas_rotate_rows(&mut res, &a, 1, ctx, &keys, s));
        rejected("rotate_cols", module.blas_rotate_cols(&mut res, &a, 1, ctx, &keys, s));
        rejected("matmul", module.blas_matmul(&mut res, &a, &a, ctx, &keys, &keys, s));
        assert_rejected(
            &format!("{label} context: prepare_left"),
            module.blas_matmul_prepare_left(&a, ctx, &keys, s),
        );
        assert_rejected(
            &format!("{label} context: prepare_right"),
            module.blas_matmul_prepare_right(&a, ctx, &keys, s),
        );
    }
    let TestKeys { atks, tsk } = keys;
    let (no_atks, no_tsk) = (
        TestKeys {
            atks: HashMap::new(),
            tsk,
        },
        TestKeys { atks, tsk: None },
    );
    let rejected = |call: &str, result: CKKSResult<()>| assert_rejected(&format!("no rotation keys: {call}"), result);
    rejected("transpose", module.blas_transpose(&mut res, &a, &ctx, &no_atks, s));
    rejected("rotate_rows", module.blas_rotate_rows(&mut res, &a, 1, &ctx, &no_atks, s));
    rejected("rotate_cols", module.blas_rotate_cols(&mut res, &a, 1, &ctx, &no_atks, s));
    rejected("matmul", module.blas_matmul(&mut res, &a, &a, &ctx, &no_atks, &no_atks, s));
    assert_rejected(
        "no rotation keys: prepare_left",
        module.blas_matmul_prepare_left(&a, &ctx, &no_atks, s),
    );
    assert_rejected(
        "no rotation keys: prepare_right",
        module.blas_matmul_prepare_right(&a, &ctx, &no_atks, s),
    );
    assert_rejected(
        "no tensor key: matmul",
        module.blas_matmul(&mut res, &a, &a, &ctx, &no_tsk, &no_tsk, s),
    );
    assert_rejected(
        "no tensor key: prepared",
        module.blas_matmul_prepared(&mut res, &left, &right, &no_tsk, s),
    );
}
