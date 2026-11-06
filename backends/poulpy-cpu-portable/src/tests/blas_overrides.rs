//! Downstream BLAS hooks: counting map, JKLS and masked-unit hooks that forward to the reference, under the reference
//! compositions and the provided transpose. The overriding backend matches the oracle bit for bit, every operation over
//! a hook enters it, and no call the API rejects enters any hook.
use crate::{ckks_encoding::EncodingFFTTable, hal_impl::delegating_backend::DifferentSamplingFFT64Portable as OverrideBackend};
use poulpy_blas::{
    api::{BlasMatIntraOps, BlasMatMulLeftOperandPreparedFactory, BlasMatMulOps, BlasMatMulRightOperandPreparedFactory},
    layouts::{BlasContextPrepared, BlasGrid, BlasGridInfos, BlasGridMut, BlasPlan, BlasPtAlgorithm, Left, Right},
    oep::{BlasJklsImpl, BlasMatIntraImpl},
    reference::{BlasJklsReference, BlasMatIntraReference},
    test_suite::{
        helpers::{
            Grid, HINTS, MASKED_PT_ALGORITHMS, MIXES, PT_ALGORITHMS, TestKeys, alloc_grid, context_and_keys, encoded_context,
            encrypt_grid, foreign_contexts, foreign_layouts, matmul_left_operand, matmul_right_operand, pt_operand,
            rejected_grids, rotation_keys,
        },
        parity::{BlasParityShapes, test_blas_inter_parity, test_blas_intra_parity, test_blas_matmul_parity},
    },
};
use poulpy_ckks::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    api::LinearTransformationPrepared,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
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
    layouts::{CyclotomicOrder, HostBytesBackend, Module, PrepareHint::Reuse, ScratchArena, ScratchOwned},
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
static PT_UNIT_CALLS: AtomicUsize = AtomicUsize::new(0);
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
    PT_UNIT_CALLS.store(0, Relaxed);
}

/// `(map hook calls, JKLS kernel calls, masked-unit hook calls)` since the last reset.
fn calls() -> (usize, usize, usize) {
    (MAP_CALLS.load(Relaxed), JKLS_CALLS.load(Relaxed), PT_UNIT_CALLS.load(Relaxed))
}

poulpy_blas::impl_blas_mat_inter_reference_full!(OverrideBackend);
poulpy_blas::impl_blas_mat_mul_reference_full!(OverrideBackend);
poulpy_blas::impl_blas_mat_vec_reference_full!(OverrideBackend);
poulpy_blas::impl_blas_encoding_reference_full!(OverrideBackend);

unsafe impl BlasMatIntraImpl for OverrideBackend {
    fn blas_map_blocks_tmp_bytes<D, S, K>(
        module: &Module<Self>,
        sources: usize,
        dst: &D,
        src: &S,
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &K,
    ) -> usize
    where
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        module.blas_map_blocks_tmp_bytes_reference(sources, dst, src, lts, keys)
    }

    fn blas_map_blocks<D, S, K>(
        module: &Module<Self>,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        MAP_CALLS.fetch_add(1, Relaxed);
        module.blas_map_blocks_reference(dst, src, lts, keys, scratch)
    }

    fn blas_rotate_rows_tmp_bytes<R, A, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        module.blas_rotate_rows_tmp_bytes_reference(res, a, ctx, keys)
    }

    fn blas_rotate_rows<R, A, K>(
        module: &Module<Self>,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        module.blas_rotate_rows_reference(k, res, a, ctx, keys, scratch)
    }

    fn blas_rotate_cols_tmp_bytes<R, A, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        module.blas_rotate_cols_tmp_bytes_reference(res, a, ctx, keys)
    }

    fn blas_rotate_cols<R, A, K>(
        module: &Module<Self>,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        module.blas_rotate_cols_reference(k, res, a, ctx, keys, scratch)
    }

    fn blas_sum_rows_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContextPrepared<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        module.blas_sum_rows_tmp_bytes_reference(res, a, ctx, keys)
    }

    fn blas_sum_rows<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        module.blas_sum_rows_reference(res, a, ctx, keys, scratch)
    }

    fn blas_sum_cols_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContextPrepared<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        module.blas_sum_cols_tmp_bytes_reference(res, a, ctx, keys)
    }

    fn blas_sum_cols<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        module.blas_sum_cols_reference(res, a, ctx, keys, scratch)
    }
}

unsafe impl BlasJklsImpl for OverrideBackend {
    fn blas_jkls_block_tmp_bytes<R, L, Rh, T>(module: &Module<Self>, terms: usize, res: &R, lhs: &L, rhs: &Rh, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        module.blas_jkls_block_tmp_bytes_reference(terms, res, lhs, rhs, tsk)
    }

    fn blas_jkls_block<R, T>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &[&CKKSLeftOperandPrepared<Self>],
        rhs: &[&CKKSRightOperandPrepared<Self>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<Self> + Sync,
    {
        JKLS_CALLS.fetch_add(1, Relaxed);
        module.blas_jkls_block_reference(res, lhs, rhs, tsk, scratch)
    }

    fn blas_jkls_pt_unit_tmp_bytes<R, S, K>(
        module: &Module<Self>,
        outputs: usize,
        res: &R,
        src: &S,
        first: &LinearTransformationPrepared<Self>,
        masks: &[&LinearTransformationPrepared<Self>],
        plaintext: &CKKSRightOperandPrepared<Self>,
        keys: &K,
    ) -> usize
    where
        R: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        module.blas_jkls_pt_unit_tmp_bytes_reference(outputs, res, src, first, masks, plaintext, keys)
    }

    fn blas_jkls_pt_unit<D, S, K>(
        module: &Module<Self>,
        res: &mut [&mut D],
        sources: &[&S],
        first: &LinearTransformationPrepared<Self>,
        masks: &[&LinearTransformationPrepared<Self>],
        plaintexts: &[&[CKKSRightOperandPrepared<Self>]],
        galois: &[i64],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync,
    {
        PT_UNIT_CALLS.fetch_add(1, Relaxed);
        module.blas_jkls_pt_unit_reference(res, sources, first, masks, plaintexts, galois, keys, scratch)
    }
}

/// Parity of the overriding backend with the oracle, every family, through every hook.
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
    let (map, jkls, pt_unit) = calls();
    assert!(
        map > 0 && jkls > 0 && pt_unit > 0,
        "parity entered the map hook {map} times, the JKLS kernel {jkls} times and the masked-unit hook {pt_unit} times"
    );
}

/// Runs `call`, which must succeed, and asserts it entered the map hook, the JKLS kernel and the masked-unit hook
/// `want` times.
fn assert_enters<T>(op: &str, want: (usize, usize, usize), call: impl FnOnce() -> CKKSResult<T>) -> T {
    let before = calls();
    let out = call().unwrap_or_else(|e| panic!("{op}: {e}"));
    let after = calls();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        want,
        "{op}: (map hook, JKLS kernel, masked-unit hook) calls"
    );
    out
}

/// Asserts `result` is an error, and that no hook was entered since the counters were reset.
fn assert_rejected<T>(label: &str, result: CKKSResult<T>) {
    assert!(result.is_err(), "{label}: accepted");
    assert_eq!(calls(), (0, 0, 0), "{label}: a rejected call entered a hook");
}

/// On `2 x 2` grids of `4 x 4` blocks, every operation over a hook enters it as the reference composes it: the
/// transpose and both shifts across blocks one map-hook call, both sums one per line for its projection, the row shift
/// and the row sum of a `1 x 2` grid holding one matrix none, a cold preparation two per block row (sigma or tau, then
/// every shift) and a hot one none, `blas_matmul_prepared` one JKLS call per block of `res` beside two map-hook calls
/// per block row of a hot `a` and per block column of a hot `b`, which it makes the terms of one at a time, and
/// `blas_matmul` those of its two hot sides, each unit once, whether it holds `a` or, beside a `2 x 1` grid, `b`; a
/// JKLS product by public matrices of the `2 x 1` grid enters the map hook once per unit, a block row on the right (two
/// calls) and a block column on the left (one), and a diagonal one never, and a masked one enters the masked-unit hook
/// once per unit, which enters the map hook twice per source block, and, without rotation keys, no hook. Then no call
/// the API rejects enters any hook: every grid [`rejected_grids`] and [`foreign_layouts`] build, a grid of another shape and one at radix 63, as an
/// input or as `res` of each of these operations, the sums included, preparations under either hint and prepared products of every mix (a
/// lone preparation input of another shape is valid; one of another rank than the keys is not), an inner-dimension
/// mismatch, foreign contexts, which a hot side's product rejects too, and key stores without rotation keys, which a hot
/// side's product needs, or without the tensor key.
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
    let foreign = foreign_contexts::<OverrideBackend, f64>(
        &params,
        &module,
        BlasPlan::new(log_d),
        &sk_raw,
        &mut atks,
        &mut scratch.borrow(),
    );
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
    let col = encrypt_grid::<OverrideBackend, f64, EncodingFFTTable<f64>>(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(log_d, 2, 1, slots >> (2 * log_d), 8),
        &mut scratch.borrow(),
    );
    let lanes = slots >> (2 * log_d);
    let one_matrix = Grid::model(log_d, 1, 2, 1, 9);
    let one = encrypt_grid::<OverrideBackend, f64, EncodingFFTTable<f64>>(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        lanes.ilog2() as usize,
        &Grid::from_fn(log_d, 1, 2, lanes, |_, r, c| one_matrix.at(0, r, c)),
        &mut scratch.borrow(),
    );
    let alloc = |rows: usize, cols: usize| alloc_grid(&params, &module, log_d, rows, cols, params.k);
    let (mut res, mut res_col, mut res_one) = (alloc(2, 2), alloc(2, 1), alloc(1, 2));
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let bytes = module
        .blas_matmul_tmp_bytes(&res, &a, &a, &ctx, &tsk_layout, &atk_layout)
        .max(module.blas_matmul_tmp_bytes(&res_col, &a, &col, &ctx, &tsk_layout, &atk_layout))
        .max(module.blas_matmul_left_operand_prepare_tmp_bytes(&a, &ctx, &atk_layout, Reuse))
        .max(module.blas_matmul_right_operand_prepare_tmp_bytes(&a, &ctx, &atk_layout, Reuse))
        .max(module.blas_transpose_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_rotate_rows_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_rotate_cols_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_sum_rows_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_sum_cols_tmp_bytes(&res, &a, &ctx, &atk_layout))
        .max(module.blas_rotate_rows_tmp_bytes(&res_one, &one, &ctx, &atk_layout))
        .max(module.blas_sum_rows_tmp_bytes(&res_one, &one, &ctx, &atk_layout));
    let mut op_scratch = ScratchOwned::<OverrideBackend>::alloc(bytes);

    assert_enters("blas_transpose", (1, 0, 0), || {
        module.blas_transpose(&mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_rotate_rows", (1, 0, 0), || {
        module.blas_rotate_rows(1, &mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_rotate_cols", (1, 0, 0), || {
        module.blas_rotate_cols(1, &mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_sum_rows", (2, 0, 0), || {
        module.blas_sum_rows(&mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_sum_cols", (2, 0, 0), || {
        module.blas_sum_cols(&mut res, &a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_rotate_rows of one matrix", (0, 0, 0), || {
        module.blas_rotate_rows(1, &mut res_one, &one, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_enters("blas_sum_rows of one matrix", (0, 0, 0), || {
        module.blas_sum_rows(&mut res_one, &one, &ctx, &keys, &mut op_scratch.borrow())
    });
    // Hot operands at `side(OneShot)`, cold ones at `side(Reuse)`.
    let side = |hint| usize::from(hint == Reuse);
    let calls = |hint| if hint == Reuse { (4, 0, 0) } else { (0, 0, 0) };
    let left = HINTS.map(|hint| {
        assert_enters(&format!("blas_matmul_left_operand_prepare {hint:?}"), calls(hint), || {
            matmul_left_operand(&module, &a, &ctx, &keys, hint, &mut op_scratch.borrow())
        })
    });
    let right = HINTS.map(|hint| {
        assert_enters(&format!("blas_matmul_right_operand_prepare {hint:?}"), calls(hint), || {
            matmul_right_operand(&module, &a, &ctx, &keys, hint, &mut op_scratch.borrow())
        })
    });
    // A hot side's terms cost the product two map-hook calls per unit, its 2 block rows of `a` or block columns of `b`.
    let hot = |hint| if hint == Reuse { 0 } else { 2 * 2 };
    for (hl, hr) in MIXES {
        let maps = hot(hl) + hot(hr);
        assert_enters(&format!("blas_matmul_prepared {hl:?} x {hr:?}"), (maps, 4, 0), || {
            let (l, r) = (&left[side(hl)], &right[side(hr)]);
            module.blas_matmul_prepared(&mut res, l, r, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    assert_enters("blas_matmul", (8, 4, 0), || {
        module.blas_matmul(&mut res, &a, &a, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });
    // Beside a `2 x 1` grid, `blas_matmul` holds its block column and streams the 2 block rows of `a`.
    assert_enters("blas_matmul by 2 x 1", (6, 2, 0), || {
        module.blas_matmul(&mut res_col, &a, &col, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });
    for algorithm in PT_ALGORITHMS {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let entry = |_: usize, r: usize, c: usize| ((r + c) % 3) as f64 - 1.0;
        let (base2k, meta) = (params.base2k.into(), params.prec().into());
        let pt = pt_operand::<OverrideBackend, f64, Right>(&module, plan, 1, 1, 1, entry, base2k, meta);
        let pt_left = pt_operand::<OverrideBackend, f64, Left>(&module, plan, 2, 2, 1, entry, base2k, meta);
        let (right_calls, left_calls) = if algorithm == BlasPtAlgorithm::Jkls { (2, 1) } else { (0, 0) };
        let bytes = module
            .blas_matmul_pt_right_tmp_bytes(&res_col, &col, &pt, &ctx, &atk_layout)
            .max(module.blas_matmul_pt_left_tmp_bytes(&res_col, &pt_left, &col, &ctx, &atk_layout));
        assert_enters(&format!("blas_matmul_pt_right {algorithm:?}"), (right_calls, 0, 0), || {
            module.blas_matmul_pt_right(&mut res_col, &col, &pt, &ctx, &keys, &mut ScratchOwned::alloc(bytes).borrow())
        });
        assert_enters(&format!("blas_matmul_pt_left {algorithm:?}"), (left_calls, 0, 0), || {
            module.blas_matmul_pt_left(
                &mut res_col,
                &pt_left,
                &col,
                &ctx,
                &keys,
                &mut ScratchOwned::alloc(bytes).borrow(),
            )
        });
    }
    // Under the masked paths, one masked-unit call per block row (column on the left) of `res`, entering the map hook
    // twice per source block, for sigma (tau) and for the masks: 2 rows of one block on the right, 1 column of 2 on the
    // left.
    let order = module.cyclotomic_order();
    for algorithm in MASKED_PT_ALGORITHMS {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let entry = |_: usize, r: usize, c: usize| ((r + c) % 3) as f64 - 1.0;
        let (base2k, meta) = (params.base2k.into(), params.prec().into());
        let masked_ctx = encoded_context::<OverrideBackend, f64>(&module, plan, base2k, meta);
        let elements = plan
            .matmul_pt_right_galois_elements(order)
            .into_iter()
            .chain(plan.matmul_pt_left_galois_elements(order));
        let masked_keys = TestKeys {
            atks: rotation_keys(&params, &module, &sk_raw, elements, &mut scratch.borrow()),
            tsk: None,
        };
        let pt = pt_operand::<OverrideBackend, f64, Right>(&module, plan, 1, 1, 1, entry, base2k, meta);
        let pt_left = pt_operand::<OverrideBackend, f64, Left>(&module, plan, 2, 2, 1, entry, base2k, meta);
        let bytes = module
            .blas_matmul_pt_right_tmp_bytes(&res_col, &col, &pt, &masked_ctx, &atk_layout)
            .max(module.blas_matmul_pt_left_tmp_bytes(&res_col, &pt_left, &col, &masked_ctx, &atk_layout));
        assert_enters(&format!("blas_matmul_pt_right {algorithm:?}"), (4, 0, 2), || {
            module.blas_matmul_pt_right(
                &mut res_col,
                &col,
                &pt,
                &masked_ctx,
                &masked_keys,
                &mut ScratchOwned::alloc(bytes).borrow(),
            )
        });
        assert_enters(&format!("blas_matmul_pt_left {algorithm:?}"), (4, 0, 1), || {
            module.blas_matmul_pt_left(
                &mut res_col,
                &pt_left,
                &col,
                &masked_ctx,
                &masked_keys,
                &mut ScratchOwned::alloc(bytes).borrow(),
            )
        });
        reset();
        let no_keys = TestKeys {
            atks: HashMap::new(),
            tsk: None,
        };
        let s = &mut ScratchOwned::alloc(bytes);
        assert_rejected(
            &format!("no rotation keys: pt_right {algorithm:?}"),
            module.blas_matmul_pt_right(&mut res_col, &col, &pt, &masked_ctx, &no_keys, &mut s.borrow()),
        );
        assert_rejected(
            &format!("no rotation keys: pt_left {algorithm:?}"),
            module.blas_matmul_pt_left(&mut res_col, &pt_left, &col, &masked_ctx, &no_keys, &mut s.borrow()),
        );
    }

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
        rejected("rotate_rows a", module.blas_rotate_rows(1, res, &bad, &ctx, &keys, s));
        rejected("rotate_rows res", module.blas_rotate_rows(1, &mut bad, &a, &ctx, &keys, s));
        rejected("rotate_cols a", module.blas_rotate_cols(1, res, &bad, &ctx, &keys, s));
        rejected("rotate_cols res", module.blas_rotate_cols(1, &mut bad, &a, &ctx, &keys, s));
        rejected("sum_rows a", module.blas_sum_rows(res, &bad, &ctx, &keys, s));
        rejected("sum_rows res", module.blas_sum_rows(&mut bad, &a, &ctx, &keys, s));
        rejected("sum_cols a", module.blas_sum_cols(res, &bad, &ctx, &keys, s));
        rejected("sum_cols res", module.blas_sum_cols(&mut bad, &a, &ctx, &keys, s));
        rejected("matmul a", module.blas_matmul(res, &bad, &a, &ctx, &keys, &keys, s));
        rejected("matmul b", module.blas_matmul(res, &a, &bad, &ctx, &keys, &keys, s));
        rejected("matmul res", module.blas_matmul(&mut bad, &a, &a, &ctx, &keys, &keys, s));
        for (hl, hr) in MIXES {
            let (l, r) = (&left[side(hl)], &right[side(hr)]);
            let call = format!("prepared {hl:?} x {hr:?} res");
            rejected(&call, module.blas_matmul_prepared(&mut bad, l, r, &ctx, &keys, &keys, s));
        }
        if label != "shape" {
            for hint in HINTS {
                assert_rejected(
                    &format!("{label}: prepare_left {hint:?}"),
                    matmul_left_operand(&module, &bad, &ctx, &keys, hint, s),
                );
                assert_rejected(
                    &format!("{label}: prepare_right {hint:?}"),
                    matmul_right_operand(&module, &bad, &ctx, &keys, hint, s),
                );
            }
        }
    }
    let s = &mut op_scratch.borrow();
    let row = alloc(1, 2);
    assert_rejected("inner: matmul", module.blas_matmul(&mut res, &a, &row, &ctx, &keys, &keys, s));
    for (label, ctx) in &foreign {
        let rejected = |call: &str, result: CKKSResult<()>| assert_rejected(&format!("{label} context: {call}"), result);
        rejected("transpose", module.blas_transpose(&mut res, &a, ctx, &keys, s));
        rejected("rotate_rows", module.blas_rotate_rows(1, &mut res, &a, ctx, &keys, s));
        rejected("rotate_cols", module.blas_rotate_cols(1, &mut res, &a, ctx, &keys, s));
        rejected("sum_rows", module.blas_sum_rows(&mut res, &a, ctx, &keys, s));
        rejected("sum_cols", module.blas_sum_cols(&mut res, &a, ctx, &keys, s));
        rejected("matmul", module.blas_matmul(&mut res, &a, &a, ctx, &keys, &keys, s));
        for hint in HINTS {
            assert_rejected(
                &format!("{label} context: prepare_left {hint:?}"),
                matmul_left_operand(&module, &a, ctx, &keys, hint, s),
            );
            assert_rejected(
                &format!("{label} context: prepare_right {hint:?}"),
                matmul_right_operand(&module, &a, ctx, &keys, hint, s),
            );
        }
        // A hot side's terms are evaluated under the product's context, which must read them.
        for (hl, hr) in MIXES.into_iter().filter(|&mix| mix != (Reuse, Reuse)) {
            let (l, r) = (&left[side(hl)], &right[side(hr)]);
            let call = format!("prepared {hl:?} x {hr:?}");
            rejected(&call, module.blas_matmul_prepared(&mut res, l, r, ctx, &keys, &keys, s));
        }
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
    rejected("rotate_rows", module.blas_rotate_rows(1, &mut res, &a, &ctx, &no_atks, s));
    rejected("rotate_cols", module.blas_rotate_cols(1, &mut res, &a, &ctx, &no_atks, s));
    rejected("sum_rows", module.blas_sum_rows(&mut res, &a, &ctx, &no_atks, s));
    rejected("sum_cols", module.blas_sum_cols(&mut res, &a, &ctx, &no_atks, s));
    rejected(
        "rotate_rows of one matrix",
        module.blas_rotate_rows(1, &mut res_one, &one, &ctx, &no_atks, s),
    );
    rejected("matmul", module.blas_matmul(&mut res, &a, &a, &ctx, &no_atks, &no_atks, s));
    for hint in HINTS {
        assert_rejected(
            &format!("no rotation keys: prepare_left {hint:?}"),
            matmul_left_operand(&module, &a, &ctx, &no_atks, hint, s),
        );
        assert_rejected(
            &format!("no rotation keys: prepare_right {hint:?}"),
            matmul_right_operand(&module, &a, &ctx, &no_atks, hint, s),
        );
    }
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        if (hl, hr) != (Reuse, Reuse) {
            rejected(
                &format!("prepared {hl:?} x {hr:?}"),
                module.blas_matmul_prepared(&mut res, l, r, &ctx, &no_atks, &no_atks, s),
            );
        }
        assert_rejected(
            &format!("no tensor key: prepared {hl:?} x {hr:?}"),
            module.blas_matmul_prepared(&mut res, l, r, &ctx, &no_tsk, &no_tsk, s),
        );
    }
    assert_rejected(
        "no tensor key: matmul",
        module.blas_matmul(&mut res, &a, &a, &ctx, &no_tsk, &no_tsk, s),
    );
}
