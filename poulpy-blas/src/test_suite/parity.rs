//! Paired backends on identical fixtures, compared bit for bit through the extension points, which the API calls and
//! then stamps with the output's batch record. Calls that poulpy-ckks rejects for their budget inside the
//! implementation return the same error on both, `res` and the inputs unchanged.
//!
//! Every call, encodings and preparations included, runs under [`with_scratch`]: exactly the bytes its query reports,
//! poisoned and guarded. Storage and ring are free; rank and `dsize` follow `params`.

use poulpy_ckks::{
    CKKSCompositionError, CKKSLayout, CKKSResult, CoeffsMeta, SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    layouts::CKKSModuleAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::TestScalar,
        parity::{
            helpers::{Snapshot, assert_untagged, fixture_ciphertext, with_scratch},
            keys::{key_layout, prepared_automorphism_key, prepared_tensor_key},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{DiagonalArithmetic, GGLWELayout, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena},
};

use super::helpers::{TestKeys, grid_infos, grid_snapshot};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContext, BlasContextUnprepared, BlasGridInfos, BlasLayout, BlasPlan,
        BlasPtOperand, BlasPtOperandUnprepared, sum_galois_elements,
    },
    oep::BlasImpl,
    reference::{Left, blas_matmul_term_layout},
};

/// Bounds of a backend taking part in blas parity: any storage, any ring.
pub trait BlasParityBackend: Backend<ZnxWord = i64> + BlasImpl {}

impl<B: Backend<ZnxWord = i64> + BlasImpl> BlasParityBackend for B {}

/// What blas parity needs from a module: fixtures, prepared keys and maps encoded at scalar `F`.
pub trait BlasParityModule<B: Backend, F: TestScalar>:
    CKKSModuleAlloc<B>
    + CKKSModuleInfos
    + CyclotomicOrder
    + GLWEMaskFill<B>
    + GLWEAutomorphismKeyPreparedFactory<B>
    + GLWETensorKeyPreparedFactory<B>
    + CKKSEncodingOps<B, F>
    + CKKSLinearTransformationOps<B>
    + CnvPVecAlloc<B>
{
}

impl<B: Backend, F: TestScalar, M> BlasParityModule<B, F> for M where
    M: CKKSModuleAlloc<B>
        + CKKSModuleInfos
        + CyclotomicOrder
        + GLWEMaskFill<B>
        + GLWEAutomorphismKeyPreparedFactory<B>
        + GLWETensorKeyPreparedFactory<B>
        + CKKSEncodingOps<B, F>
        + CKKSLinearTransformationOps<B>
        + CnvPVecAlloc<B>
{
}

/// The sweep of [`blas_parity_test_suite!`](crate::blas_parity_test_suite). Narrow it to what a backend supports; never
/// empty it.
#[derive(Clone, Debug)]
pub struct BlasParityShapes {
    /// Block dimensions, as `log2(d)`. One below 1, or whose `d^2` reaches the slot count, is skipped; at least one
    /// must run.
    pub log_d: Vec<usize>,
}

impl Default for BlasParityShapes {
    fn default() -> Self {
        Self { log_d: vec![1, 2] }
    }
}

/// The swept `log_d` whose blocks fit `slots`. The others are skipped with a message; one must remain.
fn swept_log_d(shapes: &BlasParityShapes, slots: usize) -> Vec<usize> {
    // ponytail: d^2 = slots is skipped until the matmul cases derive their counts from the slots (one matrix slot).
    let (run, skipped): (Vec<usize>, Vec<usize>) = shapes
        .log_d
        .iter()
        .partition(|&&log_d| log_d >= 1 && 2 * log_d < slots.ilog2() as usize);
    if !skipped.is_empty() {
        eprintln!("blas parity skips log_d {skipped:?}: it needs 1 <= log_d and d^2 < {slots} slots");
    }
    assert!(
        !run.is_empty(),
        "blas parity runs no log_d of {:?} on {slots} slots",
        shapes.log_d
    );
    run
}

/// Operand width: two diagonal levels and one product at `log_delta = base2k - 1` fit with room to spare.
fn width(params: CKKSTestParams) -> usize {
    6 * params.base2k + 5
}

/// Rotation keys at the operand width, `params.rank` and `params.dsize`.
fn atk_layout(params: CKKSTestParams) -> GGLWELayout {
    key_layout(params.n, params.base2k, width(params), params.dsize, params.rank, params.rank)
}

/// The tensor key: one input per product of two secret components.
fn tsk_layout(params: CKKSTestParams) -> GGLWELayout {
    let pairs = params.rank * (params.rank + 1) / 2;
    key_layout(params.n, params.base2k, width(params), params.dsize, pairs, params.rank)
}

/// A `rows x cols` grid of `2^log_d`-blocks at `params.rank`, block `t` filled from `seed + t`.
fn operand<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let la = layout(params, params.rank, width(params), params.base2k - 1, 0, SlotsKind::Complex);
    grid_of(module, &la, log_d, rows, cols, seed)
}

/// [`operand`] at a width below its scale, `base2k - 1`: poulpy-ckks rejects the budget of any operation into it.
fn narrow<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let la = layout(params, params.rank, params.base2k - 2, 0, 0, SlotsKind::Complex);
    grid_of(module, &la, log_d, rows, cols, seed)
}

fn grid_of<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    la: &CKKSLayout,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let blocks = (0..rows * cols)
        .map(|t| fixture_ciphertext(module, la, seed + t as u8))
        .collect();
    BlasCiphertext::from_blocks(blocks, log_d, rows, cols).unwrap()
}

fn fixture_keys<B>(params: CKKSTestParams, module: &Module<B>, elements: &[i64], tensor: bool) -> TestKeys<B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let key = atk_layout(params);
    TestKeys {
        atks: elements
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, prepared_automorphism_key(module, &key, p, 40 + i as u8)))
            .collect(),
        tsk: tensor.then(|| prepared_tensor_key(module, &tsk_layout(params), 39)),
    }
}

fn diagonal_meta(params: CKKSTestParams) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(params.base2k - 1, params.base2k)
}

fn context<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> BlasContext<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = with_scratch::<B, _>(BlasContextUnprepared::encode_tmp_bytes::<F>(module, log_d), |s| {
        BlasContextUnprepared::encode::<F>(module, BlasPlan::new(log_d), params.base2k.into(), diagonal_meta(params), s)
    })
    .unwrap();
    with_scratch::<B, _>(ctx.prepare_tmp_bytes(module), |s| ctx.prepare(module, s))
}

/// Every output by operation: its batch record and the snapshots of its blocks.
type Outputs = Vec<(&'static str, Option<usize>, Vec<Snapshot>)>;

fn output<B: BlasParityBackend>(op: &'static str, grid: &BlasCiphertextOwned<B>) -> (&'static str, Option<usize>, Vec<Snapshot>) {
    (op, grid.mat_slots(), grid_snapshot::<B, _>(grid))
}

fn inter<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Outputs
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar,
{
    let a = operand(params, module, log_d, 2, 2, 61);
    let mut elements = sum_galois_elements(module, &a);
    elements.extend(BlasPlan::new(log_d).rotate_galois_elements(module.cyclotomic_order(), 3));
    let keys = fixture_keys(params, module, &elements, false);
    let key = atk_layout(params);
    let mut out = Vec::new();
    let mut res = operand(params, module, log_d, 2, 2, 70);
    with_scratch::<B, _>(module.blas_rotate_tmp_bytes(&res, &a, &key), |s| {
        module.blas_rotate(&mut res, &a, 3, &keys, s)
    })
    .unwrap();
    out.push(output::<B>("blas_rotate", &res));
    with_scratch::<B, _>(module.blas_sum_tmp_bytes(&res, &a, &key), |s| {
        module.blas_sum(&mut res, &a, &keys, s)
    })
    .unwrap();
    out.push(output::<B>("blas_sum", &res));
    out
}

/// The errors of calls that fail, by operation.
type Failures = Vec<(&'static str, Option<CKKSCompositionError>)>;

/// Runs `call`, which must fail, and asserts it leaves `res`, its batch record included, and `inputs` unchanged.
fn failure<B: BlasParityBackend>(
    op: &'static str,
    res: &mut BlasCiphertextOwned<B>,
    inputs: &[&BlasCiphertextOwned<B>],
    call: impl FnOnce(&mut BlasCiphertextOwned<B>) -> CKKSResult<()>,
) -> (&'static str, Option<CKKSCompositionError>) {
    let state = |res: &BlasCiphertextOwned<B>| {
        let inputs: Vec<_> = inputs.iter().map(|g| grid_snapshot::<B, _>(g)).collect();
        (output::<B>(op, res), inputs)
    };
    let before = state(res);
    let e = call(res).expect_err(op);
    assert!(state(res) == before, "{op} changed res or an input before failing");
    (op, e.composition().cloned())
}

/// A rotation and a sum into [`narrow`] grids, which poulpy-ckks rejects for their budget inside the implementation.
fn inter_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Failures
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar,
{
    let a = operand(params, module, log_d, 2, 2, 61);
    let mut elements = sum_galois_elements(module, &a);
    elements.extend(BlasPlan::new(log_d).rotate_galois_elements(module.cyclotomic_order(), 3));
    let keys = fixture_keys(params, module, &elements, false);
    let key = atk_layout(params);
    let mut res = narrow(params, module, log_d, 2, 2, 72);
    vec![
        failure::<B>("blas_rotate", &mut res, &[&a], |res| {
            with_scratch::<B, _>(module.blas_rotate_tmp_bytes(res, &a, &key), |s| {
                module.blas_rotate(res, &a, 3, &keys, s)
            })
        }),
        failure::<B>("blas_sum", &mut res, &[&a], |res| {
            with_scratch::<B, _>(module.blas_sum_tmp_bytes(res, &a, &key), |s| {
                module.blas_sum(res, &a, &keys, s)
            })
        }),
    ]
}

fn intra<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Outputs
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module, log_d);
    let keys = fixture_keys(params, module, &ctx.plan().galois_elements(module.cyclotomic_order()), false);
    let key = atk_layout(params);
    let d = 1usize << log_d;
    let mut out = Vec::new();
    let a = operand(params, module, log_d, 2, 3, 80);
    let mut res = operand(params, module, log_d, 3, 2, 90);
    with_scratch::<B, _>(module.blas_transpose_tmp_bytes(&res, &a, &ctx, &key), |s| {
        module.blas_transpose(&mut res, &a, &ctx, &keys, s)
    })
    .unwrap();
    out.push(output::<B>("blas_transpose", &res));
    // k' = 0 is a copy, which the functional tests cover: row shifts k and column shifts k d / 2 keep k' != 0 at
    // every log_d. A one-row grid applies psi^k'.
    let a_row = operand(params, module, log_d, 1, 2, 110);
    let mut res = operand(params, module, log_d, 2, 3, 100);
    let mut res_row = operand(params, module, log_d, 1, 2, 112);
    for k in [1, d + 1] {
        with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(&res, &a, &ctx, &key), |s| {
            module.blas_rotate_rows(&mut res, &a, k, &ctx, &keys, s)
        })
        .unwrap();
        out.push(output::<B>("blas_rotate_rows", &res));
        with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(&res, &a, &ctx, &key), |s| {
            module.blas_rotate_cols(&mut res, &a, k * d / 2, &ctx, &keys, s)
        })
        .unwrap();
        out.push(output::<B>("blas_rotate_cols", &res));
    }
    with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(&res_row, &a_row, &ctx, &key), |s| {
        module.blas_rotate_rows(&mut res_row, &a_row, 1, &ctx, &keys, s)
    })
    .unwrap();
    out.push(output::<B>("blas_rotate_rows one row", &res_row));
    out
}

/// A transpose and both shifts across blocks into [`narrow`] grids, which poulpy-ckks rejects for the budget of their
/// first map inside the implementation.
fn intra_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Failures
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module, log_d);
    let keys = fixture_keys(params, module, &ctx.plan().galois_elements(module.cyclotomic_order()), false);
    let key = atk_layout(params);
    let d = 1usize << log_d;
    let a = operand(params, module, log_d, 2, 3, 80);
    let (mut res, mut res_t) = (
        narrow(params, module, log_d, 2, 3, 100),
        narrow(params, module, log_d, 3, 2, 90),
    );
    vec![
        failure::<B>("blas_transpose", &mut res_t, &[&a], |res| {
            with_scratch::<B, _>(module.blas_transpose_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_transpose(res, &a, &ctx, &keys, s)
            })
        }),
        failure::<B>("blas_rotate_rows", &mut res, &[&a], |res| {
            with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_rotate_rows(res, &a, 1, &ctx, &keys, s)
            })
        }),
        failure::<B>("blas_rotate_cols", &mut res, &[&a], |res| {
            with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_rotate_cols(res, &a, d / 2, &ctx, &keys, s)
            })
        }),
    ]
}

/// What each prepared operand reports through its infos, by operation: never its storage.
type Infos = Vec<(&'static str, (BlasLayout, Option<usize>))>;

/// Entry `(r, c)` of public matrix `m`.
fn public_entry<F: TestScalar>(m: usize, r: usize, c: usize) -> F {
    F::from_f64(((m + r * 4 + c) * 3 % 5) as f64 - 2.0).unwrap()
}

/// The public operand `encode` makes, of `count` matrices per batch, prepared.
fn prepared<B, F, S>(
    module: &Module<B>,
    log_d: usize,
    count: usize,
    encode: impl FnOnce(&mut ScratchArena<'_, B>) -> anyhow::Result<BlasPtOperandUnprepared<B, S>>,
) -> BlasPtOperand<B, S>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let pt = with_scratch::<B, _>(BlasPtOperandUnprepared::encode_tmp_bytes::<F>(module, log_d, count), encode).unwrap();
    with_scratch::<B, _>(pt.prepare_tmp_bytes(module), |s| pt.prepare(module, s))
}

/// A `2 x 1` grid of public matrices of `2^log_d x 2^log_d` blocks, `count` per batch, on the right, prepared.
fn public_operand<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize, count: usize) -> BlasPtOperand<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    prepared::<B, F, _>(module, log_d, count, |s| {
        let base2k = params.base2k.into();
        BlasPtOperandUnprepared::encode::<F>(
            module,
            log_d,
            2,
            1,
            count,
            public_entry::<F>,
            base2k,
            diagonal_meta(params),
            s,
        )
    })
}

/// As [`public_operand`], a `2 x 2` grid on the left.
fn public_left_operand<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize, count: usize) -> BlasPtOperand<B, Left>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    prepared::<B, F, _>(module, log_d, count, |s| {
        let base2k = params.base2k.into();
        BlasPtOperandUnprepared::encode_left::<F>(
            module,
            log_d,
            2,
            2,
            count,
            public_entry::<F>,
            base2k,
            diagonal_meta(params),
            s,
        )
    })
}

/// The products, both prepared operands' infos, which must equal their inputs' term layouts and batches, and the
/// prepared product run twice on the same operands, which must give `blas_matmul`'s blocks both times.
fn matmul<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> (Outputs, Infos)
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let order = module.cyclotomic_order();
    let ctx = context::<B, F>(params, module, log_d);
    // One public matrix for every matrix slot, then two alternating across them, on either side.
    let pts = [1, 2].map(|count| public_operand::<B, F>(params, module, log_d, count));
    let pts_left = [1, 2].map(|count| public_left_operand::<B, F>(params, module, log_d, count));
    let mut elements = ctx.plan().matmul_galois_elements(order);
    elements.extend(ctx.plan().matmul_pt_galois_elements(order));
    elements.extend(ctx.plan().matmul_pt_left_galois_elements(order));
    elements.sort_unstable();
    elements.dedup();
    let keys = fixture_keys(params, module, &elements, true);
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let (a, b) = (
        operand(params, module, log_d, 2, 2, 120),
        operand(params, module, log_d, 2, 1, 124),
    );
    let mut out = Vec::new();
    let mut res = operand(params, module, log_d, 2, 1, 126);
    with_scratch::<B, _>(module.blas_matmul_tmp_bytes(&res, &a, &b, &ctx, &tsk, &atk), |s| {
        module.blas_matmul(&mut res, &a, &b, &ctx, &keys, &keys, s)
    })
    .unwrap();
    out.push(output::<B>("blas_matmul", &res));
    let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk), |s| {
        module.blas_matmul_prepare_left(&a, &ctx, &keys, s)
    })
    .unwrap();
    let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk), |s| {
        module.blas_matmul_prepare_right(&b, &ctx, &keys, s)
    })
    .unwrap();
    let infos: Infos = vec![
        ("blas_matmul_prepare_left", grid_infos(&left)),
        ("blas_matmul_prepare_right", grid_infos(&right)),
    ];
    for ((op, operand), input) in infos.iter().zip([&a, &b]) {
        let term = (blas_matmul_term_layout(input, &ctx), input.mat_slots());
        assert_eq!(
            *operand, term,
            "{op}: the operand's infos are not its input's term layout and batch"
        );
    }
    for seed in [136, 137] {
        let mut res = operand(params, module, log_d, 2, 1, seed);
        let bytes = module.blas_matmul_prepared_tmp_bytes(&res, &left, &right, &tsk);
        with_scratch::<B, _>(bytes, |s| module.blas_matmul_prepared(&mut res, &left, &right, &keys, s)).unwrap();
        let prepared = output::<B>("blas_matmul_prepared", &res);
        assert!(
            prepared.1 == out[0].1 && prepared.2 == out[0].2,
            "blas_matmul_prepared differs from blas_matmul"
        );
        out.push(prepared);
    }
    for pt in &pts {
        let mut res = operand(params, module, log_d, 2, 1, 138);
        with_scratch::<B, _>(module.blas_matmul_pt_tmp_bytes(&res, &a, pt, &atk), |s| {
            module.blas_matmul_pt(&mut res, &a, pt, &keys, s)
        })
        .unwrap();
        out.push(output::<B>("blas_matmul_pt", &res));
    }
    for pt in &pts_left {
        let mut res = operand(params, module, log_d, 2, 1, 139);
        with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(&res, pt, &b, &atk), |s| {
            module.blas_matmul_pt_left(&mut res, pt, &b, &keys, s)
        })
        .unwrap();
        out.push(output::<B>("blas_matmul_pt_left", &res));
    }
    (out, infos)
}

/// A product and a prepared product into a [`narrow`] grid, which poulpy-ckks rejects for the budget of the first
/// block's JKLS kernel inside the implementation, after the preparations, and the products by public matrices on either
/// side, rejected for the budget of their first map: no block of `res` is written.
fn matmul_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Failures
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let order = module.cyclotomic_order();
    let ctx = context::<B, F>(params, module, log_d);
    let mut elements = ctx.plan().matmul_galois_elements(order);
    elements.extend(ctx.plan().matmul_pt_galois_elements(order));
    elements.extend(ctx.plan().matmul_pt_left_galois_elements(order));
    elements.sort_unstable();
    elements.dedup();
    let keys = fixture_keys(params, module, &elements, true);
    let pt = public_operand::<B, F>(params, module, log_d, 1);
    let pt_left = public_left_operand::<B, F>(params, module, log_d, 1);
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let (a, b) = (
        operand(params, module, log_d, 2, 2, 120),
        operand(params, module, log_d, 2, 1, 124),
    );
    let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk), |s| {
        module.blas_matmul_prepare_left(&a, &ctx, &keys, s)
    })
    .unwrap();
    let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk), |s| {
        module.blas_matmul_prepare_right(&b, &ctx, &keys, s)
    })
    .unwrap();
    let mut res = narrow(params, module, log_d, 2, 1, 128);
    vec![
        failure::<B>("blas_matmul", &mut res, &[&a, &b], |res| {
            with_scratch::<B, _>(module.blas_matmul_tmp_bytes(res, &a, &b, &ctx, &tsk, &atk), |s| {
                module.blas_matmul(res, &a, &b, &ctx, &keys, &keys, s)
            })
        }),
        failure::<B>("blas_matmul_prepared", &mut res, &[], |res| {
            with_scratch::<B, _>(module.blas_matmul_prepared_tmp_bytes(res, &left, &right, &tsk), |s| {
                module.blas_matmul_prepared(res, &left, &right, &keys, s)
            })
        }),
        failure::<B>("blas_matmul_pt", &mut res, &[&a], |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_tmp_bytes(res, &a, &pt, &atk), |s| {
                module.blas_matmul_pt(res, &a, &pt, &keys, s)
            })
        }),
        failure::<B>("blas_matmul_pt_left", &mut res, &[&b], |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(res, &pt_left, &b, &atk), |s| {
                module.blas_matmul_pt_left(res, &pt_left, &b, &keys, s)
            })
        }),
    ]
}

fn compare(label: &str, want: Outputs, got: Outputs) {
    assert_untagged(label, want.iter().flat_map(|(_, _, s)| s));
    assert_eq!(want, got, "{label}");
}

/// Matrix-slot rotation and sum agree bit for bit at every swept `log_d`, and fail alike on a budget poulpy-ckks rejects,
/// leaving `res` and `a` unchanged.
pub fn test_blas_inter_parity<BR, BT, F>(params: CKKSTestParams, shapes: &BlasParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, r.ckks_max_slots()) {
        let label = format!("inter, log_d {log_d}");
        compare(&label, inter::<BR, F>(params, r, log_d), inter::<BT, F>(params, t, log_d));
        let (want, got) = (
            inter_budget::<BR, F>(params, r, log_d),
            inter_budget::<BT, F>(params, t, log_d),
        );
        assert_eq!(want, got, "{label}: budget failures");
    }
}

/// Transpose and row and column rotations agree bit for bit at every swept `log_d`, and fail alike on a budget
/// poulpy-ckks rejects, leaving `res` and `a` unchanged.
pub fn test_blas_intra_parity<BR, BT, F>(params: CKKSTestParams, shapes: &BlasParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, r.ckks_max_slots()) {
        let label = format!("intra, log_d {log_d}");
        compare(&label, intra::<BR, F>(params, r, log_d), intra::<BT, F>(params, t, log_d));
        let (want, got) = (
            intra_budget::<BR, F>(params, r, log_d),
            intra_budget::<BT, F>(params, t, log_d),
        );
        assert_eq!(want, got, "{label}: budget failures");
    }
}

/// The products and the prepared product agree bit for bit at every swept `log_d`, and the prepared operands' infos,
/// `(log_d, rows, cols, block_layout, mat_slots)`, agree; prepared storage is never compared. The product, the prepared
/// product and the products by public matrices on either side fail alike on a budget poulpy-ckks rejects, leaving
/// `res` and the factors unchanged.
pub fn test_blas_matmul_parity<BR, BT, F>(params: CKKSTestParams, shapes: &BlasParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, r.ckks_max_slots()) {
        let label = format!("matmul, log_d {log_d}");
        let ((want, want_infos), (got, got_infos)) = (matmul::<BR, F>(params, r, log_d), matmul::<BT, F>(params, t, log_d));
        compare(&label, want, got);
        assert_eq!(want_infos, got_infos, "{label}: prepared operands");
        let (want, got) = (
            matmul_budget::<BR, F>(params, r, log_d),
            matmul_budget::<BT, F>(params, t, log_d),
        );
        assert_eq!(want, got, "{label}: budget failures");
    }
}
