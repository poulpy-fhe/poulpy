//! Paired backends on identical fixtures, compared bit for bit through the extension points, which the API calls and
//! then stamps with the output's batch record. Calls that poulpy-ckks rejects for their budget inside the
//! implementation return the same error on both, `res` and the inputs unchanged.
//!
//! Every call, encodings and preparations included, runs under [`with_scratch`]: exactly the bytes its query reports,
//! poisoned and guarded. Storage and ring are free; rank and `dsize` follow `params`. Each call looks its keys up in a
//! store of its operation's set alone, from [`BlasPlan`] or [`sum_galois_elements`], which records every lookup: the
//! tested backend may look up only keys, at widths, that the reference looked up in the same call. No call writes its
//! inputs, on either backend.

use std::{collections::BTreeSet, sync::Mutex};

use poulpy_ckks::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, CKKSResult, CoeffsMeta, SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    layouts::CKKSModuleAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::TestScalar,
        parity::{
            helpers::{Snapshot, assert_untagged, fixture_operand, with_scratch},
            keys::{key_layout, prepared_automorphism_key, prepared_tensor_key},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEAdd, GLWEMaskFill,
    layouts::{
        DiagonalArithmetic, GGLWELayout, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory, GetAutomorphismKey,
        GetTensorKey, LWEInfos, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
    },
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
    reference::{Left, blas_matmul_term_layout, max_lazy_terms},
};

/// Bounds of a backend taking part in blas parity: any storage, any ring.
pub trait BlasParityBackend: Backend<ZnxWord = i64> + BlasImpl {}

impl<B: Backend<ZnxWord = i64> + BlasImpl> BlasParityBackend for B {}

/// What blas parity needs from a module: fixtures, lazy ones included, prepared keys and maps encoded at scalar `F`.
pub trait BlasParityModule<B: Backend, F: TestScalar>:
    CKKSModuleAlloc<B>
    + CKKSModuleInfos
    + CyclotomicOrder
    + GLWEMaskFill<B>
    + GLWEAdd<B>
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
        + GLWEAdd<B>
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
    /// Block dimensions, as `log2(d)`. One below 1, or whose `d^2` exceeds the slot count, is skipped; at least one
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
    let (run, skipped): (Vec<usize>, Vec<usize>) = shapes
        .log_d
        .iter()
        .partition(|&&log_d| log_d >= 1 && 2 * log_d <= slots.ilog2() as usize);
    if !skipped.is_empty() {
        eprintln!("blas parity skips log_d {skipped:?}: it needs 1 <= log_d and d^2 <= {slots} slots");
    }
    assert!(
        !run.is_empty(),
        "blas parity runs no log_d of {:?} on {slots} slots",
        shapes.log_d
    );
    run
}

/// Operand width: two maps at any scale of [`scale`] and [`other_scales`] and one product at the operands' scale fit
/// with room to spare.
fn width(params: CKKSTestParams) -> usize {
    6 * params.base2k + 5
}

/// The operands' scale, `base2k - 1`, at which the maps and public matrices are encoded unless a case says otherwise.
fn scale(params: CKKSTestParams) -> usize {
    params.base2k - 1
}

/// Scales of maps and public matrices below and above the operands', half a limb off their limb grid.
fn other_scales(params: CKKSTestParams) -> [usize; 2] {
    [params.base2k / 2, params.base2k + params.base2k / 2]
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

/// Blocks at `params.rank` and torus width `k`, at the operands' [`scale`], labelled `log_sparsity`, of slot kind
/// `slots`.
fn block(params: CKKSTestParams, k: usize, log_sparsity: usize, slots: SlotsKind) -> CKKSLayout {
    layout(params, params.rank, k, scale(params), log_sparsity, slots)
}

/// A `rows x cols` grid of `2^log_d`-blocks of layout `la`, block `t` filled from `seed + t`, with lazy digits with
/// `lazy` ([`fixture_operand`]); it records the batch its label proves.
fn grid_of<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    la: &CKKSLayout,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
    lazy: bool,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    let blocks = (0..rows * cols)
        .map(|t| fixture_operand(module, la, seed + t as u8, lazy))
        .collect();
    BlasCiphertext::from_blocks(blocks, log_d, rows, cols).unwrap()
}

/// A `rows x cols` grid at the operand width, dense and complex.
fn operand<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    sized(params, module, log_d, rows, cols, width(params), seed)
}

/// [`operand`] at torus width `k`.
fn sized<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    rows: usize,
    cols: usize,
    k: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    grid_of(
        module,
        &block(params, k, 0, SlotsKind::Complex),
        log_d,
        rows,
        cols,
        seed,
        false,
    )
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
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    let la = layout(params, params.rank, params.base2k - 2, 0, 0, SlotsKind::Complex);
    grid_of(module, &la, log_d, rows, cols, seed, false)
}

/// The label of one matrix per batch, `log2(S / d^2)` on `S` slots.
fn one_slot<B: Backend>(module: &Module<B>, log_d: usize) -> usize
where
    Module<B>: CKKSModuleInfos,
{
    module.ckks_max_slots().ilog2() as usize - 2 * log_d
}

/// A `rows x cols` grid at the operand width with lazy digits ([`fixture_operand`]) on real slots, labelled 1, or 0
/// when one matrix fills the slots.
fn lazy_real<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B> + CKKSModuleInfos,
{
    let la = block(params, width(params), one_slot(module, log_d).min(1), SlotsKind::Real);
    grid_of(module, &la, log_d, rows, cols, seed, true)
}

/// Every rotation key `(Galois element, width)` and every tensor-key width a call looked up.
type Lookups = (BTreeSet<(i64, usize)>, BTreeSet<usize>);

/// [`TestKeys`] that record every lookup. Their maps answer at any width, while a store may answer per width, so each
/// call of the tested backend may look up only keys, at widths, that the same call of the reference looked up.
struct Recorded<B: Backend> {
    keys: TestKeys<B>,
    lookups: Mutex<Lookups>,
}

impl<B: Backend> Recorded<B> {
    /// The lookups since the last call, cleared.
    fn take(&self) -> Lookups {
        std::mem::take(&mut *self.lookups.lock().unwrap())
    }
}

impl<B: Backend> GetAutomorphismKey<B> for Recorded<B>
where
    TestKeys<B>: GetAutomorphismKey<B>,
{
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, B>> {
        self.lookups.lock().unwrap().0.insert((p, k.as_usize()));
        self.keys.lookup_automorphism_key(p, k)
    }
}

impl<B: Backend> GetTensorKey<B> for Recorded<B>
where
    TestKeys<B>: GetTensorKey<B>,
{
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, B>> {
        self.lookups.lock().unwrap().1.insert(k.as_usize());
        self.keys.get_tensor_key(k)
    }
}

/// Rotation keys for `elements` alone, an operation's plan set, and the tensor key with `tensor`, recording every
/// lookup.
fn fixture_keys<B>(params: CKKSTestParams, module: &Module<B>, elements: &[i64], tensor: bool) -> Recorded<B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let key = atk_layout(params);
    let keys = TestKeys {
        atks: elements
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, prepared_automorphism_key(module, &key, p, 40 + i as u8)))
            .collect(),
        tsk: tensor.then(|| prepared_tensor_key(module, &tsk_layout(params), 39)),
    };
    Recorded {
        keys,
        lookups: Mutex::default(),
    }
}

/// Diagonals at scale `p`, with a limb of room above it.
fn diagonal_meta(params: CKKSTestParams, p: usize) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(p, params.base2k)
}

/// The context of `2^log_d x 2^log_d` blocks, its maps at scale `p`.
fn context<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize, p: usize) -> BlasContext<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = with_scratch::<B, _>(BlasContextUnprepared::encode_tmp_bytes::<F>(module, log_d), |s| {
        BlasContextUnprepared::encode::<F>(
            module,
            BlasPlan::new(log_d),
            params.base2k.into(),
            diagonal_meta(params, p),
            s,
        )
    })
    .unwrap();
    with_scratch::<B, _>(ctx.prepare_tmp_bytes(module), |s| ctx.prepare(module, s))
}

/// What one call leaves: the infos of its result, `(layout, batch record)`; the snapshots of its blocks, none for a
/// prepared operand, whose storage is never compared, or for a call that fails, which leaves `res` unchanged; the
/// composition error of a call that fails; and the keys it looked up.
struct Output {
    op: &'static str,
    infos: (BlasLayout, Option<usize>),
    blocks: Vec<Snapshot>,
    error: Option<CKKSCompositionError>,
    lookups: Lookups,
}

/// The [`Output`] of `op` into `res`, with the lookups `keys` recorded since the last output.
fn output<B: BlasParityBackend>(op: &'static str, res: &BlasCiphertextOwned<B>, keys: &Recorded<B>) -> Output {
    Output {
        op,
        infos: grid_infos(res),
        blocks: grid_snapshot::<B, _>(res),
        error: None,
        lookups: keys.take(),
    }
}

/// The snapshots of `inputs`, taken when they are made and again after the last call: no call writes its inputs.
fn snapshots<B: BlasParityBackend>(inputs: &[&BlasCiphertextOwned<B>]) -> Vec<Vec<Snapshot>> {
    inputs.iter().map(|g| grid_snapshot::<B, _>(g)).collect()
}

/// Asserts the calls of `label` left `inputs` as `before`, their [`snapshots`] when made, naming the backend and the
/// first input it wrote.
fn assert_unchanged<B: BlasParityBackend>(label: &str, inputs: &[&BlasCiphertextOwned<B>], before: &[Vec<Snapshot>]) {
    if let Some(i) = snapshots::<B>(inputs).iter().zip(before).position(|(now, made)| now != made) {
        panic!("{label}: {} wrote input {i}", std::any::type_name::<B>());
    }
}

/// Rotations of a `2 x 2` grid by `r` matrix slots, 3 and -1, and by the identities 0 and the batch, copies whose noise
/// tag must clear. Sums over its batch into `res` of its width and one limb narrower, at whose width the rotation keys
/// are looked up, and over one matrix slot, a copy. Rotations by -1 and 0, a copy, and a sum of a [`lazy_real`] grid.
fn inter<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar,
{
    let (plan, order, key, w) = (
        BlasPlan::new(log_d),
        module.cyclotomic_order(),
        atk_layout(params),
        width(params),
    );
    let a = operand(params, module, log_d, 2, 2, 61);
    let one_layout = block(params, w, one_slot(module, log_d), SlotsKind::Complex);
    let one = grid_of(module, &one_layout, log_d, 2, 2, 65, false);
    let lazy = lazy_real(params, module, log_d, 2, 2, 75);
    let inputs = [&a, &one, &lazy];
    let before = snapshots::<B>(&inputs);
    let batch = a.mat_slots().unwrap() as i64;
    let mut out = Vec::new();
    for (src, r) in [(&a, 3), (&a, 0), (&a, batch), (&a, -1), (&lazy, -1), (&lazy, 0)] {
        let keys = fixture_keys(params, module, &plan.rotate_galois_elements(order, r), false);
        let mut res = operand(params, module, log_d, 2, 2, 70);
        with_scratch::<B, _>(module.blas_rotate_tmp_bytes(&res, src, &key), |s| {
            module.blas_rotate(&mut res, src, r, &keys, s)
        })
        .unwrap();
        out.push(output("blas_rotate", &res, &keys));
    }
    for (src, k) in [(&a, w), (&a, w - params.base2k), (&one, w), (&lazy, w)] {
        let keys = fixture_keys(params, module, &sum_galois_elements(module, src), false);
        let mut res = sized(params, module, log_d, 2, 2, k, 70);
        with_scratch::<B, _>(module.blas_sum_tmp_bytes(&res, src, &key), |s| {
            module.blas_sum(&mut res, src, &keys, s)
        })
        .unwrap();
        out.push(output("blas_sum", &res, &keys));
    }
    assert_unchanged::<B>(&format!("inter, log_d {log_d}"), &inputs, &before);
    out
}

/// The [`Output`] of `call` into `res`, which must fail and leave `res`, its batch record included, and `inputs`
/// unchanged, with the lookups `keys` recorded since the last output.
fn failure<B: BlasParityBackend>(
    op: &'static str,
    res: &mut BlasCiphertextOwned<B>,
    inputs: &[&BlasCiphertextOwned<B>],
    keys: &Recorded<B>,
    call: impl FnOnce(&mut BlasCiphertextOwned<B>) -> CKKSResult<()>,
) -> Output {
    let state = |res: &BlasCiphertextOwned<B>| (res.mat_slots(), grid_snapshot::<B, _>(res), snapshots::<B>(inputs));
    let before = state(res);
    let error = call(res).expect_err(op).composition().cloned();
    assert!(state(res) == before, "{op} changed res or an input before failing");
    Output {
        op,
        infos: grid_infos(res),
        blocks: Vec::new(),
        error,
        lookups: keys.take(),
    }
}

/// A rotation and a sum into [`narrow`] grids, which poulpy-ckks rejects for their budget inside the implementation.
fn inter_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar,
{
    let a = operand(params, module, log_d, 2, 2, 61);
    let rotation = BlasPlan::new(log_d).rotate_galois_elements(module.cyclotomic_order(), 3);
    let rotate = fixture_keys(params, module, &rotation, false);
    let sum = fixture_keys(params, module, &sum_galois_elements(module, &a), false);
    let key = atk_layout(params);
    let mut res = narrow(params, module, log_d, 2, 2, 72);
    vec![
        failure::<B>("blas_rotate", &mut res, &[&a], &rotate, |res| {
            with_scratch::<B, _>(module.blas_rotate_tmp_bytes(res, &a, &key), |s| {
                module.blas_rotate(res, &a, 3, &rotate, s)
            })
        }),
        failure::<B>("blas_sum", &mut res, &[&a], &sum, |res| {
            with_scratch::<B, _>(module.blas_sum_tmp_bytes(res, &a, &key), |s| {
                module.blas_sum(res, &a, &sum, s)
            })
        }),
    ]
}

/// Transposes of a `2 x 3` grid, of a [`lazy_real`] grid, and into `res` one limb narrower than its input, below the
/// transpose's width, which caps it and at whose width its giant steps are looked up. Row shifts of a `3 x 2` grid and
/// column shifts of the `2 x 3` grid, three blocks along the shifted axis, by `k`: `k mod d != 0` through both half
/// maps, within the first block and past it (column shifts by `k d / 2` keep `k mod d != 0` at `d = 2`), then `k = d`
/// and `2 d`, copies of the block one and two steps along, whose noise tag must clear. Shifts of a one-row grid (psi)
/// and of a one-column grid (phi), and of the lazy grid: rows by 1, then rows and columns by `d`, copies. Then, through
/// maps at scales below and above the operands' ([`other_scales`]), whose scale they consume in place of the operands',
/// the transpose and the row and column shifts through both half maps.
fn intra<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module, log_d, scale(params));
    let (plan, order, key, w) = (ctx.plan(), module.cyclotomic_order(), atk_layout(params), width(params));
    let d = 1usize << log_d;
    let a = operand(params, module, log_d, 2, 3, 80);
    let tall = operand(params, module, log_d, 3, 2, 123);
    let a_row = operand(params, module, log_d, 1, 2, 110);
    let a_col = operand(params, module, log_d, 2, 1, 113);
    let lazy = lazy_real(params, module, log_d, 2, 3, 117);
    let inputs = [&a, &tall, &a_row, &a_col, &lazy];
    let before = snapshots::<B>(&inputs);
    let mut out = Vec::new();
    let keys = fixture_keys(params, module, &plan.transpose_galois_elements(order), false);
    for (src, k) in [(&a, w), (&lazy, w), (&a, w - params.base2k)] {
        let mut res = sized(params, module, log_d, 3, 2, k, 90);
        with_scratch::<B, _>(module.blas_transpose_tmp_bytes(&res, src, &ctx, &key), |s| {
            module.blas_transpose(&mut res, src, &ctx, &keys, s)
        })
        .unwrap();
        out.push(output("blas_transpose", &res, &keys));
    }
    let shifts = [
        (true, &tall, 1),
        (false, &a, d / 2),
        (true, &tall, d + 1),
        (false, &a, (d + 1) * d / 2),
        (true, &tall, d),
        (false, &a, d),
        (true, &tall, 2 * d),
        (false, &a, 2 * d),
        (true, &a_row, 1),
        (false, &a_col, 1),
        (true, &lazy, 1),
        (true, &lazy, d),
        (false, &lazy, d),
    ];
    let shift = |rows: bool, src: &BlasCiphertextOwned<B>, k: usize, ctx: &BlasContext<B>| {
        let mut res = operand(params, module, log_d, src.rows(), src.cols(), 100);
        if rows {
            let keys = fixture_keys(params, module, &plan.rotate_rows_galois_elements(order, k), false);
            with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_rotate_rows(&mut res, src, k, ctx, &keys, s)
            })
            .unwrap();
            output("blas_rotate_rows", &res, &keys)
        } else {
            let keys = fixture_keys(params, module, &plan.rotate_cols_galois_elements(order, k), false);
            with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_rotate_cols(&mut res, src, k, ctx, &keys, s)
            })
            .unwrap();
            output("blas_rotate_cols", &res, &keys)
        }
    };
    out.extend(shifts.into_iter().map(|(rows, src, k)| shift(rows, src, k, &ctx)));
    for p in other_scales(params) {
        let ctx = context::<B, F>(params, module, log_d, p);
        let mut res = operand(params, module, log_d, 3, 2, 90);
        with_scratch::<B, _>(module.blas_transpose_tmp_bytes(&res, &a, &ctx, &key), |s| {
            module.blas_transpose(&mut res, &a, &ctx, &keys, s)
        })
        .unwrap();
        out.push(output("blas_transpose", &res, &keys));
        out.extend([shift(true, &tall, 1, &ctx), shift(false, &a, d / 2, &ctx)]);
    }
    assert_unchanged::<B>(&format!("intra, log_d {log_d}"), &inputs, &before);
    out
}

/// A transpose and both shifts across blocks into [`narrow`] grids, which poulpy-ckks rejects for the budget of their
/// first map inside the implementation.
fn intra_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module, log_d, scale(params));
    let (plan, order, key) = (ctx.plan(), module.cyclotomic_order(), atk_layout(params));
    let d = 1usize << log_d;
    let transpose = fixture_keys(params, module, &plan.transpose_galois_elements(order), false);
    let rows = fixture_keys(params, module, &plan.rotate_rows_galois_elements(order, 1), false);
    let cols = fixture_keys(params, module, &plan.rotate_cols_galois_elements(order, d / 2), false);
    let a = operand(params, module, log_d, 2, 3, 80);
    let (mut res, mut res_t) = (
        narrow(params, module, log_d, 2, 3, 100),
        narrow(params, module, log_d, 3, 2, 90),
    );
    vec![
        failure::<B>("blas_transpose", &mut res_t, &[&a], &transpose, |res| {
            with_scratch::<B, _>(module.blas_transpose_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_transpose(res, &a, &ctx, &transpose, s)
            })
        }),
        failure::<B>("blas_rotate_rows", &mut res, &[&a], &rows, |res| {
            with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_rotate_rows(res, &a, 1, &ctx, &rows, s)
            })
        }),
        failure::<B>("blas_rotate_cols", &mut res, &[&a], &cols, |res| {
            with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_rotate_cols(res, &a, d / 2, &ctx, &cols, s)
            })
        }),
    ]
}

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

/// A `2 x 1` grid of public matrices of `2^log_d x 2^log_d` blocks, `count` per batch, at scale `p`, on the right,
/// prepared.
fn public_operand<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize, count: usize, p: usize) -> BlasPtOperand<B>
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
            diagonal_meta(params, p),
            s,
        )
    })
}

/// As [`public_operand`], a `2 x 2` grid on the left.
fn public_left_operand<B, F>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    count: usize,
    p: usize,
) -> BlasPtOperand<B, Left>
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
            diagonal_meta(params, p),
            s,
        )
    })
}

/// `blas_matmul` of `a` by `b` into a fixture of width `k`.
fn product<B: BlasParityBackend>(
    params: CKKSTestParams,
    module: &Module<B>,
    ctx: &BlasContext<B>,
    keys: &Recorded<B>,
    (a, b): (&BlasCiphertextOwned<B>, &BlasCiphertextOwned<B>),
    k: usize,
) -> Output
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    let mut res = sized(params, module, a.log_d(), a.rows(), b.cols(), k, 126);
    let bytes = module.blas_matmul_tmp_bytes(&res, a, b, ctx, &tsk_layout(params), &atk_layout(params));
    with_scratch::<B, _>(bytes, |s| module.blas_matmul(&mut res, a, b, ctx, keys, keys, s)).unwrap();
    output("blas_matmul", &res, keys)
}

/// The [`Output`] of the preparation `op` of `input`: the operand's infos, which must be its input's term layout and
/// batch.
fn preparation<B: BlasParityBackend>(
    op: &'static str,
    operand: &impl BlasGridInfos,
    input: &BlasCiphertextOwned<B>,
    ctx: &BlasContext<B>,
    keys: &Recorded<B>,
) -> Output {
    let infos = grid_infos(operand);
    assert_eq!(
        infos,
        (blas_matmul_term_layout(input, ctx), input.mat_slots()),
        "{op}: the operand's infos are not its input's term layout and batch"
    );
    Output {
        op,
        infos,
        blocks: Vec::new(),
        error: None,
        lookups: keys.take(),
    }
}

/// Both preparations of a `2 x 2` grid and a `2 x 1` grid, then their product into `res` of their width, one limb
/// wider, and one limb narrower than the product's natural width, which the kernel's offset absorbs: each time by
/// `blas_matmul` and by `blas_matmul_prepared` on the one pair of prepared operands, which must agree; all of it
/// through maps at the operands' scale, then at scales below and above it ([`other_scales`]), whose scale the
/// preparations consume twice in place of the operands'. Through the latter, the same for factors at two scales, in
/// either order, into `res` as wide as the wider factor: one two bits lower and either as much narrower, so that their
/// terms keep one budget, or as wide, so that its terms have the larger budget (crossed scales), where the larger
/// budget plus the larger scale exceeds the wider term's width: a kernel taking its offset from that width, not from
/// that sum, changes the bits. Products into `2 x 2` grids, of factors one limb apart, in both orders,
/// since the tensor key is looked up at the wider side's width, and of a lazy grid of real slots labelled 1 by itself.
/// `1 x K` by `K x 1` products past the `2^(63 - base2k)` terms the kernel sums between normalizations pin that the
/// kernel accepts such a sum and returns the reference's bits: one of the fewest distinct blocks whose `K d` terms pass
/// the bound, where some `K <= 32` does, and, from radix 57 (58 at `log_d` 1), one of `3 * 2^(63 - base2k) / min(d, 4)`
/// copies of one block, at least three times the bound in terms. Every copy adds the same sum of `d` terms, so the
/// copies overflow an `i64` digit in a kernel that normalizes only at the end; since a term equals every term a
/// multiple of `d` away, only the distinct blocks see a kernel that sums one in place of another. The products by
/// public matrices on either side: of one matrix, of one per matrix slot, and of one on the lazy grid into `res` one
/// limb narrower than it, below the products' width, which caps them and at whose width their giant steps are looked
/// up, at the operands' scale; then of one matrix at each other scale.
fn matmul<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let (order, base2k, w, delta) = (module.cyclotomic_order(), params.base2k, width(params), scale(params));
    // The maps at the operands' scale, then at scales below and above it.
    let [low, high] = other_scales(params);
    let scales = [delta, low, high];
    let contexts = scales.map(|p| context::<B, F>(params, module, log_d, p));
    let ctx = &contexts[0];
    let plan = ctx.plan();
    let products = fixture_keys(params, module, &plan.matmul_galois_elements(order), true);
    let tensor = fixture_keys(params, module, &[], true);
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let a = operand(params, module, log_d, 2, 2, 120);
    let b = operand(params, module, log_d, 2, 1, 124);
    // `a` and `b` two bits below their scale: either times the other multiplies factors at two scales, as much
    // narrower, whose terms keep one budget, and as wide, whose terms at the lower scale have the larger budget
    // (crossed scales).
    let [(a_low, b_low), (a_crossed, b_crossed)] = [(w - 2, 128), (w, 192)].map(|(k, seed)| {
        let below = layout(params, params.rank, k, delta - 2, 0, SlotsKind::Complex);
        (
            grid_of(module, &below, log_d, 2, 2, seed, false),
            grid_of(module, &below, log_d, 2, 1, seed + 4, false),
        )
    });
    let thin = sized(params, module, log_d, 2, 2, w - base2k, 140);
    let lazy = lazy_real(params, module, log_d, 2, 2, 144);
    // Past the bound: the fewest distinct inner blocks whose `K d` terms pass it, and `3 bound / min(d, 4)` copies of
    // one block. Every copy adds the same sum of `d` distinct terms, whose digits grow like `sqrt(d)` times a term's,
    // so in a kernel that normalizes only at the end the copies overflow about 1.6% of the `i64` digits at `d = 4`,
    // and more at any other `d`.
    let (bound, d) = (max_lazy_terms(base2k), 1usize << log_d);
    let (inner, copies) = (bound / d + 1, (3 * bound).div_ceil(d.min(4)));
    let repeat = |seed, rows, cols| {
        let la = block(params, w, 0, SlotsKind::Complex);
        let blocks = (0..rows * cols).map(|_| fixture_operand(module, &la, seed, false)).collect();
        BlasCiphertext::from_blocks(blocks, log_d, rows, cols).unwrap()
    };
    let mut long = Vec::new();
    if inner <= 32 {
        long.push((
            operand(params, module, log_d, 1, inner, 160),
            operand(params, module, log_d, inner, 1, 200),
        ));
    }
    if copies <= 48 {
        long.push((repeat(150, 1, copies), repeat(152, copies, 1)));
    }
    let mut inputs = vec![&a, &b, &a_low, &b_low, &a_crossed, &b_crossed, &thin, &lazy];
    inputs.extend(long.iter().flat_map(|(x, y)| [x, y]));
    let before = snapshots::<B>(&inputs);
    let mut out = Vec::new();
    for (p, ctx) in scales.into_iter().zip(&contexts) {
        // Factors of one width and scale: poulpy-ckks's ct x ct rule leaves the terms' width less their scale.
        let term = blas_matmul_term_layout(&a, ctx).block;
        let natural = term.k().as_usize() - term.log_delta();
        let mut pairs = vec![(&a, &b, vec![w, w + base2k, natural - base2k])];
        // Through maps at a scale apart from both, factors at two scales, in either order, crossed or not.
        if p != delta {
            pairs.extend([
                (&a, &b_low, vec![w]),
                (&a_low, &b, vec![w]),
                (&a, &b_crossed, vec![w]),
                (&a_crossed, &b, vec![w]),
            ]);
        }
        for (x, y, widths) in pairs {
            let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(x, ctx, &atk), |s| {
                module.blas_matmul_prepare_left(x, ctx, &products, s)
            })
            .unwrap();
            out.push(preparation("blas_matmul_prepare_left", &left, x, ctx, &products));
            let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(y, ctx, &atk), |s| {
                module.blas_matmul_prepare_right(y, ctx, &products, s)
            })
            .unwrap();
            out.push(preparation("blas_matmul_prepare_right", &right, y, ctx, &products));
            for k in widths {
                let from_factors = product(params, module, ctx, &products, (x, y), k);
                let mut res = sized(params, module, log_d, 2, 1, k, 136);
                let bytes = module.blas_matmul_prepared_tmp_bytes(&res, &left, &right, &tsk);
                with_scratch::<B, _>(bytes, |s| module.blas_matmul_prepared(&mut res, &left, &right, &tensor, s)).unwrap();
                let from_operands = output("blas_matmul_prepared", &res, &tensor);
                assert!(
                    from_operands.infos == from_factors.infos && from_operands.blocks == from_factors.blocks,
                    "blas_matmul_prepared differs from blas_matmul into res of width {k}, maps at scale {p}"
                );
                out.extend([from_factors, from_operands]);
            }
        }
    }
    for factors in [(&a, &thin), (&thin, &a), (&lazy, &lazy)]
        .into_iter()
        .chain(long.iter().map(|(x, y)| (x, y)))
    {
        out.push(product(params, module, ctx, &products, factors, w));
    }
    let pt_keys = fixture_keys(params, module, &plan.matmul_pt_galois_elements(order), false);
    let left_keys = fixture_keys(params, module, &plan.matmul_pt_left_galois_elements(order), false);
    let batch = a.mat_slots().unwrap();
    let mut cases = vec![(&a, &b, 1, w, delta)];
    if batch > 1 {
        cases.push((&a, &b, batch, w, delta));
    }
    cases.extend([
        (&lazy, &lazy, 1, w - base2k, delta),
        (&a, &b, 1, w, low),
        (&a, &b, 1, w, high),
    ]);
    for (x, y, count, k, p) in cases {
        let pt = public_operand::<B, F>(params, module, log_d, count, p);
        let mut res = sized(params, module, log_d, x.rows(), pt.cols(), k, 138);
        with_scratch::<B, _>(module.blas_matmul_pt_tmp_bytes(&res, x, &pt, &atk), |s| {
            module.blas_matmul_pt(&mut res, x, &pt, &pt_keys, s)
        })
        .unwrap();
        out.push(output("blas_matmul_pt", &res, &pt_keys));
        let pt = public_left_operand::<B, F>(params, module, log_d, count, p);
        let mut res = sized(params, module, log_d, pt.rows(), y.cols(), k, 139);
        with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(&res, &pt, y, &atk), |s| {
            module.blas_matmul_pt_left(&mut res, &pt, y, &left_keys, s)
        })
        .unwrap();
        out.push(output("blas_matmul_pt_left", &res, &left_keys));
    }
    assert_unchanged::<B>(&format!("matmul, log_d {log_d}"), &inputs, &before);
    out
}

/// The preparations of a `2 x 2` and a `2 x 1` grid, then their product and their prepared product into a [`narrow`]
/// grid, which poulpy-ckks rejects for the budget of the first block's JKLS kernel inside the implementation, after the
/// preparations, and the products by public matrices on either side, rejected for the budget of their first map: no
/// block of `res` is written.
fn matmul_budget<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let order = module.cyclotomic_order();
    let ctx = context::<B, F>(params, module, log_d, scale(params));
    let plan = ctx.plan();
    let products = fixture_keys(params, module, &plan.matmul_galois_elements(order), true);
    let tensor = fixture_keys(params, module, &[], true);
    let pt_keys = fixture_keys(params, module, &plan.matmul_pt_galois_elements(order), false);
    let left_keys = fixture_keys(params, module, &plan.matmul_pt_left_galois_elements(order), false);
    let pt = public_operand::<B, F>(params, module, log_d, 1, scale(params));
    let pt_left = public_left_operand::<B, F>(params, module, log_d, 1, scale(params));
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let (a, b) = (
        operand(params, module, log_d, 2, 2, 120),
        operand(params, module, log_d, 2, 1, 124),
    );
    let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk), |s| {
        module.blas_matmul_prepare_left(&a, &ctx, &products, s)
    })
    .unwrap();
    let mut out = vec![preparation("blas_matmul_prepare_left", &left, &a, &ctx, &products)];
    let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk), |s| {
        module.blas_matmul_prepare_right(&b, &ctx, &products, s)
    })
    .unwrap();
    out.push(preparation("blas_matmul_prepare_right", &right, &b, &ctx, &products));
    let mut res = narrow(params, module, log_d, 2, 1, 128);
    out.extend([
        failure::<B>("blas_matmul", &mut res, &[&a, &b], &products, |res| {
            with_scratch::<B, _>(module.blas_matmul_tmp_bytes(res, &a, &b, &ctx, &tsk, &atk), |s| {
                module.blas_matmul(res, &a, &b, &ctx, &products, &products, s)
            })
        }),
        failure::<B>("blas_matmul_prepared", &mut res, &[], &tensor, |res| {
            with_scratch::<B, _>(module.blas_matmul_prepared_tmp_bytes(res, &left, &right, &tsk), |s| {
                module.blas_matmul_prepared(res, &left, &right, &tensor, s)
            })
        }),
        failure::<B>("blas_matmul_pt", &mut res, &[&a], &pt_keys, |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_tmp_bytes(res, &a, &pt, &atk), |s| {
                module.blas_matmul_pt(res, &a, &pt, &pt_keys, s)
            })
        }),
        failure::<B>("blas_matmul_pt_left", &mut res, &[&b], &left_keys, |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(res, &pt_left, &b, &atk), |s| {
                module.blas_matmul_pt_left(res, &pt_left, &b, &left_keys, s)
            })
        }),
    ]);
    out
}

/// Asserts the tested backend's outputs, `got`, equal the reference's, `want`, call by call, errors included, except
/// for the keys: each call of `got` may look up only keys, at widths, that the same call of `want` looked up.
fn compare(label: &str, want: Vec<Output>, got: Vec<Output>) {
    assert_untagged(label, want.iter().flat_map(|o| &o.blocks));
    assert_eq!(want.len(), got.len(), "{label}: calls");
    for (i, (want, got)) in want.into_iter().zip(got).enumerate() {
        let call = format!("{label}, call {i} ({})", want.op);
        assert_eq!(
            (want.op, want.infos, want.blocks, want.error),
            (got.op, got.infos, got.blocks, got.error),
            "{call}"
        );
        let ((want_atks, want_tsks), (got_atks, got_tsks)) = (want.lookups, got.lookups);
        assert!(
            got_atks.is_subset(&want_atks) && got_tsks.is_subset(&want_tsks),
            "{call}: looked up rotation keys {got_atks:?} and tensor keys at {got_tsks:?}, the reference {want_atks:?} and \
             {want_tsks:?}"
        );
    }
}

/// Matrix-slot rotations and sums agree bit for bit at every swept `log_d`, through the identity, one matrix slot, a
/// narrower `res`, and lazy real inputs, and fail alike on a budget poulpy-ckks rejects, leaving `res` and `a`
/// unchanged.
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
        compare(&format!("{label}, budget"), want, got);
    }
}

/// Transposes and row and column rotations agree bit for bit at every swept `log_d`, through every branch of the
/// shifts, a narrower `res`, lazy real inputs, and maps at scales other than the inputs', and fail alike on a budget
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
        compare(&format!("{label}, budget"), want, got);
    }
}

/// The products, the prepared product and the products by public matrices agree bit for bit at every swept `log_d`,
/// through maps and public matrices at the factors' scale and at scales other than theirs, there also of factors at two
/// scales in either order, the terms at the lower scale with the other's budget or a larger one (crossed scales), and
/// the prepared operands' infos, `(log_d, rows, cols, block_layout, mat_slots)`, agree; prepared storage is never
/// compared. Products past the `2^(63 - base2k)` terms the JKLS kernel sums between normalizations agree too: of
/// distinct blocks, where some `K <= 32` of them pass that bound, and, from radix 57 (58 at `log_d` 1), of copies of
/// one block, which overflow an `i64` digit in a kernel that normalizes only at the end.
/// The product, the prepared product and the products by public matrices on either side fail alike on a budget
/// poulpy-ckks rejects, leaving `res` and the factors unchanged.
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
        compare(&label, matmul::<BR, F>(params, r, log_d), matmul::<BT, F>(params, t, log_d));
        let (want, got) = (
            matmul_budget::<BR, F>(params, r, log_d),
            matmul_budget::<BT, F>(params, t, log_d),
        );
        compare(&format!("{label}, budget"), want, got);
    }
}
