//! Paired backends on identical fixtures, compared bit for bit through the extension points, which the API calls and
//! which stamp the output's batch record. Calls that poulpy-ckks rejects for their budget inside the implementation
//! return the same error on both, `res` and the inputs unchanged.
//!
//! Every call, encodings and preparations included, runs under [`with_scratch`]: exactly the bytes its query reports,
//! poisoned and guarded. Storage and ring are free; rank and `dsize` follow `params`. Each call looks its keys up in a
//! store of its operation's set alone, from [`BlasPlan`] or [`sum_galois_elements`], which records every lookup: the
//! tested backend may look up only keys, at widths, that the reference looked up in the same call. No call writes its
//! inputs, on either backend.
//!
//! Two tests use the same fixtures on one backend: [`test_blas_concurrency`] runs calls at once and compares each with
//! the same call run alone, and [`test_blas_max_base2k`] checks the radix helpers of the products against counts
//! made by hand.

use std::{
    collections::BTreeSet,
    sync::{Barrier, Mutex},
};

use poulpy_ckks::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, CKKSResult, CoeffsMeta, SlotsKind,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps, LinearTransformationLayout, LinearTransformationStrategy,
    },
    layouts::{CKKSModuleAlloc, CKKSPreparedRight},
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
    GLWEAdd, GLWEMaskFill, GLWENormalize,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        DiagonalArithmetic, GGLWELayout, GetAutomorphismKey, GetDegree, GetTensorKey, LWEInfos, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{
        Backend, CyclotomicOrder, MaxBase2k, Module, PrepareHint,
        PrepareHint::{OneShot, Reuse},
    },
};

use super::{
    helpers::{
        HINTS, MASKED_PT_ALGORITHMS, MIXES, PT_ALGORITHMS, TestKeys, assert_panics, grid_infos, grid_snapshot,
        matmul_left_operand, matmul_right_operand,
    },
    matvec::{
        append, append_tmp_bytes, diagonal_ciphertext_elements, diagonal_ciphertext_snapshot, matvec_layouts, matvec_operands,
        prepared_of, vector_snapshot,
    },
};
use crate::{
    api::{
        BlasContextPreparedFactory, BlasEncodingOps, BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps, BlasMatVecOps,
        BlasPtOperandPreparedFactory,
    },
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextInfos, BlasContextLayout, BlasContextPrepared, BlasDiagonalCiphertext,
        BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextLayout, BlasGridInfos, BlasLayout, BlasMatMulOperandInfos,
        BlasMatVecOperandLayout, BlasModuleAlloc, BlasPlan, BlasPtAlgorithm, BlasPtOperandInfos, BlasPtOperandLayout,
        BlasPtOperandPrepared, BlasPtSide, BlasVector, BlasVectorInfos, BlasVectorLayout, BlasVectorOwned, BlasVectorShape, Left,
        Right, blas_matmul_left_term_layout, blas_matmul_max_base2k, blas_matmul_pt_left_max_base2k,
        blas_matmul_pt_right_max_base2k, blas_matmul_right_term_layout, blas_matvec_diagonal_max_base2k,
        blas_matvec_pt_max_base2k,
        context::module_layout,
        diagonal_ciphertext::AppendAxis,
        matvec_pt_galois_elements, sum_galois_elements,
        validate::{MASKLESS, MAX_BASE2K},
    },
    oep::BlasImpl,
    reference::{sum_line, sum_line_tmp_bytes},
};

/// Bounds of a backend taking part in blas parity: any storage, any ring.
pub trait BlasParityBackend: Backend<ZnxWord = i64> + BlasImpl {}

impl<B: Backend<ZnxWord = i64> + BlasImpl> BlasParityBackend for B {}

/// What blas parity needs from a module: fixtures, lazy ones included, prepared keys and maps encoded at scalar `F`.
pub trait BlasParityModule<B: Backend, F: TestScalar + DiagonalArithmetic>:
    CKKSModuleAlloc<B>
    + CKKSModuleInfos
    + CyclotomicOrder
    + GLWEMaskFill<B>
    + GLWEAdd<B>
    + GLWEAutomorphismKeyPreparedFactory<B>
    + GLWETensorKeyPreparedFactory<B>
    + CKKSEncodingOps<B, F>
    + BlasEncodingOps<B, F>
    + CKKSLinearTransformationOps<B>
    + CKKSCopyOps<B>
    + CKKSAddOps<B>
    + CKKSAddManyOps<B>
    + GLWENormalize<B>
    + CnvPVecAlloc<B>
{
}

impl<B: Backend, F: TestScalar + DiagonalArithmetic, M> BlasParityModule<B, F> for M where
    M: CKKSModuleAlloc<B>
        + CKKSModuleInfos
        + CyclotomicOrder
        + GLWEMaskFill<B>
        + GLWEAdd<B>
        + GLWEAutomorphismKeyPreparedFactory<B>
        + GLWETensorKeyPreparedFactory<B>
        + CKKSEncodingOps<B, F>
        + BlasEncodingOps<B, F>
        + CKKSLinearTransformationOps<B>
        + CKKSCopyOps<B>
        + CKKSAddOps<B>
        + CKKSAddManyOps<B>
        + GLWENormalize<B>
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
    grid_of(module, &layout_of_narrow(params), log_d, rows, cols, seed, false)
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
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, B>> {
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

/// The context of `2^log_d x 2^log_d` blocks, its maps at scale `p`, holding the masks of the masked products.
fn context<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize, p: usize) -> BlasContextPrepared<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let layout = BlasContextLayout {
        plan: BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois),
        base2k: params.base2k.into(),
        coeffs_meta: diagonal_meta(params, p),
        n: module.ring_degree(),
        slots: module.ckks_max_slots(),
    };
    let mut ctx = module.blas_context_alloc_from_infos(&layout);
    let mut prepared = module.blas_context_prepared_alloc_from_infos(&layout);
    with_scratch::<B, _>(module.blas_context_encode_tmp_bytes(&layout), |s| {
        module.blas_context_encode(&mut ctx, s).unwrap()
    });
    with_scratch::<B, _>(module.blas_context_prepare_tmp_bytes(&layout), |s| {
        module.blas_context_prepare(&mut prepared, &ctx, s)
    });
    prepared
}

/// What one call leaves: the infos of its result, `(layout, batch record)`; the snapshots of its blocks, none for a
/// prepared operand, whose storage is never compared, or for a call that fails, which leaves `res` unchanged; the
/// composition error of a call that fails; and the keys it looked up.
struct Output {
    op: &'static str,
    infos: Infos,
    blocks: Vec<Snapshot>,
    error: Option<CKKSCompositionError>,
    lookups: Lookups,
}

/// The infos of a call's result: a grid's `(layout, batch record)`, or a vector's layout.
#[derive(Debug, PartialEq)]
enum Infos {
    Grid((BlasLayout, Option<usize>)),
    Vector(BlasVectorLayout),
    Matrix(BlasDiagonalCiphertextLayout),
}

/// The [`Output`] of `op` into `res`, with the lookups `keys` recorded since the last output.
fn output<B: BlasParityBackend>(op: &'static str, res: &BlasCiphertextOwned<B>, keys: &Recorded<B>) -> Output {
    Output {
        op,
        infos: Infos::Grid(grid_infos(res)),
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
    F: TestScalar + DiagonalArithmetic,
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
            module.blas_rotate(r, &mut res, src, &keys, s)
        })
        .unwrap();
        out.push(output("blas_rotate", &res, &keys));
    }
    for (src, k) in [(&a, w), (&a, w - params.base2k), (&one, w), (&lazy, w)] {
        let keys = fixture_keys(params, module, &sum_galois_elements(module.cyclotomic_order(), src), false);
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
        infos: Infos::Grid(grid_infos(res)),
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
    F: TestScalar + DiagonalArithmetic,
{
    let a = operand(params, module, log_d, 2, 2, 61);
    let rotation = BlasPlan::new(log_d).rotate_galois_elements(module.cyclotomic_order(), 3);
    let rotate = fixture_keys(params, module, &rotation, false);
    let sum = fixture_keys(params, module, &sum_galois_elements(module.cyclotomic_order(), &a), false);
    let key = atk_layout(params);
    let mut res = narrow(params, module, log_d, 2, 2, 72);
    vec![
        failure::<B>("blas_rotate", &mut res, &[&a], &rotate, |res| {
            with_scratch::<B, _>(module.blas_rotate_tmp_bytes(res, &a, &key), |s| {
                module.blas_rotate(3, res, &a, &rotate, s)
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
/// and of a one-column grid (phi), and of the lazy grid: rows by 1, then rows and columns by `d`, copies. Shifts of a
/// one-row grid labelled one matrix: rows by 1 and `d + 1`, rotations, and columns by 1 (phi). The row and column sums
/// of every grid, of the replicated grids, whose rows sum without the projection, and into `res` one limb narrower, and
/// their line sum of lazy, mixed and canonical blocks at bounds that normalize every addition or go block by block.
/// Then, through maps at scales below and above the operands' ([`other_scales`]), whose scale they consume in place of
/// the operands', the transpose, the row and column shifts through both half maps and both sums.
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
    let one_layout = block(params, w, one_slot(module, log_d), SlotsKind::Complex);
    let one = grid_of(module, &one_layout, log_d, 2, 3, 127, false);
    let one_row = grid_of(module, &one_layout, log_d, 1, 2, 135, false);
    let inputs = [&a, &tall, &a_row, &a_col, &lazy, &one, &one_row];
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
        (true, &one_row, 1),
        (true, &one_row, d + 1),
        (false, &one_row, 1),
    ];
    let shift = |rows: bool, src: &BlasCiphertextOwned<B>, k: usize, ctx: &BlasContextPrepared<B>| {
        let mut res = operand(params, module, log_d, src.rows(), src.cols(), 100);
        if rows {
            let keys = fixture_keys(params, module, &plan.rotate_rows_galois_elements(order, k), false);
            with_scratch::<B, _>(module.blas_rotate_rows_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_rotate_rows(k, &mut res, src, ctx, &keys, s)
            })
            .unwrap();
            output("blas_rotate_rows", &res, &keys)
        } else {
            let keys = fixture_keys(params, module, &plan.rotate_cols_galois_elements(order, k), false);
            with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_rotate_cols(k, &mut res, src, ctx, &keys, s)
            })
            .unwrap();
            output("blas_rotate_cols", &res, &keys)
        }
    };
    out.extend(shifts.into_iter().map(|(rows, src, k)| shift(rows, src, k, &ctx)));
    let sum = |rows: bool, src: &BlasCiphertextOwned<B>, k: usize, ctx: &BlasContextPrepared<B>| {
        let mut res = sized(params, module, log_d, src.rows(), src.cols(), k, 100);
        if rows {
            let keys = fixture_keys(params, module, &plan.sum_rows_galois_elements(order), false);
            with_scratch::<B, _>(module.blas_sum_rows_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_sum_rows(&mut res, src, ctx, &keys, s)
            })
            .unwrap();
            output("blas_sum_rows", &res, &keys)
        } else {
            let keys = fixture_keys(params, module, &plan.sum_cols_galois_elements(order), false);
            with_scratch::<B, _>(module.blas_sum_cols_tmp_bytes(&res, src, ctx, &key), |s| {
                module.blas_sum_cols(&mut res, src, ctx, &keys, s)
            })
            .unwrap();
            output("blas_sum_cols", &res, &keys)
        }
    };
    let narrower = w - params.base2k;
    let sums = [&a, &tall, &a_row, &a_col, &lazy, &one, &one_row]
        .map(|src| (src, w))
        .into_iter()
        .chain([(&a, narrower), (&one, narrower)]);
    for (src, k) in sums {
        out.extend([sum(true, src, k, &ctx), sum(false, src, k, &ctx)]);
    }
    // The line sum at bounds no registered radix reaches: where `max_lazy_terms` binds, at radix 61 and 62, no backend
    // key-switches exactly.
    let (lazy_line, canonical_line): (Vec<_>, Vec<_>) = (lazy.blocks().iter().collect(), a.blocks().iter().collect());
    let mixed_line: Vec<_> = lazy_line.iter().zip(&canonical_line).flat_map(|(l, c)| [*l, *c]).collect();
    let layout = a.block_layout();
    for (line, terms) in [(&lazy_line, 4), (&lazy_line, 3), (&mixed_line, 2), (&canonical_line, 2)] {
        let mut res = operand(params, module, log_d, 1, 1, 100);
        with_scratch::<B, _>(sum_line_tmp_bytes(module, &layout, terms), |s| {
            sum_line(module, &mut res.blocks_mut()[0], line, terms, s)
        })
        .unwrap();
        out.push(output("sum_line", &res, &keys));
    }
    for p in other_scales(params) {
        let ctx = context::<B, F>(params, module, log_d, p);
        let mut res = operand(params, module, log_d, 3, 2, 90);
        with_scratch::<B, _>(module.blas_transpose_tmp_bytes(&res, &a, &ctx, &key), |s| {
            module.blas_transpose(&mut res, &a, &ctx, &keys, s)
        })
        .unwrap();
        out.push(output("blas_transpose", &res, &keys));
        out.extend([shift(true, &tall, 1, &ctx), shift(false, &a, d / 2, &ctx)]);
        out.extend([sum(true, &a, w, &ctx), sum(false, &a, w, &ctx)]);
    }
    assert_unchanged::<B>(&format!("intra, log_d {log_d}"), &inputs, &before);
    out
}

/// A transpose, both shifts across blocks and both sums into [`narrow`] grids, which poulpy-ckks rejects for the budget
/// of their first map inside the implementation.
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
    let sum_rows = fixture_keys(params, module, &plan.sum_rows_galois_elements(order), false);
    let sum_cols = fixture_keys(params, module, &plan.sum_cols_galois_elements(order), false);
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
                module.blas_rotate_rows(1, res, &a, &ctx, &rows, s)
            })
        }),
        failure::<B>("blas_rotate_cols", &mut res, &[&a], &cols, |res| {
            with_scratch::<B, _>(module.blas_rotate_cols_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_rotate_cols(d / 2, res, &a, &ctx, &cols, s)
            })
        }),
        failure::<B>("blas_sum_rows", &mut res, &[&a], &sum_rows, |res| {
            with_scratch::<B, _>(module.blas_sum_rows_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_sum_rows(res, &a, &ctx, &sum_rows, s)
            })
        }),
        failure::<B>("blas_sum_cols", &mut res, &[&a], &sum_cols, |res| {
            with_scratch::<B, _>(module.blas_sum_cols_tmp_bytes(res, &a, &ctx, &key), |s| {
                module.blas_sum_cols(res, &a, &ctx, &sum_cols, s)
            })
        }),
    ]
}

/// Entry `(r, c)` of public matrix `m`.
fn public_entry<F: TestScalar>(m: usize, r: usize, c: usize) -> F {
    F::from_f64(((m + r * 4 + c) * 3 % 5) as f64 - 2.0).unwrap()
}

/// The public operand of `layout`, its entries [`public_entry`], prepared.
fn prepared<B, F, S>(module: &Module<B>, layout: BlasPtOperandLayout<S>) -> BlasPtOperandPrepared<B, S>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
    S: BlasPtSide,
{
    let mut op = module.blas_pt_operand_alloc_from_infos(&layout);
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&layout);
    with_scratch::<B, _>(module.blas_pt_operand_encode_tmp_bytes(&layout), |s| {
        module.blas_pt_operand_encode(&mut op, public_entry::<F>, s).unwrap()
    });
    with_scratch::<B, _>(module.blas_pt_operand_prepare_tmp_bytes(&layout), |s| {
        module.blas_pt_operand_prepare(&mut prepared, &op, s)
    });
    prepared
}

/// A `2 x 1` grid of public matrices of `2^log_d x 2^log_d` blocks, `count` per batch, at scale `p`, on the right,
/// prepared, for the products `algorithm` evaluates.
fn public_operand<B, F>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    count: usize,
    p: usize,
    algorithm: BlasPtAlgorithm,
) -> BlasPtOperandPrepared<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
    let base2k = params.base2k.into();
    prepared::<B, F, Right>(
        module,
        BlasPtOperandLayout::new(plan, module.ckks_max_slots(), 2, 1, count, base2k, diagonal_meta(params, p)),
    )
}

/// As [`public_operand`], a `2 x 2` grid on the left.
fn public_left_operand<B, F>(
    params: CKKSTestParams,
    module: &Module<B>,
    log_d: usize,
    count: usize,
    p: usize,
    algorithm: BlasPtAlgorithm,
) -> BlasPtOperandPrepared<B, Left>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
    let base2k = params.base2k.into();
    prepared::<B, F, Left>(
        module,
        BlasPtOperandLayout::new(plan, module.ckks_max_slots(), 2, 2, count, base2k, diagonal_meta(params, p)),
    )
}

/// `blas_matmul` of `a` by `b` into a fixture of width `k`.
fn product<B: BlasParityBackend>(
    params: CKKSTestParams,
    module: &Module<B>,
    ctx: &BlasContextPrepared<B>,
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

/// The [`Output`] of the preparation `op` of `input` under `hint`: the operand's infos, which must be its input's
/// batch and, as its layout, its input's when hot and its input's term layout when cold; its hint and its source's
/// layout, its input's, are checked too.
fn preparation<B: BlasParityBackend>(
    op: &'static str,
    operand: &impl BlasMatMulOperandInfos,
    input: &BlasCiphertextOwned<B>,
    ctx: &BlasContextPrepared<B>,
    right: bool,
    hint: PrepareHint,
    keys: &Recorded<B>,
) -> Output {
    let infos = grid_infos(operand);
    let layout = match hint {
        OneShot => BlasLayout::new(input.block_layout(), input.log_d(), input.rows(), input.cols()),
        Reuse => {
            if right {
                blas_matmul_right_term_layout(input, ctx)
            } else {
                blas_matmul_left_term_layout(input, ctx)
            }
        }
    };
    assert_eq!(
        (infos, operand.hint(), operand.source_block_layout()),
        ((layout, input.mat_slots()), hint, input.block_layout()),
        "{op} {hint:?}: the operand's infos are not its input's"
    );
    Output {
        op,
        infos: Infos::Grid(infos),
        blocks: Vec::new(),
        error: None,
        lookups: keys.take(),
    }
}

/// Both preparations of a `2 x 2` grid and a `2 x 1` grid under either hint, then their product into `res` of their
/// width, one limb wider, and one limb narrower than the product's natural width, which the kernel's offset absorbs:
/// each time by `blas_matmul` and by `blas_matmul_prepared` of every mix of hot and cold operands, which must agree,
/// with the keys of the products' set when a side is hot and the tensor key alone for two cold operands; all of it
/// through maps at the operands' scale, then at scales below and above it ([`other_scales`]), whose scale the terms
/// consume twice in place of the operands'. Through the former, the same for a `2 x 1` right factor labelled one
/// matrix, whose terms consume that scale once, into `res` of its width. Through the latter, the same for factors at
/// two scales, in either order, into `res` as wide as the wider factor: one two bits lower and either as much narrower,
/// so that their terms keep one budget, or as wide, so that its terms have the larger budget (crossed scales), where
/// the larger budget plus the larger scale exceeds the wider term's width: a kernel taking its offset from that width,
/// not from that sum, changes the bits. Products into `2 x 2` grids, of factors one limb apart, in both orders, since
/// the tensor key is looked up at the wider side's width, of a lazy grid of real slots labelled 1 by itself, and a long
/// product, `1 x 32` by `32 x 1` distinct blocks, whose `32 d` terms one JKLS kernel sums, or `1 x 16` by `16 x 1`
/// above radix 52, where `Module::max_base2k`'s model leaves `32 d` pairs less than 3 bits of slack at `d = 4`: a
/// kernel that drops, repeats or swaps a term changes the bits. The products by public matrices on either side, under
/// the diagonal algorithm and under JKLS: of one matrix, of one per matrix slot, and of one on the lazy grid into `res`
/// one limb narrower than it under one map and two under JKLS, below the products' width, which caps them and at whose
/// width their giant steps are looked up, at the operands' scale; then of one matrix at each other scale; and the same
/// under `JklsStored` and `JklsGalois`, three scales deep under the context's masks, the lazy grid's `res` three limbs
/// narrower, on the right of one public matrix only.
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
    // A right factor holding one matrix: its psi^k are rotations at no scale.
    let b_one = grid_of(
        module,
        &block(params, w, one_slot(module, log_d), SlotsKind::Complex),
        log_d,
        2,
        1,
        210,
        false,
    );
    let thin = sized(params, module, log_d, 2, 2, w - base2k, 140);
    let lazy = lazy_real(params, module, log_d, 2, 2, 144);
    // A long product: `inner d` terms of distinct blocks in one JKLS kernel.
    let inner = if base2k > 52 { 16 } else { 32 };
    let (long_a, long_b) = (
        operand(params, module, log_d, 1, inner, 160),
        operand(params, module, log_d, inner, 1, 200),
    );
    let inputs = [
        &a, &b, &a_low, &b_low, &a_crossed, &b_crossed, &b_one, &thin, &lazy, &long_a, &long_b,
    ];
    let before = snapshots::<B>(&inputs);
    let mut out = Vec::new();
    for (p, ctx) in scales.into_iter().zip(&contexts) {
        // Factors of one width and scale: poulpy-ckks's ct x ct rule leaves the terms' width less their scale.
        let term = blas_matmul_left_term_layout(&a, ctx).block;
        let natural = term.k().as_usize() - term.log_delta();
        let mut pairs = vec![(&a, &b, vec![w, w + base2k, natural - base2k])];
        if p == delta {
            pairs.push((&a, &b_one, vec![w]));
        } else {
            pairs.extend([
                (&a, &b_low, vec![w]),
                (&a_low, &b, vec![w]),
                (&a, &b_crossed, vec![w]),
                (&a_crossed, &b, vec![w]),
            ]);
        }
        for (x, y, widths) in pairs {
            // Hot operands at `side(OneShot)`, cold ones at `side(Reuse)`.
            let side = |hint| usize::from(hint == Reuse);
            let lefts = HINTS.map(|hint| {
                let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(x, ctx, &atk, hint), |s| {
                    matmul_left_operand(module, x, ctx, &products, hint, s)
                })
                .unwrap();
                out.push(preparation("blas_matmul_prepare_left", &left, x, ctx, false, hint, &products));
                left
            });
            let rights = HINTS.map(|hint| {
                let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(y, ctx, &atk, hint), |s| {
                    matmul_right_operand(module, y, ctx, &products, hint, s)
                })
                .unwrap();
                out.push(preparation(
                    "blas_matmul_prepare_right",
                    &right,
                    y,
                    ctx,
                    true,
                    hint,
                    &products,
                ));
                right
            });
            for k in widths {
                let from_factors = product(params, module, ctx, &products, (x, y), k);
                for (hl, hr) in MIXES {
                    let (left, right) = (&lefts[side(hl)], &rights[side(hr)]);
                    let keys = if (hl, hr) == (Reuse, Reuse) { &tensor } else { &products };
                    let mut res = sized(params, module, log_d, 2, 1, k, 136);
                    let bytes = module.blas_matmul_prepared_tmp_bytes(&res, left, right, ctx, &tsk, &atk);
                    with_scratch::<B, _>(bytes, |s| {
                        module.blas_matmul_prepared(&mut res, left, right, ctx, keys, keys, s)
                    })
                    .unwrap();
                    let from_operands = output("blas_matmul_prepared", &res, keys);
                    assert!(
                        from_operands.infos == from_factors.infos && from_operands.blocks == from_factors.blocks,
                        "blas_matmul_prepared {hl:?} x {hr:?} differs from blas_matmul into res of width {k}, maps at \
                         scale {p}"
                    );
                    out.push(from_operands);
                }
                out.push(from_factors);
            }
        }
    }
    for factors in [(&a, &thin), (&thin, &a), (&lazy, &lazy), (&long_a, &long_b)] {
        out.push(product(params, module, ctx, &products, factors, w));
    }
    let batch = a.mat_slots().unwrap();
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        let pt_plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let pt_keys = fixture_keys(params, module, &pt_plan.matmul_pt_right_galois_elements(order), false);
        let left_keys = fixture_keys(params, module, &pt_plan.matmul_pt_left_galois_elements(order), false);
        let capped = match algorithm {
            BlasPtAlgorithm::Diagonal => w - base2k,
            BlasPtAlgorithm::Jkls => w - 2 * base2k,
            BlasPtAlgorithm::JklsStored | BlasPtAlgorithm::JklsGalois => w - 3 * base2k,
        };
        let mut cases = vec![(&a, &b, 1, w, delta)];
        if batch > 1 {
            cases.push((&a, &b, batch, w, delta));
        }
        cases.extend([(&lazy, &lazy, 1, capped, delta), (&a, &b, 1, w, low), (&a, &b, 1, w, high)]);
        for (x, y, count, k, p) in cases {
            // The masked paths take one public matrix on the right.
            if count == 1 || !algorithm.masked() {
                let pt = public_operand::<B, F>(params, module, log_d, count, p, algorithm);
                let mut res = sized(params, module, log_d, x.rows(), pt.cols(), k, 138);
                with_scratch::<B, _>(module.blas_matmul_pt_right_tmp_bytes(&res, x, &pt, ctx, &atk), |s| {
                    module.blas_matmul_pt_right(&mut res, x, &pt, ctx, &pt_keys, s)
                })
                .unwrap();
                out.push(output("blas_matmul_pt_right", &res, &pt_keys));
            }
            let pt = public_left_operand::<B, F>(params, module, log_d, count, p, algorithm);
            let mut res = sized(params, module, log_d, pt.rows(), y.cols(), k, 139);
            with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(&res, &pt, y, ctx, &atk), |s| {
                module.blas_matmul_pt_left(&mut res, &pt, y, ctx, &left_keys, s)
            })
            .unwrap();
            out.push(output("blas_matmul_pt_left", &res, &left_keys));
        }
        // A left product of a replicated input takes the operand's replicated-input maps; the right leg has its own case.
        let pt = public_left_operand::<B, F>(params, module, log_d, 1, delta, algorithm);
        let mut res = sized(params, module, log_d, pt.rows(), b_one.cols(), w, 139);
        with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(&res, &pt, &b_one, ctx, &atk), |s| {
            module.blas_matmul_pt_left(&mut res, &pt, &b_one, ctx, &left_keys, s)
        })
        .unwrap();
        out.push(output("blas_matmul_pt_left", &res, &left_keys));
    }
    assert_unchanged::<B>(&format!("matmul, log_d {log_d}"), &inputs, &before);
    out
}

/// The cold preparations of a `2 x 2` and a `2 x 1` grid, then their product and their prepared product into a
/// [`narrow`] grid, which poulpy-ckks rejects for the budget of the first block's JKLS kernel inside the
/// implementation, after the preparations, and the products by public matrices on either side, rejected for the
/// budget of their first map; then the hot operand of a `2 x 2` grid one bit short of the `2p` its terms spend, which
/// its preparation accepts and the prepared product rejects for the budget of its first shift; then the products by
/// public matrices under `Jkls`, `JklsStored` and `JklsGalois` of grids one bit short of the first map's scale `p_B`,
/// narrower than the maps, which the validator rejects, then under `Jkls` of both maps', rejected inside the first
/// block's summed transformation, and under the masked paths of `p_B + p` and of `2 p_B + p`, rejected inside the masks
/// and the summed product: no block of `res` is written. Then the product by a `2 x 1` right factor of budget `3p - 1`
/// labelled one matrix, whose terms spend `p`, which passes, and, where the slots hold more than one matrix, by such a
/// factor holding distinct matrices, whose terms spend `2p`, rejected for the budget of the first block's JKLS kernel.
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
    let pt_plan = BlasPlan::new(log_d);
    let pt_keys = fixture_keys(params, module, &pt_plan.matmul_pt_right_galois_elements(order), false);
    let left_keys = fixture_keys(params, module, &pt_plan.matmul_pt_left_galois_elements(order), false);
    let pt = public_operand::<B, F>(params, module, log_d, 1, scale(params), BlasPtAlgorithm::Diagonal);
    let pt_left = public_left_operand::<B, F>(params, module, log_d, 1, scale(params), BlasPtAlgorithm::Diagonal);
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let (a, b) = (
        operand(params, module, log_d, 2, 2, 120),
        operand(params, module, log_d, 2, 1, 124),
    );
    let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk, Reuse), |s| {
        matmul_left_operand(module, &a, &ctx, &products, Reuse, s)
    })
    .unwrap();
    let mut out = vec![preparation(
        "blas_matmul_prepare_left",
        &left,
        &a,
        &ctx,
        false,
        Reuse,
        &products,
    )];
    let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk, Reuse), |s| {
        matmul_right_operand(module, &b, &ctx, &products, Reuse, s)
    })
    .unwrap();
    out.push(preparation(
        "blas_matmul_prepare_right",
        &right,
        &b,
        &ctx,
        true,
        Reuse,
        &products,
    ));
    let mut res = narrow(params, module, log_d, 2, 1, 128);
    out.extend([
        failure::<B>("blas_matmul", &mut res, &[&a, &b], &products, |res| {
            with_scratch::<B, _>(module.blas_matmul_tmp_bytes(res, &a, &b, &ctx, &tsk, &atk), |s| {
                module.blas_matmul(res, &a, &b, &ctx, &products, &products, s)
            })
        }),
        failure::<B>("blas_matmul_prepared", &mut res, &[], &tensor, |res| {
            with_scratch::<B, _>(
                module.blas_matmul_prepared_tmp_bytes(res, &left, &right, &ctx, &tsk, &atk),
                |s| module.blas_matmul_prepared(res, &left, &right, &ctx, &tensor, &tensor, s),
            )
        }),
        failure::<B>("blas_matmul_pt_right", &mut res, &[&a], &pt_keys, |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_right_tmp_bytes(res, &a, &pt, &ctx, &atk), |s| {
                module.blas_matmul_pt_right(res, &a, &pt, &ctx, &pt_keys, s)
            })
        }),
        failure::<B>("blas_matmul_pt_left", &mut res, &[&b], &left_keys, |res| {
            with_scratch::<B, _>(module.blas_matmul_pt_left_tmp_bytes(res, &pt_left, &b, &ctx, &atk), |s| {
                module.blas_matmul_pt_left(res, &pt_left, &b, &ctx, &left_keys, s)
            })
        }),
    ]);
    // Budget `k - log_delta` one bit short of the `2p` its terms spend, at the operands' scale `p`, into `res` of the
    // operand width.
    let p = scale(params);
    let short = sized(params, module, log_d, 2, 2, 3 * p - 1, 132);
    let hot = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&short, &ctx, &atk, OneShot), |s| {
        matmul_left_operand(module, &short, &ctx, &products, OneShot, s)
    })
    .unwrap();
    out.push(preparation(
        "blas_matmul_prepare_left",
        &hot,
        &short,
        &ctx,
        false,
        OneShot,
        &products,
    ));
    let mut res = sized(params, module, log_d, 2, 1, width(params), 134);
    out.push(failure::<B>("blas_matmul_prepared", &mut res, &[&short], &products, |res| {
        with_scratch::<B, _>(
            module.blas_matmul_prepared_tmp_bytes(res, &hot, &right, &ctx, &tsk, &atk),
            |s| module.blas_matmul_prepared(res, &hot, &right, &ctx, &products, &products, s),
        )
    }));
    let [masked_right, masked_left] = [
        plan.matmul_pt_right_galois_elements(order),
        plan.matmul_pt_left_galois_elements(order),
    ]
    .map(|elements| fixture_keys(params, module, &elements, false));
    let budgets = [(2 * p - 1, 150), (3 * p - 1, 152), (4 * p - 1, 162)];
    let paths = [(BlasPtAlgorithm::Jkls, [&pt_keys, &left_keys], &budgets[..2])]
        .into_iter()
        .chain(MASKED_PT_ALGORITHMS.map(|algorithm| (algorithm, [&masked_right, &masked_left], &budgets[..])));
    for (algorithm, [right_keys, left_keys], budgets) in paths {
        let jkls = public_operand::<B, F>(params, module, log_d, 1, p, algorithm);
        let jkls_left = public_left_operand::<B, F>(params, module, log_d, 1, p, algorithm);
        for &(k, seed) in budgets {
            let row = sized(params, module, log_d, 2, 2, k, seed);
            let column = sized(params, module, log_d, 2, 1, k, seed + 1);
            let mut res = sized(params, module, log_d, 2, 1, width(params), 154);
            out.push(failure::<B>("blas_matmul_pt_right", &mut res, &[&row], right_keys, |res| {
                with_scratch::<B, _>(module.blas_matmul_pt_right_tmp_bytes(res, &row, &jkls, &ctx, &atk), |s| {
                    module.blas_matmul_pt_right(res, &row, &jkls, &ctx, right_keys, s)
                })
            }));
            out.push(failure::<B>("blas_matmul_pt_left", &mut res, &[&column], left_keys, |res| {
                with_scratch::<B, _>(
                    module.blas_matmul_pt_left_tmp_bytes(res, &jkls_left, &column, &ctx, &atk),
                    |s| module.blas_matmul_pt_left(res, &jkls_left, &column, &ctx, left_keys, s),
                )
            }));
        }
    }
    let rep = grid_of(
        module,
        &block(params, 4 * p - 1, one_slot(module, log_d), SlotsKind::Complex),
        log_d,
        2,
        1,
        156,
        false,
    );
    out.push(product(params, module, &ctx, &products, (&a, &rep), width(params)));
    if one_slot(module, log_d) > 0 {
        let dense = sized(params, module, log_d, 2, 1, 4 * p - 1, 158);
        let mut res = sized(params, module, log_d, 2, 1, width(params), 160);
        out.push(failure::<B>("blas_matmul", &mut res, &[&a, &dense], &products, |res| {
            with_scratch::<B, _>(module.blas_matmul_tmp_bytes(res, &a, &dense, &ctx, &tsk, &atk), |s| {
                module.blas_matmul(res, &a, &dense, &ctx, &products, &products, s)
            })
        }));
    }
    out
}

/// A vector of `shape` at torus width `k` repeating every `period` slots, ciphertext `t` filled from `seed + t`, with lazy
/// digits on real slots with `lazy` ([`fixture_operand`]).
fn vector_of<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    shape: BlasVectorShape,
    period: usize,
    k: usize,
    seed: u8,
    lazy: bool,
) -> BlasVectorOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B> + CKKSModuleInfos,
{
    let label = (module.ckks_max_slots() / period).ilog2() as usize;
    let slots = if lazy { SlotsKind::Real } else { SlotsKind::Complex };
    let ct = block(params, k, label, slots);
    let cts = (0..shape.count())
        .map(|t| fixture_operand(module, &ct, seed + t as u8, lazy))
        .collect();
    BlasVector::from_cts(cts, shape).unwrap()
}

/// The [`Output`] of `op` into the vector `res`, with the lookups `keys` recorded since the last output.
fn vector_output<B: BlasParityBackend>(op: &'static str, res: &BlasVectorOwned<B>, keys: &Recorded<B>) -> Output {
    Output {
        op,
        infos: Infos::Vector(res.blas_vector_layout()),
        blocks: vector_snapshot::<B>(res),
        error: None,
        lookups: keys.take(),
    }
}

/// The [`Output`] of `call` into the vector `res`, which must fail and leave `res` and `inputs` unchanged, with the
/// lookups `keys` recorded since the last output.
fn vector_failure<B: BlasParityBackend>(
    op: &'static str,
    res: &mut BlasVectorOwned<B>,
    inputs: &[&BlasVectorOwned<B>],
    keys: &Recorded<B>,
    call: impl FnOnce(&mut BlasVectorOwned<B>) -> CKKSResult<()>,
) -> Output {
    let state = |res: &BlasVectorOwned<B>| {
        let inputs: Vec<_> = inputs.iter().map(|v| vector_snapshot::<B>(v)).collect();
        (vector_snapshot::<B>(res), inputs)
    };
    let before = state(res);
    let error = call(res).expect_err(op).composition().cloned();
    assert!(state(res) == before, "{op} changed res or an input before failing");
    Output {
        op,
        infos: Infos::Vector(res.blas_vector_layout()),
        blocks: Vec::new(),
        error,
        lookups: keys.take(),
    }
}

/// The layouts of [`matvec_layouts`] with diagonals at scale `p`.
fn matvec_layouts_at(params: CKKSTestParams, slots: usize, p: usize) -> Vec<BlasMatVecOperandLayout> {
    matvec_layouts(&params, slots)
        .into_iter()
        .map(|layout| BlasMatVecOperandLayout {
            coeffs_meta: diagonal_meta(params, p),
            ..layout
        })
        .collect()
}

/// Products of a vector by a public matrix for every layout of [`matvec_layouts`], resident then streamed, into `res`
/// of the operand width, one limb wider and one limb narrower; of a [`vector_of`] with lazy digits on real slots; and,
/// on the multi-ciphertext layout, by matrices at scales below and above the inputs'.
fn matvec<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let (slots, w, key) = (module.ckks_max_slots(), width(params), atk_layout(params));
    let mut out = Vec::new();
    let scales = std::iter::once(scale(params)).chain(other_scales(params));
    for (s, p) in scales.enumerate() {
        let layouts = matvec_layouts_at(params, slots, p);
        // The other scales run on the multi-ciphertext layout alone.
        let layouts = if s == 0 {
            layouts
        } else {
            layouts.into_iter().skip(3).collect()
        };
        for (l, layout) in layouts.into_iter().enumerate() {
            let (streamed, resident) = matvec_operands::<B, F>(module, &layout);
            let keys = fixture_keys(
                params,
                module,
                &matvec_pt_galois_elements(module.cyclotomic_order(), &layout),
                false,
            );
            let seed = 150 + 10 * l as u8;
            let a = vector_of(params, module, layout.input, layout.input_period, w, seed, false);
            let lazy = vector_of(params, module, layout.input, layout.input_period, w, seed + 5, true);
            let before = [vector_snapshot::<B>(&a), vector_snapshot::<B>(&lazy)];
            for (src, k) in [(&a, w), (&a, w + params.base2k), (&a, w - params.base2k), (&lazy, w)] {
                let mut res = vector_of(params, module, layout.output, slots, k, 140, false);
                with_scratch::<B, _>(module.blas_matvec_pt_tmp_bytes(&res, &resident, src, &key), |s| {
                    module.blas_matvec_pt(&mut res, &resident, src, &keys, s)
                })
                .unwrap();
                out.push(vector_output("blas_matvec_pt", &res, &keys));
                let mut res = vector_of(params, module, layout.output, slots, k, 140, false);
                with_scratch::<B, _>(module.blas_matvec_pt_tmp_bytes(&res, &streamed, src, &key), |s| {
                    module.blas_matvec_pt(&mut res, &streamed, src, &keys, s)
                })
                .unwrap();
                out.push(vector_output("blas_matvec_pt", &res, &keys));
            }
            assert!(
                [vector_snapshot::<B>(&a), vector_snapshot::<B>(&lazy)] == before,
                "matvec: {} wrote an input",
                std::any::type_name::<B>()
            );
        }
    }
    out
}

/// Products of a vector by a public matrix, resident and streamed, into a vector narrower than the result's scale, and
/// of an input whose budget is below the matrix's scale where its width allows one, which poulpy-ckks rejects inside
/// the implementation.
fn matvec_budget<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let (slots, w, key, p) = (module.ckks_max_slots(), width(params), atk_layout(params), scale(params));
    let layout = matvec_layouts_at(params, slots, p).pop().unwrap();
    let (streamed, resident) = matvec_operands::<B, F>(module, &layout);
    let keys = fixture_keys(
        params,
        module,
        &matvec_pt_galois_elements(module.cyclotomic_order(), &layout),
        false,
    );
    let a = vector_of(params, module, layout.input, layout.input_period, w, 190, false);
    let mut inputs = vec![(a, None)];
    // An input at the matrix's width, `p + base2k`, keeps a budget of `base2k` bits, below `p` when `base2k < p`.
    if params.base2k < p {
        let low = vector_of(
            params,
            module,
            layout.input,
            layout.input_period,
            p + params.base2k,
            195,
            false,
        );
        inputs.push((low, Some(w)));
    }
    let mut out = Vec::new();
    for (src, res_k) in &inputs {
        let mut res = match res_k {
            None => {
                let ct = layout_of_narrow(params);
                let cts = (0..layout.output.count())
                    .map(|t| fixture_operand(module, &ct, 200 + t as u8, false))
                    .collect();
                BlasVector::from_cts(cts, layout.output).unwrap()
            }
            Some(k) => vector_of(params, module, layout.output, slots, *k, 200, false),
        };
        out.push(vector_failure("blas_matvec_pt", &mut res, &[src], &keys, |res| {
            with_scratch::<B, _>(module.blas_matvec_pt_tmp_bytes(res, &resident, src, &key), |s| {
                module.blas_matvec_pt(res, &resident, src, &keys, s)
            })
        }));
        out.push(vector_failure("blas_matvec_pt", &mut res, &[src], &keys, |res| {
            with_scratch::<B, _>(module.blas_matvec_pt_tmp_bytes(res, &streamed, src, &key), |s| {
                module.blas_matvec_pt(res, &streamed, src, &keys, s)
            })
        }));
    }
    out
}

/// The [`Output`] of `op` into the matrix `m`, which `result` must leave written on `Ok`, or unchanged on `Err`, with
/// the lookups `keys` recorded since the last output.
fn diagonal_ciphertext_output<B: BlasParityBackend>(
    op: &'static str,
    m: &mut BlasDiagonalCiphertext<B>,
    keys: &Recorded<B>,
    call: impl FnOnce(&mut BlasDiagonalCiphertext<B>) -> CKKSResult<()>,
) -> Output {
    let before = diagonal_ciphertext_snapshot::<B>(m);
    let error = call(m).err().map(|e| e.composition().cloned().expect("a composition error"));
    let blocks = if error.is_some() {
        assert!(
            diagonal_ciphertext_snapshot::<B>(m) == before,
            "{op} changed the matrix before failing"
        );
        Vec::new()
    } else {
        diagonal_ciphertext_snapshot::<B>(m)
    };
    Output {
        op,
        infos: Infos::Matrix(m.blas_diagonal_ciphertext_layout()),
        blocks,
        error,
        lookups: keys.take(),
    }
}

/// A prepared mask, `1` on the slots of `set` and `0` elsewhere, encoded at scale `p` and prepared under `hint`.
fn mask_of<B, F>(params: CKKSTestParams, module: &Module<B>, set: &[usize], p: usize, hint: PrepareHint) -> CKKSPreparedRight<B>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let slots = module.ckks_max_slots();
    let zero = <F as DiagonalArithmetic>::zero();
    let mut re = vec![zero; slots];
    set.iter().for_each(|&s| re[s] = F::from_f64(1.0).unwrap());
    let meta = diagonal_meta(params, p);
    let mut pt = module.ckks_pt_vec_alloc(slots, params.base2k.into(), meta.k);
    poulpy_ckks::SetCKKSInfos::set_meta(&mut pt, meta.meta);
    with_scratch::<B, _>(module.ckks_reim_tmp_bytes(slots), |s| {
        module.ckks_encode_reim_into(&mut pt, &re, &vec![zero; slots], s)
    })
    .unwrap();
    with_scratch::<B, _>(module.ckks_prepare_tmp_bytes(&pt), |s| {
        module.ckks_prepare_right(&pt, hint, s)
    })
    .unwrap()
}

/// A diagonal ciphertext of diagonals `0..8` at giant step 4, by rotation and by powers of two: rows and columns appended
/// by a mask of one slot prepared to reuse and one of two prepared once, of a vector with lazy digits on real slots
/// too, a row and a column from a vector one limb narrower than the diagonals' frame (refused), then its product,
/// prepared then streamed, by a vector and by one with lazy digits into `res` of the operand width, one limb wider and
/// one limb narrower, and into a `res` narrower than the result's scale (refused).
fn diagonal_ciphertext<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<Output>
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let (slots, w, p, order) = (
        module.ckks_max_slots(),
        width(params),
        scale(params),
        module.cyclotomic_order(),
    );
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    let (one, two) = (
        mask_of::<B, F>(params, module, &[0], p, PrepareHint::Reuse),
        mask_of::<B, F>(params, module, &[0, slots / 2], p, PrepareHint::OneShot),
    );
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let mut out = Vec::new();
    let strategies = [
        LinearTransformationStrategy::Bsgs { giant_step: 4 },
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 },
    ];
    for strategy in strategies {
        let layout = BlasDiagonalCiphertextLayout::new(
            block(params, w - p, 0, SlotsKind::Complex),
            LinearTransformationLayout {
                indexes: (0..8).collect(),
                slots,
                strategy,
            },
        );
        let elements: Vec<i64> = diagonal_ciphertext_elements(order, &layout).into_iter().collect();
        let keys = fixture_keys(params, module, &elements, true);
        let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(&layout);
        let appends = [
            (AppendAxis::Row, 0, &one, 210, false),
            (AppendAxis::Col, 3, &one, 214, false),
            (AppendAxis::Row, -2, &two, 218, false),
            (AppendAxis::Col, 6, &two, 220, true),
        ];
        for (axis, at, mask, seed, lazy) in appends {
            let v = vector_of(params, module, whole, slots, w, seed, lazy);
            let before = vector_snapshot::<B>(&v);
            let bytes = append_tmp_bytes(module, axis, &m, &v, mask, &atk);
            out.push(diagonal_ciphertext_output(axis.op(), &mut m, &keys, |m| {
                with_scratch::<B, _>(bytes, |s| append(module, axis, m, at, &v, mask, &keys, s))
            }));
            assert!(
                vector_snapshot::<B>(&v) == before,
                "diagonal_ciphertext: {} wrote its vector",
                axis.op()
            );
        }
        let narrower = vector_of(params, module, whole, slots, w - params.base2k, 222, false);
        for axis in [AppendAxis::Row, AppendAxis::Col] {
            let bytes = append_tmp_bytes(module, axis, &m, &narrower, &one, &atk);
            out.push(diagonal_ciphertext_output(axis.op(), &mut m, &keys, |m| {
                with_scratch::<B, _>(bytes, |s| append(module, axis, m, 1, &narrower, &one, &keys, s))
            }));
        }
        let prepared = prepared_of(module, &m);
        assert!(
            prepared.blas_diagonal_ciphertext_layout() == layout,
            "diagonal_ciphertext: the prepared form's infos"
        );
        let a = vector_of(params, module, whole, slots, w, 226, false);
        let lazy = vector_of(params, module, whole, slots, w, 228, true);
        let before = [vector_snapshot::<B>(&a), vector_snapshot::<B>(&lazy)];
        for (src, k) in [(&a, w), (&a, w + params.base2k), (&a, w - params.base2k), (&lazy, w)] {
            let mut res = vector_of(params, module, whole, slots, k, 230, false);
            with_scratch::<B, _>(module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, src, &tsk, &atk), |s| {
                module.blas_matvec_diagonal(&mut res, &prepared, src, &keys, s)
            })
            .unwrap();
            out.push(vector_output("blas_matvec_diagonal", &res, &keys));
            let mut res = vector_of(params, module, whole, slots, k, 230, false);
            with_scratch::<B, _>(module.blas_matvec_diagonal_tmp_bytes(&res, &m, src, &tsk, &atk), |s| {
                module.blas_matvec_diagonal(&mut res, &m, src, &keys, s)
            })
            .unwrap();
            out.push(vector_output("blas_matvec_diagonal", &res, &keys));
        }
        assert!(
            [vector_snapshot::<B>(&a), vector_snapshot::<B>(&lazy)] == before,
            "diagonal_ciphertext: blas_matvec_diagonal wrote an input"
        );
        let ct = layout_of_narrow(params);
        let mut res = BlasVector::from_cts(vec![fixture_operand(module, &ct, 234, false)], whole).unwrap();
        out.push(vector_failure("blas_matvec_diagonal", &mut res, &[&a], &keys, |res| {
            with_scratch::<B, _>(module.blas_matvec_diagonal_tmp_bytes(res, &m, &a, &tsk, &atk), |s| {
                module.blas_matvec_diagonal(res, &m, &a, &keys, s)
            })
        }));
        out.push(vector_failure("blas_matvec_diagonal", &mut res, &[&a], &keys, |res| {
            with_scratch::<B, _>(module.blas_matvec_diagonal_tmp_bytes(res, &prepared, &a, &tsk, &atk), |s| {
                module.blas_matvec_diagonal(res, &prepared, &a, &keys, s)
            })
        }));
    }
    out
}

/// The layout of [`narrow`]'s blocks: a width below the scale of any result.
fn layout_of_narrow(params: CKKSTestParams) -> CKKSLayout {
    layout(params, params.rank, params.base2k - 2, 0, 0, SlotsKind::Complex)
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
pub fn test_blas_inter_parity<BR, BT, F>(
    params: CKKSTestParams,
    shapes: &BlasParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, module_ref.ckks_max_slots()) {
        let label = format!("inter, log_d {log_d}");
        compare(
            &label,
            inter::<BR, F>(params, module_ref, log_d),
            inter::<BT, F>(params, module_test, log_d),
        );
        let (want, got) = (
            inter_budget::<BR, F>(params, module_ref, log_d),
            inter_budget::<BT, F>(params, module_test, log_d),
        );
        compare(&format!("{label}, budget"), want, got);
    }
}

/// Transposes, row and column rotations and row and column sums agree bit for bit at every swept `log_d`, through every
/// branch of the shifts and of the sums, a narrower `res`, lazy real inputs, and maps at scales other than the inputs',
/// and fail alike on a budget poulpy-ckks rejects, leaving `res` and `a` unchanged.
pub fn test_blas_intra_parity<BR, BT, F>(
    params: CKKSTestParams,
    shapes: &BlasParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, module_ref.ckks_max_slots()) {
        let label = format!("intra, log_d {log_d}");
        compare(
            &label,
            intra::<BR, F>(params, module_ref, log_d),
            intra::<BT, F>(params, module_test, log_d),
        );
        let (want, got) = (
            intra_budget::<BR, F>(params, module_ref, log_d),
            intra_budget::<BT, F>(params, module_test, log_d),
        );
        compare(&format!("{label}, budget"), want, got);
    }
}

/// The products, the prepared product of every mix of hot and cold operands and the products by public matrices, under
/// every algorithm, the masked ones on the right of one public matrix only, agree bit for bit at every swept `log_d`,
/// through maps and public matrices at the factors' scale and at scales other than theirs, there also of factors at two
/// scales in either order, the terms at the lower scale with the other's budget or a larger one (crossed scales), and
/// the prepared operands' infos, `(log_d, rows, cols, block_layout, mat_slots)`, agree, beside their hints and their
/// inputs' layouts as their sources; prepared storage is never compared. A long product, `1 x 32` by `32 x 1` distinct
/// blocks (`1 x 16` by `16 x 1` above radix 52), agrees too, and so does a right factor holding one matrix, whose terms
/// spend one scale. The product, the prepared product and the products by public matrices on either side, under
/// `Diagonal` and `Jkls`, fail alike on a budget poulpy-ckks rejects, leaving `res` and the factors unchanged.
pub fn test_blas_matmul_parity<BR, BT, F>(
    params: CKKSTestParams,
    shapes: &BlasParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    for log_d in swept_log_d(shapes, module_ref.ckks_max_slots()) {
        let label = format!("matmul, log_d {log_d}");
        compare(
            &label,
            matmul::<BR, F>(params, module_ref, log_d),
            matmul::<BT, F>(params, module_test, log_d),
        );
        let (want, got) = (
            matmul_budget::<BR, F>(params, module_ref, log_d),
            matmul_budget::<BT, F>(params, module_test, log_d),
        );
        compare(&format!("{label}, budget"), want, got);
    }
}

/// Products of a vector by a public matrix, resident and streamed, agree bit for bit over the layouts of
/// [`matvec_layouts`] (dense and replicated vectors, one and several ciphertexts), into `res` wider and narrower than
/// the result, of lazy real inputs, and by matrices at scales other than the inputs', and fail alike on a budget
/// poulpy-ckks rejects, leaving `res` and the input unchanged; so do the appends and the product of a diagonal ciphertext,
/// by rotation and by powers of two. Vectors span 16 slots: below, it is skipped with a message.
pub fn test_blas_matvec_parity<BR, BT, F>(
    params: CKKSTestParams,
    _shapes: &BlasParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: BlasParityModule<BR, F>,
    Module<BT>: BlasParityModule<BT, F>,
    F: TestScalar + DiagonalArithmetic,
{
    if module_ref.ckks_max_slots() < 16 {
        eprintln!("matvec parity skipped: its vectors span 16 slots");
        return;
    }
    compare(
        "matvec",
        matvec::<BR, F>(params, module_ref),
        matvec::<BT, F>(params, module_test),
    );
    compare(
        "matvec, budget",
        matvec_budget::<BR, F>(params, module_ref),
        matvec_budget::<BT, F>(params, module_test),
    );
    compare(
        "matvec, diagonal ciphertext",
        diagonal_ciphertext::<BR, F>(params, module_ref),
        diagonal_ciphertext::<BT, F>(params, module_test),
    );
}

/// Asserts `got` is the largest radix `hand` allows at every failure target of `targets`, bits per output polynomial: it
/// satisfies [`Module::max_base2k`] at the products `hand` counts at that radix, and no larger one does, none at all for
/// 0.
fn assert_largest_radix<B: Backend + MaxBase2k>(
    label: &str,
    n: usize,
    targets: impl IntoIterator<Item = usize>,
    hand: impl Fn(usize) -> usize,
    got: impl Fn(usize) -> usize,
) {
    for failure_bits in targets {
        let fits = |r: usize| Module::<B>::max_base2k(n, hand(r), failure_bits, false) >= r;
        let r = got(failure_bits);
        assert!(r == 0 || fits(r), "{label}: radix {r} does not fit at {failure_bits} bits");
        assert!(
            (r + 1..=MAX_BASE2K).all(|q| !fits(q)),
            "{label}: a radix above {r} fits at {failure_bits} bits"
        );
    }
}

/// The radix helpers on `4 x 4` blocks on 32 slots, `n = 64` or `n = 32` on the conjugate-invariant ring, against
/// counts made by hand. A giant step groups at most 4 diagonals of sigma and of a right product's block maps (offsets
/// `-3..=3`, split at 29: `{0, 1, 2, 3}` and `{29, 30, 31}`), 3 of tau, `tau_rep` and a left product's block maps (split
/// at 12: `{0, 4, 8}`, `{12, 20}` and `{24, 28}`), and 1 of a shift; every diagonal is `p + base2k` bits wide. Over
/// every failure target from 1 to 1000 bits, each helper returns the largest radix at which [`Module::max_base2k`]
/// holds the count of its call at that radix:
/// - `blas_matmul`: the largest of the kernel's `2 L K d`, `L` the limbs of the narrower terms, `a.k - 2p` and
///   `b.k - 2p`, or `b.k - p` for a `b` holding one matrix, and, for each map that makes the terms, the count
///   `blas_matmul_pt_right` documents for one term, read at `a.k` or `b.k` for sigma and tau and at that less `p` for a
///   shift. At diagonals wider than the narrower term, a map's count exceeds the kernel's: `p` of 37 bits over the
///   radices of the FFT64 model, and of 103 over those of the NTT model;
/// - the products by public matrices, on either side, under every algorithm, the masked ones where `a.k` exceeds
///   `p_B + p`: the count `blas_matmul_pt_right` documents. The first map's is the larger for one inner block at
///   `a.k = 30` and `p_B = 20` over the radices of the FFT64 model, and at `a.k = 90` and `p_B = 50` over those of the
///   NTT model. A left product of an input labelled one matrix counts the operand's replicated-input maps, whose
///   largest giant group equals `left_mul`'s at every shape (the dimensions `d = 2` to `128` at one to eight matrix
///   slots were checked), so that case covers the wiring only;
///
/// - the products of a vector, over the layouts of [`matvec_layouts`]: `K g min(L, P)` over the `K` input
///   ciphertexts, `g` the most diagonals a giant step of the operand's encoded maps groups;
/// - the product by a diagonal ciphertext of diagonals `0..8` at giant step 4: `2 g min(L, P)`;
///
/// and, for a sum so long that no FFT64 radix holds it at a failure target of 2^60 bits, 0. Under the masked paths, a left
/// product's query panics on a context without the column halves of psi^k, with a static message.
pub fn test_blas_max_base2k<B, F>(params: CKKSTestParams, module: &Module<B>)
where
    B: BlasParityBackend + MaxBase2k,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let (n, log_d, p) = (params.n, 2, scale(params));
    let ctx = context::<B, F>(params, module, log_d, p);
    let layout = |k: usize, label: usize, rows: usize, cols: usize| {
        BlasLayout::new(block(params, k, label, SlotsKind::Complex), log_d, rows, cols)
    };
    let right_operand = |inner: usize, algorithm, p_b: usize| {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let (base2k, meta) = (params.base2k.into(), diagonal_meta(params, p_b));
        prepared::<B, F, Right>(
            module,
            BlasPtOperandLayout::new(plan, module.ckks_max_slots(), inner, 1, 1, base2k, meta),
        )
    };
    let left_operand = |inner: usize, algorithm, p_b: usize| {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let (base2k, meta) = (params.base2k.into(), diagonal_meta(params, p_b));
        prepared::<B, F, Left>(
            module,
            BlasPtOperandLayout::new(plan, module.ckks_max_slots(), 1, inner, 1, base2k, meta),
        )
    };
    assert_eq!(module.ckks_max_slots(), 32, "the counts below are those of 32 slots");

    // `(g, input width)` of sigma, the left shifts, tau (or `tau_rep`) and the right shifts, beside the kernel's count.
    let matmul_hand = |inner: usize, a_k: usize, b_k: usize, spent: usize, p: usize| {
        let diagonal_k = diagonal_meta(params, p).k.as_usize();
        move |r: usize| {
            let kernel = 2 * (a_k - 2 * p).min(b_k - spent).div_ceil(r) * (inner * 4);
            [(4, a_k), (1, a_k - p), (3, b_k), (1, b_k - p)]
                .into_iter()
                .fold(kernel, |count, (g, k)| {
                    count.max(g * k.div_ceil(r).min(diagonal_k.div_ceil(r)))
                })
        }
    };
    for (label, inner, a_k, b_k, b_label, p_maps) in [
        ("narrower b", 3, 77, 60, 0, p),
        ("narrower a", 3, 50, 77, 0, p),
        ("replicated b", 3, 77, 60, one_slot(module, log_d), p),
        ("wide diagonals", 1, 77, 77, 0, 37),
        ("wider diagonals", 1, 215, 215, 0, 103),
    ] {
        let ctx = context::<B, F>(params, module, log_d, p_maps);
        let (a, b) = (layout(a_k, 0, 2, inner), layout(b_k, b_label, inner, 2));
        let hand = matmul_hand(inner, a_k, b_k, if b_label == 0 { 2 * p_maps } else { p_maps }, p_maps);
        assert_largest_radix::<B>(label, n, 1..=1000, hand, |bits| {
            blas_matmul_max_base2k::<B>(&a, &b, &ctx, bits)
        });
    }
    // 2^40 inner blocks at a failure target of 2^60 bits: no radix holds the sum.
    let (a, b) = (layout(77, 0, 2, 1 << 40), layout(77, 0, 1 << 40, 2));
    assert_largest_radix::<B>("long sum", n, [1 << 60], matmul_hand(1 << 40, 77, 77, 2 * p, p), |bits| {
        blas_matmul_max_base2k::<B>(&a, &b, &ctx, bits)
    });

    // The masked paths' final step sums `terms` products by prepared plaintexts per inner block.
    let hand = |algorithm: BlasPtAlgorithm, g: usize, terms: usize, inner: usize, a_k: usize, p_b: usize| {
        let (pt_k, ctx_k) = (diagonal_meta(params, p_b).k.as_usize(), diagonal_meta(params, p).k.as_usize());
        move |r: usize| {
            let count = |g: usize, width: usize, diagonal: usize| g * width.div_ceil(r).min(diagonal.div_ceil(r));
            match algorithm {
                BlasPtAlgorithm::Diagonal => inner * count(g, a_k, pt_k),
                BlasPtAlgorithm::Jkls => (inner * count(g, a_k - p_b, pt_k)).max(count(g, a_k, pt_k)),
                BlasPtAlgorithm::JklsStored | BlasPtAlgorithm::JklsGalois => count(g, a_k, pt_k)
                    .max(count(1, a_k - p_b, ctx_k))
                    .max(inner * terms * count(1, a_k - p_b - p, pt_k)),
            }
        }
    };
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        for (inner, a_k, p_b) in [
            (1, 77, p),
            (3, 77, p),
            (3, 20, p),
            (1, 30, 20),
            (3, 30, 20),
            (1, 90, 50),
            (3, 90, 50),
        ] {
            // The masked paths' terms, at `a.k - p_B - p`, need a positive width.
            if algorithm.masked() && a_k <= p_b + p {
                continue;
            }
            let label = format!("{algorithm:?} inner={inner} a.k={a_k} p_B={p_b}");
            let (pt, a) = (right_operand(inner, algorithm, p_b), layout(a_k, 0, 2, inner));
            let hand_right = hand(algorithm, 4, 4, inner, a_k, p_b);
            assert_largest_radix::<B>(&format!("right {label}"), n, 1..=1000, hand_right, |bits| {
                blas_matmul_pt_right_max_base2k::<B>(&a, &pt, &ctx, bits)
            });
            let (pt, a) = (left_operand(inner, algorithm, p_b), layout(a_k, 0, inner, 2));
            let hand_left = hand(algorithm, 3, 7, inner, a_k, p_b);
            assert_largest_radix::<B>(&format!("left {label}"), n, 1..=1000, hand_left, |bits| {
                blas_matmul_pt_left_max_base2k::<B>(&pt, &a, &ctx, bits)
            });
            // Wiring only: the replicated-input maps group as many diagonals to a giant step as `left_mul`'s at any shape.
            let a_rep = layout(a_k, one_slot(module, log_d), inner, 2);
            let hand_rep = hand(algorithm, 3, 7, inner, a_k, p_b);
            assert_largest_radix::<B>(&format!("left replicated {label}"), n, 1..=1000, hand_rep, |bits| {
                blas_matmul_pt_left_max_base2k::<B>(&pt, &a_rep, &ctx, bits)
            });
        }
    }

    // Products of a vector: `K g min(L, P)` over the `K` input ciphertexts, `g` read off the operand's encoded maps.
    for layout in matvec_layouts_at(params, module.ckks_max_slots(), p) {
        let (_, w) = matvec_operands::<B, F>(module, &layout);
        let g = w.block(0, 0).giant_steps.iter().map(|gs| gs.diagonals.len()).max().unwrap();
        let (inputs, diagonal_k) = (layout.input.count(), layout.coeffs_meta.k.as_usize());
        for a_k in [77, 20] {
            let a = BlasVectorLayout::new(block(params, a_k, 0, SlotsKind::Complex), layout.input);
            let hand = move |r: usize| inputs * g * a_k.div_ceil(r).min(diagonal_k.div_ceil(r));
            assert_largest_radix::<B>(&format!("matvec {layout:?} a.k={a_k}"), n, 1..=1000, hand, |bits| {
                blas_matvec_pt_max_base2k::<B>(&layout, &a, bits)
            });
        }
    }

    // The product by a diagonal ciphertext: `2 g min(L, P)` per giant step, `g` read off the matrix's prepared schedule.
    let m = module.blas_diagonal_ciphertext_alloc_from_infos(&BlasDiagonalCiphertextLayout::new(
        block(params, 60, 0, SlotsKind::Complex),
        LinearTransformationLayout {
            indexes: (0..8).collect(),
            slots: module.ckks_max_slots(),
            strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
        },
    ));
    let g = m.diagonals().giant_steps.iter().map(|gs| gs.diagonals.len()).max().unwrap();
    for a_k in [77, 40] {
        let a = BlasVectorLayout::new(
            block(params, a_k, 0, SlotsKind::Complex),
            BlasVectorShape::new(module.ckks_max_slots(), module.ckks_max_slots(), 0, 1),
        );
        let hand = move |r: usize| 2 * g * a_k.div_ceil(r).min(60usize.div_ceil(r));
        assert_largest_radix::<B>(&format!("matvec a.k={a_k}"), n, 1..=1000, hand, |bits| {
            blas_matvec_diagonal_max_base2k::<B>(&m, &a, bits)
        });
    }

    let maskless = module_layout(module, BlasPlan::new(log_d), params.base2k.into(), diagonal_meta(params, p));
    for algorithm in MASKED_PT_ALGORITHMS {
        let (pt, a) = (left_operand(3, algorithm, p), layout(width(params), 0, 3, 2));
        assert_panics(&format!("{algorithm:?}: a context without masks"), MASKLESS, || {
            blas_matmul_pt_left_max_base2k::<B>(&pt, &a, &maskless, 128)
        });
    }
}

/// Products of one tested backend run at once equal the same products run alone, bit for bit: a `blas_matmul`, a
/// `blas_matmul_prepared` of cold operands and a `blas_matmul_pt_right` under JKLS, each twice, so that two calls read
/// the same prepared operands and the same public operand together, all over one module, context and key store, each
/// with its own `res` and scratch, released together.
fn concurrent<B, F>(params: CKKSTestParams, module: &Module<B>, log_d: usize)
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module, log_d, scale(params));
    let keys = fixture_keys(
        params,
        module,
        &ctx.plan().matmul_galois_elements(module.cyclotomic_order()),
        true,
    );
    let (atk, tsk) = (atk_layout(params), tsk_layout(params));
    let (a, b) = (
        operand(params, module, log_d, 2, 2, 120),
        operand(params, module, log_d, 2, 1, 124),
    );
    let pt = public_operand::<B, F>(params, module, log_d, 1, scale(params), BlasPtAlgorithm::Jkls);
    let left = with_scratch::<B, _>(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk, Reuse), |s| {
        matmul_left_operand(module, &a, &ctx, &keys, Reuse, s)
    })
    .unwrap();
    let right = with_scratch::<B, _>(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk, Reuse), |s| {
        matmul_right_operand(module, &b, &ctx, &keys, Reuse, s)
    })
    .unwrap();
    let res = || sized(params, module, log_d, 2, 1, width(params), 126);
    let matmul = || {
        let mut res = res();
        let bytes = module.blas_matmul_tmp_bytes(&res, &a, &b, &ctx, &tsk, &atk);
        with_scratch::<B, _>(bytes, |s| module.blas_matmul(&mut res, &a, &b, &ctx, &keys, &keys, s)).unwrap();
        grid_snapshot::<B, _>(&res)
    };
    let prepared = || {
        let mut res = res();
        let bytes = module.blas_matmul_prepared_tmp_bytes(&res, &left, &right, &ctx, &tsk, &atk);
        with_scratch::<B, _>(bytes, |s| {
            module.blas_matmul_prepared(&mut res, &left, &right, &ctx, &keys, &keys, s)
        })
        .unwrap();
        grid_snapshot::<B, _>(&res)
    };
    let pt_right = || {
        let mut res = res();
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &a, &pt, &ctx, &atk);
        with_scratch::<B, _>(bytes, |s| module.blas_matmul_pt_right(&mut res, &a, &pt, &ctx, &keys, s)).unwrap();
        grid_snapshot::<B, _>(&res)
    };
    let calls: [&(dyn Fn() -> Vec<Snapshot> + Sync); 3] = [&matmul, &prepared, &pt_right];
    let alone: Vec<_> = calls.iter().map(|call| call()).collect();
    for round in 0..16 {
        let start = Barrier::new(2 * calls.len());
        let at_once: Vec<_> = std::thread::scope(|scope| {
            let threads: Vec<_> = calls
                .iter()
                .chain(&calls)
                .map(|&call| {
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        call()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
                .collect()
        });
        for (i, got) in at_once.iter().enumerate() {
            let call = ["blas_matmul", "blas_matmul_prepared", "blas_matmul_pt_right"][i % calls.len()];
            assert_eq!(
                *got,
                alone[i % calls.len()],
                "{}: {call} at once, round {round}",
                std::any::type_name::<B>()
            );
        }
    }
}

/// Calls with distinct `res` and scratch run at once over one module, context, key store, public operand and prepared
/// operands give, bit for bit, what the same calls give alone, at the first swept `log_d`.
pub fn test_blas_concurrency<B, F>(params: CKKSTestParams, shapes: &BlasParityShapes, module: &Module<B>)
where
    B: BlasParityBackend,
    Module<B>: BlasParityModule<B, F>,
    F: TestScalar + DiagonalArithmetic,
{
    concurrent::<B, F>(params, module, swept_log_d(shapes, module.ckks_max_slots())[0]);
}
