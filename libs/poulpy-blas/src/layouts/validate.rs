//! The validator of every operation, from metadata alone, which its implementation runs before any write. An override
//! runs the same validators, as the references do. They check no budget: poulpy-ckks rejects those inside the
//! implementation, before any write.

use std::cell::RefCell;

use poulpy_ckks::{
    CKKSCompositionError, CKKSCtBounds, CKKSError, CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextInfos,
    api::{
        CKKSModuleInfos, LinearTransformation, LinearTransformationPrepared, LtDiagonalMeta, ckks_linear_transformation_output_k,
        ckks_linear_transformation_output_layout,
    },
    layouts::CKKSPreparedRight,
};
use poulpy_core::layouts::{
    Base2K, BsgsRotations, GGLWEInfos, GGLWELayout, GLWEInfos, GetAutomorphismKey, GetDegree, GetTensorKey, LWEInfos, Rank,
    TorusPrecision,
    prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, Ring, ScratchArena};

use super::{
    BlasContextInfos, BlasContextPrepared, BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextLayout,
    BlasDiagonalCiphertextMap, BlasGrid, BlasGridInfos, BlasGridLine, BlasMatMulOperandInfos, BlasMatMulOperandLayout,
    BlasMatVecMaps, BlasPlan, BlasPtOperandInfos, BlasPtOperandPrepared, BlasVectorInfos, BlasVectorLayout, BlasVectorRef,
    BlasVectorShape, Left,
    ciphertext::{batch_fits, layout_of, replicated_at, slots_of},
    context::{ContextMaps, reduction_projection, reduction_rotations, shift_maps, shift_rotates, sum_rotations},
    diagonal_ciphertext::{AppendAxis, append_rotations},
    grid_batch,
    matmul_operand::{
        left_operand_layout, left_term_layout, product_terms, right_maps, right_operand_layout, right_term_layout, same_shape,
    },
    pt_operand::OperandMaps,
    vector_period,
};

/// The largest radix of a product, as poulpy-ckks's prepared dot product takes: past it two canonical tensor digits can
/// overflow an `i64`.
pub(crate) const MAX_BASE2K: usize = 62;

/// The `InvalidOperand` rejection of `op` for `reason`.
pub(crate) fn reject(op: &'static str, reason: String) -> CKKSError {
    CKKSCompositionError::InvalidOperand { op, reason }.into()
}

/// Rejects `g`, whose blocks are `blocks`, unless it reads `2^log_d x 2^log_d` blocks, at least one and exactly
/// `rows x cols` of them, that share one layout, whose `log_delta` does not exceed their `k`, and whose label allows its
/// batch on ring `R`: a power of two from the batch the label proves to the slots' dense batch, with the label at most
/// `log2` of the degree.
pub(crate) fn ensure_grid<R: Ring>(
    op: &'static str,
    log_d: usize,
    g: &impl BlasGridInfos,
    blocks: &[impl CKKSCtBounds],
) -> Result<()> {
    ensure_dimension(op, log_d, g)?;
    if blocks.is_empty() || g.rows().checked_mul(g.cols()) != Some(blocks.len()) {
        return Err(reject(
            op,
            format!("{} blocks do not form a {} x {} grid", blocks.len(), g.rows(), g.cols()),
        ));
    }
    let first = g.block_layout();
    if blocks.iter().any(|block| layout_of(block) != first) {
        return Err(reject(op, "the blocks of a grid do not share one layout".into()));
    }
    ensure_record::<R>(op, g)
}

/// Rejects `g` unless it reads `2^log_d x 2^log_d` blocks.
pub(crate) fn ensure_dimension(op: &'static str, log_d: usize, g: &impl BlasGridInfos) -> Result<()> {
    if g.log_d() != log_d {
        return Err(reject(op, format!("grid dimension 2^{} differs from 2^{log_d}", g.log_d())));
    }
    Ok(())
}

/// Rejects `g` unless the `log_delta` of its block layout does not exceed its `k` and its label allows its batch on
/// ring `R`, as [`ensure_grid`] states.
pub(crate) fn ensure_record<R: Ring>(op: &'static str, g: &impl BlasGridInfos) -> Result<()> {
    let (first, log_d) = (g.block_layout(), g.log_d());
    if first.log_delta() > first.k().as_usize() {
        return Err(reject(
            op,
            format!(
                "log_delta={} exceeds the blocks' k={}",
                first.log_delta(),
                first.k().as_usize()
            ),
        ));
    }
    if !batch_fits::<R>(g) {
        return Err(reject(
            op,
            format!(
                "a batch of {} matrices of 2^{log_d} x 2^{log_d} blocks does not fit blocks of degree {} labelled log_sparsity={}",
                grid_batch::<R>(g),
                first.n().as_usize(),
                first.log_sparsity()
            ),
        ));
    }
    Ok(())
}

/// Rejects `g` unless it has `rows x cols` blocks.
pub(crate) fn ensure_shape<G: BlasGridInfos>(op: &'static str, g: &G, rows: usize, cols: usize) -> Result<()> {
    if (g.rows(), g.cols()) != (rows, cols) {
        return Err(reject(
            op,
            format!("a {} x {} grid where {rows} x {cols} is required", g.rows(), g.cols()),
        ));
    }
    Ok(())
}

/// Rejects `res` unless its blocks are of degree `n`, the module's, and `inputs` unless each holds blocks of the degree,
/// radix and rank of `res`'s.
pub(crate) fn ensure_layouts(op: &'static str, n: usize, res: &dyn BlasGridInfos, inputs: &[&dyn BlasGridInfos]) -> Result<()> {
    let ring = |g: &dyn BlasGridInfos| {
        let block = g.block_layout();
        (block.n().as_usize(), block.base2k().as_usize(), block.rank().as_usize())
    };
    if ring(res).0 != n {
        return Err(reject(
            op,
            format!("blocks of degree {} on a module of degree {n}", ring(res).0),
        ));
    }
    if let Some(input) = inputs.iter().find(|g| ring(**g) != ring(res)) {
        return Err(reject(
            op,
            format!(
                "blocks of (n, base2k, rank) = {:?} read into blocks of {:?}",
                ring(*input),
                ring(res)
            ),
        ));
    }
    Ok(())
}

/// Rejects a product whose left operand's `a_cols` block columns differ from its right operand's `b_rows` block rows.
pub(crate) fn ensure_inner(op: &'static str, a_cols: usize, b_rows: usize) -> Result<()> {
    if a_cols != b_rows {
        return Err(reject(op, format!("{a_cols} block columns against {b_rows} block rows")));
    }
    Ok(())
}

/// Rejects maps encoded on a module of `slots` slots at radix `base2k` and torus width `k` unless `module` has `slots`
/// slots and every grid of `inputs`, the grids they read, holds blocks at `base2k` no narrower than `k`: the scratch
/// queries bound a diagonal by its input block.
pub(crate) fn ensure_maps<M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    slots: usize,
    base2k: Base2K,
    k: TorusPrecision,
    inputs: &[&dyn BlasGridInfos],
) -> Result<()> {
    let inputs: Vec<CKKSLayout> = inputs.iter().map(|g| g.block_layout()).collect();
    ensure_maps_reading(op, module, slots, base2k, k, &inputs, "blocks")
}

/// [`ensure_maps`] over the layouts of the ciphertexts the maps read, named `what` in its reasons.
fn ensure_maps_reading<M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    slots: usize,
    base2k: Base2K,
    k: TorusPrecision,
    inputs: &[CKKSLayout],
    what: &str,
) -> Result<()> {
    if module.ckks_max_slots() != slots {
        return Err(reject(
            op,
            format!(
                "maps encoded for {slots} slots on a module of {} slots",
                module.ckks_max_slots()
            ),
        ));
    }
    if let Some(input) = inputs.iter().find(|ct| ct.base2k() != base2k) {
        return Err(reject(
            op,
            format!(
                "{what} at base2k={} read by maps at base2k={}",
                input.base2k().as_usize(),
                base2k.as_usize()
            ),
        ));
    }
    if let Some(input) = inputs.iter().find(|ct| ct.k() < k) {
        return Err(reject(
            op,
            format!("{what} at k={} read by maps at k={}", input.k().as_usize(), k.as_usize()),
        ));
    }
    Ok(())
}

/// Rejects `v`, whose ciphertexts are `cts`, unless its shape holds an entry, an entry per ciphertext and a positive
/// stride, and `cts` number its ciphertexts, share one layout whose `log_delta` does not exceed its `k`, and carry a
/// label, at most `log2` of the degree, proving a period on ring `R` that holds the entries.
pub(crate) fn ensure_vector<R: Ring>(op: &'static str, v: &impl BlasVectorInfos, cts: &[impl CKKSCtBounds]) -> Result<()> {
    let shape = v.vector_shape();
    if shape.len == 0 || shape.chunk == 0 || shape.stride == 0 {
        return Err(reject(
            op,
            format!(
                "a vector of {} entries, {} per ciphertext, at stride {}",
                shape.len, shape.chunk, shape.stride
            ),
        ));
    }
    if cts.len() != shape.count() {
        return Err(reject(
            op,
            format!(
                "{} ciphertexts do not hold {} entries of {} per ciphertext",
                cts.len(),
                shape.len,
                shape.chunk
            ),
        ));
    }
    let first = v.ct_layout();
    if cts.iter().any(|ct| layout_of(ct) != first) {
        return Err(reject(op, "the ciphertexts of a vector do not share one layout".into()));
    }
    if first.log_delta() > first.k().as_usize() {
        return Err(reject(
            op,
            format!(
                "log_delta={} exceeds the ciphertexts' k={}",
                first.log_delta(),
                first.k().as_usize()
            ),
        ));
    }
    let (n, s) = (first.n().as_usize(), first.log_sparsity());
    let period = vector_period::<R>(v);
    if s > n.ilog2() as usize || !shape.fits(slots_of::<R>(n), period) {
        return Err(reject(
            op,
            format!(
                "entries at slots {} + {} e for e < {} do not fit ciphertexts of degree {n} labelled log_sparsity={s}",
                shape.offset, shape.stride, shape.chunk
            ),
        ));
    }
    Ok(())
}

/// Rejects a vector of shape `have` where the operand reads or writes one of shape `want`, its `side`.
fn ensure_vector_shape(op: &'static str, side: &str, have: BlasVectorShape, want: BlasVectorShape) -> Result<()> {
    if have != want {
        return Err(reject(
            op,
            format!("a vector of {have:?} where the operand's {side} is {want:?}"),
        ));
    }
    Ok(())
}

/// The degree, radix and rank of ciphertexts of `ct`.
fn ring(ct: CKKSLayout) -> (usize, usize, usize) {
    (ct.n().as_usize(), ct.base2k().as_usize(), ct.rank().as_usize())
}

/// Rejects `res` unless its ciphertexts are of degree `n`, the module's, and `a`'s of the degree, radix and rank of
/// `res`'s.
fn ensure_vector_layouts(op: &'static str, n: usize, res: CKKSLayout, a: CKKSLayout) -> Result<()> {
    if ring(res).0 != n {
        return Err(reject(
            op,
            format!("ciphertexts of degree {} on a module of degree {n}", ring(res).0),
        ));
    }
    if ring(a) != ring(res) {
        return Err(reject(
            op,
            format!(
                "ciphertexts of (n, base2k, rank) = {:?} read into ciphertexts of {:?}",
                ring(a),
                ring(res)
            ),
        ));
    }
    Ok(())
}

/// One rotation-key look-up of an operation: the rotation, the width of the input it key-switches, and the rank of the
/// blocks. The enumerators of an operation's look-ups yield these in the order its validator checks them; a validator
/// checks each against the store, a [`BlasKeyUses`](super::BlasKeyUses) query collects them.
#[derive(Clone, Copy)]
pub(crate) struct RotationUse {
    pub(crate) rotation: i64,
    pub(crate) k: TorusPrecision,
    pub(crate) rank: Rank,
}

/// Receives the look-ups of an enumerator, which stops at the first error.
pub(crate) type UseSink<'f> = &'f mut dyn FnMut(RotationUse) -> Result<()>;

/// Fails with `MissingAutomorphismKey` unless `keys` holds a key for `u`, or with `InvalidOperand` unless it is of the
/// rank of the blocks it switches. Element 1 is the identity, which poulpy-ckks applies without a key.
pub(crate) fn ensure_rotation<BE, M, K>(op: &'static str, module: &M, keys: &K, u: RotationUse) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    K: GetAutomorphismKey<BE>,
{
    let p = module.ckks_galois_element(u.rotation);
    if p == 1 {
        return Ok(());
    }
    let key = keys
        .get_automorphism_key(p, u.k)
        .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
            op,
            rotation: u.rotation,
            k: u.k.into(),
            source,
        })?;
    ensure_key_rank(op, "rotation", key.rank(), u.rank)
}

/// Fails with `MissingRelinearizationKey` unless `keys` holds a tensor key at width `k`, then with `InvalidOperand`
/// unless it is of `rank`, that of the blocks it relinearizes.
pub(crate) fn ensure_tensor_key<BE: Backend, K: GetTensorKey<BE>>(
    op: &'static str,
    keys: &K,
    k: TorusPrecision,
    rank: Rank,
) -> Result<()> {
    let key = keys
        .get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source })?;
    ensure_key_rank(op, "tensor", key.rank(), rank)
}

/// Rejects a `kind` key of rank `key` for blocks of rank `blocks`, on which poulpy-core's key switch would panic.
fn ensure_key_rank(op: &'static str, kind: &str, key: Rank, blocks: Rank) -> Result<()> {
    if key != blocks {
        return Err(reject(op, format!("a {kind} key of rank {key} for blocks of rank {blocks}")));
    }
    Ok(())
}

/// The look-ups of evaluating `lt` from blocks of rank `rank` and width `src_k` into a ciphertext of width `dst_k`: the
/// baby steps at `src_k`, the non-empty giant steps at [`ckks_linear_transformation_output_k`].
///
/// These are poulpy-ckks's widths on inputs it evaluates. On an input it rejects with a budget error, this may name a key
/// poulpy-ckks never looks up for that input: a store holding exactly the widths in use then fails the validator with
/// `MissingAutomorphismKey` instead of that budget error.
pub(crate) fn lt_uses<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
    f: UseSink<'_>,
) -> Result<()> {
    bsgs_uses(lt, src_k, ckks_linear_transformation_output_k(lt, src_k, dst_k), rank, f)
}

/// The rotation keys poulpy-ckks looks up to evaluate `lt`, by its rotation schedule: the keys of its baby steps at
/// `baby_k`, then those of its non-empty giant steps at `giant_k`, one per rotation for
/// [`BsgsRotations::PerRotation`], the doubling passes' and the giant tree's for [`BsgsRotations::PowersOfTwo`].
fn bsgs_uses<P>(
    lt: &LinearTransformation<P>,
    baby_k: TorusPrecision,
    giant_k: TorusPrecision,
    rank: Rank,
    f: UseSink<'_>,
) -> Result<()> {
    let giants = lt.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty()).map(|gs| gs.rot);
    let giants = match lt.rotations {
        BsgsRotations::PerRotation => giants.collect(),
        rotations => rotations.giant_key_rotations(giants),
    };
    let babies = lt.rotations.baby_key_rotations(lt.baby_steps().iter().copied());
    let babies = babies.into_iter().map(|rotation| (rotation, baby_k));
    babies
        .chain(giants.into_iter().map(|rotation| (rotation, giant_k)))
        .try_for_each(|(rotation, k)| f(RotationUse { rotation, k, rank }))
}

/// Whether `uses`, an enumerator, yields a look-up [`ensure_rotation`] makes: a rotation other than the identity.
pub(crate) fn rotates<M: CKKSModuleInfos + ?Sized>(module: &M, uses: impl FnOnce(UseSink<'_>) -> Result<()>) -> bool {
    let mut any = false;
    let _ = uses(&mut |u| {
        any |= module.ckks_galois_element(u.rotation) != 1;
        Ok(())
    });
    any
}

/// Panics unless `scratch` holds the `required` bytes of `op`'s own scratch query; `op` runs it before it writes
/// anything.
pub(crate) fn assert_scratch<BE: Backend>(op: &str, required: usize, scratch: &ScratchArena<'_, BE>) {
    assert!(
        scratch.available() >= required,
        "{op} needs {required} scratch bytes, but only {} are available",
        scratch.available()
    );
}

/// `keys`, recording the layout of every key looked up through it: an operation sizes its scratch query on the keys its
/// checks look up. An implementation runs on keys its validator accepted, so it repeats the validator's look-ups through
/// it only to record their layouts.
pub(crate) struct KeyLayouts<'a, K> {
    keys: &'a K,
    layouts: RefCell<Vec<GGLWELayout>>,
}

impl<'a, K> KeyLayouts<'a, K> {
    pub(crate) fn new(keys: &'a K) -> Self {
        Self {
            keys,
            layouts: RefCell::default(),
        }
    }

    /// The largest `bytes` over the keys looked up, `0` without any.
    pub(crate) fn max_bytes(&self, bytes: impl Fn(&GGLWELayout) -> usize) -> usize {
        self.layouts.borrow().iter().map(bytes).max().unwrap_or(0)
    }

    /// The largest `bytes` over the keys looked up, `bytes(None)` without any: the query of a call that rotates nothing.
    pub(crate) fn max_bytes_or_unused(&self, bytes: impl Fn(Option<&GGLWELayout>) -> usize) -> usize {
        let layouts = self.layouts.borrow();
        if layouts.is_empty() {
            return bytes(None);
        }
        layouts.iter().map(|layout| bytes(Some(layout))).max().unwrap_or(0)
    }

    fn record<G: GGLWEInfos>(&self, key: &G) {
        let layout = key.gglwe_layout();
        let mut layouts = self.layouts.borrow_mut();
        if !layouts.contains(&layout) {
            layouts.push(layout);
        }
    }
}

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for KeyLayouts<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_automorphism_key(p, k)?;
        self.record(&key);
        Ok(key)
    }
}

impl<BE: Backend, K: GetTensorKey<BE>> GetTensorKey<BE> for KeyLayouts<'_, K> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_tensor_key(k)?;
        self.record(&key);
        Ok(key)
    }
}

/// Panics unless `scratch` holds the largest `bytes` over the layouts of the rotation keys that evaluating `lts` from
/// blocks of rank `rank` and width `src_k` into width `dst_k` looks up in `keys`, as [`assert_scratch`] does; a call
/// that looks up none asserts nothing. The keys passed `op`'s validator, so the look-ups only record their layouts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_maps_scratch<BE: Backend, M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    lts: &[&LinearTransformationPrepared<BE>],
    keys: &impl GetAutomorphismKey<BE>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
    bytes: impl Fn(&GGLWELayout) -> usize,
    scratch: &ScratchArena<'_, BE>,
) {
    let used = KeyLayouts::new(keys);
    for lt in lts {
        let _ = ensure_lt_keys(op, module, lt, &used, src_k, dst_k, rank);
    }
    assert_scratch(op, used.max_bytes(bytes), scratch);
}

/// Fails as [`ensure_rotation`] unless `keys` holds every key of [`lt_uses`], of that rank.
pub(crate) fn ensure_lt_keys<BE, M, P, K>(
    op: &'static str,
    module: &M,
    lt: &LinearTransformation<P>,
    keys: &K,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    P: LtDiagonalMeta,
    K: GetAutomorphismKey<BE>,
{
    lt_uses(lt, src_k, dst_k, rank, &mut |u| ensure_rotation(op, module, keys, u))
}

/// Checks a [`blas_rotate`](crate::api::BlasMatInterOps::blas_rotate) call, and returns the batch `res` records, `a`'s:
/// `a` and `res` are grids at `a`'s `log_d` whose blocks tile their shape and share one layout whose label allows their
/// batch record ([`BlasGrid::blocks`], [`BlasGridInfos::mat_slots`]), `res` has `a`'s shape and its blocks `a`'s
/// degree, radix and rank, and `keys` holds the rotation key at the narrower of `a`'s and `res`'s widths, of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape or layout it rejects, then `MissingAutomorphismKey`, or `InvalidOperand` for a
/// key of another rank.
pub fn validate_blas_rotate<BE: Backend>(
    module: &Module<BE>,
    r: i64,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_rotate";
    ensure_same_grids(op, module, res, a)?;
    rotate_uses(r, res, a, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// Checks a [`blas_sum`](crate::api::BlasMatInterOps::blas_sum) call as [`validate_blas_rotate`] does, with the
/// rotation keys of the batch at the narrower of `a`'s and `res`'s widths, of `a`'s rank, and returns the batch `res`
/// records, `a`'s.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape or layout it rejects, then, key by key, `MissingAutomorphismKey`, or
/// `InvalidOperand` for a key of another rank.
pub fn validate_blas_sum<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_sum";
    ensure_same_grids(op, module, res, a)?;
    sum_uses::<BE::Ring>(res, a, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// The look-up of [`validate_blas_rotate`]: the rotation by `r d^2` slots, at the narrower of `a`'s and `res`'s widths.
pub(crate) fn rotate_uses(r: i64, res: &impl BlasGridInfos, a: &impl BlasGridInfos, f: UseSink<'_>) -> Result<()> {
    f(RotationUse {
        rotation: r << (2 * a.log_d()),
        k: rotation_key_width(res, a),
        rank: a.block_layout().rank(),
    })
}

/// The look-ups of [`validate_blas_sum`]: the rotations of the batch, at the narrower of `a`'s and `res`'s widths.
pub(crate) fn sum_uses<R: Ring>(res: &impl BlasGridInfos, a: &impl BlasGridInfos, f: UseSink<'_>) -> Result<()> {
    let (k, rank) = (rotation_key_width(res, a), a.block_layout().rank());
    sum_rotations(a.log_d(), grid_batch::<R>(a)).try_for_each(|rotation| f(RotationUse { rotation, k, rank }))
}

/// `a` and `res` are grids at `a`'s `log_d`, `res` of `a`'s shape and its blocks of the module's degree and `a`'s radix
/// and rank.
fn ensure_same_grids<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, a.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, a.log_d(), res, res.blocks())?;
    ensure_shape(op, res, a.rows(), a.cols())?;
    ensure_layouts(op, module.n(), res, &[a])
}

/// The width `blas_rotate` and `blas_sum` rotate at, the narrower of `a`'s and `res`'s: `ckks_rotate_into` reads `a` at
/// a narrower `res`'s width, and `ckks_copy` leaves `res` at it.
fn rotation_key_width(res: &impl BlasGridInfos, a: &impl BlasGridInfos) -> TorusPrecision {
    a.block_layout().k().min(res.block_layout().k())
}

/// Checks a [`blas_transpose`](crate::api::BlasMatIntraOps::blas_transpose) call, and returns the batch `res` records,
/// `a`'s: `a` and `res` are grids at the context's `log_d` whose blocks tile their shape and share one layout whose
/// label allows their batch record ([`BlasGrid::blocks`], [`crate::layouts::BlasGridInfos::mat_slots`]), `res` has the
/// transposed shape of `a` and its blocks `a`'s degree, radix and rank, `ctx` was encoded on a module of `module`'s
/// slot count at `a`'s radix, and `keys` holds every key of the transpose at the width it is looked up at, of `a`'s
/// rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout or context it rejects, then, key by key, `MissingAutomorphismKey`, or
/// `InvalidOperand` for a key of another rank.
pub fn validate_blas_transpose<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_transpose";
    ensure_intra(op, module, res, a, ctx, a.cols(), a.rows())?;
    ensure_mapped(op, module, ctx, a)?;
    transpose_uses(res, a, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// Checks a [`blas_rotate_rows`](crate::api::BlasMatIntraOps::blas_rotate_rows) call as [`validate_blas_transpose`]
/// does, `res` of `a`'s shape and the keys of the maps the shift by `k` applies, or, on a replicated grid one block
/// high, of its rotation at the narrower of `a`'s and `res`'s widths, and returns the batch `res` records, `a`'s. A
/// shift that evaluates no map, by a multiple of `d` or that rotation, checks the context's slot count alone, not its
/// radix and width.
///
/// # Errors
///
/// As [`validate_blas_transpose`].
pub fn validate_blas_rotate_rows<BE: Backend>(
    module: &Module<BE>,
    k: usize,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_shift(module, k, res, a, true, ctx, keys)
}

/// [`validate_blas_rotate_rows`] for [`blas_rotate_cols`](crate::api::BlasMatIntraOps::blas_rotate_cols).
///
/// # Errors
///
/// As [`validate_blas_transpose`].
pub fn validate_blas_rotate_cols<BE: Backend>(
    module: &Module<BE>,
    k: usize,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_shift(module, k, res, a, false, ctx, keys)
}

/// `a` and `res` are grids at the context's `log_d`, `res` of `rows x cols` blocks of `a`'s degree, radix and rank, and
/// the context encoded for the module's slot count.
fn ensure_intra<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, ctx.log_d(), res, res.blocks())?;
    ensure_shape(op, res, rows, cols)?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), ctx.coeffs_meta().k, &[])
}

/// The context's maps read `a`: its blocks at their radix and at least as wide.
fn ensure_mapped<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    ctx: &BlasContextPrepared<BE>,
    a: &impl BlasGrid<BE>,
) -> Result<()> {
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), ctx.coeffs_meta().k, &[a])
}

/// Checks a [`blas_sum_rows`](crate::api::BlasMatIntraOps::blas_sum_rows) call as [`validate_blas_rotate_rows`] does,
/// with the keys of its rotations and of its projection, and returns the batch `res` records, `a`'s. The row sum of a
/// replicated grid evaluates no projection and checks the context's slot count alone, not its radix and width.
///
/// # Errors
///
/// As [`validate_blas_transpose`].
pub fn validate_blas_sum_rows<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_reduction(module, res, a, true, ctx, keys)
}

/// [`validate_blas_sum_rows`] for [`blas_sum_cols`](crate::api::BlasMatIntraOps::blas_sum_cols).
///
/// # Errors
///
/// As [`validate_blas_sum_rows`].
pub fn validate_blas_sum_cols<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_reduction(module, res, a, false, ctx, keys)
}

/// The checks of an axis reduction along the rows (`rows`) or the columns.
fn validate_reduction<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    rows: bool,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = if rows { "blas_sum_rows" } else { "blas_sum_cols" };
    ensure_intra(op, module, res, a, ctx, a.rows(), a.cols())?;
    if reduction_projection(&ctx.maps, rows, replicated_at(ctx.slots(), a)).is_some() {
        ensure_mapped(op, module, ctx, a)?;
    }
    reduction_uses(res, a, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// The look-ups of an axis reduction of `a` into `res` along its rows (`rows`) or columns, of `a`'s rank: along the rows
/// of a replicated `a`, the rotations of its tree at the narrower of `a`'s and `res`'s widths; otherwise those of its
/// tree at `a`'s width, of the projection from `a`'s width into `res`'s and of the broadcast at the width the
/// projection leaves.
pub(crate) fn reduction_uses<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    let (a_k, rank) = (block.k(), block.rank());
    let projection = reduction_projection(maps, rows, replicated_at(maps.slots(), a));
    let tree_k = if projection.is_some() {
        a_k
    } else {
        rotation_key_width(res, a)
    };
    reduction_rotations(a.log_d(), rows).try_for_each(|rotation| {
        f(RotationUse {
            rotation,
            k: tree_k,
            rank,
        })
    })?;
    let Some(projection) = projection else {
        return Ok(());
    };
    lt_uses(projection, a_k, res_k, rank, &mut *f)?;
    let k = ckks_linear_transformation_output_k(projection, a_k, res_k);
    reduction_rotations(a.log_d(), rows).try_for_each(|rotation| {
        f(RotationUse {
            rotation: -rotation,
            k,
            rank,
        })
    })
}

/// The checks of a row (`rows`) or column shift by `k`.
fn validate_shift<BE: Backend>(
    module: &Module<BE>,
    k: usize,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    rows: bool,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = if rows { "blas_rotate_rows" } else { "blas_rotate_cols" };
    ensure_intra(op, module, res, a, ctx, a.rows(), a.cols())?;
    let extent = if rows { a.rows() } else { a.cols() };
    if !k.is_multiple_of(1 << ctx.log_d()) && !shift_rotates(rows, extent, replicated_at(ctx.slots(), a)) {
        ensure_mapped(op, module, ctx, a)?;
    }
    shift_uses(k, res, a, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// The look-ups of [`validate_blas_transpose`]: the transpose map from `a`'s width into `res`'s.
pub(crate) fn transpose_uses<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    lt_uses(maps.transpose(), block.k(), res_k, block.rank(), f)
}

/// The look-ups of a row (`rows`) or column shift by `k`: the rotation by `k1 d` slots at the narrower of `a`'s and
/// `res`'s widths where it rotates ([`shift_rotates`]), or else the keys of each of its maps ([`shift_maps`]), from
/// `a`'s width into `res`'s.
pub(crate) fn shift_uses<P: LtDiagonalMeta>(
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    let (extent, k1) = (if rows { a.rows() } else { a.cols() }, k % (1 << maps.log_d()));
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    if k1 != 0 && shift_rotates(rows, extent, replicated_at(maps.slots(), a)) {
        return f(RotationUse {
            rotation: (k1 << maps.log_d()) as i64,
            k: rotation_key_width(res, a),
            rank: block.rank(),
        });
    }
    shift_maps(maps, k1, rows, extent)
        .into_iter()
        .try_for_each(|lt| lt_uses(lt, block.k(), res_k, block.rank(), &mut *f))
}

/// Whether a left product by `b` of `a` takes `b`'s replicated-input block maps: `a`'s label proves one matrix and `b`
/// holds them.
pub(crate) fn left_takes_replicated<T>(b: &OperandMaps<T>, a: &impl BlasGridInfos) -> bool {
    b.has_replicated() && replicated_at(b.slots(), a)
}

/// Why a left product under `JklsStored` or `JklsGalois` rejects a context.
pub(crate) const MASKLESS: &str =
    "a context without the column halves of psi^k: its plan selects neither JklsStored nor JklsGalois";

/// The context's masks a product by public matrices of an operand of `plan` evaluates, in the operand's term order
/// after the first: none but under `JklsStored` and `JklsGalois`; on the right phi^k, on the left `rows_left(k)` then
/// `rows_right(k)`, for `1 <= k < d`.
///
/// # Panics
///
/// Under those paths, when `maps` is at another block dimension than `plan`, and on the `left` when `maps` lacks the
/// column halves of psi^k, both of which the validators reject first.
pub(crate) fn context_masks<T>(plan: BlasPlan, maps: &ContextMaps<T>, left: bool) -> Vec<&T> {
    if !plan.pt_algorithm().masked() {
        return Vec::new();
    }
    assert!(
        maps.log_d() == plan.log_d(),
        "a context at another block dimension than the operand's"
    );
    let d = 1usize << plan.log_d();
    if left {
        assert!(maps.has_masked_rows(), "{MASKLESS}");
        (1..d).flat_map(|k| [maps.rows_left(k), maps.rows_right(k)]).collect()
    } else {
        (1..d).map(|k| maps.rotate_cols(k)).collect()
    }
}

/// Under `JklsStored` and `JklsGalois`, `ctx`'s masks read the results of the operand's first map: `ctx` is at `a`'s
/// `log_d`, encoded for the module's slot count at `a`'s radix and no wider than `a`, and on the `left` holds the
/// column halves of psi^k.
fn ensure_masks<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    ctx: &BlasContextPrepared<BE>,
    left: bool,
) -> Result<()> {
    ensure_dimension(op, ctx.log_d(), a)?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), ctx.coeffs_meta().k, &[a])?;
    if left && !ctx.maps.has_masked_rows() {
        return Err(reject(op, MASKLESS.into()));
    }
    Ok(())
}

/// Checks a [`blas_matmul`](crate::api::BlasMatMulOps::blas_matmul) call, and returns the batch `res` records, the
/// larger of `a`'s and `b`'s: `a` and `b` as their preparations check them ([`validate_blas_matmul_prepare_left`],
/// [`validate_blas_matmul_prepare_right`]), `res` against them as [`validate_blas_matmul_prepared`] checks it against
/// their operands, and `tsk` at the wider of their term widths
/// ([`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout),
/// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout)), of their rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, radix or context it rejects, then, key by key, `MissingAutomorphismKey`
/// or `MissingRelinearizationKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matmul<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    b: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    tsk: &impl GetTensorKey<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul";
    ensure_factor(op, module, a, ctx)?;
    ensure_factor(op, module, b, ctx)?;
    ensure_product(op, module, res, a, b)?;
    matmul_uses(a, b, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))?;
    let (k, rank) = matmul_tensor(a, b, &ctx.maps);
    ensure_tensor_key(op, tsk, k, rank)?;
    Ok(grid_batch::<BE::Ring>(a).max(grid_batch::<BE::Ring>(b)))
}

/// Checks a [`blas_matmul_prepare_left`](crate::api::BlasMatMulOps::blas_matmul_prepare_left) call: `a` is a grid at
/// the context's `log_d` whose blocks tile its shape and share one layout whose label allows its batch record, of the
/// module's degree, `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, `res` has the shape of the
/// infos [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) gives for `a`, `ctx` and
/// `res`'s hint (the hint, the dimensions, and the degree, radix, width and rank of the source's and the reported
/// blocks, whatever their metadata), and `keys` holds every key of sigma from `a`'s width and of each phi^k from sigma's
/// result into the terms' width, at the width it is looked up at, of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, layout or context it rejects or for `res` of another shape, then, key by key,
/// `MissingAutomorphismKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matmul_prepare_left<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasMatMulOperandInfos,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_matmul_prepare_left";
    ensure_factor(op, module, a, ctx)?;
    ensure_allocated(op, res, left_operand_layout(a, &ctx.maps, res.hint()))?;
    left_uses(a, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))
}

/// [`validate_blas_matmul_prepare_left`] for
/// [`blas_matmul_prepare_right`](crate::api::BlasMatMulOps::blas_matmul_prepare_right), with tau and psi^k, or for a
/// replicated `b` [`BlasContextPrepared::tau_rep`] and [`BlasContextPrepared::rotate_rows_rep`], and the infos
/// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) gives.
///
/// # Errors
///
/// As [`validate_blas_matmul_prepare_left`].
pub fn validate_blas_matmul_prepare_right<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasMatMulOperandInfos,
    b: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_matmul_prepare_right";
    ensure_factor(op, module, b, ctx)?;
    ensure_allocated(op, res, right_operand_layout(b, &ctx.maps, res.hint()))?;
    right_uses(b, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))
}

/// Rejects `res` unless its shape is that of `want`, the infos of a preparation of its source under its hint: the hint,
/// the dimensions and the GLWE layouts of the source's and of the reported blocks. Their metadata may differ, and a
/// preparation then reports `want`.
fn ensure_allocated(op: &'static str, res: &impl BlasMatMulOperandInfos, want: BlasMatMulOperandLayout) -> Result<()> {
    if !same_shape(res, &want) {
        return Err(reject(
            op,
            "the operand was allocated for another grid, context or hint".into(),
        ));
    }
    Ok(())
}

/// Checks a [`blas_matmul_prepared`](crate::api::BlasMatMulOps::blas_matmul_prepared) call, and returns the batch `res`
/// records, the larger of the operands': `a` and `b` are prepared operands, read through their
/// [`BlasMatMulOperandInfos`], at one `log_d`, with at least one block each and a batch record their label allows, a
/// hot operand at the `log_d` of `ctx`, which was encoded on a module of `module`'s slot count at its radix; `res` is a
/// grid at that `log_d` whose blocks tile its shape and share one layout whose label allows its batch record, of `a`'s
/// rows and `b`'s columns, of the module's degree and the operands' radix and rank, at a radix of 62 at most; `a`'s
/// columns match `b`'s rows; `keys` holds every key of a hot side's terms under `ctx`, as the preparation of its source
/// looks them up; and `tsk` holds the tensor key at the wider of the terms' widths, a cold side's own, a hot side's
/// under `ctx`, the one width the product looks it up at, of their rank. A cold side reads neither `ctx` nor `keys`, and
/// the check reads no hot side's scale: the product spends that of the maps of `ctx`.
///
/// # Errors
///
/// `InvalidOperand` for an operand without a batch record, or a grid, shape, layout, radix or context it rejects, then,
/// key by key, `MissingAutomorphismKey` or `MissingRelinearizationKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matmul_prepared<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContextPrepared<BE>,
    tsk: &impl GetTensorKey<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_prepared";
    let batch = ensure_operand::<BE>(op, a.log_d(), a)?.max(ensure_operand::<BE>(op, a.log_d(), b)?);
    ensure_hot_context(op, module, a, ctx)?;
    ensure_hot_context(op, module, b, ctx)?;
    ensure_product(op, module, res, a, b)?;
    prepared_uses(a, b, &ctx.maps, &mut |u| ensure_rotation(op, module, keys, u))?;
    let (k, rank) = prepared_tensor(a, b, &ctx.maps);
    ensure_tensor_key(op, tsk, k, rank)?;
    Ok(batch)
}

/// Checks a [`blas_matmul_pt_right`](crate::api::BlasMatMulOps::blas_matmul_pt_right) call, and returns the batch `res`
/// records, `a`'s: `a` and `res` are grids at `b`'s `log_d` whose blocks tile their shape and share one layout whose
/// label allows their batch record, `a`'s columns match `b`'s rows, `res` has `a`'s rows and `b`'s columns, its blocks
/// have the module's degree and `a`'s radix and rank, `b` was encoded on a module of `module`'s slot count at `a`'s
/// radix, `b`'s public matrices divide `a`'s batch, and `keys` holds every key of `b`'s maps at the width it is looked
/// up at, of `a`'s rank: under `Jkls` the first map's from `a`'s width, then the blocks' from the width it leaves.
/// Under `JklsStored` and `JklsGalois`, `ctx` is at `a`'s `log_d`, encoded for the module's slot count at `a`'s radix,
/// no wider than `a`, on the left holding the column halves of psi^k, and on the right `b` holds one public matrix;
/// `keys` then holds the first map's keys from `a`'s width and the masks' from the width it leaves. Otherwise `ctx` is
/// not read.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, operand, count or context it rejects, then, key by key,
/// `MissingAutomorphismKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matmul_pt_right<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    b: &BlasPtOperandPrepared<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_pt_right";
    ensure_grid::<BE::Ring>(op, b.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, b.log_d(), res, res.blocks())?;
    ensure_inner(op, a.cols(), b.rows())?;
    ensure_shape(op, res, a.rows(), b.cols())?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, b.slots(), b.base2k(), b.coeffs_meta().k, &[a])?;
    let batch = grid_batch::<BE::Ring>(a);
    if !batch.is_multiple_of(b.count()) {
        return Err(reject(
            op,
            format!("{} public matrices do not divide the {batch} matrix slots", b.count()),
        ));
    }
    if b.plan().pt_algorithm().masked() {
        if b.count() != 1 {
            return Err(reject(
                op,
                format!(
                    "{} public matrices under {:?}, whose rotations by multiples of d read past a matrix slot",
                    b.count(),
                    b.plan().pt_algorithm()
                ),
            ));
        }
        ensure_masks(op, module, a, ctx, false)?;
    }
    let masks = context_masks(b.plan(), &ctx.maps, false);
    pt_right_uses(res, a, &b.maps, &masks, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(batch)
}

/// Checks a [`blas_matmul_pt_left`](crate::api::BlasMatMulOps::blas_matmul_pt_left) call, and returns the batch `res`
/// records, `a`'s: `a` and `res` are grids at `b`'s `log_d` whose blocks tile their shape and share one layout whose
/// label allows their batch record, `b`'s columns match `a`'s rows, `res` has `b`'s rows and `a`'s columns, its blocks
/// have the module's degree and `a`'s radix and rank, `b` was encoded on a module of `module`'s slot count at `a`'s
/// radix, `b`'s public matrices divide `a`'s batch, and `keys` holds every key of `b`'s maps at the width it is looked
/// up at, of `a`'s rank: under `Jkls` the first map's from `a`'s width, then the blocks' from the width it leaves.
/// Under `JklsStored` and `JklsGalois`, `ctx` is at `a`'s `log_d`, encoded for the module's slot count at `a`'s radix,
/// no wider than `a`, and holds the column halves of psi^k; `keys` then holds the first map's keys from `a`'s width and
/// the masks' from the width it leaves. Otherwise `ctx` is not read.
///
/// # Errors
///
/// As [`validate_blas_matmul_pt_right`].
pub fn validate_blas_matmul_pt_left<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    b: &BlasPtOperandPrepared<BE, Left>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_pt_left";
    ensure_grid::<BE::Ring>(op, b.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, b.log_d(), res, res.blocks())?;
    ensure_inner(op, b.cols(), a.rows())?;
    ensure_shape(op, res, b.rows(), a.cols())?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, b.slots(), b.base2k(), b.coeffs_meta().k, &[a])?;
    let batch = grid_batch::<BE::Ring>(a);
    if !batch.is_multiple_of(b.count()) {
        return Err(reject(
            op,
            format!("{} public matrices do not divide the {batch} matrix slots", b.count()),
        ));
    }
    if b.plan().pt_algorithm().masked() {
        ensure_masks(op, module, a, ctx, true)?;
    }
    let masks = context_masks(b.plan(), &ctx.maps, true);
    pt_left_uses(res, &b.maps, &masks, a, &mut |u| ensure_rotation(op, module, keys, u))?;
    Ok(batch)
}

/// `a` is a grid at the context's `log_d` of blocks of the module's degree, which the context's maps read.
fn ensure_factor<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContextPrepared<BE>,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_layouts(op, module.n(), a, &[])?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), ctx.coeffs_meta().k, &[a])
}

/// For a hot operand `x`, whose terms the product evaluates, that `ctx` reads it: at its `log_d`, encoded for the
/// module's slot count at its radix. A cold operand brings its terms and reads no context.
fn ensure_hot_context<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    x: &impl BlasMatMulOperandInfos,
    ctx: &BlasContextPrepared<BE>,
) -> Result<()> {
    if x.hint() == PrepareHint::Reuse {
        return Ok(());
    }
    ensure_dimension(op, ctx.log_d(), x)?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), ctx.coeffs_meta().k, &[x])
}

/// `g` is a prepared operand at `log_d`, with at least one block and a batch record its label allows; returns the record.
fn ensure_operand<BE: Backend>(op: &'static str, log_d: usize, g: &impl BlasGridInfos) -> Result<usize> {
    let batch = g
        .mat_slots()
        .ok_or_else(|| reject(op, "an operand without a batch record: only a preparation records one".into()))?;
    ensure_dimension(op, log_d, g)?;
    if !matches!(g.rows().checked_mul(g.cols()), Some(1..)) {
        return Err(reject(op, format!("{} x {} blocks form no operand", g.rows(), g.cols())));
    }
    ensure_record::<BE::Ring>(op, g)?;
    Ok(batch)
}

/// `res` is a grid at `a`'s `log_d` of `a`'s rows and `b`'s columns, `a`'s columns match `b`'s rows, its radix is at
/// most [`MAX_BASE2K`], and the blocks of `res` have the module's degree and `a` and `b` its radix and rank.
fn ensure_product<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, a.log_d(), res, res.blocks())?;
    ensure_inner(op, a.cols(), b.rows())?;
    ensure_shape(op, res, a.rows(), b.cols())?;
    let base2k = res.block_layout().base2k().as_usize();
    if base2k > MAX_BASE2K {
        return Err(reject(
            op,
            format!("base2k {base2k} > {MAX_BASE2K}: two canonical tensor digits can overflow an i64"),
        ));
    }
    ensure_layouts(op, module.n(), res, &[a, b])
}

/// The look-ups of preparing `a`, of `a`'s rank: `first` (sigma or tau) from `a`'s width into a block of `a`'s layout
/// at the width it leaves, then `shift(k)` (phi^k or psi^k) of that result for every `k` in `1..d`, each into a block
/// of the terms' width, neither of which caps anything, as the reference evaluates them.
pub(crate) fn term_uses<'c, P: LtDiagonalMeta + 'c>(
    a: &impl BlasGridInfos,
    first: &LinearTransformation<P>,
    shift: impl Fn(usize) -> &'c LinearTransformation<P>,
    f: UseSink<'_>,
) -> Result<()> {
    let (a_k, rank) = (a.block_layout().k(), a.block_layout().rank());
    lt_uses(first, a_k, a_k, rank, &mut *f)?;
    let s_k = ckks_linear_transformation_output_k(first, a_k, a_k);
    (1..1usize << a.log_d()).try_for_each(|k| lt_uses(shift(k), s_k, s_k, rank, &mut *f))
}

/// The look-ups of preparing a left factor `a`: sigma and phi^k.
pub(crate) fn left_uses<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    term_uses(a, maps.sigma(), |k| maps.rotate_cols(k), f)
}

/// The look-ups of preparing a right factor `b`: tau and psi^k, or the replicated maps ([`right_maps`]).
pub(crate) fn right_uses<P: LtDiagonalMeta>(
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    let (first, shift) = right_maps(maps, replicated_at(maps.slots(), b));
    term_uses(b, first, shift, f)
}

/// The rotation look-ups of [`validate_blas_matmul`]: the terms of `a`, then those of `b`.
pub(crate) fn matmul_uses<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    left_uses(a, maps, &mut *f)?;
    right_uses(b, maps, f)
}

/// The tensor look-up of [`validate_blas_matmul`]: the wider of the terms' widths, at their rank.
pub(crate) fn matmul_tensor<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> (TorusPrecision, Rank) {
    let (left, right) = (left_term_layout(a, maps).block, right_term_layout(b, maps).block);
    (left.k().max(right.k()), left.rank())
}

/// The rotation look-ups of [`validate_blas_matmul_prepared`]: the terms of each hot side.
pub(crate) fn prepared_uses<P: LtDiagonalMeta>(
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    if a.hint() == PrepareHint::OneShot {
        left_uses(a, maps, &mut *f)?;
    }
    if b.hint() == PrepareHint::OneShot {
        right_uses(b, maps, f)?;
    }
    Ok(())
}

/// The tensor look-up of [`validate_blas_matmul_prepared`]: the wider of the terms' widths, a cold side's own, a hot
/// side's under `ctx`.
pub(crate) fn prepared_tensor<P: LtDiagonalMeta>(
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> (TorusPrecision, Rank) {
    let (left, right) = (
        product_terms(a, || left_term_layout(a, maps)),
        product_terms(b, || right_term_layout(b, maps)),
    );
    (left.k().max(right.k()), left.rank())
}

/// The look-ups of a product by public matrices of `a` into `res`, of `a`'s rank: `first`'s, if any, from `a`'s width into
/// a block of `a`'s layout at the width it leaves; then every mask of `masks` from that width into a block at the width
/// the first mask leaves; then the block maps of `maps` from the width the last stage leaves into `res`. A prepared masked
/// operand holds no block map, its terms being plaintexts read through no key; the queries read the terms' one-diagonal
/// shapes, which look up rotation `0` alone.
fn pt_uses<'m, P: LtDiagonalMeta + 'm>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    first: Option<&LinearTransformation<P>>,
    masks: &[&LinearTransformation<P>],
    maps: impl IntoIterator<Item = &'m LinearTransformation<P>>,
    f: UseSink<'_>,
) -> Result<()> {
    let (a_k, rank, res_k) = (a.block_layout().k(), a.block_layout().rank(), res.block_layout().k());
    let src_k = match first {
        None => a_k,
        Some(first) => {
            lt_uses(first, a_k, a_k, rank, &mut *f)?;
            ckks_linear_transformation_output_k(first, a_k, a_k)
        }
    };
    for &mask in masks {
        lt_uses(mask, src_k, src_k, rank, &mut *f)?;
    }
    let src_k = masks
        .first()
        .map_or(src_k, |&mask| ckks_linear_transformation_output_k(mask, src_k, src_k));
    maps.into_iter().try_for_each(|lt| lt_uses(lt, src_k, res_k, rank, &mut *f))
}

/// The look-ups of [`validate_blas_matmul_pt_right`]: `b`'s first map, the context's `masks`, `b`'s maps, row-major.
pub(crate) fn pt_right_uses<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &OperandMaps<LinearTransformation<P>>,
    masks: &[&LinearTransformation<P>],
    f: UseSink<'_>,
) -> Result<()> {
    let maps = (0..b.rows()).flat_map(|l| (0..b.cols()).map(move |j| b.at(l, j)));
    pt_uses(res, a, b.first(), masks, maps, f)
}

/// The look-ups of [`validate_blas_matmul_pt_left`]: as on the right, `b`'s replicated-input maps when `a` is
/// replicated ([`left_takes_replicated`]).
pub(crate) fn pt_left_uses<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    b: &OperandMaps<LinearTransformation<P>>,
    masks: &[&LinearTransformation<P>],
    a: &impl BlasGridInfos,
    f: UseSink<'_>,
) -> Result<()> {
    let rep = left_takes_replicated(b, a);
    let maps = (0..b.rows()).flat_map(|i| (0..b.cols()).map(move |l| b.at_for(i, l, rep)));
    pt_uses(res, a, b.first(), masks, maps, f)
}

/// Checks a [`blas_matvec_pt`](crate::api::BlasMatVecOps::blas_matvec_pt) call: `a` and `res` are vectors whose
/// ciphertexts tile their shape and share one layout whose label proves a period holding their entries, `a` of `w`'s
/// input shape and `res` of its output shape, the ciphertexts of `res` of the module's degree and those of `a` of their
/// radix and rank, `w` encoded on a module of the module's slot count at `a`'s radix and no wider than `a`, `a`
/// repeating within `w`'s input period, and `keys` holding every key of `w`'s maps, the baby steps at `a`'s width, the
/// giant steps at the width they leave into `res`, of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a vector, shape, layout, operand or period it rejects, then, key by key,
/// `MissingAutomorphismKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matvec_pt<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasVectorRef<BE>,
    w: &impl BlasMatVecMaps<BE>,
    a: &impl BlasVectorRef<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_matvec_pt";
    ensure_vector::<BE::Ring>(op, a, a.cts())?;
    ensure_vector::<BE::Ring>(op, res, res.cts())?;
    ensure_matvec_pt(op, module, &res.blas_vector_layout(), w, &a.blas_vector_layout(), keys)
}

/// The checks of a product by `w` from a vector of layout `a` into one of layout `res`, whose ciphertexts passed their
/// own: `a` of `w`'s input shape and `res` of its output shape, the ciphertexts of `res` of the module's degree and
/// those of `a` of their radix and rank, `w` encoded on a module of the module's slot count at `a`'s radix and no
/// wider than `a`, `a` repeating within `w`'s input period, and `keys` holding every key of `w`'s maps.
fn ensure_matvec_pt<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &BlasVectorLayout,
    w: &impl BlasMatVecMaps<BE>,
    a: &BlasVectorLayout,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    ensure_vector_shape(op, "input", a.shape, w.input())?;
    ensure_vector_shape(op, "output", res.shape, w.output())?;
    ensure_vector_layouts(op, module.n(), res.ct, a.ct)?;
    ensure_maps_reading(op, module, w.slots(), w.base2k(), w.coeffs_meta().k, &[a.ct], "ciphertexts")?;
    let period = vector_period::<BE::Ring>(a);
    if period > w.input_period() {
        return Err(reject(
            op,
            format!(
                "an input repeating every {period} slots read by an operand of input period {}",
                w.input_period()
            ),
        ));
    }
    matvec_pt_uses(res, w.blocks(), a, &mut |u| ensure_rotation(op, module, keys, u))
}

/// Rejects `line` unless it lies in `g`: its index below the grid's `rows d` rows or `cols d` columns and its matrix
/// below `batch`.
fn ensure_line(op: &'static str, line: BlasGridLine, g: &impl BlasGridInfos, batch: usize) -> Result<()> {
    let (index, lines) = line.index(g);
    if lines.is_none_or(|lines| index >= lines) {
        return Err(reject(op, format!("{line:?} is outside a {} x {} grid", g.rows(), g.cols())));
    }
    if line.mat_slot() >= batch {
        return Err(reject(op, format!("{line:?} is outside a batch of {batch} matrices")));
    }
    Ok(())
}

/// Checks a [`blas_grid_to_vector`](crate::api::BlasMatVecOps::blas_grid_to_vector) call: `a` is a grid whose blocks
/// tile its shape and share one layout whose label allows its batch record, `line` lies in it, its index below the
/// grid's rows or columns and its matrix below the batch, then the checks of
/// [`validate_blas_matvec_pt`] with the line's view ([`BlasGridLine::vector_layout`]) as the input, except that its
/// entries may sit past the period the label proves, in a repetition of it.
///
/// # Errors
///
/// `InvalidOperand` for a grid, line, vector, shape, layout, operand or period it rejects, then, key by key,
/// `MissingAutomorphismKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_grid_to_vector<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasVectorRef<BE>,
    w: &impl BlasMatVecMaps<BE>,
    a: &impl BlasGrid<BE>,
    line: BlasGridLine,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_grid_to_vector";
    ensure_grid::<BE::Ring>(op, a.log_d(), a, a.blocks())?;
    ensure_line(op, line, a, grid_batch::<BE::Ring>(a))?;
    ensure_vector::<BE::Ring>(op, res, res.cts())?;
    ensure_matvec_pt(op, module, &res.blas_vector_layout(), w, &line.vector_layout(a), keys)
}

/// Checks a [`blas_vector_to_grid`](crate::api::BlasMatVecOps::blas_vector_to_grid) call, and returns the batch `res`
/// records, `output_period / d^2`: `a` is a vector whose ciphertexts hold its shape and share one layout whose label
/// proves a period holding its entries, `res` a grid whose blocks tile its shape and share one layout whose label allows
/// its batch record, of one block row for a row and one block column for a column, `w`'s output period holds a matrix,
/// `line` lies in `res`, its index below the grid's rows or columns and its matrix below that batch, then the checks of
/// [`validate_blas_matvec_pt`] with the line's view ([`BlasGridLine::vector_layout`]) as `res`.
///
/// # Errors
///
/// `InvalidOperand` for a vector, grid, shape, operand, line, layout or period it rejects, then, key by key,
/// `MissingAutomorphismKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_vector_to_grid<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    line: BlasGridLine,
    w: &impl BlasMatVecMaps<BE>,
    a: &impl BlasVectorRef<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_vector_to_grid";
    ensure_vector::<BE::Ring>(op, a, a.cts())?;
    ensure_grid::<BE::Ring>(op, res.log_d(), res, res.blocks())?;
    match line {
        BlasGridLine::Row { .. } => ensure_shape(op, res, 1, res.cols())?,
        BlasGridLine::Col { .. } => ensure_shape(op, res, res.rows(), 1)?,
    }
    let log_d = res.log_d();
    let batch = w.output_period() >> (2 * log_d);
    if batch == 0 {
        return Err(reject(
            op,
            format!(
                "an operand of output period {} writes no matrix of 2^{log_d} x 2^{log_d} blocks",
                w.output_period()
            ),
        ));
    }
    ensure_line(op, line, res, batch)?;
    ensure_matvec_pt(op, module, &line.vector_layout(res), w, &a.blas_vector_layout(), keys)?;
    Ok(batch)
}

/// The look-ups of [`validate_blas_matvec_pt`]: every map of `maps`, output-major, from `a`'s width into `res`, of `a`'s
/// rank.
pub(crate) fn matvec_pt_uses<'m, P: LtDiagonalMeta + 'm>(
    res: &impl BlasVectorInfos,
    maps: impl IntoIterator<Item = &'m LinearTransformation<P>>,
    a: &impl BlasVectorInfos,
    f: UseSink<'_>,
) -> Result<()> {
    let (a_k, rank, res_k) = (a.ct_layout().k(), a.ct_layout().rank(), res.ct_layout().k());
    maps.into_iter().try_for_each(|lt| lt_uses(lt, a_k, res_k, rank, &mut *f))
}

/// Rejects `v` unless it is a vector, [`ensure_vector`], of one ciphertext whose every slot is an entry: the shape
/// `(S, S, 0, 1)` for `S` slots, which the products of a diagonal ciphertext read and write.
fn ensure_whole<R: Ring>(op: &'static str, v: &impl BlasVectorInfos, cts: &[impl CKKSCtBounds]) -> Result<()> {
    ensure_vector::<R>(op, v, cts)?;
    let slots = slots_of::<R>(v.ct_layout().n().as_usize());
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    if v.vector_shape() != whole {
        return Err(reject(
            op,
            format!(
                "a vector of {:?} where a whole ciphertext, {whole:?}, is required",
                v.vector_shape()
            ),
        ));
    }
    Ok(())
}

/// Rejects a matrix unless its layout fits the module and its diagonals have the degree, radix and rank of `ct`.
fn ensure_diagonal_ciphertext(
    op: &'static str,
    module: &(impl CKKSModuleInfos + GetDegree),
    m: &BlasDiagonalCiphertextLayout,
    ct: CKKSLayout,
) -> Result<()> {
    if m.lt.slots != module.ckks_max_slots() {
        return Err(reject(
            op,
            format!("a matrix of {} slots on a module of {}", m.lt.slots, module.ckks_max_slots()),
        ));
    }
    if ring(m.ct) != ring(ct) {
        return Err(reject(
            op,
            format!(
                "a matrix of diagonals of (n, base2k, rank) = {:?} on ciphertexts of {:?}",
                ring(m.ct),
                ring(ct)
            ),
        ));
    }
    Ok(())
}

/// Checks a [`blas_matvec_diagonal`](crate::api::BlasMatVecOps::blas_matvec_diagonal) call: `a` and `res` are vectors of one
/// ciphertext whose every slot is an entry, `res` of the module's degree and `a` of its radix and rank, at most 62, `m`,
/// resident or streamed, of the module's slots with diagonals of that degree, radix and rank, and `keys` holds the keys
/// of `m`'s rotations, the baby steps at `a`'s width and the giant steps at the width of the result, and the tensor key
/// at the wider of `a`'s and the diagonals' widths, all of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a vector, layout, radix or matrix it rejects, then, key by key, `MissingAutomorphismKey`, then
/// `MissingRelinearizationKey`, or `InvalidOperand` for a key of another rank.
pub fn validate_blas_matvec_diagonal<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasVectorRef<BE>,
    m: &impl BlasDiagonalCiphertextMap<BE>,
    a: &impl BlasVectorRef<BE>,
    keys: &(impl GetAutomorphismKey<BE> + GetTensorKey<BE>),
) -> Result<()> {
    let op = "blas_matvec_diagonal";
    ensure_whole::<BE::Ring>(op, a, a.cts())?;
    ensure_whole::<BE::Ring>(op, res, res.cts())?;
    ensure_vector_layouts(op, module.n(), res.ct_layout(), a.ct_layout())?;
    let base2k = res.ct_layout().base2k().as_usize();
    if base2k > MAX_BASE2K {
        return Err(reject(
            op,
            format!("base2k {base2k} > {MAX_BASE2K}: two canonical tensor digits can overflow an i64"),
        ));
    }
    let layout = m.blas_diagonal_ciphertext_layout();
    ensure_diagonal_ciphertext(op, module, &layout, a.ct_layout())?;
    matvec_diagonal_uses(res, m.map(), a, &mut |u| ensure_rotation(op, module, keys, u))?;
    ensure_tensor_key(op, keys, matvec_diagonal_tensor(&layout, a), a.ct_layout().rank())
}

/// The rotation look-ups of [`validate_blas_matvec_diagonal`]: `lt`'s baby steps at `a`'s width, its non-empty giant steps at the
/// width of the result into `res`, of `a`'s rank.
pub(crate) fn matvec_diagonal_uses<P: LtDiagonalMeta>(
    res: &impl BlasVectorInfos,
    lt: &LinearTransformation<P>,
    a: &impl BlasVectorInfos,
    f: UseSink<'_>,
) -> Result<()> {
    let a_ct = a.ct_layout();
    let giant_k = ckks_linear_transformation_output_layout(lt, a_ct, res.ct_layout().k()).k();
    bsgs_uses(lt, a_ct.k(), giant_k, a_ct.rank(), f)
}

/// The width of the tensor key [`validate_blas_matvec_diagonal`] looks up: the wider of `a`'s and the diagonals'.
pub(crate) fn matvec_diagonal_tensor(m: &BlasDiagonalCiphertextLayout, a: &impl BlasVectorInfos) -> TorusPrecision {
    a.ct_layout().k().max(m.ct.k())
}

/// Checks a [`blas_append_row_diagonal`](crate::api::BlasMatVecOps::blas_append_row_diagonal) call: `v` is a whole ciphertext of the
/// module's degree and of the radix and rank of `m`'s diagonals, `m` is of the module's slots, `mask` is a prepared
/// plaintext at that radix, the append keeps the diagonals' slot kind and sparsity, and `keys` holds, at `v`'s width and
/// of its rank, the keys of the rotations of `v` by the baby steps `m`'s diagonals read.
///
/// # Errors
///
/// `InvalidOperand` for a vector, layout, matrix or mask it rejects, then, key by key, `MissingAutomorphismKey`, or
/// `InvalidOperand` for a key of another rank.
pub fn validate_blas_append_row_diagonal<BE: Backend>(
    module: &Module<BE>,
    m: &BlasDiagonalCiphertext<BE>,
    v: &impl BlasVectorRef<BE>,
    mask: &CKKSPreparedRight<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    validate_append(module, AppendAxis::Row, m, v, mask, keys)
}

/// Checks a [`blas_append_col_diagonal`](crate::api::BlasMatVecOps::blas_append_col_diagonal) call as [`validate_blas_append_row_diagonal`] does,
/// `keys` holding the keys of the rotations of `v` by the opposites of `m`'s giant steps.
///
/// # Errors
///
/// As [`validate_blas_append_row_diagonal`].
pub fn validate_blas_append_col_diagonal<BE: Backend>(
    module: &Module<BE>,
    m: &BlasDiagonalCiphertext<BE>,
    v: &impl BlasVectorRef<BE>,
    mask: &CKKSPreparedRight<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    validate_append(module, AppendAxis::Col, m, v, mask, keys)
}

fn validate_append<BE: Backend>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &BlasDiagonalCiphertext<BE>,
    v: &impl BlasVectorRef<BE>,
    mask: &CKKSPreparedRight<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = axis.op();
    let (layout, v_ct) = (m.blas_diagonal_ciphertext_layout(), v.ct_layout());
    ensure_whole::<BE::Ring>(op, v, v.cts())?;
    ensure_vector_layouts(op, module.n(), v_ct, v_ct)?;
    ensure_diagonal_ciphertext(op, module, &layout, v_ct)?;
    if mask.rank().as_usize() != 0 || mask.base2k() != v_ct.base2k() {
        return Err(reject(
            op,
            "the mask must be a prepared plaintext at the radix of the vector".into(),
        ));
    }
    // poulpy-ckks's accumulate joins the slot kinds and keeps the smaller sparsity; the matrix's layout must hold both.
    let diagonal = layout.ct;
    let slots = diagonal.slots().join(v_ct.slots()).join(mask.slots());
    let log_sparsity = diagonal
        .log_sparsity()
        .min(v_ct.log_sparsity())
        .min(v_ct.log_n().saturating_sub(mask.log_n()));
    if (slots, log_sparsity) != (diagonal.slots(), diagonal.log_sparsity()) {
        return Err(reject(
            op,
            format!(
                "an append leaving the diagonals {slots:?} at log_sparsity={log_sparsity} on a matrix of {:?} at \
                 log_sparsity={}",
                diagonal.slots(),
                diagonal.log_sparsity()
            ),
        ));
    }
    append_uses(axis, &m.diagonals, v, &mut |u| ensure_rotation(op, module, keys, u))
}

/// The look-ups of [`validate_blas_append_row_diagonal`] and [`validate_blas_append_col_diagonal`]: the rotations of `v` an append along
/// `axis` makes ([`append_rotations`]), at `v`'s width and of its rank.
pub(crate) fn append_uses<P>(
    axis: AppendAxis,
    lt: &LinearTransformation<P>,
    v: &impl BlasVectorInfos,
    f: UseSink<'_>,
) -> Result<()> {
    let (k, rank) = (v.ct_layout().k(), v.ct_layout().rank());
    append_rotations(axis, lt)
        .into_iter()
        .try_for_each(|rotation| f(RotationUse { rotation, k, rank }))
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::layouts::CKKSModuleAlloc;
    use poulpy_hal::layouts::HostBytesBackend;

    use super::*;
    use crate::layouts::{BlasLayout, BlasModuleAlloc};

    #[test]
    fn maps_reject_blocks_of_another_ring_radix_or_a_narrower_width() {
        let module = Module::<HostBytesBackend>::new(256);
        let block = layout_of(&module.ckks_ciphertext_alloc(19usize.into(), 60usize.into()));
        let grid = BlasLayout::new(block, 2, 1, 1);
        let check = |slots: usize, base2k: usize, k: usize| {
            ensure_maps("op", &module, slots, base2k.into(), k.into(), &[&grid]).map_err(|err| err.composition().cloned())
        };
        assert!(check(128, 19, 60).is_ok());
        assert!(check(128, 19, 40).is_ok());
        let rejected = |reason: &str| {
            Err(Some(CKKSCompositionError::InvalidOperand {
                op: "op",
                reason: reason.into(),
            }))
        };
        assert_eq!(
            check(64, 19, 60),
            rejected("maps encoded for 64 slots on a module of 128 slots")
        );
        assert_eq!(check(128, 20, 60), rejected("blocks at base2k=19 read by maps at base2k=20"));
        assert_eq!(check(128, 19, 61), rejected("blocks at k=60 read by maps at k=61"));
    }

    /// A product into `res` at radix 63 is rejected for that radix before the factors' other radix is compared.
    #[test]
    fn products_reject_a_radix_above_62_first() {
        let module = Module::<HostBytesBackend>::new(256);
        let grid = |base2k: usize| module.blas_ciphertext_alloc(2, 1, 1, base2k.into(), 120usize.into());
        let (a, b) = (grid(19), grid(19));
        let check =
            |base2k: usize| ensure_product("op", &module, &grid(base2k), &a, &b).map_err(|err| err.composition().cloned());
        assert!(check(19).is_ok());
        assert_eq!(
            check(63),
            Err(Some(CKKSCompositionError::InvalidOperand {
                op: "op",
                reason: "base2k 63 > 62: two canonical tensor digits can overflow an i64".into(),
            }))
        );
    }
}
