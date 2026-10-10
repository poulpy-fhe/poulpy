//! Reference compositions over poulpy-ckks; the contract every override reproduces.

mod inter;
mod intra;
mod matmul;

pub use inter::BlasMatInterReference;
pub use intra::BlasMatIntraReference;
pub use matmul::BlasMatMulReference;

use std::cell::RefCell;

use poulpy_ckks::{
    CKKSCompositionError, CKKSError, CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextInfos, SetGLWEPlaintextInfos,
    api::{CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformation, LtDiagonalMeta},
};
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GGLWELayout, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision,
    prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Ring, ScratchArena, ZnxWord};

use crate::layouts::{BlasCiphertext, BlasGridInfos, layout_of};

/// The `InvalidPlan` rejection of `op` for `reason`.
pub fn reject(op: &'static str, reason: String) -> CKKSError {
    CKKSCompositionError::InvalidPlan { plan: op, reason }.into()
}

/// Rejects `g` unless it reads `2^log_d x 2^log_d` blocks that share one layout, whose `log_delta` does not exceed their
/// `k`, and whose label allows its batch record: a power of two from the batch the label proves to the slots' dense
/// batch, with the label at most `log2` of the degree.
pub fn ensure_grid<D: Data, W: ZnxWord, R: Ring>(op: &'static str, log_d: usize, g: &BlasCiphertext<D, W, R>) -> Result<()> {
    if g.log_d() != log_d {
        return Err(reject(op, format!("grid dimension 2^{} differs from 2^{log_d}", g.log_d())));
    }
    let first = g.block_layout();
    if g.blocks().iter().any(|block| layout_of(block) != first) {
        return Err(reject(op, "the blocks of a grid do not share one layout".into()));
    }
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
    if !g.batch_fits() {
        return Err(reject(
            op,
            format!(
                "a batch of {} matrices of 2^{log_d} x 2^{log_d} blocks does not fit blocks of degree {} labelled log_sparsity={}",
                g.mat_slots,
                first.n().as_usize(),
                first.log_sparsity()
            ),
        ));
    }
    Ok(())
}

/// Rejects `g` unless it has `rows x cols` blocks.
pub fn ensure_shape<G: BlasGridInfos>(op: &'static str, g: &G, rows: usize, cols: usize) -> Result<()> {
    if (g.rows(), g.cols()) != (rows, cols) {
        return Err(reject(
            op,
            format!("a {} x {} grid where {rows} x {cols} is required", g.rows(), g.cols()),
        ));
    }
    Ok(())
}

/// Rejects a product whose left operand's `a_cols` block columns differ from its right operand's `b_rows` block rows.
pub fn ensure_inner(op: &'static str, a_cols: usize, b_rows: usize) -> Result<()> {
    if a_cols != b_rows {
        return Err(reject(op, format!("{a_cols} block columns against {b_rows} block rows")));
    }
    Ok(())
}

/// Rejects a shift `k` that is not below `bound`.
pub fn ensure_shift(op: &'static str, k: usize, bound: usize) -> Result<()> {
    if k >= bound {
        return Err(reject(op, format!("shift {k} is not below {bound}")));
    }
    Ok(())
}

/// Rejects maps encoded on a module of `slots` slots at radix `base2k` unless `module` has `slots` slots and every grid
/// of `inputs`, the grids they read, holds blocks at `base2k`.
pub fn ensure_maps<M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    slots: usize,
    base2k: Base2K,
    inputs: &[&dyn BlasGridInfos],
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
    if let Some(input) = inputs.iter().find(|g| g.block_layout().base2k() != base2k) {
        return Err(reject(
            op,
            format!(
                "blocks at base2k={} read by maps at base2k={}",
                input.block_layout().base2k().as_usize(),
                base2k.as_usize()
            ),
        ));
    }
    Ok(())
}

/// Fails with `MissingAutomorphismKey` on the first rotation that has no key at width `k`.
pub fn ensure_keys<BE, M, K>(
    op: &'static str,
    module: &M,
    rotations: impl IntoIterator<Item = i64>,
    keys: &K,
    k: TorusPrecision,
) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    K: GetAutomorphismKey<BE>,
{
    for rotation in rotations {
        let p = module.ckks_galois_element(rotation);
        // Element 1 is the identity, which poulpy-ckks applies without a key.
        if p != 1 && keys.get_automorphism_key(p, k).is_err() {
            return Err(CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation,
                k: k.into(),
            }
            .into());
        }
    }
    Ok(())
}

/// Fails with `MissingRelinearizationKey` unless `keys` holds a tensor key at width `k`.
pub fn ensure_tensor_key<BE: Backend, K: GetTensorKey<BE>>(op: &'static str, keys: &K, k: TorusPrecision) -> Result<()> {
    keys.get_tensor_key(k)
        .map_err(|_| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into() })?;
    Ok(())
}

/// Width poulpy-ckks gives `lt` evaluated from a ciphertext of width `src_k` into one of width `dst_k`, on an input it
/// evaluates: the diagonals' scale comes off the source, capped at the destination.
pub fn lt_output_k<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
) -> TorusPrecision {
    let scale = lt.first_diagonal_plaintext().map_or(0, LtDiagonalMeta::lt_log_scale);
    src_k.as_usize().saturating_sub(scale).min(dst_k.as_usize()).into()
}

/// Fails with `MissingAutomorphismKey` unless `keys` holds every key evaluating `lt` from width `src_k` into a
/// ciphertext of width `dst_k` looks up: the baby steps at `src_k`, the non-empty giant steps at [`lt_output_k`].
///
/// These are poulpy-ckks's widths on inputs it evaluates. On an input it rejects with a budget error, this check may
/// ask for a key poulpy-ckks never looks up for that input: a store holding exactly the widths in use then fails here
/// with `MissingAutomorphismKey` instead of that budget error.
pub fn ensure_lt_keys<BE, M, P, K>(
    op: &'static str,
    module: &M,
    lt: &LinearTransformation<P>,
    keys: &K,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    P: LtDiagonalMeta,
    K: GetAutomorphismKey<BE>,
{
    ensure_keys(op, module, lt.baby_steps().iter().copied(), keys, src_k)?;
    let giants = lt.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty()).map(|gs| gs.rot);
    ensure_keys(op, module, giants, keys, lt_output_k(lt, src_k, dst_k))
}

/// `keys`, recording the layout of every key looked up through it: an operation sizes its scratch query on the keys its
/// checks look up, so a call that looks up no key does not assert its scratch.
pub(crate) struct KeyLayouts<'a, K> {
    keys: &'a K,
    atks: RefCell<Vec<GGLWELayout>>,
    tsks: RefCell<Vec<GGLWELayout>>,
}

impl<'a, K> KeyLayouts<'a, K> {
    pub(crate) fn new(keys: &'a K) -> Self {
        Self {
            keys,
            atks: RefCell::default(),
            tsks: RefCell::default(),
        }
    }

    /// The largest `bytes` over the rotation keys looked up, `0` without any.
    pub(crate) fn max_bytes(&self, bytes: impl Fn(&GGLWELayout) -> usize) -> usize {
        self.atks.borrow().iter().map(bytes).max().unwrap_or(0)
    }

    /// The largest `bytes` over the pairs of a tensor key and a rotation key looked up, `0` without any.
    pub(crate) fn max_pair_bytes(&self, bytes: impl Fn(&GGLWELayout, &GGLWELayout) -> usize) -> usize {
        let tsks = self.tsks.borrow();
        tsks.iter()
            .map(|tsk| self.max_bytes(|atk| bytes(tsk, atk)))
            .max()
            .unwrap_or(0)
    }
}

fn record<G: GGLWEInfos>(layouts: &RefCell<Vec<GGLWELayout>>, key: &G) {
    let layout = key.gglwe_layout();
    let mut layouts = layouts.borrow_mut();
    if !layouts.contains(&layout) {
        layouts.push(layout);
    }
}

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for KeyLayouts<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_automorphism_key(p, k)?;
        record(&self.atks, &key);
        Ok(key)
    }
}

impl<BE: Backend, K: GetTensorKey<BE>> GetTensorKey<BE> for KeyLayouts<'_, K> {
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_tensor_key(k)?;
        record(&self.tsks, &key);
        Ok(key)
    }
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

/// Scratch of one linear transformation from a block of layout `a` into a block of layout `res`.
pub(crate) fn lt_tmp_bytes<BE, M, K>(module: &M, res: &CKKSLayout, a: &CKKSLayout, key: &K) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    K: GGLWEInfos,
{
    module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(a, key)
        .max(module.ckks_eval_linear_transformation_tmp_bytes(a, key))
        .max(module.ckks_eval_linear_transformation_tmp_bytes(res, key))
}

/// Stamps the sparsity of `a` on every block of `res`: a per-matrix map keeps the batch, while poulpy-ckks's linear
/// transformations leave the destination's sparsity.
pub(crate) fn stamp_sparsity<DR: Data, DA: Data, W: ZnxWord, R: Ring>(
    res: &mut BlasCiphertext<DR, W, R>,
    a: &BlasCiphertext<DA, W, R>,
) {
    let log_sparsity = a.block_layout().log_sparsity();
    res.blocks_mut()
        .iter_mut()
        .for_each(|block| block.set_log_sparsity(log_sparsity));
}
