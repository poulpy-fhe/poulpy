//! The provided methods of [`BlasMatIntraImpl`], the transpose and the row and column shifts, each composed over the
//! map hook [`BlasMatIntraImpl::blas_map_blocks_impl`] and poulpy-ckks's copy and addition; and of [`BlasMatMulImpl`],
//! `blas_matmul` over both hot preparations and the prepared product. Each runs its validator before any write and, on
//! `Ok`, stamps the batch record of `res`, as the references do. They are public so that an override of a provided
//! method can fall back to it; their scratch assertion over the keys a map looks up is shared with the references.

use std::cell::RefCell;

use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSModuleInfos, LinearTransformation, LinearTransformationPrepared, LtDiagonalMeta},
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, GetTensorKey, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::BlasMatIntraImpl;
use crate::layouts::{
    BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut,
    context::shift_maps,
    validate::{assert_scratch, ensure_rotation, lt_uses},
    validate_blas_rotate_cols, validate_blas_rotate_rows, validate_blas_transpose,
};

/// The provided `blas_transpose_tmp_bytes_impl`: the hook's scratch for the transpose of every block of `a` into a
/// block of `res`.
#[doc(hidden)]
pub fn blas_transpose_tmp_bytes_derived<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let sources = a.rows() * a.cols();
    BE::blas_map_blocks_tmp_bytes_impl(
        module,
        &res.block_layout(),
        &a.block_layout(),
        sources,
        &[ctx.transpose()],
        atk,
    )
}

/// The provided `blas_transpose_impl`, `res_ji = transpose(a_ij)`: one hook call, its sources in the order of `res`'s
/// blocks.
#[doc(hidden)]
pub fn blas_transpose_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = validate_blas_transpose(module, &*res, a, ctx, atk)?;
    let lts = [ctx.transpose()];
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| blas_transpose_tmp_bytes_derived(module, &*res, a, ctx, key);
    assert_maps_scratch("blas_transpose", module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let (rows, cols, blocks) = (a.rows(), a.cols(), a.blocks());
    let src: Vec<_> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| &blocks[i * cols + j]))
        .collect();
    BE::blas_map_blocks_impl(module, res.blocks_mut(), &src, &lts, atk, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The provided `blas_rotate_rows_tmp_bytes_impl`: the scratch of a row shift by any `k`, as [`shift_tmp_bytes`].
#[doc(hidden)]
pub fn blas_rotate_rows_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, true, ctx, atk)
}

/// The provided `blas_rotate_cols_tmp_bytes_impl`: the scratch of a column shift by any `k`, as [`shift_tmp_bytes`].
#[doc(hidden)]
pub fn blas_rotate_cols_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, false, ctx, atk)
}

/// Scratch of a row (`rows`) or column shift by any `k`: the copy for `k' = 0`, otherwise for every `k'` the hook with
/// psi^k' (phi^k') along one block, or with both half maps plus the addition along several.
fn shift_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let extent = if rows { a.rows() } else { a.cols() };
    let (sources, res, a) = (a.rows() * a.cols(), res.block_layout(), a.block_layout());
    (1..1 << ctx.log_d())
        .map(|k1| {
            let lts = shift_maps(&ctx.maps, k1, rows, extent);
            let hook = BE::blas_map_blocks_tmp_bytes_impl(module, &res, &a, sources, &lts, atk);
            if lts.len() == 2 {
                hook.max(module.ckks_add_tmp_bytes(res.size()))
            } else {
                hook
            }
        })
        .fold(module.ckks_copy_tmp_bytes(&res, &a), usize::max)
}

/// The provided `blas_rotate_rows_impl`, as [`shift`].
#[doc(hidden)]
pub fn blas_rotate_rows_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_rows", res, a, k, true, ctx, atk, scratch)
}

/// The provided `blas_rotate_cols_impl`, as [`shift`].
#[doc(hidden)]
pub fn blas_rotate_cols_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_cols", res, a, k, false, ctx, atk, scratch)
}

/// Shifts every matrix by `k mod m d = q d + k'` rows (`rows`) or columns, along the block axis of extent `m`. With
/// `src(t, o)` the block `o` steps past block `t` along that axis: `k' = 0` copies `src(t, q)` into block `t`, then
/// clears its noise tag; `m = 1` is one hook call with psi^k' (phi^k'); otherwise one hook call evaluates both half
/// maps of every block of `a` into a heap grid `h` of `res`'s block layout, and block `t` is
/// `h[first, src(t, q)] + h[second, src(t, q + 1)]`.
#[allow(clippy::too_many_arguments)]
fn shift<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut R,
    a: &A,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = if rows {
        validate_blas_rotate_rows(module, &*res, a, k, ctx, atk)?
    } else {
        validate_blas_rotate_cols(module, &*res, a, k, ctx, atk)?
    };
    shift_blocks(module, op, res, a, k, rows, ctx, atk, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The blocks of [`shift`], on inputs its validator accepted.
#[allow(clippy::too_many_arguments)]
fn shift_blocks<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut R,
    a: &A,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let d = 1usize << ctx.log_d();
    let (a_rows, a_cols) = (a.rows(), a.cols());
    let extent = if rows { a_rows } else { a_cols };
    // `k mod extent d = q d + k1`, without forming `extent d`.
    let (q, k1) = (k / d % extent, k % d);
    // `res` has `a`'s shape, so block `t` of both sits at row `t / a_cols`, column `t % a_cols`.
    let src = |t: usize, o: usize| {
        let (i, j) = (t / a_cols, t % a_cols);
        if rows {
            (i + o) % a_rows * a_cols + j
        } else {
            i * a_cols + (j + o) % a_cols
        }
    };
    if k1 == 0 {
        for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
            module.ckks_copy(dst, &a.blocks()[src(t, q)], scratch)?;
            dst.set_noise(None);
        }
        return Ok(());
    }
    let lts = shift_maps(&ctx.maps, k1, rows, extent);
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| shift_tmp_bytes(module, &*res, a, rows, ctx, key);
    assert_maps_scratch(op, module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let sources: Vec<_> = a.blocks().iter().collect();
    if lts.len() == 1 {
        // One block along the axis: `q = 0`, so block `t` maps into block `t`.
        return BE::blas_map_blocks_impl(module, res.blocks_mut(), &sources, &lts, atk, scratch);
    }
    // ponytail: one hook call over the whole heap grid, two blocks per block of `a`; an override chunks it.
    let layout = res.block_layout();
    let mut halves: Vec<_> = (0..2 * sources.len())
        .map(|_| module.ckks_ciphertext_alloc_from_infos(&layout))
        .collect();
    BE::blas_map_blocks_impl(module, &mut halves, &sources, &lts, atk, scratch)?;
    for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
        module.ckks_add_into(dst, &halves[2 * src(t, q)], &halves[2 * src(t, q + 1) + 1], scratch)?;
    }
    Ok(())
}

/// `keys`, recording the layout of every key looked up through it: an operation sizes its scratch query on the keys its
/// checks look up, so a call that looks up no key does not assert its scratch. An implementation runs on keys its
/// validator accepted, so it repeats the validator's look-ups through it only to record their layouts.
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
/// blocks of rank `rank` and width `src_k` into width `dst_k` looks up in `atk`, as [`assert_scratch`] does; a call
/// that looks up none asserts nothing. The keys passed `op`'s validator, so the look-ups only record their layouts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_maps_scratch<BE: Backend, M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    lts: &[&LinearTransformationPrepared<BE>],
    atk: &impl GetAutomorphismKey<BE>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
    bytes: impl Fn(&GGLWELayout) -> usize,
    scratch: &ScratchArena<'_, BE>,
) {
    let used = KeyLayouts::new(atk);
    for lt in lts {
        let _ = ensure_lt_keys(op, module, lt, &used, src_k, dst_k, rank);
    }
    assert_scratch(op, used.max_bytes(bytes), scratch);
}

/// Fails as [`ensure_rotation`] unless `keys` holds every key of [`lt_uses`], of that rank.
fn ensure_lt_keys<BE, M, P, K>(
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
