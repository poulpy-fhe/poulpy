use std::marker::PhantomData;

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSCopyOps, CKKSDotProductOps, CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformationBabySteps,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    execution::worker_count,
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    KeyLayouts, WORKERS, assert_maps_scratch, assert_scratch, ensure_dimension, ensure_grid, ensure_inner, ensure_layouts,
    ensure_lt_keys, ensure_maps, ensure_record, ensure_shape, ensure_tensor_key, lt_output_k, max_lazy_terms, reject, run_tasks,
    tasks_tmp_bytes,
};
use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasPtOperand, grid_batch, layout_of},
    oep::{BlasJklsImpl, BlasMatIntraImpl},
};

/// The left side of a product, `A` in `A B`: sigma, then phi^k (`BlasContext::rotate_cols`).
pub struct Left;

/// The right side of a product, `B` in `A B`: tau, then psi^k (`BlasContext::rotate_rows`).
pub struct Right;

/// The reference's prepared operand, which [`crate::impl_blas_matmul_reference!`] selects: the `d` terms of every block
/// of the source grid that the JKLS product multiplies, all of one layout. On the [`Left`] side, term `0` of block
/// `(i, l)` is sigma of the source block, narrowed exactly to the terms' width, and term `k` its phi^k; on the [`Right`]
/// side, tau and psi^k. Only a preparation makes one.
///
/// Its [`BlasGridInfos`] report the source grid's dimensions, the terms' layout, which [`blas_matmul_term_layout`]
/// gives before preparing, and the source grid's batch.
pub struct BlasMatMulOperand<BE: Backend, S> {
    /// Block `(i, j)`'s term `k` at `(i * cols + j) * d + k`.
    terms: Vec<CKKSCiphertextOwned<BE>>,
    log_d: usize,
    rows: usize,
    cols: usize,
    mat_slots: usize,
    _side: PhantomData<fn() -> S>,
}

impl<BE: Backend, S> BlasMatMulOperand<BE, S> {
    /// Term `k` of block `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics unless `row < rows`, `col < cols` and `k < d`.
    pub fn term(&self, row: usize, col: usize, k: usize) -> &CKKSCiphertextOwned<BE> {
        assert!(
            row < self.rows && col < self.cols && k < 1 << self.log_d,
            "term {k} of block ({row}, {col}) is outside {} x {} blocks of {} terms",
            self.rows,
            self.cols,
            1usize << self.log_d
        );
        &self.terms[((row * self.cols + col) << self.log_d) + k]
    }
}

/// The operand's infos and term count; the terms' digits may live on a device.
impl<BE: Backend, S> std::fmt::Debug for BlasMatMulOperand<BE, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlasMatMulOperand")
            .field("side", &std::any::type_name::<S>())
            .field("log_d", &self.log_d)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("mat_slots", &self.mat_slots)
            .field("term_layout", &self.block_layout())
            .field("terms", &self.terms.len())
            .finish()
    }
}

impl<BE: Backend, S> BlasGridInfos for BlasMatMulOperand<BE, S> {
    fn log_d(&self) -> usize {
        self.log_d
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        layout_of(&self.terms[0])
    }

    fn mat_slots(&self) -> Option<usize> {
        Some(self.mat_slots)
    }
}

/// The layout every prepared operand of `a` under `ctx` reports ([`BlasGridInfos`]): `a`'s dimensions, and the block
/// layout of every term, `a`'s at the width sigma then a shift leave of it, `a.k - 2p` for the maps' scale `p`, its
/// slots joined with theirs. The context's maps share one scale and slot kind, so both sides share this layout; it
/// carries no batch record.
pub fn blas_matmul_term_layout<BE: Backend>(a: &impl BlasGridInfos, ctx: &BlasContext<BE>) -> BlasLayout {
    let sigma = mapped_block(a.block_layout(), ctx.sigma());
    BlasLayout::new(mapped_block(sigma, ctx.rotate_cols(1)), a.log_d(), a.rows(), a.cols())
}

/// `block` after `lt`: at the width `lt` leaves of it, its slots joined with the map's.
fn mapped_block<BE: Backend>(mut block: CKKSLayout, lt: &LinearTransformationPrepared<BE>) -> CKKSLayout {
    block.glwe_layout.k = lt_output_k(lt, block.k(), block.k());
    if let Some(diagonal) = lt.first_diagonal_plaintext() {
        block.meta.slots = block.meta.slots.join(diagonal.lt_slots());
    }
    block
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul`] call from metadata alone, before any implementation runs, and
/// returns the batch `res` records, the larger of `a`'s and `b`'s: `a` and `b` as their preparations check them
/// ([`validate_blas_matmul_prepare_left`], [`validate_blas_matmul_prepare_right`]), `res` against them as
/// [`validate_blas_matmul_prepared`] checks it against their operands, and `tsk` at the wider of their term widths
/// ([`blas_matmul_term_layout`]), of their rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, radix or context it rejects, then, key by key, `MissingAutomorphismKey`
/// or `MissingRelinearizationKey`, or `InvalidOperand` for a key of another rank. It checks no budget: poulpy-ckks
/// rejects those inside the implementation, before any write to `res`.
pub fn validate_blas_matmul<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    b: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    tsk: &impl GetTensorKey<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul";
    ensure_factor(op, module, a, ctx)?;
    ensure_factor(op, module, b, ctx)?;
    ensure_product(op, module, res, a, b)?;
    ensure_term_keys(op, module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk)?;
    ensure_term_keys(op, module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk)?;
    let (left, right) = (blas_matmul_term_layout(a, ctx).block, blas_matmul_term_layout(b, ctx).block);
    ensure_tensor_key(op, tsk, left.k().max(right.k()), left.rank())?;
    Ok(grid_batch::<BE::Ring>(a).max(grid_batch::<BE::Ring>(b)))
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`] call from metadata alone, before any implementation
/// runs: `a` is a grid at the context's `log_d` whose blocks tile its shape and share one layout whose label allows its
/// batch record, of the module's degree, `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, and
/// `atk` holds every key of sigma from `a`'s width and of each phi^k from sigma's result into the terms' width, at the
/// width it is looked up at, of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, layout or context it rejects, then, key by key, `MissingAutomorphismKey`, or
/// `InvalidOperand` for a key of another rank. It checks no budget: poulpy-ckks rejects those inside the
/// implementation.
pub fn validate_blas_matmul_prepare_left<BE: Backend>(
    module: &Module<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_matmul_prepare_left";
    ensure_factor(op, module, a, ctx)?;
    ensure_term_keys(op, module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk)
}

/// [`validate_blas_matmul_prepare_left`] for [`crate::api::BlasMatMulOps::blas_matmul_prepare_right`], with tau and
/// psi^k.
///
/// # Errors
///
/// As [`validate_blas_matmul_prepare_left`].
pub fn validate_blas_matmul_prepare_right<BE: Backend>(
    module: &Module<BE>,
    b: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let op = "blas_matmul_prepare_right";
    ensure_factor(op, module, b, ctx)?;
    ensure_term_keys(op, module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk)
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_prepared`] call from metadata alone, before any implementation runs,
/// and returns the batch `res` records, the larger of the operands': `a` and `b` are prepared operands, read through
/// their [`BlasGridInfos`], at one `log_d`, with at least one block each and a batch record their label allows; `res` is
/// a grid at that `log_d` whose blocks tile its shape and share one layout whose label allows its batch record, of `a`'s
/// rows and `b`'s columns, of the module's degree and the operands' radix and rank, at a radix of 62 at most; `a`'s
/// columns match `b`'s rows; and `tsk` holds the tensor key at the wider of the operands' term widths, the one width the
/// product looks it up at, of their rank.
///
/// # Errors
///
/// `InvalidOperand` for an operand without a batch record, or a grid, shape, layout or radix it rejects, then
/// `MissingRelinearizationKey`, or `InvalidOperand` for a tensor key of another rank. It checks no budget: poulpy-ckks
/// rejects those inside the implementation, before any write to `res`.
pub fn validate_blas_matmul_prepared<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    tsk: &impl GetTensorKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_prepared";
    let batch = ensure_operand::<BE>(op, a.log_d(), a)?.max(ensure_operand::<BE>(op, a.log_d(), b)?);
    ensure_product(op, module, res, a, b)?;
    let (left, right) = (a.block_layout(), b.block_layout());
    ensure_tensor_key(op, tsk, left.k().max(right.k()), left.rank())?;
    Ok(batch)
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_pt`] call from metadata alone, before any implementation runs, and
/// returns the batch `res` records, `a`'s: `a` and `res` are grids at `b`'s `log_d` whose blocks tile their shape and
/// share one layout whose label allows their batch record, `a`'s columns match `b`'s rows, `res` has `a`'s rows and
/// `b`'s columns, its blocks have the module's degree and `a`'s radix and rank, `b` was encoded on a module of
/// `module`'s slot count at `a`'s radix, `b`'s public matrices divide `a`'s batch, and `atk` holds every key of every
/// map of `b` at the width it is looked up at, of `a`'s rank.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, operand or count it rejects, then, key by key, `MissingAutomorphismKey`,
/// or `InvalidOperand` for a key of another rank. It checks no budget: poulpy-ckks rejects those inside the
/// implementation, before any write.
pub fn validate_blas_matmul_pt<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    b: &BlasPtOperand<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_pt";
    ensure_grid::<BE::Ring>(op, b.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, b.log_d(), res, res.blocks())?;
    ensure_inner(op, a.cols(), b.rows())?;
    ensure_shape(op, res, a.rows(), b.cols())?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, b.slots(), b.base2k(), &[a])?;
    let batch = grid_batch::<BE::Ring>(a);
    if !batch.is_multiple_of(b.count()) {
        return Err(reject(
            op,
            format!("{} public matrices do not divide the {batch} matrix slots", b.count()),
        ));
    }
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    for l in 0..b.rows() {
        for j in 0..b.cols() {
            ensure_lt_keys(op, module, b.at(l, j), atk, block.k(), res_k, block.rank())?;
        }
    }
    Ok(batch)
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_pt_left`] call from metadata alone, before any implementation runs,
/// and returns the batch `res` records, `a`'s: `a` and `res` are grids at `b`'s `log_d` whose blocks tile their shape
/// and share one layout whose label allows their batch record, `b`'s columns match `a`'s rows, `res` has `b`'s rows and
/// `a`'s columns, its blocks have the module's degree and `a`'s radix and rank, `b` was encoded on a module of
/// `module`'s slot count at `a`'s radix, `b`'s public matrices divide `a`'s batch, and `atk` holds every key of every
/// map of `b` at the width it is looked up at, of `a`'s rank.
///
/// # Errors
///
/// As [`validate_blas_matmul_pt`].
pub fn validate_blas_matmul_pt_left<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    b: &BlasPtOperand<BE, Left>,
    a: &impl BlasGrid<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_pt_left";
    ensure_grid::<BE::Ring>(op, b.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, b.log_d(), res, res.blocks())?;
    ensure_inner(op, b.cols(), a.rows())?;
    ensure_shape(op, res, b.rows(), a.cols())?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, b.slots(), b.base2k(), &[a])?;
    let batch = grid_batch::<BE::Ring>(a);
    if !batch.is_multiple_of(b.count()) {
        return Err(reject(
            op,
            format!("{} public matrices do not divide the {batch} matrix slots", b.count()),
        ));
    }
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    for i in 0..b.rows() {
        for l in 0..b.cols() {
            ensure_lt_keys(op, module, b.at(i, l), atk, block.k(), res_k, block.rank())?;
        }
    }
    Ok(batch)
}

/// `a` is a grid at the context's `log_d` of blocks of the module's degree, which the context's maps read.
fn ensure_factor<BE: Backend>(op: &'static str, module: &Module<BE>, a: &impl BlasGrid<BE>, ctx: &BlasContext<BE>) -> Result<()> {
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_layouts(op, module.n(), a, &[])?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[a])
}

/// `g` is a prepared operand at `log_d`, with at least one block and a batch record its label allows; returns the record.
fn ensure_operand<BE: Backend>(op: &'static str, log_d: usize, g: &impl BlasGridInfos) -> Result<usize> {
    let batch = g
        .mat_slots()
        .ok_or_else(|| reject(op, "an operand without a batch record: only a preparation makes one".into()))?;
    ensure_dimension(op, log_d, g)?;
    if !matches!(g.rows().checked_mul(g.cols()), Some(1..)) {
        return Err(reject(op, format!("{} x {} blocks form no operand", g.rows(), g.cols())));
    }
    ensure_record::<BE::Ring>(op, g)?;
    Ok(batch)
}

/// `res` is a grid at `a`'s `log_d` of `a`'s rows and `b`'s columns, `a`'s columns match `b`'s rows, its radix is at
/// most 62, so that its digits hold the two lazy terms the JKLS kernel sums between normalizations, and the blocks of
/// `res` have the module's degree and `a` and `b` its radix and rank.
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
    if max_lazy_terms(base2k) < 2 {
        return Err(reject(op, format!("2 lazy terms risk i64 overflow at base2k={base2k}")));
    }
    ensure_layouts(op, module.n(), res, &[a, b])
}

/// Checks the keys preparing `a` looks up, of `a`'s rank: `first` (sigma or tau) from `a`'s width into a block of `a`'s
/// layout, then `shift(k)` (phi^k or psi^k) of that result for every `k` in `1..d`, each into a block of the terms'
/// width, which caps nothing, as the reference evaluates them.
fn ensure_term_keys<'c, BE: Backend + 'c>(
    op: &'static str,
    module: &(impl CKKSModuleInfos + ?Sized),
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let (a_k, rank) = (a.block_layout().k(), a.block_layout().rank());
    ensure_lt_keys(op, module, first, atk, a_k, a_k, rank)?;
    let s_k = lt_output_k(first, a_k, a_k);
    (1..1usize << a.log_d()).try_for_each(|k| ensure_lt_keys(op, module, shift(k), atk, s_k, s_k, rank))
}

/// Reference compositions of [`crate::api::BlasMatMulOps`] and the JKLS kernel, on inputs their validators accepted:
/// they check nothing but their scratch, and return poulpy-ckks's budget errors before any write to `res`. The
/// preparations make [`BlasMatMulOperand`]s through the map hook of [`BlasMatIntraImpl`]; the prepared product reads
/// them one output block at a time through the JKLS kernel of [`BlasJklsImpl`], the blocks split over up to four workers
/// of `BE`'s task executor, each on its own scratch slice, and returns the error of the lowest failing block, as a
/// serial loop would. The products by public matrices evaluate one poulpy-ckks summed linear transformation per output
/// block, their block rows (columns, on the left) split the same way.
pub trait BlasMatMulReference<BE: Backend> {
    /// The largest of the map hook's queries through sigma, from `a`'s blocks into blocks of their layout, and through
    /// every shift, from those into blocks of the terms' layout, and of the copy of one into a block of the terms'
    /// layout.
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// The terms of every block of `a`, row-major. One hook call evaluates `S = sigma(a_il)` for every block into a heap
    /// block of `a`'s layout; one hook call evaluates `phi^k(S)` for every `S` and every `k` in `1..d` into heap blocks
    /// of the terms' layout, one baby-step cache per `S`; then `ckks_copy` narrows each `S` into a block of the terms'
    /// layout, its term `0`, exactly: it charges the shift to the budget. Every term has the layout
    /// [`blas_matmul_term_layout`] gives. The operand records `a`'s batch, [`grid_batch`](crate::layouts::grid_batch).
    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Left>>
    where
        BE: BlasMatIntraImpl,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_prepare_left_tmp_bytes_reference`], with tau and psi^k.
    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// As [`Self::blas_matmul_prepare_left_reference`], with tau and psi^k.
    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Right>>
    where
        BE: BlasMatIntraImpl,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The JKLS kernel's query for a block of `res` and the `K d` pairs of terms of the operands `a` and `b`, or of the
    /// layouts [`blas_matmul_term_layout`] gives, on each worker's slice.
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        BE: BlasJklsImpl,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos;

    /// For every block `(i, j)` of `res`, one JKLS kernel, [`BlasJklsImpl::blas_jkls_block_impl`], over the `K d` pairs
    /// `(a_il[k], b_lj[k])`, `l`-major, the blocks in parallel.
    fn blas_matmul_prepared_reference<R, T>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, Left>,
        b: &BlasMatMulOperand<BE, Right>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync;

    /// poulpy-ckks's `ckks_dot_product_ct` query for `terms` pairs.
    fn blas_jkls_block_tmp_bytes_reference<R, L, Rh, T>(&self, res: &R, lhs: &L, rhs: &Rh, terms: usize, tsk: &T) -> usize
    where
        Self: CKKSDotProductOps<BE>,
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_dot_product_ct_tmp_bytes(terms, res, lhs, rhs, tsk)
    }

    /// The JKLS kernel: `res = sum_t lhs[t] rhs[t]` by poulpy-ckks's `ckks_dot_product_ct`. On pairs that share one
    /// layout per side, it rounds each tensor product at `max(lhs.k, rhs.k)`, sums them exactly, normalizing every
    /// `2^(63 - base2k)` terms and once at the end, stamps `res` and relinearizes once, with the tensor key at that
    /// width.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds less than the query reports for the tensor key it looks up.
    fn blas_jkls_block_reference<R, L, Rh, T>(
        &self,
        res: &mut R,
        lhs: &[&L],
        rhs: &[&Rh],
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDotProductOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        L: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        Rh: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        T: GetTensorKey<BE> + Sync,
    {
        if let (Some(&l), Some(&r)) = (lhs.first(), rhs.first()) {
            let op = "blas_jkls_block";
            let used = KeyLayouts::new(tsk);
            let _ = ensure_tensor_key(op, &used, l.k().max(r.k()), l.rank());
            let bytes = used.max_tensor_bytes(|tsk| self.blas_jkls_block_tmp_bytes_reference(&*res, l, r, lhs.len(), tsk));
            assert_scratch(op, bytes, scratch);
        }
        self.ckks_dot_product_ct(res, lhs, rhs, tsk, scratch)
    }

    /// The baby steps of one block of `a` and one summed linear transformation from blocks of `a`'s layout into a block
    /// of `res`'s, on each worker's slice, one task per block row of `res`.
    fn blas_matmul_pt_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let block = summed_lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), atk);
        tasks_tmp_bytes::<BE>(res.rows(), block)
    }

    /// For every block row `i` of `res`: one baby-step cache per block `a_il`, `l < K`, covering the baby steps of every
    /// map of `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K` terms
    /// `(a_il, b_lj)`, in ascending `l`, which shares the giant steps across them, stamped with `a`'s label. The block
    /// rows split over up to four workers of `BE`'s task executor, in runs of consecutive rows, each worker with its own
    /// caches and scratch slice.
    fn blas_matmul_pt_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized + Sync,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let (inner, n) = (a.cols(), b.cols());
        let maps: Vec<_> = (0..inner).flat_map(|l| (0..n).map(move |j| b.at(l, j))).collect();
        let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
        let bytes = |key: &GGLWELayout| self.blas_matmul_pt_tmp_bytes_reference(&*res, a, b, key);
        assert_maps_scratch("blas_matmul_pt", self, &maps, atk, src_k, dst_k, rank, bytes, scratch);
        let blocks = a.blocks();
        let rows = res.blocks_mut().chunks_mut(n).map(|row| row.iter_mut().collect()).collect();
        let source = |i: usize, l: usize| &blocks[i * inner + l];
        shared_giants(self, rows, inner, source, |_, j, l| b.at(l, j), &maps, atk, scratch)
    }

    /// As [`Self::blas_matmul_pt_tmp_bytes_reference`], one task per block column of `res`.
    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(&self, res: &R, _b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let block = summed_lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), atk);
        tasks_tmp_bytes::<BE>(res.cols(), block)
    }

    /// `res = b a`. For every block column `j` of `res`: one baby-step cache per block `a_lj`, `l < K`, covering the baby
    /// steps of every map of `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K`
    /// terms `(a_lj, b_il)`, in ascending `l`, which shares the giant steps across them, stamped with `a`'s label. The
    /// block columns split over up to four workers of `BE`'s task executor, in runs of consecutive columns, each worker
    /// with its own caches and scratch slice.
    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized + Sync,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let (inner, n) = (a.rows(), a.cols());
        let maps: Vec<_> = (0..b.rows()).flat_map(|i| (0..inner).map(move |l| b.at(i, l))).collect();
        let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
        let bytes = |key: &GGLWELayout| self.blas_matmul_pt_left_tmp_bytes_reference(&*res, b, a, key);
        assert_maps_scratch("blas_matmul_pt_left", self, &maps, atk, src_k, dst_k, rank, bytes, scratch);
        let mut columns: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
        for (t, block) in res.blocks_mut().iter_mut().enumerate() {
            columns[t % n].push(block);
        }
        let blocks = a.blocks();
        let source = |j: usize, l: usize| &blocks[l * n + j];
        shared_giants(self, columns, inner, source, |_, i, l| b.at(i, l), &maps, atk, scratch)
    }
}

impl<BE: Backend> BlasMatMulReference<BE> for Module<BE> {
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk)
    }

    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Left>>
    where
        BE: BlasMatIntraImpl,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        prepare(
            self,
            "blas_matmul_prepare_left",
            a,
            ctx.sigma(),
            |k| ctx.rotate_cols(k),
            atk,
            scratch,
        )
    }

    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, b, ctx.tau(), |k| ctx.rotate_rows(k), atk)
    }

    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Right>>
    where
        BE: BlasMatIntraImpl,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        prepare(
            self,
            "blas_matmul_prepare_right",
            b,
            ctx.tau(),
            |k| ctx.rotate_rows(k),
            atk,
            scratch,
        )
    }

    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        BE: BlasJklsImpl,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
    {
        let (dst, lhs, rhs) = (res.block_layout(), a.block_layout(), b.block_layout());
        let block = BE::blas_jkls_block_tmp_bytes_impl(self, &dst, &lhs, &rhs, a.cols() << a.log_d(), tsk);
        tasks_tmp_bytes::<BE>(res.rows() * res.cols(), block)
    }

    fn blas_matmul_prepared_reference<R, T>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, Left>,
        b: &BlasMatMulOperand<BE, Right>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
    {
        let op = "blas_matmul_prepared";
        let used = KeyLayouts::new(tsk);
        let (left, right) = (a.block_layout(), b.block_layout());
        let _ = ensure_tensor_key(op, &used, left.k().max(right.k()), left.rank());
        let bytes = used.max_tensor_bytes(|tsk| self.blas_matmul_prepared_tmp_bytes_reference(&*res, a, b, tsk));
        assert_scratch(op, bytes, scratch);
        let (d, n) = (1usize << a.log_d(), b.cols());
        let mut blocks: Vec<_> = res.blocks_mut().iter_mut().enumerate().collect();
        run_tasks::<BE::TaskExecutor, _, _>(&mut blocks, scratch, |(t, dst), scratch| {
            let (i, j) = (*t / n, *t % n);
            let pairs = (0..a.cols()).flat_map(|l| (0..d).map(move |k| (l, k)));
            let (lhs, rhs): (Vec<_>, Vec<_>) = pairs.map(|(l, k)| (a.term(i, l, k), b.term(l, j, k))).unzip();
            BE::blas_jkls_block_impl(self, *dst, &lhs, &rhs, tsk, scratch)
        })
    }
}

/// Scratch of a preparation of `a` with `first` (sigma or tau) and `shift(k)` (phi^k or psi^k): the largest of the map
/// hook's queries through `first`, from `a`'s blocks into blocks of their layout, and through every shift, from those
/// into blocks of the terms' layout, and of the copy of one into a block of the terms' layout.
fn prepare_tmp_bytes<'c, BE, A, K>(
    module: &Module<BE>,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let (block, sources) = (a.block_layout(), a.rows() * a.cols());
    let s = mapped_block(block, first);
    let term = mapped_block(s, shift(1));
    let shifts: Vec<_> = (1..1usize << a.log_d()).map(shift).collect();
    BE::blas_map_blocks_tmp_bytes_impl(module, &block, &block, sources, &[first], atk)
        .max(BE::blas_map_blocks_tmp_bytes_impl(module, &term, &s, sources, &shifts, atk))
        .max(module.ckks_copy_tmp_bytes(&term, &s))
}

/// The terms of every block of `a`, row-major, through the map hook: `S = first(a_ij)` for every block into a heap block
/// of `a`'s layout, then `shift(k)` of every `S` for every `k` in `1..d` into heap blocks of the terms' layout, then each
/// `S` copied into a block of the terms' layout, its term `0`.
#[allow(clippy::too_many_arguments)]
fn prepare<'c, BE, A, K, S>(
    module: &Module<BE>,
    op: &'static str,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<BlasMatMulOperand<BE, S>>
where
    BE: BlasMatIntraImpl + 'c,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let used = KeyLayouts::new(atk);
    let _ = ensure_term_keys(op, module, a, first, &shift, &used);
    assert_scratch(
        op,
        used.max_bytes(|key| prepare_tmp_bytes(module, a, first, &shift, key)),
        scratch,
    );
    let (block, d) = (a.block_layout(), 1usize << a.log_d());
    let term = mapped_block(mapped_block(block, first), shift(1));
    let alloc = |layout: &CKKSLayout, count: usize| -> Vec<CKKSCiphertextOwned<BE>> {
        (0..count).map(|_| module.ckks_ciphertext_alloc_from_infos(layout)).collect()
    };
    let sources: Vec<_> = a.blocks().iter().collect();
    let mut sigma = alloc(&block, sources.len());
    BE::blas_map_blocks_impl(module, &mut sigma, &sources, &[first], atk, scratch)?;
    let (shifts, sources): (Vec<_>, Vec<_>) = ((1..d).map(&shift).collect(), sigma.iter().collect());
    let mut shifted = alloc(&term, sigma.len() * (d - 1));
    BE::blas_map_blocks_impl(module, &mut shifted, &sources, &shifts, atk, scratch)?;
    let mut shifted = shifted.into_iter();
    let mut terms = Vec::with_capacity(sigma.len() * d);
    for s in &sigma {
        let mut narrowed = module.ckks_ciphertext_alloc_from_infos(&term);
        module.ckks_copy(&mut narrowed, s, scratch)?;
        terms.push(narrowed);
        terms.extend(shifted.by_ref().take(d - 1));
    }
    Ok(BlasMatMulOperand {
        terms,
        log_d: a.log_d(),
        rows: a.rows(),
        cols: a.cols(),
        mat_slots: grid_batch::<BE::Ring>(a),
        _side: PhantomData,
    })
}

/// Scratch of the baby steps of a block of layout `a` and of a summed linear transformation from blocks of layout `a`
/// into a block of layout `res`.
fn summed_lt_tmp_bytes<BE, M, K>(module: &M, res: &impl CKKSCtBounds, a: &impl CKKSCtBounds, key: &K) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    K: GGLWEInfos,
{
    module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(a, key)
        .max(module.ckks_eval_linear_transformation_sum_tmp_bytes(a, key))
        .max(module.ckks_eval_linear_transformation_sum_tmp_bytes(res, key))
}

/// The products by public matrices, which share their giant steps across their `inner` terms: group `g` of `groups`
/// holds the blocks `o` of `res` it writes, block `o` receiving poulpy-ckks's summed linear transformation of the terms
/// `(source(g, l), map(g, o, l))` in ascending `l`, then the label of `source(g, 0)`. The `inner` sources of a group get
/// one baby-step cache each, covering the baby steps of every map of `maps`, which serve all its blocks. The groups
/// split over up to four workers of `BE`'s task executor, in runs of consecutive groups, each worker with its own
/// caches and scratch slice, and the error of the lowest failing group is returned, as a serial loop would.
#[allow(clippy::too_many_arguments)]
fn shared_giants<'s, BE, M, D, S, K>(
    module: &M,
    groups: Vec<Vec<&mut D>>,
    inner: usize,
    source: impl Fn(usize, usize) -> &'s S + Sync,
    map: impl Fn(usize, usize, usize) -> &'s LinearTransformationPrepared<BE> + Sync,
    maps: &[&LinearTransformationPrepared<BE>],
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sync,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync + 's,
    K: GetAutomorphismKey<BE> + Sync,
{
    let rotations: Vec<i64> = maps.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
    let mut groups: Vec<(usize, Vec<&mut D>)> = groups.into_iter().enumerate().collect();
    // ponytail: splits whole groups only, so fewer than four leave workers idle; split their blocks if that matters.
    let run = groups.len().div_ceil(worker_count::<BE::TaskExecutor>(WORKERS, groups.len()));
    let mut runs: Vec<_> = groups.chunks_mut(run).collect();
    run_tasks::<BE::TaskExecutor, _, _>(&mut runs, scratch, |run, scratch| {
        let mut caches: Vec<_> = (0..inner)
            .map(|_| LinearTransformationBabySteps::alloc(module, &rotations, source(0, 0)))
            .collect();
        for (g, blocks) in run.iter_mut() {
            for (l, cache) in caches.iter_mut().enumerate() {
                module.ckks_prepare_linear_transformation_baby_steps(cache, source(*g, l), atk, scratch)?;
            }
            for (o, dst) in blocks.iter_mut().enumerate() {
                let terms: Vec<_> = caches
                    .iter()
                    .enumerate()
                    .map(|(l, cache)| (source(*g, l), cache, map(*g, o, l)))
                    .collect();
                module.ckks_eval_linear_transformation_sum_into(&mut **dst, &terms, atk, scratch)?;
                dst.set_log_sparsity(source(*g, 0).log_sparsity());
            }
        }
        Ok(())
    })
}
