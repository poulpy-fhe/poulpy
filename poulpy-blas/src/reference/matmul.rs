use std::marker::PhantomData;

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSDotProductOps, CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformationBabySteps,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    KeyLayouts, assert_scratch, ensure_dimension, ensure_grid, ensure_inner, ensure_layouts, ensure_lt_keys, ensure_maps,
    ensure_record, ensure_shape, ensure_tensor_key, lt_output_k, lt_tmp_bytes, max_lazy_terms, reject, stamp_sparsity,
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
/// ([`blas_matmul_term_layout`]).
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, radix or context it rejects, then `MissingAutomorphismKey` or
/// `MissingRelinearizationKey`. It checks no budget: poulpy-ckks rejects those inside the implementation, before any
/// write to `res`.
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
    let (left, right) = (blas_matmul_term_layout(a, ctx), blas_matmul_term_layout(b, ctx));
    ensure_tensor_key(op, tsk, left.block.k().max(right.block.k()))?;
    Ok(grid_batch::<BE::Ring>(a).max(grid_batch::<BE::Ring>(b)))
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`] call from metadata alone, before any implementation
/// runs: `a` is a grid at the context's `log_d` whose blocks tile its shape and share one layout whose label allows its
/// batch record, of the module's degree, `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, and
/// `atk` holds every key of sigma from `a`'s width and of each phi^k from sigma's result into the terms' width, at the
/// width it is looked up at.
///
/// # Errors
///
/// `InvalidOperand` for a grid, layout or context it rejects, then `MissingAutomorphismKey`. It checks no budget:
/// poulpy-ckks rejects those inside the implementation.
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
/// product looks it up at.
///
/// # Errors
///
/// `InvalidOperand` for an operand without a batch record, or a grid, shape, layout or radix it rejects, then
/// `MissingRelinearizationKey`. It checks no budget: poulpy-ckks rejects those inside the implementation, before any
/// write to `res`.
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
    ensure_tensor_key(op, tsk, a.block_layout().k().max(b.block_layout().k()))?;
    Ok(batch)
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_pt`] call from metadata alone, before any implementation runs, and
/// returns the batch `res` records, `a`'s: `a` and `res` are grids at `b`'s `log_d` whose blocks tile their shape and
/// share one layout whose label allows their batch record, `a`'s columns match `b`'s rows, `res` has `a`'s rows and
/// `b`'s columns, a block of `res` holds the lazy terms of two inner indices, its blocks have the module's degree and
/// `a`'s radix and rank, `b` was encoded on a module of `module`'s slot count at `a`'s radix, `b`'s public matrices
/// divide `a`'s batch, and `atk` holds every key of every map of `b` at the width it is looked up at.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, operand, count or lazy sum it rejects, then `MissingAutomorphismKey`. It
/// checks no budget: poulpy-ckks rejects those inside the implementation, before any write.
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
    ensure_lazy_sums(op, res.block_layout().base2k().as_usize(), a.cols(), 1)?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, b.slots(), b.base2k(), &[a])?;
    let batch = grid_batch::<BE::Ring>(a);
    if !batch.is_multiple_of(b.count()) {
        return Err(reject(
            op,
            format!("{} public matrices do not divide the {batch} matrix slots", b.count()),
        ));
    }
    let res_k = res.block_layout().k();
    for l in 0..b.rows() {
        for j in 0..b.cols() {
            ensure_lt_keys(op, module, b.at(l, j), atk, a.block_layout().k(), res_k)?;
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

/// Checks the keys preparing `a` looks up: `first` (sigma or tau) from `a`'s width into a block of `a`'s layout, then
/// `shift(k)` (phi^k or psi^k) of that result for every `k` in `1..d`, each into a block of the terms' width, which caps
/// nothing, as the reference evaluates them.
fn ensure_term_keys<'c, BE: Backend + 'c>(
    op: &'static str,
    module: &(impl CKKSModuleInfos + ?Sized),
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    let a_k = a.block_layout().k();
    ensure_lt_keys(op, module, first, atk, a_k, a_k)?;
    let s_k = lt_output_k(first, a_k, a_k);
    (1..1usize << a.log_d()).try_for_each(|k| ensure_lt_keys(op, module, shift(k), atk, s_k, s_k))
}

/// Reference compositions of [`crate::api::BlasMatMulOps`] and the JKLS kernel, on inputs their validators accepted:
/// they check nothing but their scratch, and return poulpy-ckks's budget errors before any write to `res`. The
/// preparations make [`BlasMatMulOperand`]s through the map hook of [`BlasMatIntraImpl`]; the prepared product reads
/// them one output block at a time through the JKLS kernel of [`BlasJklsImpl`].
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
    /// layouts [`blas_matmul_term_layout`] gives.
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        BE: BlasJklsImpl,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos;

    /// For every block `(i, j)` of `res`, row-major, one JKLS kernel, [`BlasJklsImpl::blas_jkls_block_impl`], over the
    /// `K d` pairs `(a_il[k], b_lj[k])`, `l`-major.
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
            let _ = ensure_tensor_key(op, &used, l.k().max(r.k()));
            let bytes = used.max_tensor_bytes(|tsk| self.blas_jkls_block_tmp_bytes_reference(&*res, l, r, lhs.len(), tsk));
            assert_scratch(op, bytes, scratch);
        }
        self.ckks_dot_product_ct(res, lhs, rhs, tsk, scratch)
    }

    /// One block-shaped temporary on top of the linear transformation, the addition and the normalization.
    fn blas_matmul_pt_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (res, a) = (res.block_layout(), a.block_layout());
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + lt_tmp_bytes(self, &res, &a, atk)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.glwe_normalize_tmp_bytes())
    }

    /// One baby-step cache per call, holding the baby steps of every block of `b`; for `i` in `0..m`, `l` in `0..K`:
    /// the cache prepared on `a_il` and shared by every `b_lj`, then `res_ij = b_lj(a_il)` for `l = 0`, otherwise
    /// `b_lj(a_il)` into a temporary added to `res_ij`. Row `i` of `res` is normalized before an `l` whose term would take
    /// it past `2^(63 - base2k)` lazy terms.
    fn blas_matmul_pt_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE>
            + CKKSAddOps<BE>
            + CKKSModuleInfos
            + GLWEBytesOf<BE>
            + GLWENormalize<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_pt";
        let maps = || (0..b.rows()).flat_map(|l| (0..b.cols()).map(move |j| b.at(l, j)));
        let res_layout = res.block_layout();
        let used = KeyLayouts::new(atk);
        for lt in maps() {
            let _ = ensure_lt_keys(op, self, lt, &used, a.block_layout().k(), res_layout.k());
        }
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_tmp_bytes_reference(&*res, a, b, key));
        assert_scratch(op, bytes, scratch);
        let (a_cols, n) = (a.cols(), b.cols());
        let rotations: Vec<i64> = maps().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, &a.blocks()[0]);
        for i in 0..a.rows() {
            let mut held = 0;
            for l in 0..a_cols {
                let src = &a.blocks()[i * a_cols + l];
                self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, atk, scratch)?;
                let row = &mut res.blocks_mut()[i * n..][..n];
                held = make_room(self, row, held, 1, scratch);
                for (j, dst) in row.iter_mut().enumerate() {
                    if l == 0 {
                        self.ckks_eval_linear_transformation_into(dst, src, &babies, b.at(l, j), atk, scratch)?;
                        continue;
                    }
                    scratch.scope(|s| -> Result<()> {
                        let (mut tmp, mut s) = s.take_ckks_ciphertext_like_scratch(&res_layout);
                        self.ckks_eval_linear_transformation_into(&mut tmp, src, &babies, b.at(l, j), atk, &mut s)?;
                        self.ckks_add_assign(dst, &tmp, &mut s)
                    })?;
                }
            }
        }
        stamp_sparsity(res.blocks_mut(), a);
        Ok(())
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
        let (res, lhs, rhs) = (res.block_layout(), a.block_layout(), b.block_layout());
        BE::blas_jkls_block_tmp_bytes_impl(self, &res, &lhs, &rhs, a.cols() << a.log_d(), tsk)
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
        let _ = ensure_tensor_key(op, &used, a.block_layout().k().max(b.block_layout().k()));
        let bytes = used.max_tensor_bytes(|tsk| self.blas_matmul_prepared_tmp_bytes_reference(&*res, a, b, tsk));
        assert_scratch(op, bytes, scratch);
        let (d, n) = (1usize << a.log_d(), b.cols());
        for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
            let (i, j) = (t / n, t % n);
            let pairs = (0..a.cols()).flat_map(|l| (0..d).map(move |k| (l, k)));
            let (lhs, rhs): (Vec<_>, Vec<_>) = pairs.map(|(l, k)| (a.term(i, l, k), b.term(l, j, k))).unzip();
            BE::blas_jkls_block_impl(self, dst, &lhs, &rhs, tsk, scratch)?;
        }
        Ok(())
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

/// Rejects `inner` steps of `step` lazy terms each into blocks at radix `base2k` when one step passes [`max_lazy_terms`]
/// even on a normalized block, which holds one term from the second step on.
fn ensure_lazy_sums(op: &'static str, base2k: usize, inner: usize, step: usize) -> Result<()> {
    let terms = step + usize::from(inner > 1);
    if terms > max_lazy_terms(base2k) {
        return Err(reject(op, format!("{terms} lazy terms risk i64 overflow at base2k={base2k}")));
    }
    Ok(())
}

/// The lazy terms each of `blocks` holds once `step` more land on its `held`, `0` before the first: blocks the step would
/// take past [`max_lazy_terms`] are normalized first, then hold one.
fn make_room<BE, M, D>(module: &M, blocks: &mut [D], held: usize, step: usize, scratch: &mut ScratchArena<'_, BE>) -> usize
where
    BE: Backend,
    M: GLWENormalize<BE> + ?Sized,
    D: GLWEToBackendMut<BE> + LWEInfos,
{
    if held + step <= max_lazy_terms(blocks[0].base2k().as_usize()) {
        return held + step;
    }
    for block in blocks {
        module.glwe_normalize_assign(block, scratch);
    }
    1 + step
}
