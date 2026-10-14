use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSCopyOps, CKKSDotProductOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformationBabySteps,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, PrepareHint, ScratchArena},
};

use super::{
    KeyLayouts, assert_maps_scratch, assert_scratch, ensure_dimension, ensure_grid, ensure_inner, ensure_layouts, ensure_lt_keys,
    ensure_maps, ensure_record, ensure_shape, ensure_tensor_key, lt_output_k, reject,
};
use crate::{
    layouts::{
        BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasMatMulOperandInfos, BlasMatMulOperandLayout,
        BlasPtOperand, grid_batch, layout_of,
    },
    oep::{BlasJklsImpl, BlasMatIntraImpl},
};

/// The left side of a product by public matrices, `B` in `B A`: a [`BlasPtOperand`] on the left.
pub struct Left;

/// The right side of a product by public matrices, `B` in `A B`, the default side of a [`BlasPtOperand`].
pub struct Right;

/// The reference's prepared operand, which [`crate::impl_blas_matmul_reference!`] selects, of terms `P`:
/// [`CKKSPreparedLeft`] on the left and [`CKKSPreparedRight`] on the right, so that swapping the operands of a product
/// does not compile. Only a preparation makes one:
/// - hot, `OneShot`: a verbatim copy of every block of its source, row-major. The product evaluates their terms under
///   its own context;
/// - cold, `Reuse`: the `d` terms of every block of its source, all of one layout, each converted once. On the left,
///   term `0` of block `(i, l)` is sigma of the source block, narrowed exactly to the terms' width, and term `k` its
///   phi^k; on the right, tau and psi^k.
///
/// Its [`BlasMatMulOperandInfos`] report its hint, the source grid's dimensions, batch and block layout, and as its
/// block layout the source's when hot and the terms' when cold, as [`blas_matmul_operand_layout`] gives them before
/// preparing.
pub struct BlasMatMulOperand<BE: Backend, P> {
    terms: Terms<BE, P>,
    log_d: usize,
    rows: usize,
    cols: usize,
    mat_slots: usize,
    /// The source grid's block layout.
    source: CKKSLayout,
}

/// What an operand holds, after its hint.
enum Terms<BE: Backend, P> {
    /// Hot, `OneShot`: the source's blocks, copied verbatim, row-major.
    Source(Vec<CKKSCiphertextOwned<BE>>),
    /// Cold, `Reuse`: block `(i, j)`'s term `k` at `(i * cols + j) * d + k`, converted once.
    Prepared(Vec<P>),
}

/// The operand's infos and the count of what it holds; its digits may live on a device.
impl<BE: Backend, P: GLWEInfos + CKKSInfos> std::fmt::Debug for BlasMatMulOperand<BE, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let held = match &self.terms {
            Terms::Source(blocks) => blocks.len(),
            Terms::Prepared(terms) => terms.len(),
        };
        f.debug_struct("BlasMatMulOperand")
            .field("terms", &std::any::type_name::<P>())
            .field("hint", &self.hint())
            .field("log_d", &self.log_d)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("mat_slots", &self.mat_slots)
            .field("block_layout", &self.block_layout())
            .field("source_block_layout", &self.source)
            .field("held", &held)
            .finish()
    }
}

impl<BE: Backend, P: GLWEInfos + CKKSInfos> BlasGridInfos for BlasMatMulOperand<BE, P> {
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
        match &self.terms {
            Terms::Source(_) => self.source,
            Terms::Prepared(terms) => layout_of(&terms[0]),
        }
    }

    fn mat_slots(&self) -> Option<usize> {
        Some(self.mat_slots)
    }
}

impl<BE: Backend, P: GLWEInfos + CKKSInfos> BlasMatMulOperandInfos for BlasMatMulOperand<BE, P> {
    fn hint(&self) -> PrepareHint {
        match self.terms {
            Terms::Source(_) => PrepareHint::OneShot,
            Terms::Prepared(_) => PrepareHint::Reuse,
        }
    }

    fn source_block_layout(&self) -> CKKSLayout {
        self.source
    }
}

/// The layout of the terms of `a` under `ctx` ([`BlasGridInfos`]): `a`'s dimensions, and the block layout of every
/// term, `a`'s at the width sigma then a shift leave of it, `a.k - 2p` for the maps' scale `p`, its slots joined with
/// theirs. The context's maps share one scale and slot kind, so both sides share this layout; it carries no batch
/// record. A cold operand of `a` under `ctx` reports it.
pub fn blas_matmul_term_layout<BE: Backend>(a: &impl BlasGridInfos, ctx: &BlasContext<BE>) -> BlasLayout {
    let sigma = mapped_block(a.block_layout(), ctx.sigma());
    BlasLayout::new(mapped_block(sigma, ctx.rotate_cols(1)), a.log_d(), a.rows(), a.cols())
}

/// The infos a preparation of `a` under `ctx` and `hint` reports ([`BlasMatMulOperandInfos`]), before preparing it:
/// `hint`, `a`'s dimensions and its block layout as the source's, and as the block layout `a`'s when hot, `OneShot`,
/// and the terms' [`blas_matmul_term_layout`] gives when cold, `Reuse`. It carries no batch record.
pub fn blas_matmul_operand_layout<BE: Backend>(
    a: &impl BlasGridInfos,
    ctx: &BlasContext<BE>,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    let source = a.block_layout();
    let grid = match hint {
        PrepareHint::OneShot => BlasLayout::new(source, a.log_d(), a.rows(), a.cols()),
        PrepareHint::Reuse => blas_matmul_term_layout(a, ctx),
    };
    BlasMatMulOperandLayout { grid, source, hint }
}

/// The layout of the terms a product under `ctx` multiplies on the side of `x`: a cold operand's own, a hot one's from
/// its source under `ctx`.
fn product_terms<BE: Backend>(x: &impl BlasMatMulOperandInfos, ctx: &BlasContext<BE>) -> CKKSLayout {
    match x.hint() {
        PrepareHint::OneShot => blas_matmul_term_layout(x, ctx).block,
        PrepareHint::Reuse => x.block_layout(),
    }
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
/// their [`BlasMatMulOperandInfos`], at one `log_d`, with at least one block each and a batch record their label allows,
/// a hot operand at the `log_d` of `ctx`, which was encoded on a module of `module`'s slot count at its radix; `res` is
/// a grid at that `log_d` whose blocks tile its shape and share one layout whose label allows its batch record, of `a`'s
/// rows and `b`'s columns, of the module's degree and the operands' radix and rank, at a radix of 62 at most; `a`'s
/// columns match `b`'s rows; `atk` holds every key of a hot side's terms under `ctx`, as the preparation of its source
/// looks them up; and `tsk` holds the tensor key at the wider of the terms' widths, a cold side's own, a hot side's
/// under `ctx`, the one width the product looks it up at, of their rank. A cold side reads neither `ctx` nor `atk`, and
/// the check reads no hot side's scale: the product spends that of the maps of `ctx`.
///
/// # Errors
///
/// `InvalidOperand` for an operand without a batch record, or a grid, shape, layout, radix or context it rejects, then,
/// key by key, `MissingAutomorphismKey` or `MissingRelinearizationKey`, or `InvalidOperand` for a key of another rank.
/// It checks no budget: poulpy-ckks rejects those inside the implementation, before any write to `res`.
pub fn validate_blas_matmul_prepared<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &impl GetTensorKey<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_matmul_prepared";
    let batch = ensure_operand::<BE>(op, a.log_d(), a)?.max(ensure_operand::<BE>(op, a.log_d(), b)?);
    ensure_hot_context(op, module, a, ctx)?;
    ensure_hot_context(op, module, b, ctx)?;
    ensure_product(op, module, res, a, b)?;
    if a.hint() == PrepareHint::OneShot {
        ensure_term_keys(op, module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk)?;
    }
    if b.hint() == PrepareHint::OneShot {
        ensure_term_keys(op, module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk)?;
    }
    let (left, right) = (product_terms(a, ctx), product_terms(b, ctx));
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

/// For a hot operand `x`, whose terms the product evaluates, that `ctx` reads it: at its `log_d`, encoded for the
/// module's slot count at its radix. A cold operand brings its terms and reads no context.
fn ensure_hot_context<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    x: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
) -> Result<()> {
    if x.hint() == PrepareHint::Reuse {
        return Ok(());
    }
    ensure_dimension(op, ctx.log_d(), x)?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[x])
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
/// most 62, so that its digits hold the two lazy terms the JKLS kernel's two-pass circuit sums at least, and the blocks
/// of `res` have the module's degree and `a` and `b` its radix and rank.
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
    if base2k > 62 {
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
/// they check nothing but their scratch, and return poulpy-ckks's budget errors before any write to `res`. A hot
/// preparation copies its input; a cold one makes its terms through the map hook of [`BlasMatIntraImpl`], block row
/// after block row, each converted once by poulpy-ckks's preparations. The prepared product makes a hot side's terms
/// the same way under its context, then reads the terms one output block at a time through the JKLS kernel of
/// [`BlasJklsImpl`]. The products by public matrices evaluate one poulpy-ckks summed linear transformation per output
/// block. They run block after block on one scratch, and the first error stops them.
pub trait BlasMatMulReference<BE: Backend> {
    /// Under `OneShot`, the copy of a block of `a` into a block of its layout. Under `Reuse`, the largest of the map
    /// hook's queries through sigma, from a block row of `a` into blocks of its layout, and through every shift, from
    /// those into blocks of the terms' layout, of the copy of one into a block of the terms' layout, and of the
    /// conversion of a term.
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Under `OneShot`, a verbatim copy of every block of `a`: `ckks_copy` into a block that holds every limb of its
    /// allocation, which keeps its digits, metadata, canonical flag and noise tag. Under `Reuse`, the terms of every
    /// block of `a`, row-major, block row after block row: one hook call evaluates `S = sigma(a_il)` for every block of
    /// the row into a heap block of `a`'s layout; one hook call evaluates `phi^k(S)` for every `S` and every `k` in
    /// `1..d` into heap blocks of the terms' layout, one baby-step cache per `S`; then `ckks_copy` narrows each `S` into
    /// a block of the terms' layout, its term `0`, exactly: it charges the shift to the budget. Each term is converted
    /// once by `ckks_prepare_left` under `Reuse`, then dropped, and has the layout [`blas_matmul_term_layout`] gives.
    /// Either operand records `a`'s batch, [`grid_batch`](crate::layouts::grid_batch).
    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_prepare_left_tmp_bytes_reference`], with tau and psi^k.
    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// As [`Self::blas_matmul_prepare_left_reference`], with tau and psi^k, converted by `ckks_prepare_right`.
    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, CKKSPreparedRight<BE>>>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The largest of the JKLS kernel's query for a block of `res` and the `K d` pairs of terms of the operands `a` and
    /// `b`, or of the layouts [`blas_matmul_operand_layout`] gives, a hot side's terms at the layout
    /// [`blas_matmul_term_layout`] gives under `ctx`, and for each hot side, the cold preparation's query of its source.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// The terms of each hot side under `ctx`, the left side's first, as the cold preparation of its source makes them
    /// but converted under `OneShot`, held for the call; then for every block `(i, j)` of `res`, row-major, one JKLS
    /// kernel, [`BlasJklsImpl::blas_jkls_block_impl`], over the `K d` pairs `(a_il[k], b_lj[k])`, `l`-major: ascending
    /// `l`, then ascending `k` within each `l`.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// poulpy-ckks's `ckks_dot_product_ct` query for `terms` pairs, which covers `ckks_dot_product_prepared` on pairs
    /// prepared from terms of these layouts.
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

    /// The JKLS kernel: `res = sum_t lhs[t] rhs[t]` by poulpy-ckks's `ckks_dot_product_prepared`, which is
    /// `ckks_dot_product_ct` of the terms the pairs were prepared from, bit for bit, with no operand transformed. On
    /// pairs prepared from terms that share one layout per side, it relinearizes once, with the tensor key at
    /// `max(lhs.k, rhs.k)`. On rank 1, with one `base2k` across `res`, the terms and the key, and an offset of at least
    /// `base2k` in poulpy-ckks's ct x ct rule, that is poulpy-core's `glwe_mul_prepared_relinearize_sum`: the pairs'
    /// products are added in the transform domain, in slice order, before `res` is stamped. Otherwise it rounds each
    /// `glwe_tensor_apply_prepared` at that width, sums them exactly, normalizes the sum once, right before the
    /// relinearization reads it, and stamps `res` before relinearizing.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds less than the query reports for the tensor key it looks up.
    fn blas_jkls_block_reference<R, T>(
        &self,
        res: &mut R,
        lhs: &[&CKKSPreparedLeft<BE>],
        rhs: &[&CKKSPreparedRight<BE>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDotProductOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<BE> + Sync,
    {
        if let (Some(&l), Some(&r)) = (lhs.first(), rhs.first()) {
            let op = "blas_jkls_block";
            let used = KeyLayouts::new(tsk);
            let _ = ensure_tensor_key(op, &used, l.k().max(r.k()), l.rank());
            let bytes = used.max_tensor_bytes(|tsk| self.blas_jkls_block_tmp_bytes_reference(&*res, l, r, lhs.len(), tsk));
            assert_scratch(op, bytes, scratch);
        }
        self.ckks_dot_product_prepared(res, lhs, rhs, tsk, scratch)
    }

    /// The baby steps of one block of `a` and one summed linear transformation from blocks of `a`'s layout into a block
    /// of `res`'s.
    fn blas_matmul_pt_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        summed_lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), atk)
    }

    /// For every block row `i` of `res`, in order: one baby-step cache per block `a_il`, `l < K`, covering the baby
    /// steps of every map of `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K`
    /// terms `(a_il, b_lj)`, in ascending `l`, which shares the giant steps across them, stamped with `a`'s label.
    fn blas_matmul_pt_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized,
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

    /// As [`Self::blas_matmul_pt_tmp_bytes_reference`].
    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(&self, res: &R, _b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        summed_lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), atk)
    }

    /// `res = b a`. For every block column `j` of `res`, in order: one baby-step cache per block `a_lj`, `l < K`,
    /// covering the baby steps of every map of `b`, then every block `res_ij` is poulpy-ckks's summed linear
    /// transformation of the `K` terms `(a_lj, b_il)`, in ascending `l`, which shares the giant steps across them,
    /// stamped with `a`'s label.
    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized,
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
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk, hint)
    }

    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let convert = |term: &CKKSCiphertextOwned<BE>, hint, s: &mut ScratchArena<'_, BE>| self.ckks_prepare_left(term, hint, s);
        let (first, shift) = (ctx.sigma(), |k| ctx.rotate_cols(k));
        prepare(self, "blas_matmul_prepare_left", a, first, shift, atk, hint, convert, scratch)
    }

    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, b, ctx.tau(), |k| ctx.rotate_rows(k), atk, hint)
    }

    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, CKKSPreparedRight<BE>>>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let convert = |term: &CKKSCiphertextOwned<BE>, hint, s: &mut ScratchArena<'_, BE>| self.ckks_prepare_right(term, hint, s);
        let (first, shift) = (ctx.tau(), |k| ctx.rotate_rows(k));
        prepare(
            self,
            "blas_matmul_prepare_right",
            b,
            first,
            shift,
            atk,
            hint,
            convert,
            scratch,
        )
    }

    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        kernel_tmp_bytes(self, res, a, b, ctx, tsk).max(hot_tmp_bytes(self, a, b, ctx, atk))
    }

    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_prepared";
        let (tsks, atks) = (KeyLayouts::new(tsk), KeyLayouts::new(atk));
        let (left, right) = (product_terms(a, ctx), product_terms(b, ctx));
        let _ = ensure_tensor_key(op, &tsks, left.k().max(right.k()), left.rank());
        if a.hint() == PrepareHint::OneShot {
            let _ = ensure_term_keys(op, self, a, ctx.sigma(), |k| ctx.rotate_cols(k), &atks);
        }
        if b.hint() == PrepareHint::OneShot {
            let _ = ensure_term_keys(op, self, b, ctx.tau(), |k| ctx.rotate_rows(k), &atks);
        }
        let bytes = tsks
            .max_tensor_bytes(|tsk| kernel_tmp_bytes(self, &*res, a, b, ctx, tsk))
            .max(atks.max_bytes(|atk| hot_tmp_bytes(self, a, b, ctx, atk)));
        assert_scratch(op, bytes, scratch);
        // ponytail: every term of a hot side, held for the call; streaming one block row or column bounds the heap.
        let (log_d, hot) = (a.log_d, PrepareHint::OneShot);
        let hot_left;
        let lhs = match &a.terms {
            Terms::Prepared(terms) => terms,
            Terms::Source(blocks) => {
                let convert =
                    |term: &CKKSCiphertextOwned<BE>, hint, s: &mut ScratchArena<'_, BE>| self.ckks_prepare_left(term, hint, s);
                let (first, shift) = (ctx.sigma(), |k| ctx.rotate_cols(k));
                hot_left = prepare_terms(self, blocks, a.cols, log_d, first, shift, atk, hot, convert, scratch)?;
                &hot_left
            }
        };
        let hot_right;
        let rhs = match &b.terms {
            Terms::Prepared(terms) => terms,
            Terms::Source(blocks) => {
                let convert =
                    |term: &CKKSCiphertextOwned<BE>, hint, s: &mut ScratchArena<'_, BE>| self.ckks_prepare_right(term, hint, s);
                let (first, shift) = (ctx.tau(), |k| ctx.rotate_rows(k));
                hot_right = prepare_terms(self, blocks, b.cols, log_d, first, shift, atk, hot, convert, scratch)?;
                &hot_right
            }
        };
        let (inner, n) = (a.cols, b.cols);
        for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
            let (i, j) = (t / n, t % n);
            let pairs = (0..inner).flat_map(|l| (0..1usize << log_d).map(move |k| (l, k)));
            let (lhs, rhs): (Vec<_>, Vec<_>) = pairs
                .map(|(l, k)| (&lhs[((i * inner + l) << log_d) + k], &rhs[((l * n + j) << log_d) + k]))
                .unzip();
            BE::blas_jkls_block_impl(self, dst, &lhs, &rhs, tsk, scratch)?;
        }
        Ok(())
    }
}

/// The JKLS kernel's query for a block of `res` and the `K d` pairs of the terms of `a` and `b` in a product under `ctx`.
fn kernel_tmp_bytes<BE: BlasJklsImpl>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &impl GGLWEInfos,
) -> usize {
    let (res, lhs, rhs) = (res.block_layout(), product_terms(a, ctx), product_terms(b, ctx));
    BE::blas_jkls_block_tmp_bytes_impl(module, &res, &lhs, &rhs, a.cols() << a.log_d(), tsk)
}

/// The scratch of the terms of the hot sides of `a` and `b` in a product under `ctx`, block row after block row, as their
/// cold preparations make them; `0` without a hot side.
fn hot_tmp_bytes<BE>(
    module: &Module<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    atk: &impl GGLWEInfos,
) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE>,
{
    let left = match a.hint() {
        PrepareHint::OneShot => unit_tmp_bytes(module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk, a.cols()),
        PrepareHint::Reuse => 0,
    };
    let right = match b.hint() {
        PrepareHint::OneShot => unit_tmp_bytes(module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk, b.cols()),
        PrepareHint::Reuse => 0,
    };
    left.max(right)
}

/// Scratch of a preparation of `a` under `hint` with `first` (sigma or tau) and `shift(k)` (phi^k or psi^k): the copy
/// of a block of `a` into a block of its layout when hot, the query of one block row's terms when cold.
fn prepare_tmp_bytes<'c, BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    hint: PrepareHint,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    K: GGLWEInfos,
{
    match hint {
        PrepareHint::OneShot => {
            let block = a.block_layout();
            module.ckks_copy_tmp_bytes(&block, &block)
        }
        PrepareHint::Reuse => unit_tmp_bytes(module, a, first, shift, atk, a.cols()),
    }
}

/// Scratch of the terms of `sources` blocks of `a`'s layout through `first` and `shift(k)`, each converted once: the
/// largest of the map hook's queries through `first`, from those blocks into blocks of their layout, and through every
/// shift, from those into blocks of the terms' layout, of the copy of one into a block of the terms' layout, and of the
/// conversion of a term.
fn unit_tmp_bytes<'c, BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    sources: usize,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    K: GGLWEInfos,
{
    let block = a.block_layout();
    let s = mapped_block(block, first);
    let term = mapped_block(s, shift(1));
    let shifts: Vec<_> = (1..1usize << a.log_d()).map(shift).collect();
    BE::blas_map_blocks_tmp_bytes_impl(module, &block, &block, sources, &[first], atk)
        .max(BE::blas_map_blocks_tmp_bytes_impl(module, &term, &s, sources, &shifts, atk))
        .max(module.ckks_copy_tmp_bytes(&term, &s))
        .max(module.ckks_prepare_tmp_bytes(&term))
}

/// The operand of `a` under `hint`: hot, a verbatim copy of every block of `a`; cold, the terms of every block, block row
/// after block row ([`prepare_terms`]), each converted once by `convert` under `Reuse`.
#[allow(clippy::too_many_arguments)]
fn prepare<'c, BE, A, K, P>(
    module: &Module<BE>,
    op: &'static str,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    hint: PrepareHint,
    convert: impl Fn(&CKKSCiphertextOwned<BE>, PrepareHint, &mut ScratchArena<'_, BE>) -> Result<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<BlasMatMulOperand<BE, P>>
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let terms = match hint {
        PrepareHint::OneShot => {
            let block = a.block_layout();
            assert_scratch(op, module.ckks_copy_tmp_bytes(&block, &block), scratch);
            let blocks = a.blocks().iter().map(|block| verbatim(module, block, scratch));
            Terms::Source(blocks.collect::<Result<_>>()?)
        }
        PrepareHint::Reuse => {
            let used = KeyLayouts::new(atk);
            let _ = ensure_term_keys(op, module, a, first, &shift, &used);
            assert_scratch(
                op,
                used.max_bytes(|key| prepare_tmp_bytes(module, a, first, &shift, key, hint)),
                scratch,
            );
            Terms::Prepared(prepare_terms(
                module,
                a.blocks(),
                a.cols(),
                a.log_d(),
                first,
                shift,
                atk,
                hint,
                convert,
                scratch,
            )?)
        }
    };
    Ok(BlasMatMulOperand {
        terms,
        log_d: a.log_d(),
        rows: a.rows(),
        cols: a.cols(),
        mat_slots: grid_batch::<BE::Ring>(a),
        source: a.block_layout(),
    })
}

/// A heap copy of `block`, verbatim: poulpy-ckks's `ckks_copy` into a block that holds every limb of `block`'s
/// allocation, which keeps its digits, metadata, canonical flag and noise tag.
fn verbatim<BE, S>(module: &Module<BE>, block: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: BlasMatIntraImpl,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let mut layout = layout_of(block);
    layout.glwe_layout.k = block.max_k();
    let mut copy = module.ckks_ciphertext_alloc_from_infos(&layout);
    module.ckks_copy(&mut copy, block, scratch)?;
    Ok(copy)
}

/// The terms of `blocks`, a grid of `cols` columns, row-major, one block row after another through [`unit_terms`]:
/// block `(i, j)`'s term `k` at `(i * cols + j) * d + k`.
#[allow(clippy::too_many_arguments)]
fn prepare_terms<'c, BE, S, K, P>(
    module: &Module<BE>,
    blocks: &[S],
    cols: usize,
    log_d: usize,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    hint: PrepareHint,
    convert: impl Fn(&CKKSCiphertextOwned<BE>, PrepareHint, &mut ScratchArena<'_, BE>) -> Result<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<P>>
where
    BE: BlasMatIntraImpl + 'c,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let mut terms = Vec::with_capacity(blocks.len() << log_d);
    for row in blocks.chunks(cols) {
        let sources: Vec<_> = row.iter().collect();
        terms.extend(unit_terms(
            module, &sources, log_d, first, &shift, atk, hint, &convert, scratch,
        )?);
    }
    Ok(terms)
}

/// The terms of `sources`, blocks of one layout, through the map hook: `S = first(s)` for every source into a heap block
/// of the sources' layout, then `shift(k)` of every `S` for every `k` in `1..d` into heap blocks of the terms' layout,
/// then each `S` narrowed by `ckks_copy` into a block of the terms' layout, its term `0`, and dropped. Each term is
/// converted once by `convert` under `hint`, then dropped: term `k` of source `t` lands at `t * d + k`.
#[allow(clippy::too_many_arguments)]
fn unit_terms<'c, BE, S, K, P>(
    module: &Module<BE>,
    sources: &[&S],
    log_d: usize,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    hint: PrepareHint,
    convert: impl Fn(&CKKSCiphertextOwned<BE>, PrepareHint, &mut ScratchArena<'_, BE>) -> Result<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<P>>
where
    BE: BlasMatIntraImpl + 'c,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (block, d) = (layout_of(sources[0]), 1usize << log_d);
    let term = mapped_block(mapped_block(block, first), shift(1));
    let alloc = |layout: &CKKSLayout, count: usize| -> Vec<CKKSCiphertextOwned<BE>> {
        (0..count).map(|_| module.ckks_ciphertext_alloc_from_infos(layout)).collect()
    };
    let mut sigma = alloc(&block, sources.len());
    BE::blas_map_blocks_impl(module, &mut sigma, sources, &[first], atk, scratch)?;
    let (shifts, results): (Vec<_>, Vec<_>) = ((1..d).map(&shift).collect(), sigma.iter().collect());
    let mut shifted = alloc(&term, sigma.len() * (d - 1));
    BE::blas_map_blocks_impl(module, &mut shifted, &results, &shifts, atk, scratch)?;
    let mut shifted = shifted.into_iter();
    let mut terms = Vec::with_capacity(sigma.len() * d);
    for s in sigma {
        let mut narrowed = module.ckks_ciphertext_alloc_from_infos(&term);
        module.ckks_copy(&mut narrowed, &s, scratch)?;
        drop(s);
        terms.push(convert(&narrowed, hint, scratch)?);
        for shifted in shifted.by_ref().take(d - 1) {
            terms.push(convert(&shifted, hint, scratch)?);
        }
    }
    Ok(terms)
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
/// one baby-step cache each, covering the baby steps of every map of `maps`, which serve all its blocks. The groups run
/// in order, refilling the same caches, and the first error stops them.
#[allow(clippy::too_many_arguments)]
fn shared_giants<'s, BE, M, D, S, K>(
    module: &M,
    groups: Vec<Vec<&mut D>>,
    inner: usize,
    source: impl Fn(usize, usize) -> &'s S,
    map: impl Fn(usize, usize, usize) -> &'s LinearTransformationPrepared<BE>,
    maps: &[&LinearTransformationPrepared<BE>],
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + 's,
    K: GetAutomorphismKey<BE>,
{
    let rotations: Vec<i64> = maps.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
    let mut caches: Vec<_> = (0..inner)
        .map(|_| LinearTransformationBabySteps::alloc(module, &rotations, source(0, 0)))
        .collect();
    for (g, blocks) in groups.into_iter().enumerate() {
        for (l, cache) in caches.iter_mut().enumerate() {
            module.ckks_prepare_linear_transformation_baby_steps(cache, source(g, l), atk, scratch)?;
        }
        for (o, dst) in blocks.into_iter().enumerate() {
            let terms: Vec<_> = caches
                .iter()
                .enumerate()
                .map(|(l, cache)| (source(g, l), cache, map(g, o, l)))
                .collect();
            module.ckks_eval_linear_transformation_sum_into(&mut *dst, &terms, atk, scratch)?;
            dst.set_log_sparsity(source(g, 0).log_sparsity());
        }
    }
    Ok(())
}
