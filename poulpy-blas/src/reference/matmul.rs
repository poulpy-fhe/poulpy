use std::marker::PhantomData;

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformationBabySteps,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    KeyLayouts, assert_scratch, ensure_dimension, ensure_grid, ensure_inner, ensure_layouts, ensure_lt_keys, ensure_maps,
    ensure_record, ensure_shape, ensure_tensor_key, lt_output_k, lt_tmp_bytes, max_lazy_terms, reject, stamp_sparsity,
};
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasPtOperand, grid_batch, layout_of};

/// The left side of a product, `A` in `A B`: sigma, then phi^k (`BlasContext::rotate_cols`).
pub struct Left;

/// The right side of a product, `B` in `A B`: tau, then psi^k (`BlasContext::rotate_rows`).
pub struct Right;

/// The reference's prepared operand, which [`crate::impl_blas_matmul_reference!`] selects: the `d` terms of every block
/// of the source grid that the JKLS product multiplies. On the [`Left`] side, term `0` of block `(i, l)` is sigma of the
/// source block and term `k` its phi^k; on the [`Right`] side, tau and psi^k. Only a preparation makes one.
///
/// Its [`BlasGridInfos`] report the source grid's dimensions, the layout of term `0`, which
/// [`blas_matmul_term_layout`] gives before preparing, and the source grid's batch.
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

/// The layout every prepared operand of `a` under `ctx` reports ([`BlasGridInfos`]): `a`'s dimensions, and term `0`'s
/// block layout, `a`'s at the width the context's maps leave of it, `a.k - p` for their scale `p`, its slots joined with
/// theirs. The context's maps share one scale and slot kind, so both sides share this layout; it carries no batch
/// record.
pub fn blas_matmul_term_layout<BE: Backend>(a: &impl BlasGridInfos, ctx: &BlasContext<BE>) -> BlasLayout {
    let (mut block, sigma) = (a.block_layout(), ctx.sigma());
    block.glwe_layout.k = lt_output_k(sigma, block.k(), block.k());
    if let Some(diagonal) = sigma.first_diagonal_plaintext() {
        block.meta.slots = block.meta.slots.join(diagonal.lt_slots());
    }
    BlasLayout::new(block, a.log_d(), a.rows(), a.cols())
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul`] call from metadata alone, before any implementation runs, and
/// returns the batch `res` records, the larger of `a`'s and `b`'s: `a` and `b` as their preparations check them
/// ([`validate_blas_matmul_prepare_left`], [`validate_blas_matmul_prepare_right`]), `res` against them as
/// [`validate_blas_matmul_prepared`] checks it against their operands, and `tsk` at the width of each product of terms.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout, context or lazy sum it rejects, then `MissingAutomorphismKey` or
/// `MissingRelinearizationKey`. It checks no budget: poulpy-ckks rejects those inside the implementation.
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
    let left = ensure_term_keys(op, module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk)?;
    let right = ensure_term_keys(op, module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk)?;
    for (s_k, t_k) in left.into_iter().zip(right) {
        ensure_tensor_key(op, tsk, s_k.max(t_k))?;
    }
    Ok(grid_batch::<BE::Ring>(a).max(grid_batch::<BE::Ring>(b)))
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`] call from metadata alone, before any implementation
/// runs: `a` is a grid at the context's `log_d` whose blocks tile its shape and share one layout whose label allows its
/// batch record, of the module's degree, `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, and
/// `atk` holds every key of sigma from `a`'s width and of each phi^k from sigma's result, at the width it is looked up
/// at.
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
    ensure_term_keys(op, module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk).map(drop)
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
    ensure_term_keys(op, module, b, ctx.tau(), |k| ctx.rotate_rows(k), atk).map(drop)
}

/// Checks a [`crate::api::BlasMatMulOps::blas_matmul_prepared`] call from metadata alone, before any implementation runs,
/// and returns the batch `res` records, the larger of the operands': `a` and `b` are prepared operands, read through
/// their [`BlasGridInfos`], at one `log_d`, with at least one block each and a batch record their label allows; `res` is
/// a grid at that `log_d` whose blocks tile its shape and share one layout whose label allows its batch record, of `a`'s
/// rows and `b`'s columns, of the module's degree and the operands' radix and rank; `a`'s columns match `b`'s rows;
/// a block of `res` holds the lazy terms of two inner indices (`d + 1`, `2^(63 - base2k)` at most); and `tsk` holds the
/// tensor key at the wider of the operands' term widths.
///
/// # Errors
///
/// `InvalidOperand` for an operand without a batch record, or a grid, shape, layout or lazy sum it rejects, then
/// `MissingRelinearizationKey`. It checks no budget: poulpy-ckks rejects those inside the implementation.
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

/// `res` is a grid at `a`'s `log_d` of `a`'s rows and `b`'s columns, `a`'s columns match `b`'s rows, a block of `res`
/// holds the `d` lazy terms of an inner index on top of a normalized one ([`ensure_lazy_sums`]), and the blocks of `res`
/// have the module's degree and `a` and `b` its radix and rank.
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
    ensure_lazy_sums(op, res.block_layout().base2k().as_usize(), a.cols(), 1 << a.log_d())?;
    ensure_layouts(op, module.n(), res, &[a, b])
}

/// Checks the keys preparing `a` looks up: `first` (sigma or tau) from `a`'s width into a block of `a`'s layout, then
/// `shift(k)` (phi^k or psi^k) of that result for every `k` in `1..d`, each into a block of the result's layout, as the
/// reference evaluates them. Returns the widths of the `d` terms.
fn ensure_term_keys<'c, BE: Backend + 'c>(
    op: &'static str,
    module: &(impl CKKSModuleInfos + ?Sized),
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<Vec<TorusPrecision>> {
    let a_k = a.block_layout().k();
    ensure_lt_keys(op, module, first, atk, a_k, a_k)?;
    let s_k = lt_output_k(first, a_k, a_k);
    let mut widths = vec![s_k];
    for k in 1..1usize << a.log_d() {
        ensure_lt_keys(op, module, shift(k), atk, s_k, s_k)?;
        widths.push(lt_output_k(shift(k), s_k, s_k));
    }
    Ok(widths)
}

/// Reference compositions of [`crate::api::BlasMatMulOps`], on inputs their validators accepted: they check nothing but
/// their scratch and, in the prepared product, the tensor key at each term's width, and return poulpy-ckks's budget
/// errors. The preparations make [`BlasMatMulOperand`]s, which the prepared product reads.
pub trait BlasMatMulReference<BE: Backend> {
    /// Sigma from a block of `a`'s layout into one, or a shift from a term into one.
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, a, ctx, atk)
    }

    /// For every block of `a`, row-major: `S = sigma(a_il)` into a block of `a`'s layout, then for `k` in `1..d`,
    /// `phi^k(S)` into a block of `S`'s layout. The operand records `a`'s batch,
    /// [`grid_batch`](crate::layouts::grid_batch).
    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Left>>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSModuleInfos,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_prepare_left";
        prepare(self, op, a, ctx.sigma(), |k| ctx.rotate_cols(k), ctx, atk, scratch)
    }

    /// As [`Self::blas_matmul_prepare_left_tmp_bytes_reference`], with tau and psi^k.
    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, b, ctx, atk)
    }

    /// As [`Self::blas_matmul_prepare_left_reference`], with tau and psi^k.
    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulOperand<BE, Right>>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSModuleInfos,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_prepare_right";
        prepare(self, op, b, ctx.tau(), |k| ctx.rotate_rows(k), ctx, atk, scratch)
    }

    /// One block-shaped product on top of the multiplication, the addition and the normalization, from operands or
    /// term layouts `a` and `b`.
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        Self: CKKSMulOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
    {
        let (res, a, b) = (res.block_layout(), a.block_layout(), b.block_layout());
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + self
                .ckks_mul_tmp_bytes(&res, &a, &b, tsk)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.glwe_normalize_tmp_bytes())
    }

    /// For `l` in `0..K` and `k` in `0..d`, for every `(i, j)` in row-major order: `res_ij = a_il[k] (.) b_lj[k]` on the
    /// first term, `res_ij += a_il[k] (.) b_lj[k]` through a temporary otherwise. Every block of `res` is normalized
    /// before an `l` whose `d` terms would take it past `2^(63 - base2k)` lazy terms. The terms are not yet one width, so
    /// it first checks `tsk` at the width of each shift's products, below the reported one the validator checked, and
    /// fails with `MissingRelinearizationKey` before any write.
    fn blas_matmul_prepared_reference<R, T>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, Left>,
        b: &BlasMatMulOperand<BE, Right>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSMulOps<BE> + CKKSAddOps<BE> + GLWEBytesOf<BE> + GLWENormalize<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
    {
        let op = "blas_matmul_prepared";
        let d = 1usize << a.log_d();
        let used = KeyLayouts::new(tsk);
        for k in 0..d {
            let looked_up = ensure_tensor_key(op, &used, a.term(0, 0, k).k().max(b.term(0, 0, k).k()));
            // The validator checked term 0's width, the operands' reported one; this one only records its layout.
            if k > 0 {
                looked_up?;
            }
        }
        let bytes = used.max_tensor_bytes(|tsk| self.blas_matmul_prepared_tmp_bytes_reference(&*res, a, b, tsk));
        assert_scratch(op, bytes, scratch);
        let res_layout = res.block_layout();
        let mut held = 0;
        for l in 0..a.cols() {
            held = scratch.scope(|s| -> Result<usize> {
                let (mut prod, mut s) = s.take_ckks_ciphertext_like_scratch(&res_layout);
                let terms = make_room(self, res.blocks_mut(), held, d, &mut s);
                for k in 0..d {
                    let lhs: Vec<_> = (0..a.rows()).map(|i| a.term(i, l, k)).collect();
                    let rhs: Vec<_> = (0..b.cols()).map(|j| b.term(l, j, k)).collect();
                    accumulate(
                        self,
                        res.blocks_mut(),
                        &lhs,
                        &rhs,
                        held == 0 && k == 0,
                        &mut prod,
                        tsk,
                        &mut s,
                    )?;
                }
                Ok(terms)
            })?;
        }
        Ok(())
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

impl<BE: Backend> BlasMatMulReference<BE> for Module<BE> {}

/// Scratch of a preparation of `a` under `ctx`: sigma (tau) from a block of `a`'s layout into one, or a shift from a
/// term into one.
fn prepare_tmp_bytes<BE, M, A, K>(module: &M, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let (a_layout, term) = (a.block_layout(), blas_matmul_term_layout(a, ctx).block);
    lt_tmp_bytes(module, &a_layout, &a_layout, atk).max(lt_tmp_bytes(module, &term, &term, atk))
}

/// The terms of every block of `a`, row-major: `first(a_ij)` into a block of `a`'s layout, then `shift(k)` of that for
/// every `k` in `1..d`, each into a block of the first's layout.
#[allow(clippy::too_many_arguments)]
fn prepare<'c, BE, M, A, K, S>(
    module: &M,
    op: &'static str,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<BlasMatMulOperand<BE, S>>
where
    BE: Backend + 'c,
    M: CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSModuleInfos + ?Sized,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE>,
{
    let used = KeyLayouts::new(atk);
    let _ = ensure_term_keys(op, module, a, first, &shift, &used);
    let bytes = used.max_bytes(|key| prepare_tmp_bytes(module, a, ctx, key));
    assert_scratch(op, bytes, scratch);
    let (layout, d) = (a.block_layout(), 1usize << a.log_d());
    let mut terms = Vec::with_capacity(a.blocks().len() * d);
    for src in a.blocks() {
        let mut s = module.ckks_ciphertext_alloc_from_infos(&layout);
        module.ckks_eval_linear_transformation_self_into(&mut s, src, first, atk, scratch)?;
        let at = terms.len();
        terms.push(s);
        for k in 1..d {
            let mut term = module.ckks_ciphertext_alloc_from_infos(&terms[at]);
            module.ckks_eval_linear_transformation_self_into(&mut term, &terms[at], shift(k), atk, scratch)?;
            terms.push(term);
        }
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

/// For every `(i, j)` in row-major order, block `i * rhs.len() + j` of `res`: `res_ij = lhs_i (.) rhs_j` when `first`,
/// otherwise `prod = lhs_i (.) rhs_j` and `res_ij += prod`.
#[allow(clippy::too_many_arguments)]
fn accumulate<BE, M, D, L, Rh, P, T>(
    module: &M,
    res: &mut [D],
    lhs: &[&L],
    rhs: &[&Rh],
    first: bool,
    prod: &mut P,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSMulOps<BE> + CKKSAddOps<BE> + ?Sized,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    L: GLWEToBackendRef<BE> + CKKSCtBounds,
    Rh: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    T: GetTensorKey<BE>,
{
    let n = rhs.len();
    for (i, l) in lhs.iter().enumerate() {
        for (j, r) in rhs.iter().enumerate() {
            let dst = &mut res[i * n + j];
            if first {
                module.ckks_mul_into(dst, *l, *r, tsk, scratch)?;
            } else {
                module.ckks_mul_into(prod, *l, *r, tsk, scratch)?;
                module.ckks_add_assign(dst, &*prod, scratch)?;
            }
        }
    }
    Ok(())
}
