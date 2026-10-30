//! Public operands: grids of products by public matrices, allocated from a layout, encoded and prepared in place.

use std::marker::PhantomData;

use num_traits::One;
use poulpy_ckks::{
    CoeffsMeta,
    api::{CKKSModuleInfos, LinearTransformation, LinearTransformationPrepared},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, CKKSPreparedRight},
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic, GetDegree, LinearTransformationStrategy};
use poulpy_hal::layouts::{Backend, PrepareHint};

use super::{
    BlasPlan, BlasPtAlgorithm,
    context::assert_maps_fit,
    encoding::{Encoded, MapLayout, Scheduled, real_map},
    maps,
};

/// The left side of a product by public matrices, `B` in `B A`: a [`BlasPtOperand`] on the left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Left;

/// The right side of a product by public matrices, `B` in `A B`, the default side of a [`BlasPtOperand`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Right;

/// The side of a product by public matrices: [`Right`], `A B`, or [`Left`], `B A`.
pub trait BlasPtSide: Copy + Eq + std::fmt::Debug + Send + Sync + 'static {
    /// Whether the public matrices multiply on the left.
    const LEFT: bool;
}

impl BlasPtSide for Right {
    const LEFT: bool = false;
}

impl BlasPtSide for Left {
    const LEFT: bool = true;
}

/// A public operand's plan and dimensions, without its maps: `rows x cols` blocks of `count` public matrices per batch,
/// on a module of `slots` slots, every map encoded at `coeffs_meta` and radix `base2k`, on side `S`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasPtOperandLayout<S = Right> {
    /// The block dimension and how the products are evaluated.
    pub plan: BlasPlan,
    /// Slots of the module the maps are encoded on.
    pub slots: usize,
    /// Number of block rows.
    pub rows: usize,
    /// Number of block columns.
    pub cols: usize,
    /// Number of distinct public matrices, matrix slot `s` using matrix `s mod count`.
    pub count: usize,
    /// Radix of the maps' diagonals.
    pub base2k: Base2K,
    /// Torus width and metadata the maps are encoded at.
    pub coeffs_meta: CoeffsMeta,
    side: PhantomData<fn() -> S>,
}

impl<S> BlasPtOperandLayout<S> {
    /// The layout of an operand of these fields on side `S`.
    pub fn new(
        plan: BlasPlan,
        slots: usize,
        rows: usize,
        cols: usize,
        count: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
    ) -> Self {
        Self {
            plan,
            slots,
            rows,
            cols,
            count,
            base2k,
            coeffs_meta,
            side: PhantomData,
        }
    }
}

/// A public operand's plan and dimensions, read from a layout, an encoded operand or a prepared one.
pub trait BlasPtOperandInfos {
    /// The side of the products it enters.
    type Side: BlasPtSide;

    /// A plain-data snapshot of these infos.
    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<Self::Side>;

    /// [`BlasPtOperandLayout::plan`].
    fn plan(&self) -> BlasPlan {
        self.blas_pt_operand_layout().plan
    }

    /// `log2` of the block dimension.
    fn log_d(&self) -> usize {
        self.plan().log_d()
    }

    /// [`BlasPtOperandLayout::slots`].
    fn slots(&self) -> usize {
        self.blas_pt_operand_layout().slots
    }

    /// [`BlasPtOperandLayout::rows`].
    fn rows(&self) -> usize {
        self.blas_pt_operand_layout().rows
    }

    /// [`BlasPtOperandLayout::cols`].
    fn cols(&self) -> usize {
        self.blas_pt_operand_layout().cols
    }

    /// [`BlasPtOperandLayout::count`].
    fn count(&self) -> usize {
        self.blas_pt_operand_layout().count
    }

    /// [`BlasPtOperandLayout::base2k`].
    fn base2k(&self) -> Base2K {
        self.blas_pt_operand_layout().base2k
    }

    /// [`BlasPtOperandLayout::coeffs_meta`].
    fn coeffs_meta(&self) -> CoeffsMeta {
        self.blas_pt_operand_layout().coeffs_meta
    }
}

impl<T: BlasPtOperandInfos + ?Sized> BlasPtOperandInfos for &T {
    type Side = T::Side;

    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<Self::Side> {
        (**self).blas_pt_operand_layout()
    }
}

impl<T: BlasPtOperandInfos + ?Sized> BlasPtOperandInfos for &mut T {
    type Side = T::Side;

    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<Self::Side> {
        (**self).blas_pt_operand_layout()
    }
}

impl<S: BlasPtSide> BlasPtOperandInfos for BlasPtOperandLayout<S> {
    type Side = S;

    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<S> {
        *self
    }
}

/// The maps of a public operand, in the order [`BlasPtOperandUnprepared::maps`] states.
pub(crate) struct OperandMaps<T> {
    slots: usize,
    cols: usize,
    pub(crate) terms: usize,
    pub(crate) first: Option<T>,
    pub(crate) blocks: Vec<T>,
    pub(crate) blocks_rep: Option<Vec<T>>,
}

impl<T> OperandMaps<T> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> + Clone {
        self.first.iter().chain(&self.blocks).chain(self.blocks_rep.iter().flatten())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.first
            .iter_mut()
            .chain(&mut self.blocks)
            .chain(self.blocks_rep.iter_mut().flatten())
    }

    /// `f` of every map, in walking order.
    pub(crate) fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> OperandMaps<U> {
        OperandMaps {
            slots: self.slots,
            cols: self.cols,
            terms: self.terms,
            first: self.first.as_ref().map(&mut f),
            blocks: self.blocks.iter().map(&mut f).collect(),
            blocks_rep: self.blocks_rep.as_ref().map(|blocks| blocks.iter().map(&mut f).collect()),
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    pub(crate) fn rows(&self) -> usize {
        self.blocks.len() / (self.cols * self.terms)
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn first(&self) -> Option<&T> {
        self.first.as_ref()
    }

    /// Block `(l, j)`, its first term under `JklsStored`, without a bounds assertion.
    pub(crate) fn at(&self, l: usize, j: usize) -> &T {
        &self.blocks[(l * self.cols + j) * self.terms]
    }

    /// Block `(l, j)` as a product reads it: its replicated-input map when `replicated` and the operand holds them.
    pub(crate) fn at_for(&self, l: usize, j: usize, replicated: bool) -> &T {
        match &self.blocks_rep {
            Some(blocks) if replicated => &blocks[l * self.cols + j],
            _ => self.at(l, j),
        }
    }

    pub(crate) fn has_replicated(&self) -> bool {
        self.blocks_rep.is_some()
    }
}

/// A grid of products by public matrices, encoded and not yet prepared: on the right of an encrypted grid `A`, `A B`
/// ([`Right`]), or on its left, `B A` ([`Left`]).
pub struct BlasPtOperandUnprepared<BE: Backend, S = Right> {
    layout: BlasPtOperandLayout<S>,
    pub(crate) maps: OperandMaps<Encoded<BE>>,
}

/// A prepared grid of products by public matrices, on the right of an encrypted grid ([`Right`], which
/// `blas_matmul_pt_right` takes) or on its left ([`Left`], which `blas_matmul_pt_left` takes).
pub struct BlasPtOperand<BE: Backend, S = Right> {
    layout: BlasPtOperandLayout<S>,
    pub(crate) maps: OperandMaps<LinearTransformationPrepared<BE>>,
    /// Under `JklsStored` and `JklsGalois`, the terms of every block, row-major, each block's in term order; `maps` then
    /// holds the first map alone. Empty under the other paths.
    pub(crate) plaintexts: Vec<CKKSPreparedRight<BE>>,
}

impl<BE: Backend, S: BlasPtSide> BlasPtOperandInfos for BlasPtOperandUnprepared<BE, S> {
    type Side = S;

    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<S> {
        self.layout
    }
}

impl<BE: Backend, S: BlasPtSide> BlasPtOperandInfos for BlasPtOperand<BE, S> {
    type Side = S;

    fn blas_pt_operand_layout(&self) -> BlasPtOperandLayout<S> {
        self.layout
    }
}

impl<BE: Backend, S> BlasPtOperandUnprepared<BE, S> {
    /// Every encoded map, in the order encoding and preparation walk them: the first map under JKLS and the masked
    /// paths, the blocks row-major, each block's terms in order, then the replicated-input maps if any.
    pub fn maps(&self) -> impl Iterator<Item = &LinearTransformation<CKKSPlaintextOwned<BE>>> {
        self.maps.iter()
    }
}

impl<BE: Backend, S> BlasPtOperand<BE, S> {
    /// Every prepared map, in the order of [`BlasPtOperandUnprepared::maps`]; under [`BlasPtAlgorithm::JklsStored`]
    /// and [`BlasPtAlgorithm::JklsGalois`] the first map alone, the terms being [`Self::block_plaintexts`].
    pub fn maps(&self) -> impl Iterator<Item = &LinearTransformationPrepared<BE>> {
        self.maps.iter()
    }

    /// The map every block of the encrypted grid goes through first, once, under [`BlasPtAlgorithm::Jkls`],
    /// [`BlasPtAlgorithm::JklsStored`] and [`BlasPtAlgorithm::JklsGalois`]: sigma on the right, tau on the left; `None`
    /// under [`BlasPtAlgorithm::Diagonal`].
    pub fn first(&self) -> Option<&LinearTransformationPrepared<BE>> {
        self.maps.first()
    }

    /// The prepared map of block `(l, j)`: under `Diagonal`, `A -> A B_lj` on the right and `A -> B_lj A` on the left;
    /// under `Jkls`, the same products of [`Self::first`]'s results. Panics if `(l, j)` is outside the grid, and under
    /// `JklsStored` and `JklsGalois`, whose blocks are [`Self::block_plaintexts`].
    pub fn block(&self, l: usize, j: usize) -> &LinearTransformationPrepared<BE> {
        self.assert_block(l, j);
        assert!(self.plaintexts.is_empty(), "the operand's blocks are plaintexts");
        self.maps.at(l, j)
    }

    /// The prepared plaintexts of block `(l, j)` under [`BlasPtAlgorithm::JklsStored`] and
    /// [`BlasPtAlgorithm::JklsGalois`], in term order: under `JklsStored`, `d` on the right and `2d - 1` on the left, tau
    /// (sigma) of the block rotated by each rotation of its terms; under `JklsGalois`, that of the block, unrotated.
    /// Empty under the other paths. Panics if `(l, j)` is outside the grid.
    pub fn block_plaintexts(&self, l: usize, j: usize) -> &[CKKSPreparedRight<BE>] {
        self.assert_block(l, j);
        self.plaintexts_at(l, j)
    }

    /// [`Self::block_plaintexts`] without a bounds assertion.
    pub(crate) fn plaintexts_at(&self, l: usize, j: usize) -> &[CKKSPreparedRight<BE>] {
        let per = self.plaintexts.len() / (self.layout.rows * self.layout.cols);
        let t = (l * self.layout.cols + j) * per;
        &self.plaintexts[t..t + per]
    }

    fn assert_block(&self, l: usize, j: usize) {
        assert!(
            l < self.layout.rows && j < self.layout.cols,
            "the block is outside the operand's grid"
        );
    }
}

impl<BE: Backend> BlasPtOperand<BE, Left> {
    /// The prepared map of block `(i, l)` for a replicated input, with `d` diagonals at the offsets `j d` for `j < d`
    /// ([`maps::left_mul_rep`]), scheduled at `left_mul`'s giant step, so it rotates within the left product's key
    /// set; `None` unless the operand was encoded under [`BlasPtAlgorithm::Diagonal`] with one public matrix on more
    /// than one matrix slot. A left product takes these maps instead of [`Self::block`]'s exactly when its input is
    /// replicated ([`crate::layouts::replicated`], read from the input's label) and the operand holds them. Panics if
    /// `(i, l)` is outside the grid.
    pub fn block_replicated(&self, i: usize, l: usize) -> Option<&LinearTransformationPrepared<BE>> {
        self.assert_block(i, l);
        self.maps.blocks_rep.as_ref().map(|blocks| &blocks[i * self.layout.cols + l])
    }
}

/// Panics unless `layout` fits: its maps as [`assert_maps_fit`] states, at least one block and a block count that a
/// `usize` holds, and a count of public matrices that divides the matrix slots.
fn assert_fits<S>(layout: &BlasPtOperandLayout<S>) {
    let log_d = layout.plan.log_d();
    assert_maps_fit("operand", log_d, layout.slots, layout.coeffs_meta, layout.base2k);
    assert!(
        matches!(layout.rows.checked_mul(layout.cols), Some(1..)),
        "the operand's grid has no block or more than a usize counts"
    );
    let mat_slots = layout.slots >> (2 * log_d);
    assert!(
        layout.count > 0 && mat_slots.is_multiple_of(layout.count),
        "the operand's count does not divide its matrix slots"
    );
}

/// Panics unless `layout` is of `module`'s slots.
pub(crate) fn assert_module_slots<S>(module: &impl CKKSModuleInfos, layout: &BlasPtOperandLayout<S>) {
    assert!(
        layout.slots == module.ckks_max_slots(),
        "the operand's slots are not the module's"
    );
}

/// Whether an operand of `layout` holds replicated-input maps: one public matrix on the left of more than one matrix
/// slot, under the diagonal products, where they merge diagonals.
fn holds_replicated<S: BlasPtSide>(layout: &BlasPtOperandLayout<S>) -> bool {
    let mat_slots = layout.slots >> (2 * layout.plan.log_d());
    S::LEFT && layout.plan.pt_algorithm() == BlasPtAlgorithm::Diagonal && layout.count == 1 && mat_slots > 1
}

/// The rotations of the public slot vectors the terms of a block read under `JklsStored` and `JklsGalois`, in term
/// order: on the right `k d` for `k < d`; on the left `0`, then `k` and `k - d` for `1 <= k < d`.
pub(crate) fn masked_rotations(log_d: usize, left: bool) -> Vec<i64> {
    let d = 1i64 << log_d;
    if left {
        std::iter::once(0).chain((1..d).flat_map(|k| [k, k - d])).collect()
    } else {
        (0..d).map(|k| k * d).collect()
    }
}

/// Maps per block: one per rotation of [`masked_rotations`] under `JklsStored`, one otherwise.
fn terms_per_block<S: BlasPtSide>(layout: &BlasPtOperandLayout<S>) -> usize {
    let d = 1usize << layout.plan.log_d();
    match (layout.plan.pt_algorithm(), S::LEFT) {
        (BlasPtAlgorithm::JklsStored, false) => d,
        (BlasPtAlgorithm::JklsStored, true) => 2 * d - 1,
        _ => 1,
    }
}

/// `entry` on block `t`, row-major: `b(m, r, c)` is entry `(r, c)` of the block of matrix `m`.
fn block_entry<'a, F, S>(
    layout: &BlasPtOperandLayout<S>,
    entry: &'a impl Fn(usize, usize, usize) -> F,
    t: usize,
) -> impl Fn(usize, usize, usize) -> F + 'a {
    let d = 1usize << layout.plan.log_d();
    let (i, j) = (t / layout.cols, t % layout.cols);
    move |m, r, c| entry(m, i * d + r, j * d + c)
}

/// Tau (sigma on the left) of block `t`'s public matrices, the slot vector the masked paths' plaintexts rotate.
pub(crate) fn public_values<F, S: BlasPtSide>(
    layout: &BlasPtOperandLayout<S>,
    entry: &impl Fn(usize, usize, usize) -> F,
    t: usize,
) -> Vec<F> {
    let (log_d, b) = (layout.plan.log_d(), block_entry(layout, entry, t));
    if S::LEFT {
        maps::sigma_values(log_d, layout.count, b)
    } else {
        maps::tau_values(log_d, layout.count, b)
    }
}

/// The map every block of the encrypted grid goes through first under JKLS and the masked paths: sigma on the right, tau
/// on the left.
pub(crate) fn first_map<F, S>(layout: &BlasPtOperandLayout<S>) -> Option<Scheduled<F>>
where
    F: DiagonalArithmetic + One + PartialEq,
    S: BlasPtSide,
{
    let (plan, slots) = (layout.plan, layout.slots);
    (plan.pt_algorithm() != BlasPtAlgorithm::Diagonal).then(|| {
        let map = if S::LEFT {
            maps::tau(slots, plan.log_d())
        } else {
            maps::sigma(slots, plan.log_d())
        };
        let strategy = plan.strategy(&map, false);
        real_map(map, strategy, layout.coeffs_meta)
    })
}

/// The map of block `t`, row-major, holding `entry`: `A -> A B` on the right and `A -> B A` on the left, of the first
/// map's results under JKLS. Under the masked paths, the product by tau (sigma) of the block, its first term. Every
/// block shares one schedule, so `strategy`, when given, skips its search.
pub(crate) fn block_map<F, S>(
    layout: &BlasPtOperandLayout<S>,
    entry: &impl Fn(usize, usize, usize) -> F,
    t: usize,
    strategy: Option<LinearTransformationStrategy>,
) -> Scheduled<F>
where
    F: DiagonalArithmetic + PartialEq,
    S: BlasPtSide,
{
    let (plan, slots, count) = (layout.plan, layout.slots, layout.count);
    let log_d = plan.log_d();
    let b = block_entry(layout, entry, t);
    let map = match (S::LEFT, plan.pt_algorithm()) {
        (false, BlasPtAlgorithm::Diagonal) => maps::right_mul(slots, log_d, count, b),
        (false, BlasPtAlgorithm::Jkls) => maps::right_jkls(slots, log_d, count, b),
        (true, BlasPtAlgorithm::Diagonal) => maps::left_mul(slots, log_d, count, b),
        (true, BlasPtAlgorithm::Jkls) => maps::left_jkls(slots, log_d, count, b),
        (_, BlasPtAlgorithm::JklsStored | BlasPtAlgorithm::JklsGalois) => {
            maps::slot_product(slots, &public_values(layout, entry, t), 0)
        }
    };
    let strategy = strategy.unwrap_or_else(|| plan.strategy(&map, false));
    real_map(map, strategy, layout.coeffs_meta)
}

/// The replicated-input map of block `t`, scheduled by `strategy`, the plan's `left_mul_strategy`, so that it rotates
/// within the left product's key set.
pub(crate) fn rep_map<F, S>(
    layout: &BlasPtOperandLayout<S>,
    entry: &impl Fn(usize, usize, usize) -> F,
    t: usize,
    strategy: LinearTransformationStrategy,
) -> Scheduled<F>
where
    F: DiagonalArithmetic + PartialEq,
{
    let map = maps::left_mul_rep(layout.slots, layout.plan.log_d(), block_entry(layout, entry, t));
    real_map(map, strategy, layout.coeffs_meta)
}

/// The shape of every map of an operand of `layout` on `module`, once `layout` fits and is of the module's slots. The
/// blocks share one, whatever their values, so it is computed once.
fn operand_layouts<S: BlasPtSide>(module: &impl CKKSModuleInfos, layout: &BlasPtOperandLayout<S>) -> OperandMaps<MapLayout> {
    assert_module_slots(module, layout);
    layouts_of(layout)
}

/// [`operand_layouts`] split as a prepared operand holds them: the maps, and under `JklsStored` and `JklsGalois` the
/// terms, taken out of the blocks, which are prepared as plaintexts.
pub(crate) fn prepared_shapes<S: BlasPtSide>(
    module: &impl CKKSModuleInfos,
    layout: &BlasPtOperandLayout<S>,
) -> (OperandMaps<MapLayout>, Vec<MapLayout>) {
    let mut shapes = operand_layouts(module, layout);
    let terms = if layout.plan.pt_algorithm().masked() {
        std::mem::take(&mut shapes.blocks)
    } else {
        Vec::new()
    };
    (shapes, terms)
}

/// [`operand_layouts`] of a layout alone.
pub(crate) fn layouts_of<S: BlasPtSide>(layout: &BlasPtOperandLayout<S>) -> OperandMaps<MapLayout> {
    assert_fits(layout);
    let one = |_: usize, _: usize, _: usize| 1.0f64;
    let blocks = layout.rows * layout.cols;
    let terms = terms_per_block(layout);
    OperandMaps {
        slots: layout.slots,
        cols: layout.cols,
        terms,
        first: first_map::<f64, S>(layout).map(|map| map.layout()),
        blocks: vec![block_map(layout, &one, 0, None).layout(); blocks * terms],
        blocks_rep: holds_replicated(layout)
            .then(|| vec![rep_map(layout, &one, 0, layout.plan.left_mul_strategy(layout.slots)).layout(); blocks]),
    }
}

/// [`super::BlasModuleAlloc::blas_pt_operand_alloc_from_infos`].
pub(crate) fn alloc<BE, M, S>(module: &M, layout: &BlasPtOperandLayout<S>) -> BlasPtOperandUnprepared<BE, S>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GetDegree,
    S: BlasPtSide,
{
    let maps = operand_layouts(module, layout).map(|map| map.alloc::<BE, M>(module, layout.base2k));
    BlasPtOperandUnprepared { layout: *layout, maps }
}

/// [`crate::api::BlasPtOperandPreparedFactory::blas_pt_operand_prepared_alloc_from_infos`].
pub(crate) fn alloc_prepared<BE, M, S>(module: &M, layout: &BlasPtOperandLayout<S>) -> BlasPtOperand<BE, S>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GetDegree,
    S: BlasPtSide,
{
    let (shapes, terms) = prepared_shapes(module, layout);
    let plaintexts = terms
        .iter()
        .map(|term| {
            module.ckks_prepared_right_alloc_from_infos(&term.plaintext::<BE, M>(module, layout.base2k), PrepareHint::Reuse)
        })
        .collect();
    let maps = shapes.map(|map| map.alloc_prepared::<BE, M>(module, layout.base2k));
    BlasPtOperand {
        layout: *layout,
        maps,
        plaintexts,
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use poulpy_hal::layouts::{HostBytesBackend, Module};

    use super::*;
    use crate::layouts::BlasModuleAlloc;

    fn module() -> Module<HostBytesBackend> {
        Module::<HostBytesBackend>::new(256)
    }

    /// `2 x 3` blocks of `4 x 4` matrices on the module's 128 slots, eight matrix slots.
    fn layout<S: BlasPtSide>(
        module: &Module<HostBytesBackend>,
        algorithm: BlasPtAlgorithm,
        count: usize,
    ) -> BlasPtOperandLayout<S> {
        let plan = BlasPlan::new(2).with_pt_algorithm(algorithm);
        BlasPtOperandLayout::new(
            plan,
            module.ckks_max_slots(),
            2,
            3,
            count,
            Base2K(19),
            CoeffsMeta::from_delta_budget(20, 19),
        )
    }

    #[test]
    fn operands_allocate_every_map_of_their_layout() {
        let module = module();
        for algorithm in [BlasPtAlgorithm::Diagonal, BlasPtAlgorithm::Jkls] {
            for count in [1, 2] {
                let right = module.blas_pt_operand_alloc_from_infos(&layout::<Right>(&module, algorithm, count));
                let left = module.blas_pt_operand_alloc_from_infos(&layout::<Left>(&module, algorithm, count));
                let first = usize::from(algorithm == BlasPtAlgorithm::Jkls);
                // One public matrix on the left of eight matrix slots holds the replicated-input maps.
                let rep = usize::from(algorithm == BlasPtAlgorithm::Diagonal && count == 1);
                assert_eq!(right.maps().count(), first + 6, "right {algorithm:?} count={count}");
                assert_eq!(left.maps().count(), first + 6 + 6 * rep, "left {algorithm:?} count={count}");
                assert_eq!(right.blas_pt_operand_layout(), layout::<Right>(&module, algorithm, count));
            }
        }
        // Masked: a first map, then per block `d` (left `2d - 1`) maps under `JklsStored`, one under `JklsGalois`.
        for (algorithm, right_terms, left_terms) in [(BlasPtAlgorithm::JklsStored, 4, 7), (BlasPtAlgorithm::JklsGalois, 1, 1)] {
            let right = module.blas_pt_operand_alloc_from_infos(&layout::<Right>(&module, algorithm, 1));
            let left = module.blas_pt_operand_alloc_from_infos(&layout::<Left>(&module, algorithm, 2));
            assert_eq!(right.maps().count(), 1 + 6 * right_terms, "right {algorithm:?}");
            assert_eq!(left.maps().count(), 1 + 6 * left_terms, "left {algorithm:?}");
        }
    }

    /// The terms of a masked operand read rotations `k d` on the right, and `0`, then `k` and `k - d` on the left.
    #[test]
    fn masked_rotations_follow_the_term_order() {
        assert_eq!(masked_rotations(2, false), vec![0, 4, 8, 12]);
        assert_eq!(masked_rotations(2, true), vec![0, 1, -3, 2, -2, 3, -1]);
    }

    #[test]
    fn allocation_panics_on_layouts_that_do_not_fit() {
        let module = module();
        let base = layout::<Right>(&module, BlasPtAlgorithm::Diagonal, 1);
        let bad = [
            BlasPtOperandLayout {
                plan: BlasPlan::new(4),
                ..base
            },
            BlasPtOperandLayout {
                plan: BlasPlan::new(0),
                ..base
            },
            BlasPtOperandLayout { rows: 0, ..base },
            BlasPtOperandLayout {
                rows: usize::MAX / 4 + 2,
                cols: 4,
                ..base
            },
        ];
        for layout in bad {
            let alloc = catch_unwind(AssertUnwindSafe(|| module.blas_pt_operand_alloc_from_infos(&layout)));
            assert!(alloc.is_err(), "{layout:?}");
        }
        // Slots that fit the blocks but are not the module's panic on the slot check, before anything is allocated.
        let other_slots = BlasPtOperandLayout { slots: 64, ..base };
        let Err(payload) = catch_unwind(AssertUnwindSafe(|| module.blas_pt_operand_alloc_from_infos(&other_slots))) else {
            panic!("an operand of other slots allocated");
        };
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"the operand's slots are not the module's")
        );
    }
}
