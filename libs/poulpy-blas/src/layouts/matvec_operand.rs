//! Public operands of products of a vector by a public matrix: one linear transformation per pair of an output and an
//! input ciphertext, allocated from a layout, encoded and prepared in place.

use poulpy_ckks::{
    CoeffsMeta, SlotsKind,
    api::{CKKSModuleInfos, LinearTransformation, LinearTransformationPrepared, LtDiagonalMeta, PreparedDiagonal},
    layouts::{CKKSLinearTransformationAlloc, CKKSPlaintextOwned},
};
use poulpy_core::{
    GLWEDiagonalProd,
    layouts::{
        Base2K, DiagonalArithmetic, Diagonals, GetDegree, LinearTransformationLayout, LinearTransformationStrategy,
        optimal_bsgs_giant_step,
    },
};
use poulpy_hal::{api::CnvPVecAlloc, layouts::Backend};

use super::{
    BlasVectorShape,
    encoding::{Encoded, MapLayout, Scheduled, real_map, ring_slots},
};

/// A public operand's dimensions, without its maps: a matrix of `output.len` rows and `input.len` columns, from vectors
/// of shape `input` repeating every `input_period` slots into vectors of shape `output` repeating every `output_period`
/// slots, on a module of `slots` slots, every map encoded at `coeffs_meta` and radix `base2k` and evaluated on the
/// schedule of `strategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasMatVecOperandLayout {
    /// Slots of the module the maps are encoded on.
    pub slots: usize,
    /// Where the entries of the input vector sit.
    pub input: BlasVectorShape,
    /// The period the input vector repeats with, at least the one its label proves.
    pub input_period: usize,
    /// Where the entries of the result sit.
    pub output: BlasVectorShape,
    /// The period the result repeats with, which its label then proves.
    pub output_period: usize,
    /// Radix of the maps' diagonals.
    pub base2k: Base2K,
    /// Torus width and metadata the maps are encoded at.
    pub coeffs_meta: CoeffsMeta,
    /// The schedule every map shares, with a key per rotation: one giant step per diagonal (`Direct`, the default of
    /// [`Self::new`]) or a BSGS giant step (`Bsgs`), which [`Self::with_optimal_bsgs`] sets to the cost-optimal one.
    pub strategy: LinearTransformationStrategy,
}

impl BlasMatVecOperandLayout {
    /// The layout of an operand of these fields, on the `Direct` schedule.
    pub fn new(
        slots: usize,
        input: BlasVectorShape,
        input_period: usize,
        output: BlasVectorShape,
        output_period: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
    ) -> Self {
        Self {
            slots,
            input,
            input_period,
            output,
            output_period,
            base2k,
            coeffs_meta,
            strategy: LinearTransformationStrategy::Direct,
        }
    }

    /// This layout on the schedule of `strategy`.
    pub fn with_strategy(self, strategy: LinearTransformationStrategy) -> Self {
        Self { strategy, ..self }
    }

    /// This layout on the BSGS schedule at the cost-optimal giant step of its maps' diagonals.
    ///
    /// # Panics
    ///
    /// As [`super::BlasModuleAlloc::blas_matvec_operand_alloc`], for a layout that does not fit.
    pub fn with_optimal_bsgs(self) -> Self {
        let giant_step = optimal_bsgs_giant_step(map_indexes(&self), self.slots);
        self.with_strategy(LinearTransformationStrategy::Bsgs { giant_step })
    }

    /// The period every map repeats with, the larger of the two.
    pub(crate) fn map_period(&self) -> usize {
        self.input_period.max(self.output_period)
    }

    /// Number of maps, one per output and input ciphertext.
    pub(crate) fn blocks(&self) -> usize {
        self.output.count() * self.input.count()
    }
}

/// A public operand's dimensions, read from a layout, an encoded operand or a prepared one.
pub trait BlasMatVecOperandInfos {
    /// A plain-data snapshot of these infos.
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout;

    /// [`BlasMatVecOperandLayout::slots`].
    fn slots(&self) -> usize {
        self.blas_matvec_operand_layout().slots
    }

    /// [`BlasMatVecOperandLayout::input`].
    fn input(&self) -> BlasVectorShape {
        self.blas_matvec_operand_layout().input
    }

    /// [`BlasMatVecOperandLayout::input_period`].
    fn input_period(&self) -> usize {
        self.blas_matvec_operand_layout().input_period
    }

    /// [`BlasMatVecOperandLayout::output`].
    fn output(&self) -> BlasVectorShape {
        self.blas_matvec_operand_layout().output
    }

    /// [`BlasMatVecOperandLayout::output_period`].
    fn output_period(&self) -> usize {
        self.blas_matvec_operand_layout().output_period
    }

    /// [`BlasMatVecOperandLayout::base2k`].
    fn base2k(&self) -> Base2K {
        self.blas_matvec_operand_layout().base2k
    }

    /// [`BlasMatVecOperandLayout::coeffs_meta`].
    fn coeffs_meta(&self) -> CoeffsMeta {
        self.blas_matvec_operand_layout().coeffs_meta
    }
}

impl<T: BlasMatVecOperandInfos + ?Sized> BlasMatVecOperandInfos for &T {
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout {
        (**self).blas_matvec_operand_layout()
    }
}

impl<T: BlasMatVecOperandInfos + ?Sized> BlasMatVecOperandInfos for &mut T {
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout {
        (**self).blas_matvec_operand_layout()
    }
}

impl BlasMatVecOperandInfos for BlasMatVecOperandLayout {
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout {
        *self
    }
}

/// A public matrix encoded for products of a vector, not yet prepared: its diagonals are plaintexts, which a product
/// prepares at each use (streamed).
pub struct BlasMatVecOperand<BE: Backend> {
    layout: BlasMatVecOperandLayout,
    pub(crate) maps: Vec<Encoded<BE>>,
}

/// A public matrix prepared for products of a vector: its diagonals are resident.
pub struct BlasMatVecOperandPrepared<BE: Backend> {
    layout: BlasMatVecOperandLayout,
    pub(crate) maps: Vec<LinearTransformationPrepared<BE>>,
}

impl<BE: Backend> BlasMatVecOperandInfos for BlasMatVecOperand<BE> {
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout {
        self.layout
    }
}

impl<BE: Backend> BlasMatVecOperandInfos for BlasMatVecOperandPrepared<BE> {
    fn blas_matvec_operand_layout(&self) -> BlasMatVecOperandLayout {
        self.layout
    }
}

/// The maps of a public operand of products of a vector, in either form: [`BlasMatVecOperandPrepared`] (resident) or
/// [`BlasMatVecOperand`] (streamed).
pub trait BlasMatVecMaps<BE: Backend>: BlasMatVecOperandInfos + Sync {
    /// The form of a diagonal.
    type Diagonal: GLWEDiagonalProd<BE> + LtDiagonalMeta + Sync;

    /// Whether the diagonals are plaintexts a product prepares at each use.
    const STREAMED: bool;

    /// The map of output ciphertext `o` from input ciphertext `i` at `o * input.count() + i`, every map sharing one
    /// schedule.
    fn blocks(&self) -> &[LinearTransformation<Self::Diagonal>];
}

impl<BE: Backend> BlasMatVecMaps<BE> for BlasMatVecOperand<BE>
where
    CKKSPlaintextOwned<BE>: GLWEDiagonalProd<BE> + Sync,
{
    type Diagonal = CKKSPlaintextOwned<BE>;
    const STREAMED: bool = true;

    fn blocks(&self) -> &[Encoded<BE>] {
        &self.maps
    }
}

impl<BE: Backend> BlasMatVecMaps<BE> for BlasMatVecOperandPrepared<BE> {
    type Diagonal = PreparedDiagonal<BE::OwnedBuf, BE>;
    const STREAMED: bool = false;

    fn blocks(&self) -> &[LinearTransformationPrepared<BE>] {
        &self.maps
    }
}

impl<BE: Backend> BlasMatVecOperand<BE> {
    /// The encoded map of output ciphertext `o` from input ciphertext `i`. Panics outside the operand.
    pub fn block(&self, o: usize, i: usize) -> &LinearTransformation<CKKSPlaintextOwned<BE>> {
        &self.maps[block_index(&self.layout, o, i)]
    }
}

impl<BE: Backend> BlasMatVecOperandPrepared<BE> {
    /// The prepared map of output ciphertext `o` from input ciphertext `i`. Panics outside the operand.
    pub fn block(&self, o: usize, i: usize) -> &LinearTransformationPrepared<BE> {
        &self.maps[block_index(&self.layout, o, i)]
    }
}

fn block_index(layout: &BlasMatVecOperandLayout, o: usize, i: usize) -> usize {
    let (outputs, inputs) = (layout.output.count(), layout.input.count());
    assert!(o < outputs && i < inputs, "the block is outside the operand");
    o * inputs + i
}

/// Panics unless `layout` fits: slots a power of two, each vector's shape within its period, which divides the slots,
/// a block count that a `usize` holds, a torus width, a radix in `1..=63` and a `Direct` or `Bsgs` schedule, its giant
/// step at least 1.
pub(crate) fn assert_fits(layout: &BlasMatVecOperandLayout) {
    let slots = layout.slots;
    assert!(slots.is_power_of_two(), "the operand's slots are not a power of two");
    assert!(
        layout.input.fits(slots, layout.input_period),
        "the operand's input does not fit its period"
    );
    assert!(
        layout.output.fits(slots, layout.output_period),
        "the operand's output does not fit its period"
    );
    assert!(
        layout.output.count().checked_mul(layout.input.count()).is_some(),
        "the operand has more blocks than a usize counts"
    );
    assert!(layout.coeffs_meta.k.as_usize() > 0, "the operand's maps have no torus width");
    assert!(
        (1..=63).contains(&layout.base2k.as_usize()),
        "the operand's radix is outside 1..=63"
    );
    match layout.strategy {
        LinearTransformationStrategy::Direct => {}
        LinearTransformationStrategy::Bsgs { giant_step } => assert!(giant_step > 0, "the operand's BSGS giant step is zero"),
        LinearTransformationStrategy::BsgsPowersOfTwo { .. } => {
            panic!("the operand's schedule takes a key per rotation: Direct or Bsgs")
        }
    }
}

/// Panics unless `layout` fits and is of `module`'s slots.
pub(crate) fn assert_module_slots(module: &impl CKKSModuleInfos, layout: &BlasMatVecOperandLayout) {
    assert_fits(layout);
    assert!(
        layout.slots == module.ckks_max_slots(),
        "the operand's slots are not the module's"
    );
}

/// The terms of every map, `(s, j, r, c)`: output entry `r` of a ciphertext at slot `s`, one per repetition of the
/// output within the map's period `p`, reads input entry `c` at slot `j`, the slot of that entry at or after `s` that
/// the input's period reaches, so that every term sits on a diagonal `(j - s) mod p_in` below the input's period. Rows
/// and columns past the matrix keep their terms, at zero, so the maps share one shape whatever their values.
fn terms(layout: &BlasMatVecOperandLayout) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
    let (input, output) = (layout.input, layout.output);
    let (p_in, p_out, slots) = (layout.input_period, layout.output_period, layout.slots);
    (0..layout.map_period() / p_out).flat_map(move |q| {
        (0..output.chunk).flat_map(move |r| {
            let s = output.offset + output.stride * r + q * p_out;
            (0..input.chunk).map(move |c| {
                let at = input.offset + input.stride * c;
                let j = (s + (at + p_in - s % p_in) % p_in) % slots;
                (s, j, r, c)
            })
        })
    })
}

/// The map of block `(o, i)`, entry `(row, col)` of the matrix reading `entry(row, col)` for `row < output.len`, `col <
/// input.len`, on the layout's schedule, which every block shares.
pub(crate) fn block_map<F: DiagonalArithmetic>(
    layout: &BlasMatVecOperandLayout,
    entry: &impl Fn(usize, usize) -> F,
    o: usize,
    i: usize,
) -> Scheduled<F> {
    real_map(block_diagonals(layout, entry, o, i), layout.strategy, layout.coeffs_meta)
}

/// The host diagonals of [`block_map`].
fn block_diagonals<F: DiagonalArithmetic>(
    layout: &BlasMatVecOperandLayout,
    entry: &impl Fn(usize, usize) -> F,
    o: usize,
    i: usize,
) -> Diagonals<F> {
    let (input, output) = (layout.input, layout.output);
    let terms = terms(layout).map(|(s, j, r, c)| {
        let (row, col) = (o * output.chunk + r, i * input.chunk + c);
        let w = if row < output.len && col < input.len {
            entry(row, col)
        } else {
            <F as DiagonalArithmetic>::zero()
        };
        (s, j, w)
    });
    Diagonals::from_slot_map(layout.slots, layout.map_period(), terms)
}

/// The diagonals every map of an operand of `layout` has, whatever its values, ascending.
fn map_indexes(layout: &BlasMatVecOperandLayout) -> Vec<i64> {
    assert_fits(layout);
    let (p_in, slots) = (layout.input_period, layout.slots);
    let mut used = vec![false; p_in];
    for (s, j, _, _) in terms(layout) {
        used[(j + slots - s) % slots] = true;
    }
    (0..p_in).filter(|&d| used[d]).map(|d| d as i64).collect()
}

/// The shape every map of an operand of `layout` shares, [`block_map`]'s, from the layout alone.
pub(crate) fn map_layout(layout: &BlasMatVecOperandLayout) -> MapLayout {
    let mut meta = layout.coeffs_meta;
    meta.meta.slots = SlotsKind::Real;
    MapLayout::new(
        LinearTransformationLayout {
            strategy: layout.strategy,
            indexes: map_indexes(layout),
            slots: layout.slots,
        },
        layout.map_period(),
        meta,
    )
}

/// The label a product by an operand of `infos` stamps on its result: the output's period proved, `log2(slots /
/// output_period)`.
pub(crate) fn output_label(infos: &impl BlasMatVecOperandInfos) -> usize {
    (infos.slots() / infos.output_period()).ilog2() as usize
}

/// Galois elements [`blas_matvec_pt`](crate::api::BlasMatVecOps::blas_matvec_pt) looks up for an operand of `infos`,
/// for any matrix values, on a ring of cyclotomic order `cyclotomic_order`: the baby and giant rotations of the maps'
/// shared schedule. Empty when the ring's slots are not the operand's.
///
/// # Panics
///
/// As [`super::BlasModuleAlloc::blas_matvec_operand_alloc`], for a layout that does not fit.
pub fn matvec_pt_galois_elements(cyclotomic_order: i64, infos: &impl BlasMatVecOperandInfos) -> Vec<i64> {
    let layout = infos.blas_matvec_operand_layout();
    if ring_slots(cyclotomic_order) != Some(layout.slots) {
        return Vec::new();
    }
    map_layout(&layout).lt().galois_elements(cyclotomic_order)
}

/// [`super::BlasModuleAlloc::blas_matvec_operand_alloc_from_infos`].
pub(crate) fn alloc<BE, M>(module: &M, layout: &BlasMatVecOperandLayout) -> BlasMatVecOperand<BE>
where
    BE: Backend,
    M: CKKSLinearTransformationAlloc<BE> + GetDegree,
{
    assert_module_slots(module, layout);
    let map = map_layout(layout);
    let maps = (0..layout.blocks())
        .map(|_| map.alloc::<BE, M>(module, layout.base2k))
        .collect();
    BlasMatVecOperand { layout: *layout, maps }
}

/// [`crate::api::BlasMatVecOperandPreparedFactory::blas_matvec_operand_prepared_alloc_from_infos`].
pub(crate) fn alloc_prepared<BE, M>(module: &M, layout: &BlasMatVecOperandLayout) -> BlasMatVecOperandPrepared<BE>
where
    BE: Backend,
    M: CnvPVecAlloc<BE> + CKKSModuleInfos + GetDegree,
{
    assert_module_slots(module, layout);
    let map = map_layout(layout);
    let maps = (0..layout.blocks())
        .map(|_| map.alloc_prepared::<BE, M>(module, layout.base2k))
        .collect();
    BlasMatVecOperandPrepared { layout: *layout, maps }
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::Evaluate;

    use super::*;

    fn layout(input: BlasVectorShape, p_in: usize, output: BlasVectorShape, p_out: usize) -> BlasMatVecOperandLayout {
        BlasMatVecOperandLayout::new(
            64,
            input,
            p_in,
            output,
            p_out,
            Base2K(19),
            CoeffsMeta::from_delta_budget(20, 19),
        )
    }

    /// `x`'s slots holding the input's entries `v` at every repetition of `p_in`, 0 elsewhere.
    fn slots_of(shape: BlasVectorShape, period: usize, slots: usize, t: usize, v: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; slots];
        for e in 0..shape.chunk {
            let entry = t * shape.chunk + e;
            if entry < v.len() {
                for q in 0..slots / period {
                    x[shape.offset + shape.stride * e + q * period] = v[entry];
                }
            }
        }
        x
    }

    /// Every block evaluated in the clear is the matrix product on the output's slots, at every repetition, and 0 on
    /// the other slots; its shape is the layout's.
    #[test]
    fn blocks_compute_the_product_at_every_repetition() {
        let entry = |row: usize, col: usize| ((row * 7 + col * 3) % 11) as f64 / 8.0 - 0.5;
        let cases = [
            // One ciphertext each, dense.
            layout(BlasVectorShape::new(5, 5, 1, 3), 64, BlasVectorShape::new(3, 3, 0, 5), 64),
            // A replicated input into a dense output, and the reverse.
            layout(BlasVectorShape::new(4, 4, 0, 2), 8, BlasVectorShape::new(6, 6, 1, 4), 64),
            layout(BlasVectorShape::new(6, 6, 2, 5), 32, BlasVectorShape::new(4, 4, 0, 1), 16),
            // Several ciphertexts on each side, the last ones partial.
            layout(BlasVectorShape::new(7, 3, 1, 2), 16, BlasVectorShape::new(5, 2, 0, 3), 8),
        ];
        for layout in cases {
            let (input, output) = (layout.input, layout.output);
            let v: Vec<f64> = (0..input.len).map(|c| c as f64 / 4.0 - 0.75).collect();
            let shape = map_layout(&layout);
            for o in 0..output.count() {
                let mut y = vec![0.0; layout.slots];
                for i in 0..input.count() {
                    assert_eq!(
                        block_map(&layout, &entry, o, i).layout(),
                        shape,
                        "{layout:?}: block ({o}, {i})"
                    );
                    let x = slots_of(input, layout.input_period, layout.slots, i, &v);
                    let t = block_diagonals(&layout, &entry, o, i).evaluate(&x, LinearTransformationStrategy::Direct);
                    y.iter_mut().zip(t).for_each(|(y, t)| *y += t);
                }
                let want: Vec<f64> = (0..output.len)
                    .map(|row| (0..input.len).map(|col| entry(row, col) * v[col]).sum())
                    .collect();
                let want = slots_of(output, layout.output_period, layout.slots, o, &want);
                for (s, (have, want)) in y.iter().zip(&want).enumerate() {
                    assert!(
                        (have - want).abs() < 1e-9,
                        "{layout:?}: block row {o}, slot {s}: {have} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn layouts_that_do_not_fit_panic() {
        let base = layout(BlasVectorShape::new(4, 4, 0, 2), 8, BlasVectorShape::new(4, 4, 0, 1), 64);
        let bad = [
            BlasMatVecOperandLayout { slots: 48, ..base },
            BlasMatVecOperandLayout { input_period: 4, ..base },
            BlasMatVecOperandLayout {
                output_period: 24,
                ..base
            },
            BlasMatVecOperandLayout {
                input: BlasVectorShape::new(4, 0, 0, 2),
                ..base
            },
            BlasMatVecOperandLayout {
                base2k: Base2K(64),
                ..base
            },
            base.with_strategy(LinearTransformationStrategy::Bsgs { giant_step: 0 }),
            base.with_strategy(LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 2 }),
        ];
        for layout in bad {
            assert!(std::panic::catch_unwind(|| map_layout(&layout)).is_err(), "{layout:?}");
        }
    }
}
