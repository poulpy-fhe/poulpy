//! The plan of the circuit, its per-operation key sets, and the per-matrix maps encoded once, at a caller-chosen
//! precision.

use std::collections::BTreeSet;

use num_traits::One;
use poulpy_ckks::{
    CoeffsMeta, GLWEPlaintextMeta, Scale,
    api::{CKKSModuleInfos, LinearTransformation, LinearTransformationPrepared, optimal_bsgs_giant_step},
    layouts::{CKKSLinearTransformationAlloc, CKKSPlaintextOwned},
};
use poulpy_core::layouts::{
    Base2K, Degree, DiagonalArithmetic, Diagonals, GetDegree, LinearTransformationLayout, LinearTransformationStrategy,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, galois_elements_from_rotations},
};

use super::{
    BlasGridInfos,
    ciphertext::grid_batch_at,
    encoding::{DiagonalShape, Encoded, Scheduled, real_map, ring_slots, shape},
    maps,
};

/// How the products by public matrices are evaluated, which a plan records for the public operands encoded from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlasPtAlgorithm {
    /// One map per block of the public operand, `2d - 1` diagonals, `d` on the left when one matrix fills the slots
    /// ([`maps::right_mul`], [`maps::left_mul`]): one scale. A left operand of one public matrix on more than one matrix
    /// slot also holds, per block, the `d`-diagonal map for a replicated input ([`maps::left_mul_rep`]), `d^2` slot
    /// values per diagonal: `(3d - 1) d^2` slot values per block where the other operands hold `(2d - 1) d^2`.
    #[default]
    Diagonal,
    /// JKLS: sigma (on the right) or tau (on the left) of every block of the encrypted grid, once, then one map per block
    /// of the public operand ([`maps::right_jkls`], [`maps::left_jkls`]): two scales. Against `Diagonal`, for the same
    /// keys, it spends a second scale, evaluates one more map per block of the encrypted grid, its first, whose noise it
    /// adds, and holds that map in the operand; for one public matrix (`count = 1`) its blocks on the right store `d^2`
    /// slots per diagonal where [`maps::right_mul`] stores `d`, up to `d` times the memory.
    Jkls,
    /// JKLS with its masks on the ciphertext: sigma (tau on the left) of every block of the encrypted grid, once, then
    /// the context's masks of its results, phi^k on the right and the column halves of psi^k on the left
    /// ([`maps::rows_left`], [`maps::rows_right`]), then one sum of products by prepared public plaintexts, `d` per
    /// block on the right, `psi^k(tau(B))`, and `2d - 1` on the left, `Rot_j(sigma(B))` for `-d < j < d`: three scales.
    /// On the right it takes one public matrix.
    JklsStored,
    /// [`Self::JklsStored`] with one public plaintext per block, tau (sigma) of its matrices, read through the Galois
    /// element of each rotation.
    JklsGalois,
}

impl BlasPtAlgorithm {
    /// Whether the products put JKLS's masks on the ciphertext, reading them from the context.
    pub(crate) fn masked(self) -> bool {
        matches!(self, Self::JklsStored | Self::JklsGalois)
    }
}

/// The block dimension and the schedule of every map: what a client and a server agree on, and the one source of the
/// rotation keys each operation looks up.
///
/// Shift maps (phi^k, psi^k, their halves and psi^k on a replicated input) are evaluated `Direct`, one giant step per
/// diagonal; sigma, tau, the transpose and the products by public matrices take the cost-optimal BSGS giant step, and
/// tau on a replicated input takes tau's. The key sets take the ring's cyclotomic order, a power of two whose quarter
/// is the slot count, and build the maps on the host without encoding them, so a client plans its keys with no module,
/// context or public matrix. A pipeline needs the union of its operations' sets, [`sum_galois_elements`] for each grid
/// it sums, and a tensor key if and only if it calls `blas_matmul` or `blas_matmul_prepared`. Every set is empty when a
/// block does not fit the slots. The public operands encoded from a plan follow its [`BlasPtAlgorithm`]; their key sets
/// are the same under `Diagonal` and `Jkls`, and add the context's masks under `JklsStored` and `JklsGalois`, whose
/// context then also encodes the column halves of psi^k.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasPlan {
    log_d: usize,
    pt: BlasPtAlgorithm,
}

impl BlasPlan {
    /// The circuit on `2^log_d x 2^log_d` blocks, with the products by public matrices evaluated by
    /// [`BlasPtAlgorithm::Diagonal`].
    pub fn new(log_d: usize) -> Self {
        Self {
            log_d,
            pt: BlasPtAlgorithm::Diagonal,
        }
    }

    /// `log2` of the block dimension.
    pub fn log_d(&self) -> usize {
        self.log_d
    }

    /// This plan with the products by public matrices evaluated by `algorithm`.
    pub fn with_pt_algorithm(self, algorithm: BlasPtAlgorithm) -> Self {
        Self { pt: algorithm, ..self }
    }

    /// How the products by public matrices are evaluated.
    pub fn pt_algorithm(&self) -> BlasPtAlgorithm {
        self.pt
    }

    /// The schedule of `map`: `Direct` for a shift map, the cost-optimal BSGS giant step otherwise.
    pub(crate) fn strategy<F>(&self, map: &Diagonals<F>, shift: bool) -> LinearTransformationStrategy {
        if shift {
            LinearTransformationStrategy::Direct
        } else {
            LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(map.indexes(), map.slots()),
            }
        }
    }

    /// The schedule of the left product's maps over `slots` slots: the cost-optimal giant step of [`maps::left_mul`]'s
    /// offsets, which [`maps::left_mul_rep`] shares so that it rotates within the left product's key set.
    pub(crate) fn left_mul_strategy(&self, slots: usize) -> LinearTransformationStrategy {
        self.strategy(&maps::left_mul(slots, self.log_d, 1, |_, _, _| 1.0f64), false)
    }

    /// Galois elements of every map a context encodes, which cover every operation's set but `blas_rotate`'s and
    /// `blas_sum`'s.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            let mut list = self.matmul_maps(slots);
            list.push((maps::transpose(slots, self.log_d), false));
            for k in 1..1 << self.log_d {
                list.extend(self.shift_maps(slots, k, true));
                list.extend(self.shift_maps(slots, k, false));
                if self.pt.masked() {
                    list.extend(self.masked_rows(slots, k));
                }
            }
            list
        })
    }

    /// Galois elements `blas_transpose` looks up.
    pub fn transpose_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| vec![(maps::transpose(slots, self.log_d), false)])
    }

    /// Galois elements `blas_rotate_rows` by `k` looks up, on a grid of any height; none when `d` divides `k`.
    pub fn rotate_rows_galois_elements(&self, cyclotomic_order: i64, k: usize) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.shift_maps(slots, k, true))
    }

    /// Galois elements `blas_rotate_cols` by `k` looks up, on a grid of any width; none when `d` divides `k`.
    pub fn rotate_cols_galois_elements(&self, cyclotomic_order: i64, k: usize) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.shift_maps(slots, k, false))
    }

    /// Galois elements `blas_sum_rows` looks up, on a grid of any label: its tree's rotations by `2^i d` and, off a
    /// replicated grid, its broadcast's by `-2^i d`, for `i < log2(d)`.
    pub fn sum_rows_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.reduction_elements(cyclotomic_order, true)
    }

    /// Galois elements `blas_sum_cols` looks up: its tree's rotations by `2^i` and its broadcast's by `-2^i`, for
    /// `i < log2(d)`.
    pub fn sum_cols_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.reduction_elements(cyclotomic_order, false)
    }

    /// Galois elements `blas_rotate` by `r` matrix slots looks up; none when the matrix slot count divides `r`.
    pub fn rotate_galois_elements(&self, cyclotomic_order: i64, r: i64) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let mat_slots = (slots >> (2 * self.log_d)) as i64;
        galois_elements_from_rotations([r.rem_euclid(mat_slots) << (2 * self.log_d)], cyclotomic_order)
    }

    /// Galois elements `blas_matmul`, `blas_matmul_left_operand_prepare` and `blas_matmul_right_operand_prepare` look up, and
    /// `blas_matmul_prepared` for a hot operand, whose terms it evaluates; of two cold operands it looks up none.
    pub fn matmul_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.matmul_maps(slots))
    }

    /// Galois elements `blas_matmul_pt_right` looks up, for public matrices of any values, grid shape and count: under
    /// `Diagonal` and `Jkls` those of the right product, which stores a diagonal at every offset `l - c`, where sigma and
    /// the JKLS block maps sit too; under `JklsStored` and `JklsGalois` sigma's and the context's phi^k, `Direct`.
    pub fn matmul_pt_right_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            let log_d = self.log_d;
            if !self.pt.masked() {
                return vec![(maps::right_mul(slots, log_d, 1, |_, _, _| 1.0), false)];
            }
            let mut list = vec![(maps::sigma(slots, log_d), false)];
            list.extend((1..1 << log_d).map(|k| (maps::rotate_cols(slots, log_d, k), true)));
            list
        })
    }

    /// Galois elements `blas_matmul_pt_left` looks up, for public matrices of any values, grid shape and count: under
    /// `Diagonal` and `Jkls` those of the left product, which stores a diagonal at every offset `(l - r) d`, tau's
    /// offsets, scheduled alike, where the JKLS block maps sit too; under `JklsStored` and `JklsGalois` tau's and the
    /// context's column halves of psi^k, `Direct`.
    pub fn matmul_pt_left_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            if !self.pt.masked() {
                return vec![(maps::left_mul(slots, self.log_d, 1, |_, _, _| 1.0), false)];
            }
            let mut list = vec![(maps::tau(slots, self.log_d), false)];
            list.extend((1..1 << self.log_d).flat_map(|k| self.masked_rows(slots, k)));
            list
        })
    }

    /// The rotations of an axis reduction along the rows (`rows`) or the columns, tree and broadcast; none when a block
    /// does not fit the slots.
    fn reduction_elements(&self, cyclotomic_order: i64, rows: bool) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let tree = reduction_rotations(self.log_d, rows);
        galois_elements_from_rotations(tree.flat_map(|r| [r, (-r).rem_euclid(slots as i64)]), cyclotomic_order)
    }

    /// The slot count of a ring of cyclotomic order `cyclotomic_order`, if a block fits it.
    fn slots(&self, cyclotomic_order: i64) -> Option<usize> {
        let slots = ring_slots(cyclotomic_order).filter(|&slots| slots > 0)?;
        (self.log_d <= slots.ilog2() as usize / 2).then_some(slots)
    }

    /// The Galois elements of the maps `maps(slots)` lists, each flagged a shift map, scheduled as the context encodes
    /// them; none when a block does not fit the slots.
    fn elements(&self, cyclotomic_order: i64, maps: impl FnOnce(usize) -> Vec<(Diagonals<f64>, bool)>) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let mut elements = BTreeSet::new();
        for (map, shift) in maps(slots) {
            let layout = LinearTransformationLayout {
                indexes: map.indexes(),
                slots,
                strategy: self.strategy(&map, shift),
            };
            elements.extend(layout.galois_elements(cyclotomic_order));
        }
        elements.into_iter().collect()
    }

    /// Sigma and tau, then phi^k and psi^k for `1 <= k < d`.
    fn matmul_maps(&self, slots: usize) -> Vec<(Diagonals<f64>, bool)> {
        let log_d = self.log_d;
        let mut list = vec![(maps::sigma(slots, log_d), false), (maps::tau(slots, log_d), false)];
        for k in 1..1 << log_d {
            list.push((maps::rotate_cols(slots, log_d, k), true));
            list.push((maps::rotate_rows(slots, log_d, k), true));
        }
        list
    }

    /// With `k' = k mod d`: psi^k' and its two halves (`rows`), else phi^k' and its two halves; none for `k' = 0`, a
    /// copy.
    fn shift_maps(&self, slots: usize, k: usize, rows: bool) -> Vec<(Diagonals<f64>, bool)> {
        let (log_d, k) = (self.log_d, k % (1 << self.log_d));
        if k == 0 {
            return Vec::new();
        }
        let family = if rows {
            [maps::rotate_rows, maps::rows_upper, maps::rows_lower]
        } else {
            [maps::rotate_cols, maps::cols_left, maps::cols_right]
        };
        family.into_iter().map(|map| (map(slots, log_d, k), true)).collect()
    }

    /// The column halves of psi^k, `1 <= k < d`.
    fn masked_rows(&self, slots: usize, k: usize) -> [(Diagonals<f64>, bool); 2] {
        [
            (maps::rows_left(slots, self.log_d, k), true),
            (maps::rows_right(slots, self.log_d, k), true),
        ]
    }
}

/// The host diagonals of a shift map at `k`, on `slots` slots and blocks of `2^log_d`.
type ShiftMap<F> = fn(usize, usize, usize) -> Diagonals<F>;

/// One map of a context: one of its six single maps, or the map at `k` of one of its seven families of shifts, nine
/// under `JklsStored` and `JklsGalois`.
#[derive(Clone, Copy)]
pub(crate) enum MapId {
    Single(usize),
    Shift(usize, usize),
}

const SIGMA: usize = 0;
const TAU: usize = 1;
const TAU_REP: usize = 2;
const TRANSPOSE: usize = 3;
const FIRST_ROW: usize = 4;
const FIRST_COL: usize = 5;
const SINGLES: usize = 6;

const ROTATE_ROWS: usize = 0;
const ROTATE_ROWS_REP: usize = 1;
const ROTATE_COLS: usize = 2;
const ROWS_UPPER: usize = 3;
const ROWS_LOWER: usize = 4;
const COLS_LEFT: usize = 5;
const COLS_RIGHT: usize = 6;
const ROWS_LEFT: usize = 7;
const ROWS_RIGHT: usize = 8;
const FAMILIES: usize = 9;

impl MapId {
    /// Every map of a context of `plan`, in the order [`ContextMaps`] holds them.
    pub(crate) fn all(plan: BlasPlan) -> impl Iterator<Item = Self> {
        let (log_d, families) = (plan.log_d(), family_count(plan));
        let shifts = (0..families).flat_map(move |family| (1..1usize << log_d).map(move |k| Self::Shift(family, k)));
        (0..SINGLES).map(Self::Single).chain(shifts)
    }
}

/// The shift families a context of `plan` holds: the last two, the column halves of psi^k, only when its products by
/// public matrices read them.
fn family_count(plan: BlasPlan) -> usize {
    if plan.pt_algorithm().masked() { FAMILIES } else { ROWS_LEFT }
}

/// The maps of a context, in the order [`BlasContext::maps`] states.
pub(crate) struct ContextMaps<T> {
    log_d: usize,
    slots: usize,
    singles: [T; SINGLES],
    shifts: [Vec<T>; FAMILIES],
}

/// The maps in walking order, clonable so that preparation can size its scratch before it walks them.
pub(crate) type MapsIter<'a, T> = std::iter::Chain<std::slice::Iter<'a, T>, std::iter::Flatten<std::slice::Iter<'a, Vec<T>>>>;

impl<T> ContextMaps<T> {
    /// `make(id)` for every map of a context of `plan`, in walking order.
    pub(crate) fn build(plan: BlasPlan, slots: usize, mut make: impl FnMut(MapId) -> T) -> Self {
        let (log_d, families) = (plan.log_d(), family_count(plan));
        let singles = std::array::from_fn(|i| make(MapId::Single(i)));
        let shifts = std::array::from_fn(|family| {
            if family < families {
                (1..1usize << log_d).map(|k| make(MapId::Shift(family, k))).collect()
            } else {
                Vec::new()
            }
        });
        Self {
            log_d,
            slots,
            singles,
            shifts,
        }
    }

    pub(crate) fn log_d(&self) -> usize {
        self.log_d
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    pub(crate) fn iter(&self) -> MapsIter<'_, T> {
        self.singles.iter().chain(self.shifts.iter().flatten())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.singles.iter_mut().chain(self.shifts.iter_mut().flatten())
    }

    pub(crate) fn sigma(&self) -> &T {
        &self.singles[SIGMA]
    }

    pub(crate) fn tau(&self) -> &T {
        &self.singles[TAU]
    }

    pub(crate) fn tau_rep(&self) -> &T {
        &self.singles[TAU_REP]
    }

    pub(crate) fn transpose(&self) -> &T {
        &self.singles[TRANSPOSE]
    }

    pub(crate) fn first_row(&self) -> &T {
        &self.singles[FIRST_ROW]
    }

    pub(crate) fn first_col(&self) -> &T {
        &self.singles[FIRST_COL]
    }

    pub(crate) fn rotate_rows(&self, k: usize) -> &T {
        &self.shifts[ROTATE_ROWS][k - 1]
    }

    pub(crate) fn rotate_rows_rep(&self, k: usize) -> &T {
        &self.shifts[ROTATE_ROWS_REP][k - 1]
    }

    pub(crate) fn rotate_cols(&self, k: usize) -> &T {
        &self.shifts[ROTATE_COLS][k - 1]
    }

    pub(crate) fn rows_upper(&self, k: usize) -> &T {
        &self.shifts[ROWS_UPPER][k - 1]
    }

    pub(crate) fn rows_lower(&self, k: usize) -> &T {
        &self.shifts[ROWS_LOWER][k - 1]
    }

    pub(crate) fn cols_left(&self, k: usize) -> &T {
        &self.shifts[COLS_LEFT][k - 1]
    }

    pub(crate) fn cols_right(&self, k: usize) -> &T {
        &self.shifts[COLS_RIGHT][k - 1]
    }

    pub(crate) fn rows_left(&self, k: usize) -> &T {
        &self.shifts[ROWS_LEFT][k - 1]
    }

    pub(crate) fn rows_right(&self, k: usize) -> &T {
        &self.shifts[ROWS_RIGHT][k - 1]
    }

    /// Whether it holds the column halves of psi^k: its plan selects `JklsStored` or `JklsGalois`.
    pub(crate) fn has_masked_rows(&self) -> bool {
        !self.shifts[ROWS_LEFT].is_empty()
    }
}

/// `meta` at no scale, a product by 1.
fn unit(meta: CoeffsMeta) -> CoeffsMeta {
    CoeffsMeta {
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(0),
            ..meta.meta
        },
        ..meta
    }
}

/// Map `id` of a context of `layout`, scheduled as the plan schedules it and at the context's metadata, but tau on a
/// replicated input, scheduled at tau's giant step, psi^k on a replicated input, at no scale, and the projections on
/// row 0 and column 0, one diagonal at offset 0, `Direct`.
pub(crate) fn context_map<F: DiagonalArithmetic + One + PartialEq>(layout: &BlasContextLayout, id: MapId) -> Scheduled<F> {
    let (plan, slots, meta) = (layout.plan, layout.slots, layout.coeffs_meta);
    let log_d = plan.log_d();
    let scheduled = |diagonals: Diagonals<F>, shift: bool, meta: CoeffsMeta| {
        let strategy = plan.strategy(&diagonals, shift);
        real_map(diagonals, strategy, meta)
    };
    match id {
        MapId::Single(SIGMA) => scheduled(maps::sigma(slots, log_d), false, meta),
        MapId::Single(TAU) => scheduled(maps::tau(slots, log_d), false, meta),
        MapId::Single(TAU_REP) => {
            let strategy = plan.strategy(&maps::tau::<F>(slots, log_d), false);
            real_map(maps::tau_rep(slots, log_d), strategy, meta)
        }
        MapId::Single(TRANSPOSE) => scheduled(maps::transpose(slots, log_d), false, meta),
        MapId::Single(FIRST_ROW) => scheduled(maps::first_row(slots, log_d), true, meta),
        MapId::Single(_) => scheduled(maps::first_col(slots, log_d), true, meta),
        MapId::Shift(family, k) => {
            let families: [ShiftMap<F>; FAMILIES] = [
                maps::rotate_rows,
                maps::rotate_rows_rep,
                maps::rotate_cols,
                maps::rows_upper,
                maps::rows_lower,
                maps::cols_left,
                maps::cols_right,
                maps::rows_left,
                maps::rows_right,
            ];
            let meta = if family == ROTATE_ROWS_REP { unit(meta) } else { meta };
            scheduled(families[family](slots, log_d, k), true, meta)
        }
    }
}

/// A context's plan and encoding, without its maps: blocks of `2^log_d x 2^log_d` on the ring of degree `n` and `slots`
/// slots, every map encoded at `coeffs_meta` and radix `base2k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasContextLayout {
    /// The block dimension and the schedule of every map.
    pub plan: BlasPlan,
    /// Radix of the maps' diagonals.
    pub base2k: Base2K,
    /// Torus width and metadata the maps are encoded at.
    pub coeffs_meta: CoeffsMeta,
    /// Degree of the ring the maps are encoded on.
    pub n: Degree,
    /// Slots of the ring the maps are encoded on.
    pub slots: usize,
}

/// A context's plan and encoding, read from a layout, an encoded context or a prepared one.
pub trait BlasContextInfos {
    /// A plain-data snapshot of these infos.
    fn blas_context_layout(&self) -> BlasContextLayout;

    /// [`BlasContextLayout::plan`].
    fn plan(&self) -> BlasPlan {
        self.blas_context_layout().plan
    }

    /// `log2` of the block dimension.
    fn log_d(&self) -> usize {
        self.plan().log_d()
    }

    /// [`BlasContextLayout::base2k`].
    fn base2k(&self) -> Base2K {
        self.blas_context_layout().base2k
    }

    /// [`BlasContextLayout::coeffs_meta`].
    fn coeffs_meta(&self) -> CoeffsMeta {
        self.blas_context_layout().coeffs_meta
    }

    /// [`BlasContextLayout::n`].
    fn n(&self) -> Degree {
        self.blas_context_layout().n
    }

    /// [`BlasContextLayout::slots`].
    fn slots(&self) -> usize {
        self.blas_context_layout().slots
    }
}

impl<T: BlasContextInfos + ?Sized> BlasContextInfos for &T {
    fn blas_context_layout(&self) -> BlasContextLayout {
        (**self).blas_context_layout()
    }
}

impl<T: BlasContextInfos + ?Sized> BlasContextInfos for &mut T {
    fn blas_context_layout(&self) -> BlasContextLayout {
        (**self).blas_context_layout()
    }
}

impl BlasContextInfos for BlasContextLayout {
    fn blas_context_layout(&self) -> BlasContextLayout {
        *self
    }
}

/// The JKLS maps of one matrix dimension, their one-diagonal halves and their variants on a replicated input, encoded
/// and not yet prepared.
pub struct BlasContext<BE: Backend> {
    layout: BlasContextLayout,
    pub(crate) maps: ContextMaps<Encoded<BE>>,
}

/// The prepared JKLS maps of one matrix dimension, their one-diagonal halves and their variants on a replicated input.
pub struct BlasContextPrepared<BE: Backend> {
    layout: BlasContextLayout,
    pub(crate) maps: ContextMaps<LinearTransformationPrepared<BE>>,
}

impl<BE: Backend> BlasContextInfos for BlasContext<BE> {
    fn blas_context_layout(&self) -> BlasContextLayout {
        self.layout
    }
}

impl<BE: Backend> BlasContextInfos for BlasContextPrepared<BE> {
    fn blas_context_layout(&self) -> BlasContextLayout {
        self.layout
    }
}

impl<BE: Backend> BlasContext<BE> {
    /// Every encoded map, in the order encoding and preparation walk them: sigma, tau, tau on a replicated input, the
    /// transpose and the projections on row 0 and column 0, then psi^k, psi^k on a replicated input, phi^k and the
    /// halves of psi^k and of phi^k, then, under `JklsStored` and `JklsGalois`, the column halves of psi^k
    /// ([`BlasContextPrepared::rows_left`], [`BlasContextPrepared::rows_right`]), each family for `k = 1..d`.
    pub fn maps(&self) -> impl Iterator<Item = &LinearTransformation<CKKSPlaintextOwned<BE>>> {
        self.maps.iter()
    }
}

impl<BE: Backend> BlasContextPrepared<BE> {
    /// Every prepared map, in the order of [`BlasContext::maps`].
    pub fn maps(&self) -> impl Iterator<Item = &LinearTransformationPrepared<BE>> {
        self.maps.iter()
    }

    /// JKLS sigma.
    pub fn sigma(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.sigma()
    }

    /// JKLS tau.
    pub fn tau(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.tau()
    }

    /// JKLS tau of a replicated input ([`maps::tau_rep`]): `d` diagonals, scheduled at tau's giant step.
    pub fn tau_rep(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.tau_rep()
    }

    /// Per-matrix transpose.
    pub fn transpose(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.transpose()
    }

    /// Projection on row 0 of every matrix ([`maps::first_row`]): one diagonal at offset 0, `Direct`.
    pub fn first_row(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.first_row()
    }

    /// Projection on column 0 of every matrix ([`maps::first_col`]): one diagonal at offset 0, `Direct`.
    pub fn first_col(&self) -> &LinearTransformationPrepared<BE> {
        self.maps.first_col()
    }

    /// psi^k. Panics unless `1 <= k < d`.
    pub fn rotate_rows(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rotate_rows(k)
    }

    /// psi^k of a replicated input ([`maps::rotate_rows_rep`]): the rotation by `k d`, at no scale. Panics unless
    /// `1 <= k < d`.
    pub fn rotate_rows_rep(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rotate_rows_rep(k)
    }

    /// phi^k. Panics unless `1 <= k < d`.
    pub fn rotate_cols(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rotate_cols(k)
    }

    /// The rows of psi^k that stay in the block. Panics unless `1 <= k < d`.
    pub fn rows_upper(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rows_upper(k)
    }

    /// The rows of psi^k that wrap around the block. Panics unless `1 <= k < d`.
    pub fn rows_lower(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rows_lower(k)
    }

    /// The columns of phi^k that stay in the block. Panics unless `1 <= k < d`.
    pub fn cols_left(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.cols_left(k)
    }

    /// The columns of phi^k that wrap around the block. Panics unless `1 <= k < d`.
    pub fn cols_right(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.cols_right(k)
    }

    /// The columns of psi^k that phi^k keeps in the block, `c + k < d` ([`maps::rows_left`]). Panics unless
    /// `1 <= k < d` and the plan selects [`BlasPtAlgorithm::JklsStored`] or [`BlasPtAlgorithm::JklsGalois`].
    pub fn rows_left(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rows_left(k)
    }

    /// The columns of psi^k that phi^k wraps, `c + k >= d` ([`maps::rows_right`]). Panics as [`Self::rows_left`].
    pub fn rows_right(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        self.maps.rows_right(k)
    }
}

/// Panics unless the blocks of `layout`, at least `2 x 2`, fit its slots, and its maps have a torus width and a radix in
/// `1..=63`.
pub(crate) fn assert_fits(layout: &BlasContextLayout) {
    assert_maps_fit(
        "context",
        layout.plan.log_d(),
        layout.slots,
        layout.coeffs_meta,
        layout.base2k,
    );
}

/// The maps of a context of `infos` as the planning queries read them, without encoding any.
///
/// # Panics
///
/// As [`assert_fits`].
pub(crate) fn context_shapes(infos: &impl BlasContextInfos) -> ContextMaps<LinearTransformation<DiagonalShape>> {
    let layout = infos.blas_context_layout();
    assert_fits(&layout);
    ContextMaps::build(layout.plan, layout.slots, |id| {
        shape(&context_map::<f64>(&layout, id).layout())
    })
}

/// Panics unless maps of `owner` (a context or an operand) on blocks of dimension `2^log_d` fit: blocks at least `2 x 2`
/// that fit `slots`, a torus width and a radix in `1..=63`.
pub(crate) fn assert_maps_fit(owner: &str, log_d: usize, slots: usize, coeffs_meta: CoeffsMeta, base2k: Base2K) {
    assert!(
        log_d >= 1 && slots > 0 && log_d <= slots.ilog2() as usize / 2,
        "the {owner}'s blocks do not fit its slots"
    );
    assert!(coeffs_meta.k.as_usize() > 0, "the {owner}'s maps have no torus width");
    assert!((1..=63).contains(&base2k.as_usize()), "the {owner}'s radix is outside 1..=63");
}

/// Panics unless `layout` fits, as [`assert_fits`] states, and is of `module`'s ring.
pub(crate) fn assert_fits_module(module: &(impl GetDegree + CKKSModuleInfos), layout: &BlasContextLayout) {
    assert_fits(layout);
    assert!(
        layout.n == module.ring_degree() && layout.slots == module.ckks_max_slots(),
        "the context's ring is not the module's"
    );
}

/// The layout of a context of `plan` on `module`'s ring.
pub(crate) fn module_layout(
    module: &(impl GetDegree + CKKSModuleInfos),
    plan: BlasPlan,
    base2k: Base2K,
    coeffs_meta: CoeffsMeta,
) -> BlasContextLayout {
    BlasContextLayout {
        plan,
        base2k,
        coeffs_meta,
        n: module.ring_degree(),
        slots: module.ckks_max_slots(),
    }
}

/// Allocation of contexts.
pub trait BlasContextAlloc<BE: Backend>: CKKSLinearTransformationAlloc<BE> + GetDegree + Sized {
    /// A context of `plan` on this module's ring, every diagonal zero, which
    /// [`BlasEncodingOps::blas_context_encode`](crate::api::BlasEncodingOps::blas_context_encode) fills.
    ///
    /// # Panics
    ///
    /// Panics if the blocks of `plan` are smaller than `2 x 2` or do not fit the module's slots, if `coeffs_meta.k` is
    /// `0` or if `base2k` is outside `1..=63`.
    fn blas_context_alloc(&self, plan: BlasPlan, base2k: Base2K, coeffs_meta: CoeffsMeta) -> BlasContext<BE> {
        self.blas_context_alloc_from_infos(&module_layout(self, plan, base2k, coeffs_meta))
    }

    /// A context of `infos`, which must be of this module's ring.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_context_alloc`], and if the degree or the slots of `infos` are not the module's.
    fn blas_context_alloc_from_infos<A: BlasContextInfos>(&self, infos: &A) -> BlasContext<BE> {
        alloc(self, &infos.blas_context_layout())
    }
}

impl<BE: Backend> BlasContextAlloc<BE> for Module<BE> where Module<BE>: CKKSLinearTransformationAlloc<BE> {}

/// [`BlasContextAlloc::blas_context_alloc_from_infos`].
fn alloc<BE, M>(module: &M, layout: &BlasContextLayout) -> BlasContext<BE>
where
    BE: Backend,
    M: CKKSLinearTransformationAlloc<BE> + GetDegree,
{
    assert_fits_module(module, layout);
    let maps = ContextMaps::build(layout.plan, layout.slots, |id| {
        context_map::<f64>(layout, id).layout().alloc::<BE, M>(module, layout.base2k)
    });
    BlasContext { layout: *layout, maps }
}

/// [`crate::api::BlasContextPreparedFactory::blas_context_prepared_alloc_from_infos`].
pub(crate) fn alloc_prepared<BE, M>(module: &M, layout: &BlasContextLayout) -> BlasContextPrepared<BE>
where
    BE: Backend,
    M: CnvPVecAlloc<BE> + CKKSModuleInfos + GetDegree,
{
    assert_fits_module(module, layout);
    let maps = ContextMaps::build(layout.plan, layout.slots, |id| {
        context_map::<f64>(layout, id)
            .layout()
            .alloc_prepared::<BE, M>(module, layout.base2k)
    });
    BlasContextPrepared { layout: *layout, maps }
}

/// Galois elements `blas_sum` needs on a grid of layout `infos` on a ring of `cyclotomic_order`: at its recorded batch,
/// or for a layout without one the batch its label proves; none when one block does not fit the slots.
pub fn sum_galois_elements(cyclotomic_order: i64, infos: &impl BlasGridInfos) -> Vec<i64> {
    let slots = ring_slots(cyclotomic_order).unwrap_or(0);
    galois_elements_from_rotations(sum_rotations(infos.log_d(), grid_batch_at(slots, infos)), cyclotomic_order)
}

/// The rotations of `blas_sum` on `2^log_d x 2^log_d` blocks holding `mat_slots` matrices: `2^i d^2` for
/// `i < log2(mat_slots)`.
pub(crate) fn sum_rotations(log_d: usize, mat_slots: usize) -> impl Iterator<Item = i64> {
    (0..mat_slots.max(1).ilog2()).map(move |i| 1i64 << (2 * log_d + i as usize))
}

/// The maps a row (`rows`) or column shift by `k1 = k mod d` applies along a block axis of extent `extent`: none for
/// `k1 = 0`, a copy; psi^k1 (phi^k1) on one block; otherwise both half maps, the first of block `q`, the second of
/// block `q + 1`.
pub(crate) fn shift_maps<T>(maps: &ContextMaps<T>, k1: usize, rows: bool, extent: usize) -> Vec<&T> {
    match (k1, extent, rows) {
        (0, ..) => Vec::new(),
        (_, 1, true) => vec![maps.rotate_rows(k1)],
        (_, 1, false) => vec![maps.rotate_cols(k1)],
        (_, _, true) => vec![maps.rows_upper(k1), maps.rows_lower(k1)],
        (_, _, false) => vec![maps.cols_left(k1), maps.cols_right(k1)],
    }
}

/// Whether a row shift (`rows`) by `k1 = k mod d` on a grid one block high (`extent = 1`) whose label proves one matrix
/// (`replicated`) rotates by `k1 d` slots in place of psi^k1: the rotation wraps inside a block, at no scale.
pub(crate) fn shift_rotates(rows: bool, extent: usize, replicated: bool) -> bool {
    rows && extent == 1 && replicated
}

/// The tree of an axis reduction on `2^log_d x 2^log_d` blocks: by `2^i` rows (`rows`) or columns for `i < log_d`,
/// `2^i d` or `2^i` slots. Its broadcast rotates by their opposites.
pub(crate) fn reduction_rotations(log_d: usize, rows: bool) -> impl Iterator<Item = i64> {
    let stride = if rows { log_d } else { 0 };
    (0..log_d).map(move |i| 1i64 << (i + stride))
}

/// The projection an axis reduction along the rows (`rows`) or the columns evaluates after its tree: on row 0 or on
/// column 0. None along the rows of a `replicated` grid, where the tree leaves the sums in every row.
pub(crate) fn reduction_projection<T>(maps: &ContextMaps<T>, rows: bool, replicated: bool) -> Option<&T> {
    match (rows, replicated) {
        (true, true) => None,
        (true, false) => Some(maps.first_row()),
        (false, _) => Some(maps.first_col()),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::layouts::{BlasCiphertextAlloc, BlasLayout, replicated};
    use poulpy_ckks::{CKKSInfos, SetGLWEPlaintextInfos};
    use poulpy_hal::layouts::{CyclotomicOrder, HostBytesBackend, Standard};

    #[test]
    fn contexts_allocate_every_map_of_their_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let layout = module_layout(&module, BlasPlan::new(2), Base2K(19), CoeffsMeta::from_delta_budget(20, 19));
        let ctx = module.blas_context_alloc_from_infos(&layout);
        // Sigma, tau, tau on a replicated input, the transpose and both projections, then seven families of shifts for
        // k = 1, 2, 3.
        assert_eq!(ctx.maps().count(), 6 + 7 * 3);
        assert_eq!(ctx.blas_context_layout(), layout);
        // psi^k on a replicated input is encoded at no scale, every other map at the layout's.
        let scales: Vec<usize> = ctx
            .maps()
            .map(|lt| lt.first_diagonal_plaintext().unwrap().log_delta())
            .collect();
        assert_eq!(scales.iter().filter(|&&s| s == 0).count(), 3);
        assert_eq!(scales.iter().filter(|&&s| s == 20).count(), 6 + 6 * 3);
    }

    /// A context holds the column halves of psi^k, two more families, exactly when its plan's products by public
    /// matrices read them.
    #[test]
    fn masked_plans_encode_the_masked_row_shifts() {
        let module = Module::<HostBytesBackend>::new(256);
        let meta = CoeffsMeta::from_delta_budget(20, 19);
        for algorithm in [
            BlasPtAlgorithm::Diagonal,
            BlasPtAlgorithm::Jkls,
            BlasPtAlgorithm::JklsStored,
            BlasPtAlgorithm::JklsGalois,
        ] {
            let layout = module_layout(&module, BlasPlan::new(2).with_pt_algorithm(algorithm), Base2K(19), meta);
            let ctx = module.blas_context_alloc_from_infos(&layout);
            let families = if algorithm.masked() { 9 } else { 7 };
            assert_eq!(ctx.maps().count(), 6 + families * 3, "{algorithm:?}");
            assert_eq!(MapId::all(layout.plan).count(), ctx.maps().count(), "{algorithm:?}");
        }
    }

    #[test]
    fn context_allocation_panics_on_layouts_that_do_not_fit() {
        let module = Module::<HostBytesBackend>::new(256);
        let base = module_layout(&module, BlasPlan::new(2), Base2K(19), CoeffsMeta::from_delta_budget(20, 19));
        for layout in [
            BlasContextLayout {
                plan: BlasPlan::new(0),
                ..base
            },
            BlasContextLayout { n: Degree(512), ..base },
            BlasContextLayout { slots: 64, ..base },
        ] {
            let alloc = catch_unwind(AssertUnwindSafe(|| module.blas_context_alloc_from_infos(&layout)));
            assert!(alloc.is_err(), "{layout:?}");
        }
    }

    #[test]
    fn sum_keys_plan_from_a_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let mut ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        let mut layout = BlasLayout::new(ct.block_layout(), 2, 2, 2);
        let order = module.cyclotomic_order();
        // 128 slots hold 8 matrix slots of 4 x 4 blocks: rotations by 16, 32 and 64.
        let want = |rotations: &[i64]| -> Vec<i64> { rotations.iter().map(|&r| module.ckks_galois_element(r)).collect() };
        let sorted = |mut elements: Vec<i64>| {
            elements.sort_unstable();
            elements
        };
        assert_eq!(sum_galois_elements(order, &layout), sorted(want(&[16, 32, 64])));
        assert_eq!(sum_galois_elements(order, &ct), sorted(want(&[16, 32, 64])));
        layout.block.meta.log_sparsity = 1;
        assert_eq!(sum_galois_elements(order, &layout), sorted(want(&[16, 32])));
        // A grid plans from its record, not from its label.
        ct.block_mut(0, 0).set_log_sparsity(1);
        assert_eq!(sum_galois_elements(order, &ct), sorted(want(&[16, 32, 64])));
        ct.mat_slots = 2;
        assert_eq!(sum_galois_elements(order, &ct), want(&[16]));
        // No block fits, or a shift is past the word width: no key, no panic.
        for log_sparsity in [64, 65] {
            layout.block.meta.log_sparsity = log_sparsity;
            assert!(sum_galois_elements(order, &layout).is_empty(), "log_sparsity={log_sparsity}");
        }
        layout.block.meta.log_sparsity = 0;
        for log_d in [4, 40, usize::MAX] {
            layout.log_d = log_d;
            assert!(sum_galois_elements(order, &layout).is_empty(), "log_d={log_d}");
        }
    }

    /// The left product's replicated-input maps, scheduled at `left_mul`'s giant step, rotate within the left key set,
    /// so a client plans no new key.
    #[test]
    fn left_rep_maps_rotate_within_the_left_set() {
        for (order, log_d) in [
            (512, 1),
            (512, 2),
            (512, 3),
            (1 << 10, 3),
            (1 << 11, 4),
            (1 << 12, 4),
            (1 << 13, 5),
            (1 << 14, 5),
            (1 << 15, 6),
            (1 << 16, 6),
            (1 << 17, 7),
            (1 << 18, 7),
        ] {
            let (plan, slots) = (BlasPlan::new(log_d), (order / 4) as usize);
            let set = plan.matmul_pt_left_galois_elements(order);
            let map = maps::left_mul_rep::<f64>(slots, log_d, |_, _, _| 1.0);
            let layout = LinearTransformationLayout {
                indexes: map.indexes(),
                slots,
                strategy: plan.left_mul_strategy(slots),
            };
            for p in layout.galois_elements(order) {
                assert!(set.contains(&p), "order={order} log_d={log_d} p={p}");
            }
        }
    }

    /// 128 slots hold 8 matrix slots of 4 x 4 blocks: label 3 proves one matrix, so a grid is replicated; label 5 fits
    /// no block, so it is vacuously so.
    #[test]
    fn replicated_reads_the_label() {
        let module = Module::<HostBytesBackend>::new(256);
        let ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        let mut layout = BlasLayout::new(ct.block_layout(), 2, 1, 1);
        for (log_sparsity, want) in [(0, false), (2, false), (3, true), (5, true)] {
            layout.block.meta.log_sparsity = log_sparsity;
            assert_eq!(replicated::<Standard>(&layout), want, "log_sparsity={log_sparsity}");
        }
    }

    /// The right product's diagonals sit at sigma's offsets, `l - c`, and the left product's at tau's, multiples of `d`,
    /// so each shares that map's schedule and key set, at every dimension a block fits, one matrix slot included.
    #[test]
    fn public_products_share_sigma_and_tau_keys() {
        for (order, log_d) in [(512, 1), (512, 2), (512, 3), (256, 3)] {
            let plan = BlasPlan::new(log_d);
            let sigma = plan.elements(order, |slots| vec![(maps::sigma(slots, log_d), false)]);
            let tau = plan.elements(order, |slots| vec![(maps::tau(slots, log_d), false)]);
            assert!(!sigma.is_empty() && !tau.is_empty(), "order={order} log_d={log_d}");
            assert_eq!(
                plan.matmul_pt_right_galois_elements(order),
                sigma,
                "order={order} log_d={log_d}"
            );
            assert_eq!(plan.matmul_pt_left_galois_elements(order), tau, "order={order} log_d={log_d}");
        }
    }

    /// The key sets on 128 slots (cyclotomic order 512), by hand: a shift map rotates by each of its diagonals, psi^k'
    /// by `k' d` and `k' d - d^2`, phi^k' by `k'` and `k' - d`; BSGS rotates by the baby and giant parts of the others'
    /// diagonals, at the cost-optimal giant step. At `d = 2` those parts are the diagonals themselves. At `d = 4` the
    /// transpose's diagonals, the multiples of 3 from -9 to 9, split at giant step 6; sigma's and the right product's,
    /// -3 to 3, at 125; tau's, the multiples of 4 from -12 to 12, at 116.
    #[test]
    fn plan_sets_by_hand() {
        const ORDER: i64 = 512;
        let rotations = |r: &[i64]| galois_elements_from_rotations(r.iter().copied(), ORDER);

        let plan = BlasPlan::new(1);
        assert_eq!(plan.log_d(), 1);
        assert_eq!(plan.transpose_galois_elements(ORDER), rotations(&[1, 127]));
        assert_eq!(plan.matmul_pt_right_galois_elements(ORDER), rotations(&[1, 127]));
        assert_eq!(plan.matmul_pt_left_galois_elements(ORDER), rotations(&[2, 126]));
        assert_eq!(plan.matmul_galois_elements(ORDER), rotations(&[1, 2, 126, 127]));
        assert_eq!(plan.galois_elements(ORDER), rotations(&[1, 2, 126, 127]));
        assert_eq!(plan.sum_rows_galois_elements(ORDER), rotations(&[2, 126]));
        assert_eq!(plan.sum_cols_galois_elements(ORDER), rotations(&[1, 127]));
        for k in [1, 3] {
            assert_eq!(plan.rotate_rows_galois_elements(ORDER, k), rotations(&[2, 126]), "k={k}");
            assert_eq!(plan.rotate_cols_galois_elements(ORDER, k), rotations(&[1, 127]), "k={k}");
        }
        for k in [0, 2] {
            assert!(plan.rotate_rows_galois_elements(ORDER, k).is_empty(), "k={k}");
            assert!(plan.rotate_cols_galois_elements(ORDER, k).is_empty(), "k={k}");
        }
        // 32 matrix slots.
        for (r, rotation) in [(1, 4), (-1, 124), (33, 4)] {
            assert_eq!(plan.rotate_galois_elements(ORDER, r), rotations(&[rotation]), "r={r}");
        }
        assert!(plan.rotate_galois_elements(ORDER, 32).is_empty());

        let plan = BlasPlan::new(2);
        assert_eq!(plan.transpose_galois_elements(ORDER), rotations(&[2, 3, 5, 6, 114, 120]));
        assert_eq!(plan.matmul_pt_right_galois_elements(ORDER), rotations(&[1, 2, 3, 125]));
        assert_eq!(plan.matmul_pt_left_galois_elements(ORDER), rotations(&[4, 8, 12, 116]));
        // Sigma, tau, then phi^k and psi^k for k = 1, 2, 3.
        let matmul = [1, 2, 3, 125, 4, 8, 12, 116, 1, 125, 2, 126, 3, 127, 4, 116, 8, 120, 12, 124];
        assert_eq!(plan.matmul_galois_elements(ORDER), rotations(&matmul));
        let every: Vec<i64> = matmul.into_iter().chain([2, 3, 5, 6, 114, 120]).collect();
        assert_eq!(plan.galois_elements(ORDER), rotations(&every));
        // The reductions: trees by 4 and 8 (1 and 2), broadcasts by their opposites, within the context's set.
        assert_eq!(plan.sum_rows_galois_elements(ORDER), rotations(&[4, 8, 120, 124]));
        assert_eq!(plan.sum_cols_galois_elements(ORDER), rotations(&[1, 2, 126, 127]));
        for set in [plan.sum_rows_galois_elements(ORDER), plan.sum_cols_galois_elements(ORDER)] {
            assert!(set.iter().all(|p| plan.galois_elements(ORDER).contains(p)));
        }
        for k in 1..8 {
            let k1 = (k % 4) as i64;
            let (rows, cols) = (
                plan.rotate_rows_galois_elements(ORDER, k),
                plan.rotate_cols_galois_elements(ORDER, k),
            );
            if k1 == 0 {
                assert!(rows.is_empty() && cols.is_empty(), "k={k}");
            } else {
                assert_eq!(rows, rotations(&[4 * k1, 128 + 4 * k1 - 16]), "k={k}");
                assert_eq!(cols, rotations(&[k1, 128 + k1 - 4]), "k={k}");
            }
        }
        // 8 matrix slots.
        for (r, rotation) in [(1, 16), (-1, 112), (9, 16)] {
            assert_eq!(plan.rotate_galois_elements(ORDER, r), rotations(&[rotation]), "r={r}");
        }
        assert!(plan.rotate_galois_elements(ORDER, 8).is_empty());
        // Masked: sigma's (tau's) set plus the masks, `Direct`: phi^k by `k`, `k - 4`; psi^k's halves by `4k`, `4k - 16`.
        for algorithm in [BlasPtAlgorithm::JklsStored, BlasPtAlgorithm::JklsGalois] {
            let masked = BlasPlan::new(2).with_pt_algorithm(algorithm);
            assert_eq!(
                masked.matmul_pt_right_galois_elements(ORDER),
                rotations(&[1, 2, 3, 125, 126, 127])
            );
            assert_eq!(
                masked.matmul_pt_left_galois_elements(ORDER),
                rotations(&[4, 8, 12, 116, 120, 124])
            );
            assert_eq!(masked.galois_elements(ORDER), plan.galois_elements(ORDER));
        }

        // A block wider than the slots, or a dimension past the word width: no key, no panic.
        for log_d in [4, 40, usize::MAX] {
            let plan = BlasPlan::new(log_d);
            let sets = [
                plan.galois_elements(ORDER),
                plan.transpose_galois_elements(ORDER),
                plan.rotate_rows_galois_elements(ORDER, 1),
                plan.rotate_cols_galois_elements(ORDER, 1),
                plan.rotate_galois_elements(ORDER, 1),
                plan.sum_rows_galois_elements(ORDER),
                plan.sum_cols_galois_elements(ORDER),
                plan.matmul_galois_elements(ORDER),
                plan.matmul_pt_right_galois_elements(ORDER),
                plan.matmul_pt_left_galois_elements(ORDER),
            ];
            assert!(sets.iter().all(Vec::is_empty), "log_d={log_d}");
        }
    }
}
