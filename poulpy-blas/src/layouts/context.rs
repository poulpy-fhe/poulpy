//! The plan of the circuit, its per-operation key sets, and the per-matrix maps encoded once, at a caller-chosen
//! precision.

use std::collections::BTreeSet;

use num_traits::One;
use poulpy_ckks::{
    CKKSResult, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale,
    api::{
        CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformation,
        LinearTransformationPrepared, optimal_bsgs_giant_step,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        Base2K, Degree, DiagonalArithmetic, Diagonals, GetDegree, LinearTransformationLayout, LinearTransformationStrategy,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf},
    layouts::{Backend, Module, ScratchArena, galois_elements_from_rotations},
};

use super::{
    BlasGridInfos,
    ciphertext::mat_slots_in,
    encoding::{Encoded, Scheduled, encode_map_into, prepare_maps_into, real_map},
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
/// are the same under both.
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

    /// Galois elements `blas_rotate` by `r` matrix slots looks up; none when the matrix slot count divides `r`.
    pub fn rotate_galois_elements(&self, cyclotomic_order: i64, r: i64) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let mat_slots = (slots >> (2 * self.log_d)) as i64;
        galois_elements_from_rotations([r.rem_euclid(mat_slots) << (2 * self.log_d)], cyclotomic_order)
    }

    /// Galois elements `blas_matmul`, `blas_matmul_prepare_left` and `blas_matmul_prepare_right` look up, and
    /// `blas_matmul_prepared` for a hot operand, whose terms it evaluates; of two cold operands it looks up none.
    pub fn matmul_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.matmul_maps(slots))
    }

    /// Galois elements `blas_matmul_pt_right` looks up, for public matrices of any values, grid shape and count: the right
    /// product stores a diagonal at every offset `l - c`, whatever the entries, and sigma and the JKLS block maps sit at
    /// those offsets too, so the set is the same under either [`BlasPtAlgorithm`].
    pub fn matmul_pt_right_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            vec![(maps::right_mul(slots, self.log_d, 1, |_, _, _| 1.0), false)]
        })
    }

    /// Galois elements `blas_matmul_pt_left` looks up, for public matrices of any values, grid shape and count: the left
    /// product stores a diagonal at every offset `(l - r) d`, whatever the entries. These are tau's offsets, scheduled
    /// alike, and the JKLS block maps sit at them too, so the set is tau's under either [`BlasPtAlgorithm`].
    pub fn matmul_pt_left_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            vec![(maps::left_mul(slots, self.log_d, 1, |_, _, _| 1.0), false)]
        })
    }

    /// The slot count of a ring of cyclotomic order `cyclotomic_order`, if a block fits it.
    fn slots(&self, cyclotomic_order: i64) -> Option<usize> {
        let slots = usize::try_from(cyclotomic_order / 4).ok().filter(|&slots| slots > 0)?;
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
}

/// The host diagonals of a shift map at `k`, on `slots` slots and blocks of `2^log_d`.
type ShiftMap<F> = fn(usize, usize, usize) -> Diagonals<F>;

/// One map of a context: one of its four single maps, or the map at `k` of one of its seven families of shifts.
#[derive(Clone, Copy)]
pub(crate) enum MapId {
    Single(usize),
    Shift(usize, usize),
}

const SIGMA: usize = 0;
const TAU: usize = 1;
const TAU_REP: usize = 2;
const TRANSPOSE: usize = 3;
const SINGLES: usize = 4;

const ROTATE_ROWS: usize = 0;
const ROTATE_ROWS_REP: usize = 1;
const ROTATE_COLS: usize = 2;
const ROWS_UPPER: usize = 3;
const ROWS_LOWER: usize = 4;
const COLS_LEFT: usize = 5;
const COLS_RIGHT: usize = 6;
const FAMILIES: usize = 7;

impl MapId {
    /// Every map of a context of `2^log_d x 2^log_d` blocks, in the order [`ContextMaps`] holds them.
    fn all(log_d: usize) -> impl Iterator<Item = Self> {
        let shifts = (0..FAMILIES).flat_map(move |family| (1..1usize << log_d).map(move |k| Self::Shift(family, k)));
        (0..SINGLES).map(Self::Single).chain(shifts)
    }
}

/// The maps of a context, in the order encoding and preparation walk them: sigma, tau, tau on a replicated input and
/// the transpose, then psi^k, psi^k on a replicated input, phi^k, the halves of psi^k and those of phi^k, each family
/// for `k = 1..d`.
pub(crate) struct ContextMaps<T> {
    log_d: usize,
    slots: usize,
    singles: [T; SINGLES],
    shifts: [Vec<T>; FAMILIES],
}

/// The maps in walking order, clonable so that preparation can size its scratch before it walks them.
pub(crate) type MapsIter<'a, T> = std::iter::Chain<std::slice::Iter<'a, T>, std::iter::Flatten<std::slice::Iter<'a, Vec<T>>>>;

impl<T> ContextMaps<T> {
    /// `make(id)` for every map of a context of `2^log_d x 2^log_d` blocks, in walking order.
    pub(crate) fn build(log_d: usize, slots: usize, mut make: impl FnMut(MapId) -> T) -> Self {
        let singles = std::array::from_fn(|i| make(MapId::Single(i)));
        let shifts = std::array::from_fn(|family| (1..1usize << log_d).map(|k| make(MapId::Shift(family, k))).collect());
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
/// replicated input, scheduled at tau's giant step, and psi^k on a replicated input, at no scale.
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
        MapId::Single(_) => scheduled(maps::transpose(slots, log_d), false, meta),
        MapId::Shift(family, k) => {
            let families: [ShiftMap<F>; FAMILIES] = [
                maps::rotate_rows,
                maps::rotate_rows_rep,
                maps::rotate_cols,
                maps::rows_upper,
                maps::rows_lower,
                maps::cols_left,
                maps::cols_right,
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
    /// The block dimension and the schedule of every map.
    fn plan(&self) -> BlasPlan;

    /// `log2` of the block dimension.
    fn log_d(&self) -> usize {
        self.plan().log_d()
    }

    /// Radix of the maps' diagonals.
    fn base2k(&self) -> Base2K;

    /// Torus width and metadata the maps are encoded at.
    fn coeffs_meta(&self) -> CoeffsMeta;

    /// Degree of the ring the maps are encoded on.
    fn n(&self) -> Degree;

    /// Slots of the ring the maps are encoded on.
    fn slots(&self) -> usize;

    /// A plain-data snapshot of these infos.
    fn blas_context_layout(&self) -> BlasContextLayout {
        BlasContextLayout {
            plan: self.plan(),
            base2k: self.base2k(),
            coeffs_meta: self.coeffs_meta(),
            n: self.n(),
            slots: self.slots(),
        }
    }
}

impl<T: BlasContextInfos + ?Sized> BlasContextInfos for &T {
    fn plan(&self) -> BlasPlan {
        (**self).plan()
    }

    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn coeffs_meta(&self) -> CoeffsMeta {
        (**self).coeffs_meta()
    }

    fn n(&self) -> Degree {
        (**self).n()
    }

    fn slots(&self) -> usize {
        (**self).slots()
    }
}

impl<T: BlasContextInfos + ?Sized> BlasContextInfos for &mut T {
    fn plan(&self) -> BlasPlan {
        (**self).plan()
    }

    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn coeffs_meta(&self) -> CoeffsMeta {
        (**self).coeffs_meta()
    }

    fn n(&self) -> Degree {
        (**self).n()
    }

    fn slots(&self) -> usize {
        (**self).slots()
    }
}

impl BlasContextInfos for BlasContextLayout {
    fn plan(&self) -> BlasPlan {
        self.plan
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn coeffs_meta(&self) -> CoeffsMeta {
        self.coeffs_meta
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn slots(&self) -> usize {
        self.slots
    }
}

/// The JKLS maps of one matrix dimension, their one-diagonal halves and their variants on a replicated input, encoded
/// and not yet prepared.
pub struct BlasContextUnprepared<BE: Backend> {
    layout: BlasContextLayout,
    pub(crate) maps: ContextMaps<Encoded<BE>>,
}

/// The prepared JKLS maps of one matrix dimension, their one-diagonal halves and their variants on a replicated input.
pub struct BlasContext<BE: Backend> {
    layout: BlasContextLayout,
    pub(crate) maps: ContextMaps<LinearTransformationPrepared<BE>>,
}

impl<BE: Backend> BlasContextInfos for BlasContextUnprepared<BE> {
    fn plan(&self) -> BlasPlan {
        self.layout.plan
    }

    fn base2k(&self) -> Base2K {
        self.layout.base2k
    }

    fn coeffs_meta(&self) -> CoeffsMeta {
        self.layout.coeffs_meta
    }

    fn n(&self) -> Degree {
        self.layout.n
    }

    fn slots(&self) -> usize {
        self.layout.slots
    }
}

impl<BE: Backend> BlasContextInfos for BlasContext<BE> {
    fn plan(&self) -> BlasPlan {
        self.layout.plan
    }

    fn base2k(&self) -> Base2K {
        self.layout.base2k
    }

    fn coeffs_meta(&self) -> CoeffsMeta {
        self.layout.coeffs_meta
    }

    fn n(&self) -> Degree {
        self.layout.n
    }

    fn slots(&self) -> usize {
        self.layout.slots
    }
}

impl<BE: Backend> BlasContextUnprepared<BE> {
    /// Every encoded map, in the order encoding and preparation walk them: sigma, tau, tau on a replicated input and
    /// the transpose, then psi^k, psi^k on a replicated input, phi^k and the halves of psi^k and of phi^k, each family
    /// for `k = 1..d`.
    pub fn maps(&self) -> impl Iterator<Item = &LinearTransformation<CKKSPlaintextOwned<BE>>> {
        self.maps.iter()
    }
}

impl<BE: Backend> BlasContext<BE> {
    /// Every prepared map, in the order of [`BlasContextUnprepared::maps`].
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
}

/// Panics unless the blocks of `layout`, at least `2 x 2`, fit its slots, and its maps have a torus width and a radix in
/// `1..=63`.
pub(crate) fn assert_fits(layout: &BlasContextLayout) {
    let log_d = layout.plan.log_d();
    assert!(
        log_d >= 1 && layout.slots > 0 && log_d <= layout.slots.ilog2() as usize / 2,
        "the context's blocks do not fit its slots"
    );
    assert!(layout.coeffs_meta.k.as_usize() > 0, "the context's maps have no torus width");
    assert!(
        (1..=63).contains(&layout.base2k.as_usize()),
        "the context's radix is outside 1..=63"
    );
}

/// Panics unless `layout` fits, as [`assert_fits`] states, and is of `module`'s ring.
fn assert_fits_module(module: &(impl GetDegree + CKKSModuleInfos), layout: &BlasContextLayout) {
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

/// [`super::BlasModuleAlloc::blas_context_alloc_from_infos`].
pub(crate) fn alloc<BE, M>(module: &M, layout: &BlasContextLayout) -> BlasContextUnprepared<BE>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GetDegree,
{
    assert_fits_module(module, layout);
    let maps = ContextMaps::build(layout.plan.log_d(), layout.slots, |id| {
        context_map::<f64>(layout, id).layout().alloc::<BE, M>(module, layout.base2k)
    });
    BlasContextUnprepared { layout: *layout, maps }
}

/// [`super::BlasBytesOf::blas_context_bytes_of_from_infos`].
pub(crate) fn bytes_of<BE, M>(module: &M, layout: &BlasContextLayout) -> usize
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GLWEBytesOf<BE> + GetDegree,
{
    assert_fits_module(module, layout);
    MapId::all(layout.plan.log_d())
        .map(|id| {
            context_map::<f64>(layout, id)
                .layout()
                .bytes_of::<BE, M>(module, layout.base2k)
        })
        .sum()
}

/// [`super::BlasEncodingOps::blas_context_encode_into`].
pub(crate) fn encode_into<BE, M, F>(
    module: &M,
    res: &mut BlasContextUnprepared<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GetDegree + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let layout = res.layout;
    assert_fits_module(module, &layout);
    for (id, lt) in MapId::all(layout.plan.log_d()).zip(res.maps.iter_mut()) {
        encode_map_into(module, lt, context_map::<F>(&layout, id), scratch)?;
    }
    Ok(())
}

/// Allocation, sizing and preparation of prepared contexts.
pub trait BlasContextPreparedFactory<BE: Backend>:
    CKKSModuleAlloc<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CnvPVecBytesOf + GetDegree + Sized
{
    /// A prepared context of `plan` on this module's ring, which [`Self::blas_context_prepare`] fills.
    ///
    /// # Panics
    ///
    /// As [`super::BlasModuleAlloc::blas_context_alloc`].
    fn blas_context_prepared_alloc(&self, plan: BlasPlan, base2k: Base2K, coeffs_meta: CoeffsMeta) -> BlasContext<BE> {
        self.blas_context_prepared_alloc_from_infos(&module_layout(self, plan, base2k, coeffs_meta))
    }

    /// A prepared context of `infos`, which must be of this module's ring.
    ///
    /// # Panics
    ///
    /// As [`super::BlasModuleAlloc::blas_context_alloc_from_infos`].
    fn blas_context_prepared_alloc_from_infos<A: BlasContextInfos>(&self, infos: &A) -> BlasContext<BE> {
        let layout = infos.blas_context_layout();
        assert_fits_module(self, &layout);
        let maps = ContextMaps::build(layout.plan.log_d(), layout.slots, |id| {
            context_map::<f64>(&layout, id)
                .layout()
                .alloc_prepared::<BE, Self>(self, layout.base2k)
        });
        BlasContext { layout, maps }
    }

    /// Bytes of [`Self::blas_context_prepared_alloc`].
    fn blas_context_prepared_bytes_of(&self, plan: BlasPlan, base2k: Base2K, coeffs_meta: CoeffsMeta) -> usize {
        self.blas_context_prepared_bytes_of_from_infos(&module_layout(self, plan, base2k, coeffs_meta))
    }

    /// Bytes of [`Self::blas_context_prepared_alloc_from_infos`].
    fn blas_context_prepared_bytes_of_from_infos<A: BlasContextInfos>(&self, infos: &A) -> usize {
        let layout = infos.blas_context_layout();
        assert_fits_module(self, &layout);
        MapId::all(layout.plan.log_d())
            .map(|id| {
                context_map::<f64>(&layout, id)
                    .layout()
                    .prepared_bytes_of::<BE, Self>(self, layout.base2k)
            })
            .sum()
    }

    /// Scratch bytes [`Self::blas_context_prepare`] needs for a context of `infos`.
    fn blas_context_prepare_tmp_bytes<A: BlasContextInfos>(&self, infos: &A) -> usize {
        let layout = infos.blas_context_layout();
        assert_fits_module(self, &layout);
        MapId::all(layout.plan.log_d())
            .map(|id| {
                context_map::<f64>(&layout, id)
                    .layout()
                    .prepare_tmp_bytes::<BE, Self>(self, layout.base2k)
            })
            .max()
            .unwrap_or(0)
    }

    /// Prepares every map of `other` into `res`, in place: preparing again after a new encoding refills `res` with no
    /// allocation.
    ///
    /// # Panics
    ///
    /// Panics if `res` and `other` were allocated from different layouts, or if `scratch` holds less than
    /// [`Self::blas_context_prepare_tmp_bytes`].
    fn blas_context_prepare(
        &self,
        res: &mut BlasContext<BE>,
        other: &BlasContextUnprepared<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        assert!(
            res.layout == other.layout,
            "the prepared context was allocated for another layout"
        );
        prepare_maps_into(self, "blas_context_prepare", res.maps.iter_mut(), other.maps.iter(), scratch);
    }
}

impl<BE: Backend> BlasContextPreparedFactory<BE> for Module<BE> where
    Module<BE>: CKKSModuleAlloc<BE> + CKKSLinearTransformationOps<BE>
{
}

/// Galois elements `blas_sum` needs on a grid of layout `infos`: at its recorded batch, or for a layout without one
/// the batch its label proves; none when one block does not fit the slots.
pub fn sum_galois_elements<BE: Backend>(module: &Module<BE>, infos: &impl BlasGridInfos) -> Vec<i64> {
    let mat_slots = infos
        .mat_slots()
        .unwrap_or_else(|| mat_slots_in(module.ckks_max_slots(), infos.log_d(), infos.block_layout().log_sparsity()));
    sum_rotations(infos.log_d(), mat_slots)
        .map(|rotation| module.ckks_galois_element(rotation))
        .collect()
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

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::layouts::{BlasBytesOf, BlasLayout, BlasModuleAlloc, replicated};
    use poulpy_ckks::{CKKSInfos, SetGLWEPlaintextInfos};
    use poulpy_hal::layouts::{HostBytesBackend, Standard};

    /// The bytes every diagonal of `ctx` holds.
    fn held(ctx: &BlasContextUnprepared<HostBytesBackend>) -> usize {
        ctx.maps()
            .flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| &gs.diagonals))
            .map(|d| <HostBytesBackend as Backend>::len_bytes(d.plaintext.data().data()))
            .sum()
    }

    #[test]
    fn contexts_allocate_every_map_of_their_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let layout = module_layout(&module, BlasPlan::new(2), Base2K(19), CoeffsMeta::from_delta_budget(20, 19));
        let ctx = module.blas_context_alloc_from_infos(&layout);
        // Sigma, tau, tau on a replicated input and the transpose, then seven families of shifts for k = 1, 2, 3.
        assert_eq!(ctx.maps().count(), 4 + 7 * 3);
        assert_eq!(module.blas_context_bytes_of_from_infos(&layout), held(&ctx));
        assert_eq!(
            module.blas_context_bytes_of(BlasPlan::new(2), Base2K(19), layout.coeffs_meta),
            held(&ctx)
        );
        assert_eq!(ctx.blas_context_layout(), layout);
        // psi^k on a replicated input is encoded at no scale, every other map at the layout's.
        let scales: Vec<usize> = ctx
            .maps()
            .map(|lt| lt.first_diagonal_plaintext().unwrap().log_delta())
            .collect();
        assert_eq!(scales.iter().filter(|&&s| s == 0).count(), 3);
        assert_eq!(scales.iter().filter(|&&s| s == 20).count(), 4 + 6 * 3);
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
            BlasContextLayout {
                plan: BlasPlan::new(4),
                ..base
            },
            BlasContextLayout {
                plan: BlasPlan::new(40),
                ..base
            },
            BlasContextLayout {
                coeffs_meta: CoeffsMeta::from_delta_budget(0, 0),
                ..base
            },
            BlasContextLayout {
                base2k: Base2K(0),
                ..base
            },
            BlasContextLayout { n: Degree(512), ..base },
            BlasContextLayout { slots: 64, ..base },
        ] {
            let alloc = catch_unwind(AssertUnwindSafe(|| module.blas_context_alloc_from_infos(&layout)));
            assert!(alloc.is_err(), "{layout:?}");
            let bytes = catch_unwind(AssertUnwindSafe(|| module.blas_context_bytes_of_from_infos(&layout)));
            assert!(bytes.is_err(), "bytes_of {layout:?}");
        }
    }

    #[test]
    fn sum_keys_plan_from_a_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let mut ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        let mut layout = BlasLayout::new(ct.block_layout(), 2, 2, 2);
        // 128 slots hold 8 matrix slots of 4 x 4 blocks: rotations by 16, 32 and 64.
        let want: Vec<i64> = [16, 32, 64].map(|r| module.ckks_galois_element(r)).into();
        assert_eq!(sum_galois_elements(&module, &layout), want);
        assert_eq!(sum_galois_elements(&module, &ct), want);
        layout.block.meta.log_sparsity = 1;
        assert_eq!(sum_galois_elements(&module, &layout), want[..2]);
        // A grid plans from its record, not from its label.
        ct.block_mut(0, 0).set_log_sparsity(1);
        assert_eq!(sum_galois_elements(&module, &ct), want);
        ct.mat_slots = 2;
        assert_eq!(sum_galois_elements(&module, &ct), want[..1]);
        // No block fits, or a shift is past the word width: no key, no panic.
        for log_sparsity in [64, 65] {
            layout.block.meta.log_sparsity = log_sparsity;
            assert!(
                sum_galois_elements(&module, &layout).is_empty(),
                "log_sparsity={log_sparsity}"
            );
        }
        layout.block.meta.log_sparsity = 0;
        for log_d in [4, 40, usize::MAX] {
            layout.log_d = log_d;
            assert!(sum_galois_elements(&module, &layout).is_empty(), "log_d={log_d}");
        }
    }

    /// The replicated-input maps rotate within the plan's product set, tau_rep at tau's giant step, so a client plans
    /// no new key.
    #[test]
    fn rep_maps_rotate_within_the_matmul_set() {
        for (order, log_d) in [(512, 1), (512, 2), (512, 3), (256, 3)] {
            let (plan, slots) = (BlasPlan::new(log_d), (order / 4) as usize);
            let set = plan.matmul_galois_elements(order);
            let mut rep = vec![(
                maps::tau_rep::<f64>(slots, log_d),
                plan.strategy(&maps::tau::<f64>(slots, log_d), false),
            )];
            for k in 1..1 << log_d {
                let map = maps::rotate_rows_rep::<f64>(slots, log_d, k);
                let strategy = plan.strategy(&map, true);
                rep.push((map, strategy));
            }
            for (map, strategy) in rep {
                let layout = LinearTransformationLayout {
                    indexes: map.indexes(),
                    slots,
                    strategy,
                };
                for p in layout.galois_elements(order) {
                    assert!(set.contains(&p), "order={order} log_d={log_d} p={p}");
                }
            }
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

        // A block wider than the slots, or a dimension past the word width: no key, no panic.
        for log_d in [4, 40, usize::MAX] {
            let plan = BlasPlan::new(log_d);
            let sets = [
                plan.galois_elements(ORDER),
                plan.transpose_galois_elements(ORDER),
                plan.rotate_rows_galois_elements(ORDER, 1),
                plan.rotate_cols_galois_elements(ORDER, 1),
                plan.rotate_galois_elements(ORDER, 1),
                plan.matmul_galois_elements(ORDER),
                plan.matmul_pt_right_galois_elements(ORDER),
                plan.matmul_pt_left_galois_elements(ORDER),
            ];
            assert!(sets.iter().all(Vec::is_empty), "log_d={log_d}");
        }
    }
}
