//! Linear maps and conversions of a replicated slot layout, the
//! small-batch scheme of [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)
//! (see `docs/guides/replicated.md`).
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    Base2K, Degree, Diagonals, GLWELayout, LinearTransformation, LinearTransformationLayout, LinearTransformationPlan,
    LinearTransformationPrepared, LinearTransformationStrategy, Rank, TorusPrecision, optimal_bsgs_giant_step,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, CKKSLayout, CKKSModuleInfos, CoeffsMeta, ReplicatedLayout, SlotLayout,
    api::{CKKSEncodingOps, CKKSEncodingScalar, CKKSInnerSumOps, CKKSLinearTransformationOps},
    encoding::linear_transformation::{
        ckks_linear_transformation_encode_from_diagonals, ckks_linear_transformation_encode_tmp_bytes, period_degree,
    },
    layouts::{
        CKKSLinearTransformationAlloc, CKKSPlaintextOwned, ComplexDiagonals,
        dft::{DftScalar, charge_factor_keys, charge_key},
    },
};

/// Diagonals of the conversions between a CKKS ciphertext of period `k` and
/// a [`ReplicatedLayout`] of `k` values, each one linear transform.
///
/// With `n = kR` slots and `h = S/k`, `S` the ring's slots (`N/2` on the
/// standard ring, `N` on the conjugate-invariant ring):
///
/// - `to`: `k` diagonals writing value `z_(t/R mod k)` on the data slots `R | t`
///   and zero on the guards, quantized with shaping;
/// - `from` (when `R | k`): the gather diagonals, of weight `R`, reading only
///   data slots so that the normalized trace onto period `k` returns the
///   average of the `h/R` copies of each value, followed by the mean of the
///   `h` rotations by multiples of `k`.
///
/// [`Self::new_factored`] builds each map as the move between consecutive
/// slots and the data slots in `placement_factors` factors of about
/// `k^(1/placement_factors)` 0/1 mask diagonals, and `from` needs no `R | k`.
/// The last factor of `to` is quantized with shaping at `log_delta_diag`
/// bits, its others plainly at `inner_log_delta_diag` bits, enough for their
/// error to stay under the working precision. Every factor of `from` takes
/// `log_delta_diag` bits.
///
/// Both conversions compute `extra_bits` bits under the scale and consume
/// their diagonals' bits of budget. `log_delta` is the scale of the
/// replicated side.
pub struct ReplicatedConversion<BE: Backend, Rg: Ring = Standard> {
    /// The degree of the ciphertexts the conversions take.
    pub(crate) n: Degree,
    pub(crate) layout: ReplicatedLayout,
    pub(crate) log_delta: usize,
    pub(crate) extra_bits: usize,
    pub(crate) to: ReplicatedChain<BE, Rg>,
    pub(crate) from: Option<ReplicatedChain<BE, Rg>>,
    /// Levels of the trace of the return onto period `k`: `log2(S/k)`.
    pub(crate) trace_levels: usize,
}

/// A linear map on the `n = kR` slots of a replicated layout's period, as a
/// chain of prepared factors `y = Σ_o D_o ⊙ rot_o(x)` in evaluation order.
pub(crate) struct ReplicatedChain<BE: Backend, Rg: Ring> {
    /// Each factor with its schedule, kept so that a query builds none.
    pub(crate) factors: Vec<(LinearTransformationPrepared<BE, Rg>, LinearTransformationPlan)>,
    /// The bits of budget the factors consume.
    pub(crate) consumed: usize,
    /// Each key's Galois element, ascending, with the bits below the chain's input of its widest use, the trace's included.
    pub(crate) keys: Vec<(i64, usize)>,
}

/// One factor of a replicated map before encoding, of the schedule [`factor_layout`] gives it: its giant step fitted to
/// its offsets, or the power of two the dense maps of consecutive offsets share, and with it their keys.
pub(crate) struct ReplicatedFactor<F> {
    pub(crate) diagonals: ComplexDiagonals<F>,
    pub(crate) log_delta_diag: usize,
    pub(crate) shaped: bool,
    pub(crate) fitted_giant_step: bool,
}

impl<F> ReplicatedFactor<F> {
    /// The factors of a factored map from its `diagonals` in evaluation order: plain at `log_delta_diag` bits with their
    /// giant steps fitted, the last quantized with shaping at `shaped_log_delta_diag` bits when given.
    pub(crate) fn factored(
        diagonals: impl IntoIterator<Item = ComplexDiagonals<F>>,
        log_delta_diag: usize,
        shaped_log_delta_diag: Option<usize>,
    ) -> Vec<Self> {
        let mut factors: Vec<Self> = diagonals
            .into_iter()
            .map(|diagonals| Self {
                diagonals,
                log_delta_diag,
                shaped: false,
                fitted_giant_step: true,
            })
            .collect();
        if let (Some(last), Some(bits)) = (factors.last_mut(), shaped_log_delta_diag) {
            (last.log_delta_diag, last.shaped) = (bits, true);
        }
        factors
    }
}

/// Bits a diagonal holds above its scale: the largest weight, a sign and a carry.
const MAX_DIAGONAL_HEADROOM: usize = usize::BITS as usize + 2;

impl<BE: Backend, Rg: Ring> ReplicatedConversion<BE, Rg> {
    /// Scratch of [`Self::new`] and [`Self::new_factored`], for diagonals of at
    /// most `log_delta_diag` bits: for [`Self::new_factored`], the larger of its
    /// two diagonal scales. It holds the encoding of one diagonal and its
    /// preparation, at the widest diagonal a weight can need.
    ///
    /// # Errors
    /// When `layout` needs more slots than the module has.
    pub fn new_tmp_bytes<F: CKKSEncodingScalar>(
        module: &Module<BE>,
        layout: &ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
    ) -> Result<usize>
    where
        Module<BE>: ReplicatedConversionModule<BE, F, Rg>,
    {
        let (_, n) = check_period_at::<BE, Rg>(module, module.n().into(), layout)?;
        let diagonal = GLWELayout {
            n: period_degree::<BE, _, Rg>(module, n),
            base2k,
            k: TorusPrecision((log_delta_diag + MAX_DIAGONAL_HEADROOM) as u32),
            rank: Rank(0),
        };
        Ok(ckks_linear_transformation_encode_tmp_bytes::<BE, _, F, Rg>(module, n)
            .max(module.ckks_linear_transformation_rhs_prepare_tmp_bytes(&diagonal)))
    }

    /// Encodes and prepares the conversion maps on `module` for ciphertexts of degree `n`, within the caller's
    /// `scratch` of [`Self::new_tmp_bytes`].
    ///
    /// # Errors
    /// When `n` exceeds the degree of `module` or `layout` needs more slots than the ring of degree `n` has.
    #[expect(clippy::too_many_arguments, reason = "the maps, their degree, layout, scales and arena")]
    pub fn new<F: CKKSEncodingScalar>(
        module: &Module<BE>,
        n: Degree,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        extra_bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: ReplicatedConversionModule<BE, F, Rg>,
    {
        let (k, _) = check_period_at::<BE, Rg>(module, n, &layout)?;
        let r = 1usize << layout.log_r();
        let select = |s: usize, j: usize| {
            let read = s.is_multiple_of(r) && (s / r) % k == j;
            (if read { F::one() } else { F::zero() }, F::zero())
        };
        let to = vec![period_factor(layout, k, log_delta_diag, select)?];
        let identity = |c: usize, j: usize| (if c == j { F::one() } else { F::zero() }, F::zero());
        let from = (r <= k)
            .then(|| gather_factor(layout, log_delta_diag, identity).map(|gather| vec![gather]))
            .transpose()?;
        Self::from_factors(module, n, layout, base2k, log_delta, extra_bits, to, from, scratch)
    }

    /// [`Self::new`] with each map factored into the move between consecutive
    /// slots and the data slots, in `placement_factors` factors of about
    /// `k^(1/placement_factors)` mask diagonals, each costing about the square
    /// root of its diagonals in key switches, for the factors' bits.
    #[expect(clippy::too_many_arguments, reason = "the maps, their degree, layout, scales and arena")]
    pub fn new_factored<F: CKKSEncodingScalar>(
        module: &Module<BE>,
        n: Degree,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        inner_log_delta_diag: usize,
        extra_bits: usize,
        placement_factors: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: ReplicatedConversionModule<BE, F, Rg>,
    {
        let (k, _) = check_period_at::<BE, Rg>(module, n, &layout)?;
        ensure!(
            (1..=layout.log_k()).contains(&placement_factors),
            "the placement takes between one and log2(k) = {} factors",
            layout.log_k()
        );
        let r = 1usize << layout.log_r();
        let to = placement_spread::<F>(k, r, placement_factors);
        let to = ReplicatedFactor::factored(to, inner_log_delta_diag, Some(log_delta_diag));
        let weight = F::from_usize(r).context("the period is a small integer")?;
        let from = placement_compress::<F>(k, r, placement_factors, weight);
        let from = ReplicatedFactor::factored(from, log_delta_diag, None);
        Self::from_factors(module, n, layout, base2k, log_delta, extra_bits, to, Some(from), scratch)
    }

    /// The conversion of the maps `to` and `from`, chains of factors in evaluation order, encoded and prepared on
    /// `module` for ciphertexts of degree `n`, which the caller has checked to hold `layout`.
    #[expect(clippy::too_many_arguments, reason = "the maps, their degree, layout, scales and arena")]
    pub(crate) fn from_factors<F: CKKSEncodingScalar>(
        module: &Module<BE>,
        n: Degree,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        extra_bits: usize,
        to: Vec<ReplicatedFactor<F>>,
        from: Option<Vec<ReplicatedFactor<F>>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: ReplicatedConversionModule<BE, F, Rg>,
    {
        let order = Rg::CYCLOTOMIC_ORDER_FACTOR * n.as_usize() as i64;
        let trace_levels = (module.ckks_slots_of_degree::<Rg>(n.as_usize()) >> layout.log_k()).trailing_zeros() as usize;
        let trace = module.ckks_inner_sum_galois_elements(n, layout.log_k(), trace_levels);
        // Baby keys switch a factor's input, giant keys its output, trace keys what the factors leave.
        let mut chain = |factors: Vec<ReplicatedFactor<F>>, trace: &[i64]| -> Result<ReplicatedChain<BE, Rg>> {
            let mut offsets = BTreeMap::new();
            let mut consumed = 0;
            let mut prepared = Vec::with_capacity(factors.len());
            for factor in factors {
                let scale = factor.log_delta_diag;
                let lt = encode_factor(module, layout, base2k, factor, scratch)?;
                let plan = lt.index();
                let first = lt.first_diagonal_plaintext().context("a transform has a diagonal")?;
                let mut factor = LinearTransformationPrepared::<BE, Rg>::alloc_prepared_from_index(module, &plan, first);
                module.ckks_linear_transformation_rhs_prepare(&mut factor, &lt, scratch)?;
                charge_factor_keys(&mut offsets, &plan, consumed, scale, order);
                consumed += scale;
                prepared.push((factor, plan));
            }
            for &p in trace {
                charge_key(&mut offsets, p, consumed);
            }
            Ok(ReplicatedChain {
                factors: prepared,
                consumed,
                keys: offsets.into_iter().collect(),
            })
        };
        let to = chain(to, &[])?;
        let from = from.map(|from| chain(from, &trace)).transpose()?;
        Ok(Self {
            n,
            layout,
            log_delta,
            extra_bits,
            to,
            from,
            trace_levels,
        })
    }

    /// Galois elements of the automorphism keys the conversions look up, on ciphertexts of the degree they take.
    pub fn galois_elements(&self) -> Vec<i64> {
        let keys = self.to.keys.iter().chain(self.from.iter().flat_map(|from| &from.keys));
        keys.map(|&(p, _)| p).collect::<BTreeSet<_>>().into_iter().collect()
    }
}

/// Encodes `factor` for a ciphertext of `layout`'s period without preparing it, its shaped diagonals under `layout`.
pub(crate) fn encode_factor<BE: Backend, F: CKKSEncodingScalar, Rg: Ring>(
    module: &Module<BE>,
    layout: ReplicatedLayout,
    base2k: Base2K,
    factor: ReplicatedFactor<F>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<LinearTransformation<CKKSPlaintextOwned<BE, Rg>>>
where
    Module<BE>: CKKSLinearTransformationAlloc<BE> + CKKSEncodingOps<BE, F, Rg>,
{
    let diagonals = &factor.diagonals;
    let strategy = factor_layout(diagonals.indexes(), diagonals.slots(), factor.fitted_giant_step).strategy;
    let coeffs_meta = factor_coeffs_meta(diagonals, factor.log_delta_diag, factor.shaped.then_some(layout));
    ckks_linear_transformation_encode_from_diagonals(module, base2k, coeffs_meta, diagonals, strategy, false, scratch)
}

/// The metadata of a factor of `diagonals` at scale `2^log_delta`: a budget of the bits of its largest weight, a sign and
/// a carry, and the shaping of `shaping` when given.
pub(crate) fn factor_coeffs_meta<F: DftScalar>(
    diagonals: &ComplexDiagonals<F>,
    log_delta: usize,
    shaping: Option<ReplicatedLayout>,
) -> CoeffsMeta {
    let values = |o: i64| {
        let (re, im) = (diagonals.re.get(o), diagonals.im.get(o));
        re.into_iter().chain(im).flatten().copied()
    };
    let largest = diagonals
        .indexes()
        .into_iter()
        .flat_map(values)
        .map(|v| v.abs())
        .fold(F::zero(), F::max);
    let headroom = (usize::BITS - (largest.to_f64().unwrap_or(1.0).ceil() as usize).max(1).leading_zeros()) as usize + 2;
    debug_assert!(headroom <= MAX_DIAGONAL_HEADROOM);
    let mut coeffs_meta = CoeffsMeta::from_delta_budget(log_delta, headroom);
    if let Some(layout) = shaping {
        coeffs_meta.meta.slot_layout = SlotLayout::Replicated(layout);
    }
    coeffs_meta
}

/// The schedule of a factor of the diagonals at `indexes`, in any order, on `slots` slots: direct up to two, else BSGS,
/// of the giant step fitted to the indexes when `fitted`, or the power of two whose square covers their count.
pub(crate) fn factor_layout(indexes: impl IntoIterator<Item = i64>, slots: usize, fitted: bool) -> LinearTransformationLayout {
    let indexes: Vec<i64> = indexes.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
    let step = || match fitted {
        true => optimal_bsgs_giant_step(indexes.iter().copied(), slots),
        false => 1 << indexes.len().next_power_of_two().ilog2().div_ceil(2),
    };
    let strategy = match indexes.len() {
        ..=2 => LinearTransformationStrategy::Direct,
        _ => LinearTransformationStrategy::Bsgs { giant_step: step() },
    };
    LinearTransformationLayout {
        indexes,
        slots,
        strategy,
    }
}

/// The shaped factor of the map `y_s = Σ_(j<m) a(s, j)·x_j` from slots of period `m`, a divisor of the `n = kR` slots
/// of `layout`'s period, to those slots: `D_o[s] = a(s, (s + o) mod m)` for `0 <= o < m`.
pub(crate) fn period_factor<F: DftScalar, A: Fn(usize, usize) -> (F, F)>(
    layout: ReplicatedLayout,
    m: usize,
    log_delta_diag: usize,
    a: A,
) -> Result<ReplicatedFactor<F>> {
    let n = 1usize << (layout.log_k() + layout.log_r());
    ensure!(n.is_multiple_of(m), "the input period {m} must divide the {n} slots");
    let weights = (0..m).map(|o| (o, (0..n).map(|s| a(s, (s + o) % m)).collect())).collect();
    Ok(ReplicatedFactor {
        diagonals: diagonals_from_weights(n, weights),
        log_delta_diag,
        shaped: true,
        fitted_giant_step: false,
    })
}

/// The plain factor of the gather `y` reading only the data slots, such that the normalized
/// trace onto period `k`, which averages the `R` positions `c + k·i` of
/// fiber `c`, returns `(Az)_c = Σ_j a(c, j)·z_j` for the data values `z`.
/// Offset `o` of [`gather_offsets`] at `c + k·i` reads data value
/// `m = ⌊c/R⌋ + (k/R)·i + u mod k`, weighted `R·a(c, m)`. Needs `R | k`.
pub(crate) fn gather_factor<F: DftScalar, A: Fn(usize, usize) -> (F, F)>(
    layout: ReplicatedLayout,
    log_delta_diag: usize,
    a: A,
) -> Result<ReplicatedFactor<F>> {
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let n = k * r;
    ensure!(r <= k, "a gather needs the period {r} to divide the batch size {k}");
    let weight = F::from_usize(r).unwrap();
    let mut weights = BTreeMap::new();
    for s in 0..n {
        let (c, i) = (s % k, s / k);
        for (u, o) in gather_offsets(k, r, c) {
            let (re, im) = a(c, (c / r + (k / r) * i + u) % k);
            weights.entry(o).or_insert_with(|| vec![(F::zero(), F::zero()); n])[s] = (weight * re, weight * im);
        }
    }
    Ok(ReplicatedFactor {
        diagonals: diagonals_from_weights(n, weights),
        log_delta_diag,
        shaped: false,
        fitted_giant_step: false,
    })
}

/// The offsets `o = R·u − (c mod R)`, modulo the `kR` slots, at which the gather reads fiber `c`, each with its `u < k/R`.
pub(crate) fn gather_offsets(k: usize, r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..k / r).map(move |u| (u, (r * u + k * r - c % r) % (k * r)))
}

/// What building a [`ReplicatedConversion`] on the ring `Rg` needs of a module, blanket-implemented for every such module.
pub trait ReplicatedConversionModule<BE: Backend, F: CKKSEncodingScalar, Rg: Ring = Standard>:
    CKKSLinearTransformationAlloc<BE>
    + CKKSEncodingOps<BE, F, Rg>
    + CKKSLinearTransformationOps<BE, Rg>
    + CKKSInnerSumOps<BE, Rg>
    + CnvPVecAlloc<BE>
{
}

impl<BE: Backend, F: CKKSEncodingScalar, Rg: Ring, M> ReplicatedConversionModule<BE, F, Rg> for M where
    M: CKKSLinearTransformationAlloc<BE>
        + CKKSEncodingOps<BE, F, Rg>
        + CKKSLinearTransformationOps<BE, Rg>
        + CKKSInnerSumOps<BE, Rg>
        + CnvPVecAlloc<BE>
{
}

/// The layout of `a` at `k` bits, a working copy of `a` that computes under its scale.
pub(crate) fn layout_at<A: CKKSCtBounds>(a: &A, k: usize) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            k: k.into(),
            ..a.glwe_layout()
        },
        meta: a.meta(),
    }
}

/// `(k, kR)` of a layout that fits the slots of degree `n` of the ring `Rg`, at most the module's degree.
pub(crate) fn check_period_at<BE: Backend, Rg: Ring>(
    module: &Module<BE>,
    n: Degree,
    layout: &ReplicatedLayout,
) -> Result<(usize, usize)> {
    ensure!(
        n.as_usize() <= module.n() && n.as_usize().is_power_of_two(),
        "the degree {} is not a power of two at most the module's {}",
        n.as_usize(),
        module.n()
    );
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let slots = module.ckks_slots_of_degree::<Rg>(n.as_usize());
    ensure!(
        k * r <= slots,
        "{k} values of period {r} need more than the {slots} slots of degree {}",
        n.as_usize()
    );
    Ok((k, k * r))
}

/// The diagonals of `n` slots given by their full values, by offset.
fn diagonals_from_weights<F: DftScalar>(n: usize, weights: BTreeMap<usize, Vec<(F, F)>>) -> ComplexDiagonals<F> {
    let mut diagonals = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
    for (o, w) in weights {
        let (re, im): (Vec<F>, Vec<F>) = w.into_iter().unzip();
        set_complex(&mut diagonals, o as i64, (re, im));
    }
    diagonals
}

/// Bits of `t` selected by `group`, kept in place.
fn group_bits(t: usize, group: &[usize]) -> usize {
    group.iter().map(|&b| t & (1 << b)).sum()
}

/// The bits of `m` values in `factors` near-equal groups in the order the placement moves them, from the highest or, when
/// `compress`, from the lowest, each with the values `v` its bits take and the index, modulo the `m·R` slots, of the
/// diagonal moving them: `-(R - 1)·v`, or `(R - 1)·v` when `compress`.
fn placement_moves(m: usize, r: usize, factors: usize, compress: bool) -> Vec<(BTreeMap<usize, i64>, Vec<usize>)> {
    let log_m = m.trailing_zeros() as usize;
    let mut bits = (0..log_m).rev();
    let mut groups: Vec<Vec<usize>> = split_depth(log_m, factors)
        .into_iter()
        .map(|size| bits.by_ref().take(size).collect())
        .collect();
    if compress {
        groups.reverse();
    }
    let offset = |v: usize| (((r - 1) * v) as i64 * if compress { 1 } else { -1 }).rem_euclid((m * r) as i64);
    let moves = |group: &[usize]| (0..m).map(|t| group_bits(t, group)).map(|v| (v, offset(v))).collect();
    groups.into_iter().map(|group| (moves(&group), group)).collect()
}

/// Where value `t` sits once the bits of `moved` are strided: `t + (R - 1)·t_moved`.
fn placement_position(t: usize, moved: &[usize], r: usize) -> usize {
    t + (r - 1) * group_bits(t, moved)
}

/// A diagonal of `n` slots, `value` at the positions of `on` and zero elsewhere.
fn mask_diagonal<F: DftScalar>(n: usize, on: impl IntoIterator<Item = usize>, value: F) -> (Vec<F>, Vec<F>) {
    let mut re = vec![F::zero(); n];
    for s in on {
        re[s] = value;
    }
    (re, vec![F::zero()])
}

pub(crate) fn set_complex<F>(diagonals: &mut ComplexDiagonals<F>, index: i64, (re, im): (Vec<F>, Vec<F>)) {
    diagonals.re.set(index, re);
    diagonals.im.set(index, im);
}

/// The placement of `m` values from slots of period `k` onto the data slots of
/// stride `R`, `y_(R·t) = x_(t mod k)` for `t < m` and zero elsewhere, over the
/// `n = m·R` slots. The `log2(m)` bits of `t` are split into `factors`
/// near-equal groups, and one factor per group, from the highest, moves the
/// values whose group bits are `v` by `(R - 1)·v`. Taking the high bits first
/// keeps the positions `t + (R - 1)·t_moved` distinct, so every factor is a
/// set of 0/1 masks, `2^g` diagonals for a group of `g` bits.
pub(crate) fn placement_spread<F: DftScalar>(m: usize, r: usize, factors: usize) -> Vec<ComplexDiagonals<F>> {
    let n = m * r;
    let mut moved = Vec::new();
    let mut out = Vec::new();
    for (moves, group) in placement_moves(m, r, factors, false) {
        let next: Vec<usize> = moved.iter().chain(&group).copied().collect();
        let mut factor = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
        for (v, offset) in moves {
            let on = (0..m)
                .filter(|&t| group_bits(t, &group) == v)
                .map(|t| placement_position(t, &next, r));
            set_complex(&mut factor, offset, mask_diagonal(n, on, F::one()));
        }
        out.push(factor);
        moved = next;
    }
    out
}

/// The inverse of [`placement_spread`] for `m = k`: `y_t = weight·x_(R·t)` for
/// `t < k` and zero elsewhere, over the `n = k·R` slots, reading only the data
/// slots `R | s`. The groups are undone from the lowest, the last factor
/// carrying `weight`.
pub(crate) fn placement_compress<F: DftScalar>(k: usize, r: usize, factors: usize, weight: F) -> Vec<ComplexDiagonals<F>> {
    let n = k * r;
    let groups = placement_moves(k, r, factors, true);
    let mut moved: Vec<usize> = groups.iter().flat_map(|(_, group)| group).copied().collect();
    let mut out = Vec::new();
    for (i, (moves, group)) in groups.iter().enumerate() {
        moved.retain(|b| !group.contains(b));
        let value = if i + 1 == groups.len() { weight } else { F::one() };
        let mut factor = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
        for (&v, &offset) in moves {
            let on = (0..k)
                .filter(|&t| group_bits(t, group) == v)
                .map(|t| placement_position(t, &moved, r));
            set_complex(&mut factor, offset, mask_diagonal(n, on, value));
        }
        out.push(factor);
    }
    out
}

/// The diagonal indexes of each factor of [`placement_spread`] of `m` values, or of [`placement_compress`] when
/// `compress`, on the `m·R` slots.
pub(crate) fn placement_indexes(m: usize, r: usize, factors: usize, compress: bool) -> Vec<Vec<i64>> {
    let moves = placement_moves(m, r, factors, compress).into_iter();
    moves.map(|(moves, _)| moves.into_values().collect()).collect()
}

/// `log_m` bits in `factors` near-equal groups, the larger first.
pub(crate) fn split_depth(log_m: usize, factors: usize) -> Vec<usize> {
    let factors = factors.clamp(1, log_m.max(1));
    (0..factors)
        .map(|i| log_m / factors + usize::from(i < log_m % factors))
        .collect()
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{Evaluate, LinearTransformationStrategy};

    use super::*;
    use crate::shaping::ShapingUnit;

    fn apply(factors: &[ComplexDiagonals<f64>], x: &[f64]) -> Vec<f64> {
        let mut re = x.to_vec();
        let mut im = vec![0.0; x.len()];
        for factor in factors {
            (re, im) = factor.evaluate((&re, &im), LinearTransformationStrategy::Direct);
        }
        assert!(im.iter().all(|&v| v == 0.0));
        re
    }

    fn pseudo_random(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 2654435761) % 1000) as f64 / 7.0 + 1.0).collect()
    }

    #[test]
    fn placement_spread_writes_the_data_slots() {
        for (k, m, r, factors) in [
            (8, 8, 4, 3),
            (16, 16, 8, 2),
            (64, 64, 16, 2),
            (64, 64, 2, 1),
            (32, 64, 16, 3),
            (16, 32, 32, 5),
        ] {
            let x = pseudo_random(k);
            let input: Vec<f64> = (0..m * r).map(|s| x[s % k]).collect();
            let log_group = (m.trailing_zeros() as usize).div_ceil(factors);
            let factors = placement_spread::<f64>(m, r, factors);
            assert!(factors.iter().all(|f| f.indexes().len() <= 1 << log_group));
            let want: Vec<f64> = (0..m * r).map(|s| if s % r == 0 { x[(s / r) % k] } else { 0.0 }).collect();
            assert_eq!(apply(&factors, &input), want, "k = {k}, m = {m}, R = {r}");
        }
    }

    #[test]
    fn placement_compress_reads_only_the_data_slots() {
        for (k, r, factors) in [(8, 4, 3), (16, 8, 2), (64, 16, 2), (32, 32, 1)] {
            let x = pseudo_random(k);
            // The guards hold garbage the compression must not read.
            let input: Vec<f64> = (0..k * r)
                .map(|s| if s % r == 0 { x[s / r] } else { 1e6 + s as f64 })
                .collect();
            let factors = placement_compress::<f64>(k, r, factors, r as f64);
            let want: Vec<f64> = (0..k * r).map(|s| if s < k { r as f64 * x[s] } else { 0.0 }).collect();
            assert_eq!(apply(&factors, &input), want, "k = {k}, R = {r}");
        }
    }

    #[test]
    fn factors_refuse_a_period_off_the_layout() {
        let reason = |factor: Result<ReplicatedFactor<f64>>| factor.err().unwrap().to_string();
        let zero = |_: usize, _: usize| (0.0, 0.0);
        let period = period_factor(ShapingUnit::U16_40.layout(3).unwrap(), 3, 14, zero);
        assert_eq!(reason(period), "the input period 3 must divide the 64 slots");
        let gather = gather_factor(ShapingUnit::U16_40.layout(2).unwrap(), 30, zero);
        assert_eq!(reason(gather), "a gather needs the period 8 to divide the batch size 4");
    }

    #[test]
    fn split_depth_covers_the_bits() {
        assert_eq!(split_depth(10, 3), [4, 3, 3]);
        assert_eq!(split_depth(10, 1), [10]);
        assert_eq!(split_depth(2, 5), [1, 1]);
    }
}
