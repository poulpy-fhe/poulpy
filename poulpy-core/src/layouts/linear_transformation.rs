//! Unprepared GLWE linear transformation (BSGS) data and schedule.
//!
//! Holds the unprepared transform (`LinearTransformation`, encoded diagonals
//! bucketed by giant step) and the integer-level BSGS schedule types
//! (`LinearTransformationLayout`, `LinearTransformationPlan`,
//! `LinearTransformationStrategy`) plus their derivation, which is pure integer
//! math (cf. docs/guides/linear_transformation.md).
//!
//! Every automorphism of a transform, a diagonal, a baby step or a giant step, is a Galois
//! element at [`GALOIS_ORDER`](crate::layouts::GALOIS_ORDER) (see [`crate::layouts::galois_parts`]),
//! its rotation taken modulo the slot count of the transform.
//!
//! The *prepared* (convolution-domain) caches live in
//! [`crate::layouts::prepared`]; the HAL-dependent allocators and the
//! prepare/eval reference algorithms live in
//! [`crate::reference::linear_transformation`].

use std::collections::{BTreeMap, BTreeSet};

use poulpy_hal::layouts::MAX_RING_DEGREE;

use crate::layouts::{galois_at, galois_from_parts, galois_is_identity, galois_of_rotation, galois_parts};

// ===================================================================
// Unprepared transform
// ===================================================================

/// One non-zero diagonal of the linear map, attached to a giant step.
///
/// `plaintext` is the pre-rotated diagonal `u~_{g,b} = sigma_g^-1(diag_{g b})`, already encoded
/// as a plaintext polynomial. `baby_galois` is the Galois element `b` of the baby step whose
/// prepared `sigma_b(v)` it multiplies.
pub struct LinearTransformationDiagonal<P> {
    /// Galois element of the baby step.
    pub baby_galois: i64,
    /// Pre-rotated diagonal, encoded as a plaintext polynomial.
    pub plaintext: P,
}

/// A single giant step `g`: its inner sum is mapped by `sigma_g` and added to the output
/// (`galois == 1` is the identity giant step).
pub struct LinearTransformationGiantStep<P> {
    /// Galois element of the giant step.
    pub galois: i64,
    /// The non-zero diagonals contributing to this giant step.
    pub diagonals: Vec<LinearTransformationDiagonal<P>>,
}

/// A linear transformation in baby-step / giant-step form.
///
/// `P` is the encoded-plaintext container.
pub struct LinearTransformation<P> {
    /// Distinct baby-step Galois elements `b`; `baby_galois[0] == 1` (the identity).
    pub baby_galois: Vec<i64>,
    /// The giant steps.
    pub giant_steps: Vec<LinearTransformationGiantStep<P>>,
    /// How the evaluation reaches the baby and giant steps.
    pub rotations: BsgsRotations,
}

impl<P> LinearTransformation<P> {
    /// The distinct baby-step Galois elements (`baby_galois[0] == 1`, the identity); the set
    /// the prepared baby cache must hold. Accessor over the [`baby_galois`](Self::baby_galois)
    /// field for callers that prefer a method.
    pub fn baby_galois(&self) -> &[i64] {
        &self.baby_galois
    }

    /// The first encoded diagonal across all giant steps, or `None` if the
    /// transform is empty. The diagonals share a uniform plaintext shape, so this
    /// is the canonical place callers read `base2k` / `k` / `log_delta` from.
    pub fn first_diagonal_plaintext(&self) -> Option<&P> {
        self.giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| &d.plaintext)
            .next()
    }

    /// Derives the BSGS index schedule implied by this transform's actual
    /// baby and giant steps. Useful for one-shot allocation of the prepared
    /// cache directly from an unprepared transform.
    ///
    /// The output is canonical, giant steps sorted and per-giant baby steps
    /// sorted and de-duplicated in the order of their [`galois_parts`](crate::layouts::galois_parts), so it matches the schedule
    /// [`LinearTransformationLayout::index`] would derive from the same diagonal
    /// set. Only baby steps actually referenced by at least one non-empty
    /// giant step are included; the field `self.baby_galois` may declare more
    /// steps than the transform's data populates, and those extras would
    /// otherwise force the caller to provide automorphism keys that the
    /// schedule does not actually need.
    pub fn index(&self) -> LinearTransformationPlan {
        let mut by_giant: BTreeMap<(bool, i64), BTreeSet<(bool, i64)>> = BTreeMap::new();
        let mut used_babies: Vec<i64> = Vec::new();
        for gs in &self.giant_steps {
            if gs.diagonals.is_empty() {
                continue;
            }
            let babies = by_giant.entry(galois_parts(gs.galois)).or_default();
            for d in &gs.diagonals {
                used_babies.push(d.baby_galois);
                babies.insert(galois_parts(d.baby_galois));
            }
        }
        let baby_galois = self.rotations.baby_closure(used_babies);
        let mut giant_galois = Vec::with_capacity(by_giant.len());
        let mut index = Vec::with_capacity(by_giant.len());
        for ((conjugate, rotation), babies) in by_giant {
            giant_galois.push(galois_from_parts(conjugate, rotation));
            index.push(babies.into_iter().map(|(c, r)| galois_from_parts(c, r)).collect());
        }
        LinearTransformationPlan {
            baby_galois,
            giant_galois,
            index,
            rotations: self.rotations,
        }
    }

    /// The Galois elements whose automorphism keys are required to evaluate this
    /// transform, at `cyclotomic_order`: one per non-identity baby and giant step.
    ///
    /// Automorphism keys are keyed by Galois element throughout the engine (cf.
    /// [`LinearTransformationPlan::galois_elements`]); pass the result here to
    /// index the key store the eval entry points look up.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let babies = self
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| d.baby_galois);
        let giants = self
            .giant_steps
            .iter()
            .filter(|gs| !gs.diagonals.is_empty())
            .map(|gs| gs.galois);
        self.rotations.galois_elements(babies, giants, cyclotomic_order)
    }
}

// ===================================================================
// Rotation schedule
// ===================================================================

/// How a BSGS evaluation reaches its baby and giant steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BsgsRotations {
    /// One key per baby step, applied to the input, and one per giant step.
    #[default]
    PerRotation,
    /// Keys of the powers of two of the two step generators only, each applied to several
    /// ciphertexts at once; no step conjugates. The baby step `baby_step^k` is
    /// `baby_step^(k - 2^i)` mapped by `baby_step^(2^i)`, `2^i` the top bit of `k`, so the
    /// steps of one pass share a key. The giant steps are summed as a binary tree over the
    /// two's-complement digits of `j`, the giant step `giant_step^j` over `slots` slots:
    /// `T(u; G) = T(u_even; G^2) + sigma_G(T(u_odd; G^2))`, the top digit of weight `-2^(b-1)`,
    /// so the steps of one level share a key. For the baby steps `baby_step^k`, `k < n1`, and
    /// contiguous giant digits, the key switches are those of [`Self::PerRotation`], up to one
    /// more per level of the tree ([`Self::giant_key_switches`]); a product goes through more
    /// of them.
    PowersOfTwo {
        /// Galois element every baby step is a power of.
        baby_step: i64,
        /// Galois element every giant step is a power of.
        giant_step: i64,
        /// The slot count, the modulus of the rotations.
        slots: i64,
    },
}

/// One level of the giant-step tree of [`BsgsRotations::PowersOfTwo`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GiantTreeLevel {
    /// The Galois element of the level's key.
    pub(crate) galois: i64,
    /// `(even, odd)` node pairs, parent `even / 2`: node `odd` mapped by `galois` is added to
    /// node `even`, absent when no giant step lies under it.
    pub(crate) pairs: Vec<(Option<usize>, usize)>,
}

impl BsgsRotations {
    /// The baby steps the cache holds for the needed baby steps `babies`: themselves and,
    /// with [`Self::PowersOfTwo`], every non-identity step a doubling reads; sorted by
    /// their [`galois_parts`](crate::layouts::galois_parts).
    pub(crate) fn baby_closure(&self, babies: impl IntoIterator<Item = i64>) -> Vec<i64> {
        let mut closure: BTreeSet<(bool, i64)> = babies.into_iter().map(galois_parts).collect();
        if let Self::PowersOfTwo { baby_step, .. } = *self {
            let step = step_rotation(baby_step);
            for (_, baby) in closure.clone() {
                let mut k = baby_multiple(baby, step);
                while k != 0 {
                    closure.insert((false, k * step));
                    k &= !(1 << k.ilog2());
                }
            }
        }
        closure.into_iter().map(|(c, r)| galois_from_parts(c, r)).collect()
    }

    /// The doubling passes of [`Self::PowersOfTwo`] over the cached baby steps `babies` (a
    /// [`Self::baby_closure`]): `(key Galois element, [(source, target)])`, pass `i` mapping by
    /// `baby_step^(2^i)` the targets whose top bit is `2^i`. Empty for [`Self::PerRotation`].
    pub(crate) fn baby_passes(&self, babies: impl IntoIterator<Item = i64>) -> Vec<(i64, Vec<(i64, i64)>)> {
        let Self::PowersOfTwo { baby_step, .. } = *self else {
            return Vec::new();
        };
        let step = step_rotation(baby_step);
        let mut passes: BTreeMap<u32, Vec<(i64, i64)>> = BTreeMap::new();
        for baby in babies {
            let k = baby_multiple(step_rotation(baby), step);
            if k != 0 {
                let top = k.ilog2();
                passes
                    .entry(top)
                    .or_default()
                    .push((galois_of_rotation((k - (1 << top)) * step), baby));
            }
        }
        passes
            .into_iter()
            .map(|(top, pairs)| (galois_of_rotation((1i64 << top) * step), pairs))
            .collect()
    }

    /// The giant-step tree of [`Self::PowersOfTwo`] over the giant steps `giants`: the
    /// leaf of each giant step, and the levels from the leaves up; the root is node 0.
    /// `None` for [`Self::PerRotation`].
    pub(crate) fn giant_tree(&self, giants: impl IntoIterator<Item = i64>) -> Option<(Vec<usize>, Vec<GiantTreeLevel>)> {
        let Self::PowersOfTwo { giant_step, slots, .. } = *self else {
            return None;
        };
        let step = step_rotation(giant_step);
        let digits: Vec<i64> = giants
            .into_iter()
            .map(|giant| giant_digit(step_rotation(giant), step, slots))
            .collect();
        // The fewest digits whose two's complement holds every `j`.
        let bits = (1..63)
            .find(|&b| digits.iter().all(|&j| (-(1i64 << (b - 1))..(1i64 << (b - 1))).contains(&j)))
            .unwrap();
        let leaves: Vec<usize> = digits.iter().map(|&j| j.rem_euclid(1 << bits) as usize).collect();
        let mut nodes: BTreeSet<usize> = leaves.iter().copied().collect();
        let mut levels = Vec::with_capacity(bits);
        for i in 0..bits {
            let rotation = (if i + 1 == bits { -(1i64 << i) } else { 1i64 << i } * step).rem_euclid(slots);
            let pairs = nodes
                .iter()
                .filter(|&&node| node & 1 == 1)
                .map(|&odd| (nodes.contains(&(odd - 1)).then_some(odd - 1), odd))
                .collect();
            levels.push(GiantTreeLevel {
                galois: galois_of_rotation(rotation),
                pairs,
            });
            nodes = nodes.into_iter().map(|node| node >> 1).collect();
        }
        Some((leaves, levels))
    }

    /// The non-identity Galois elements whose keys the preparation of the baby steps `babies`
    /// switches with.
    pub fn baby_key_galois(&self, babies: impl IntoIterator<Item = i64>) -> Vec<i64> {
        match self {
            Self::PerRotation => babies.into_iter().filter(|&p| !galois_is_identity(p)).collect(),
            Self::PowersOfTwo { .. } => self
                .baby_passes(self.baby_closure(babies))
                .into_iter()
                .map(|(galois, _)| galois)
                .collect(),
        }
    }

    /// The non-identity Galois elements whose keys the evaluation of the giant steps `giants`
    /// switches with.
    pub fn giant_key_galois(&self, giants: impl IntoIterator<Item = i64>) -> Vec<i64> {
        match self.giant_tree(giants) {
            None => Vec::new(),
            Some((_, levels)) => levels
                .into_iter()
                .filter(|level| !level.pairs.is_empty())
                .map(|level| level.galois)
                .collect(),
        }
    }

    /// The key switches the evaluation of the giant steps `giants` takes: one per non-identity
    /// step for [`Self::PerRotation`], one per pair of a level of the tree for
    /// [`Self::PowersOfTwo`], which can be more.
    pub fn giant_key_switches(&self, giants: impl IntoIterator<Item = i64>) -> usize {
        match self {
            Self::PerRotation => giants.into_iter().filter(|&p| !galois_is_identity(p)).count(),
            Self::PowersOfTwo { .. } => self
                .giant_tree(giants)
                .map_or(0, |(_, levels)| levels.iter().map(|level| level.pairs.len()).sum()),
        }
    }

    /// The Galois elements, at `cyclotomic_order`, of the keys the evaluation of the baby
    /// steps `babies` and the giant steps `giants` switches with, sorted and de-duplicated.
    pub fn galois_elements(
        &self,
        babies: impl IntoIterator<Item = i64>,
        giants: impl IntoIterator<Item = i64>,
        cyclotomic_order: i64,
    ) -> Vec<i64> {
        let giants: Vec<i64> = match self {
            Self::PerRotation => giants.into_iter().filter(|&p| !galois_is_identity(p)).collect(),
            Self::PowersOfTwo { .. } => self.giant_key_galois(giants),
        };
        let mut elements: Vec<i64> = self
            .baby_key_galois(babies)
            .into_iter()
            .chain(giants)
            .map(|p| galois_at(p, cyclotomic_order))
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }
}

/// The rotation of a step of [`BsgsRotations::PowersOfTwo`], which does not conjugate.
fn step_rotation(p: i64) -> i64 {
    let (conjugate, rotation) = galois_parts(p);
    assert!(!conjugate, "a power-of-two schedule does not conjugate");
    rotation
}

/// `k` of the baby rotation `k·baby_step`.
fn baby_multiple(baby: i64, baby_step: i64) -> i64 {
    assert!(
        baby >= 0 && baby % baby_step == 0,
        "baby rotation {baby} is not a multiple of the baby step {baby_step}"
    );
    baby / baby_step
}

/// `j` of the giant rotation `j·giant_step`, the representative of least magnitude modulo
/// `slots` when `giant_step` divides it, `-slots / (2·giant_step)` included.
fn giant_digit(rot: i64, giant_step: i64, slots: i64) -> i64 {
    let rot = rot.rem_euclid(slots);
    assert!(
        rot % giant_step == 0,
        "giant rotation {rot} is not a multiple of the giant step {giant_step}"
    );
    let j = rot / giant_step;
    let period = slots / giant_step;
    if slots % giant_step == 0 && 2 * j >= period {
        j - period
    } else {
        j
    }
}

// ===================================================================
// Schedule (strategy / layout / index) + derivation
// ===================================================================

/// Strategy used to derive a linear-transformation evaluation schedule from
/// non-zero diagonals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTransformationStrategy {
    /// Use an explicit BSGS giant-step width, in slots. For the cost-optimal width, call
    /// [`optimal_bsgs_giant_step`] and pass the result here.
    Bsgs { giant_step: usize },
    /// The BSGS schedule of [`Self::Bsgs`], evaluated with the keys of
    /// [`BsgsRotations::PowersOfTwo`].
    BsgsPowersOfTwo { giant_step: usize },
    /// Use one giant step per diagonal and no baby steps.
    Direct,
    /// The BSGS schedule of the diagonals on the hypercube of the generators `5^(2^k)` and
    /// `X -> X^-1`: the dimensions the diagonals need, adjacent ones that step both ways merged,
    /// are each split into a baby and a giant part, cheapest first, so the baby count stays at
    /// the square root of the product of their costs; a dimension that steps both ways steps
    /// back by giants. When those dimensions do not cover every diagonal, the whole group is
    /// split: one dimension of the powers of 5 and `X -> X^-1` when needed. Each diagonal goes
    /// to the first baby, then giant, in the order of their [`galois_parts`](crate::layouts::galois_parts), that covers it.
    /// Evaluated with [`BsgsRotations::PerRotation`].
    Hypercube,
}

/// Scheme-agnostic specification of a linear transformation.
///
/// Carries only the integer-level information needed to derive the BSGS
/// schedule: the Galois elements of the non-zero diagonals, the slot count, and the
/// schedule-selection strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTransformationLayout {
    /// Galois elements of the non-zero diagonals of the matrix.
    pub galois: Vec<i64>,
    /// Number of slots (typically `n / 2` for a CKKS plaintext over `C^{n/2}`), a power of two:
    /// the rotations are taken modulo it.
    pub slots: usize,
    /// Strategy used to pick the BSGS schedule.
    pub strategy: LinearTransformationStrategy,
}

impl LinearTransformationLayout {
    /// Returns the BSGS index schedule implied by this layout.
    pub fn index(&self) -> LinearTransformationPlan {
        linear_transform_index(self.galois.iter().copied(), self.slots, self.strategy)
    }

    /// Returns the BSGS schedule for an explicit `giant_step`, ignoring `strategy`.
    pub fn plan(&self, giant_step: usize) -> LinearTransformationPlan {
        linear_transformation_plan(self.galois.iter().copied(), self.slots, giant_step)
    }

    /// Distinct baby-step Galois elements used by the schedule.
    ///
    /// This is the set of steps the prepared baby cache must hold; pass it
    /// to `LinearTransformationBabySteps::alloc` to size the cache up-front.
    pub fn baby_galois(&self) -> Vec<i64> {
        self.index().baby_galois
    }

    /// Galois elements, at `cyclotomic_order`, required for all non-identity baby and giant steps.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.index().galois_elements(cyclotomic_order)
    }
}

/// BSGS index schedule for a linear transformation. The default is an empty schedule with a
/// key per step, which sizes the scratch of any transform with a key per step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinearTransformationPlan {
    /// Distinct baby-step Galois elements used by the schedule.
    pub baby_galois: Vec<i64>,
    /// Giant-step Galois elements used by the schedule.
    pub giant_galois: Vec<i64>,
    /// Baby-step Galois elements grouped by giant step.
    ///
    /// `index[g]` contains the baby steps `b` used with `giant_galois[g]`. The corresponding
    /// diagonal is their composition over the slot count
    /// ([`galois_compose`](crate::layouts::galois_compose)).
    pub index: Vec<Vec<i64>>,
    /// How the evaluation reaches the steps.
    pub rotations: BsgsRotations,
}

impl LinearTransformationPlan {
    /// Galois elements, at `cyclotomic_order`, required for all non-identity baby and giant steps.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.rotations.galois_elements(
            self.baby_galois.iter().copied(),
            self.giant_galois.iter().copied(),
            cyclotomic_order,
        )
    }
}

/// The conjugation bit and the rotation, in `[0, slots)`, of a diagonal.
fn diagonal_parts(galois: i64, slots: usize) -> (bool, i64) {
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    assert!(
        slots.is_power_of_two() && slots as u64 <= MAX_RING_DEGREE,
        "the slot count must be a power of two at most MAX_RING_DEGREE"
    );
    let (conjugate, rotation) = galois_parts(galois);
    (conjugate, rotation % slots as i64)
}

/// Returns a BSGS schedule for the provided non-zero diagonals.
///
/// Internal helper; external callers go through
/// [`LinearTransformationLayout::plan`].
pub(crate) fn linear_transformation_plan<I>(diagonals: I, slots: usize, giant_step: usize) -> LinearTransformationPlan
where
    I: IntoIterator<Item = i64>,
{
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    assert!(giant_step > 0, "linear transformation giant step must be non-zero");

    let giant_step = giant_step as i64;
    let mut by_giant: BTreeMap<(bool, i64), Vec<(i64, i64)>> = BTreeMap::new();
    let mut baby_rots: BTreeSet<i64> = BTreeSet::from([0]);
    for diagonal in diagonals {
        let (conjugate, rotation) = diagonal_parts(diagonal, slots);
        let baby_rot = rotation % giant_step;
        baby_rots.insert(baby_rot);
        by_giant
            .entry((conjugate, rotation - baby_rot))
            .or_default()
            .push((rotation, baby_rot));
    }

    let baby_galois: Vec<i64> = baby_rots.iter().map(|&rot| galois_of_rotation(rot)).collect();

    let mut giant_galois = Vec::with_capacity(by_giant.len());
    let mut index = Vec::with_capacity(by_giant.len());
    for ((conjugate, rot), mut diagonals) in by_giant {
        diagonals.sort_unstable();
        diagonals.dedup_by_key(|(diagonal, _)| *diagonal);
        giant_galois.push(galois_from_parts(conjugate, rot));
        index.push(
            diagonals
                .into_iter()
                .map(|(_, baby_rot)| galois_of_rotation(baby_rot))
                .collect(),
        );
    }

    LinearTransformationPlan {
        baby_galois,
        giant_galois,
        index,
        rotations: BsgsRotations::PerRotation,
    }
}

/// Returns the optimal BSGS giant-step width, in slots.
///
/// Only evaluates candidates that are multiples of the minimum gap between
/// consecutive sorted diagonal rotations. This prunes the search space from
/// `O(slots)` to `O(slots / min_gap)`, which is significant for structured
/// sparse matrices (e.g. stride-k diagonals).
pub fn optimal_bsgs_giant_step<I>(diagonals: I, slots: usize) -> usize
where
    I: IntoIterator<Item = i64>,
{
    assert!(slots > 0, "linear transformation slot count must be non-zero");

    let diagonals: BTreeSet<(bool, i64)> = diagonals
        .into_iter()
        .map(|diagonal| diagonal_parts(diagonal, slots))
        .collect();
    let rotations: Vec<usize> = diagonals
        .iter()
        .map(|&(_, rotation)| rotation as usize)
        .collect::<BTreeSet<usize>>()
        .into_iter()
        .collect();

    if diagonals.len() <= 1 {
        return 1;
    }

    // Minimum gap between consecutive sorted diagonal rotations.
    // Candidate giant steps only need to be multiples of this value.
    let min_gap = rotations.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(1).max(1);

    let mut best_cost = usize::MAX;
    let mut best_step = min_gap;

    let mut step = min_gap;
    while step < slots {
        let mut baby_rots = BTreeSet::new();
        let mut giant_rots = BTreeSet::new();
        for &(conjugate, rotation) in &diagonals {
            let baby_rot = rotation as usize % step;
            baby_rots.insert(baby_rot);
            giant_rots.insert((conjugate, rotation as usize - baby_rot));
        }
        let n1 = baby_rots.len();
        let n2 = giant_rots.len();
        let cost = (n1 + n2) + n1.abs_diff(n2);
        if cost <= best_cost {
            best_step = step;
            best_cost = cost;
        }
        step += min_gap;
    }

    best_step
}

/// Derives an index schedule from diagonals and a strategy.
fn linear_transform_index<I>(diagonals: I, slots: usize, strategy: LinearTransformationStrategy) -> LinearTransformationPlan
where
    I: IntoIterator<Item = i64>,
{
    match strategy {
        LinearTransformationStrategy::Bsgs { giant_step } => linear_transformation_plan(diagonals, slots, giant_step),
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step } => {
            let plan = linear_transformation_plan(diagonals, slots, giant_step);
            let baby_step = plan
                .baby_galois
                .iter()
                .fold(0, |gcd, &baby| gcd_i64(gcd, step_rotation(baby)))
                .max(1);
            let rotations = BsgsRotations::PowersOfTwo {
                baby_step: galois_of_rotation(baby_step),
                giant_step: galois_of_rotation(giant_step as i64),
                slots: slots as i64,
            };
            LinearTransformationPlan {
                baby_galois: rotations.baby_closure(plan.baby_galois),
                rotations,
                ..plan
            }
        }
        LinearTransformationStrategy::Direct => {
            let diagonals: BTreeSet<(bool, i64)> = diagonals
                .into_iter()
                .map(|diagonal| diagonal_parts(diagonal, slots))
                .collect();
            LinearTransformationPlan {
                baby_galois: vec![galois_of_rotation(0)],
                giant_galois: diagonals.iter().map(|&(c, r)| galois_from_parts(c, r)).collect(),
                index: diagonals.into_iter().map(|_| vec![galois_of_rotation(0)]).collect(),
                rotations: BsgsRotations::PerRotation,
            }
        }
        LinearTransformationStrategy::Hypercube => hypercube_plan(diagonals, slots),
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd_i64(b, a % b) }
}

/// The plan of [`LinearTransformationStrategy::Hypercube`].
fn hypercube_plan<I>(diagonals: I, slots: usize) -> LinearTransformationPlan
where
    I: IntoIterator<Item = i64>,
{
    let coordinates: BTreeSet<(i64, i64)> = diagonals
        .into_iter()
        .map(|diagonal| {
            let (conjugate, rotation) = diagonal_parts(diagonal, slots);
            (i64::from(conjugate), rotation)
        })
        .collect();
    let schedule = hypercube_bsgs_schedule(&coordinates, slots as i64);
    let element = |(i, j): (i64, i64)| galois_from_parts(i == 1, j);
    let babies: BTreeSet<(i64, i64)> = std::iter::once((0, 0))
        .chain(schedule.values().flatten().map(|&(baby, _)| baby))
        .collect();
    LinearTransformationPlan {
        baby_galois: babies.into_iter().map(element).collect(),
        giant_galois: schedule.keys().map(|&giant| element(giant)).collect(),
        index: schedule
            .values()
            .map(|steps| {
                let babies: BTreeSet<(i64, i64)> = steps.iter().map(|&(baby, _)| baby).collect();
                babies.into_iter().map(element).collect()
            })
            .collect(),
        rotations: BsgsRotations::PerRotation,
    }
}

// ===================================================================
// Hypercube schedule of the automorphisms (-1)^i 5^j
// ===================================================================

/// The coordinate `(i, j)` of the automorphism `X -> X^((-1)^i 5^j)`, `i` modulo 2 and `j`
/// modulo the order of a [`hypercube_bsgs_schedule`].
type HypercubeCoordinate = (i64, i64);

/// A schedule of [`hypercube_bsgs_schedule`]: per giant automorphism, the automorphisms it covers,
/// each with its baby automorphism, as `(baby, covered)` coordinates; `covered` is the sum of the
/// two coordinates.
type HypercubeSchedule = BTreeMap<HypercubeCoordinate, Vec<(HypercubeCoordinate, HypercubeCoordinate)>>;

/// The schedule of [`LinearTransformationStrategy::Hypercube`] for the automorphisms at
/// `coordinates`, in `{0, 1} x [0, order)`, `order` a power of two.
fn hypercube_bsgs_schedule(coordinates: &BTreeSet<HypercubeCoordinate>, order: i64) -> HypercubeSchedule {
    if let Some(schedule) = hypercube_assign(
        coordinates,
        &hypercube_split(&hypercube_dimensions(coordinates, order), order),
        order,
    ) {
        return schedule;
    }
    let mut dims: Vec<HypercubeDimension> = Vec::new();
    if order > 1 {
        dims.push(HypercubeDimension::good(1, order));
    }
    if coordinates.iter().any(|c| c.0 == 1) {
        dims.push(HypercubeDimension::good(-1, 2));
    }
    hypercube_assign(coordinates, &hypercube_split(&dims, order), order)
        .expect("the split of the whole group covers every automorphism")
}

/// The baby and the giant automorphisms of a split.
type HypercubeSplit = (BTreeSet<HypercubeCoordinate>, BTreeSet<HypercubeCoordinate>);

/// The schedule of `coordinates` on the split `(babies, giants)`, if it covers them all.
fn hypercube_assign(
    coordinates: &BTreeSet<HypercubeCoordinate>,
    (babies, giants): &HypercubeSplit,
    order: i64,
) -> Option<HypercubeSchedule> {
    let mut schedule: HypercubeSchedule = BTreeMap::new();
    let mut uncovered: BTreeSet<HypercubeCoordinate> = coordinates.clone();
    for &baby in babies {
        for &giant in giants {
            let e: HypercubeCoordinate = ((baby.0 + giant.0) % 2, (baby.1 + giant.1) % order);
            if uncovered.remove(&e) {
                schedule.entry(giant).or_default().push((baby, e));
            }
        }
    }
    uncovered.is_empty().then_some(schedule)
}

/// A dimension of the hypercube of automorphisms: the powers of its generator, `5^exp` or,
/// for `exp = -1`, `X -> X^-1`, of which `size` are needed, forward (`5^(k exp)`), backward
/// (`5^(-k exp)`) or both, a bad dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HypercubeDimension {
    exp: i64,
    size: i64,
    forward: bool,
    backward: bool,
}

impl HypercubeDimension {
    fn good(exp: i64, size: i64) -> Self {
        Self {
            exp,
            size,
            forward: true,
            backward: false,
        }
    }

    fn is_bad(&self) -> bool {
        self.forward && self.backward
    }

    fn cost(&self) -> i64 {
        if self.is_bad() { 2 * self.size } else { self.size }
    }
}

/// The dimensions the automorphisms at `coordinates`, `j` modulo
/// `order`, need: for every power of two `2^k < order`, whether `5^(2^k)` and `5^(-2^k)` are
/// among them, adjacent bad dimensions merged, the last two merged into a good one when they
/// fill the group, and `X -> X^-1` when it is among them.
fn hypercube_dimensions(coordinates: &BTreeSet<HypercubeCoordinate>, order: i64) -> Vec<HypercubeDimension> {
    let exps: BTreeSet<i64> = coordinates.iter().filter(|c| c.0 == 0).map(|c| c.1).collect();
    let exps_five: Vec<i64> = (0..order.max(1).ilog2()).map(|k| 1i64 << k).collect();
    let last: usize = exps_five.len().saturating_sub(1);
    let mut dims: Vec<HypercubeDimension> = Vec::new();
    for (index, &exp) in exps_five.iter().enumerate() {
        let forward: bool = exps.contains(&exp);
        let backward: bool = index < last && exps.contains(&(-exp).rem_euclid(order));
        match dims.last_mut() {
            _ if !(forward || backward) => continue,
            Some(prev) if forward && backward && prev.is_bad() && prev.exp * prev.size == exp => prev.size *= 2,
            _ => dims.push(HypercubeDimension {
                exp,
                size: 2,
                forward,
                backward,
            }),
        }
    }
    if let [.., second, last] = dims[..]
        && 2 * last.exp == order
        && 2 * second.exp * second.size == order
    {
        dims.pop();
        *dims.last_mut().unwrap() = HypercubeDimension::good(second.exp, 2 * second.size);
    }
    if coordinates.contains(&(1, 0)) {
        dims.push(HypercubeDimension::good(-1, 2));
    }
    dims
}

/// The baby and giant automorphisms of the dimensions `dims`, every one
/// split into a baby and a giant part, cheapest first, so the babies stay at the square root of
/// the product of their costs; a bad dimension also steps back, by giants.
fn hypercube_split(dims: &[HypercubeDimension], order: i64) -> HypercubeSplit {
    let target: f64 = (dims.iter().map(HypercubeDimension::cost).product::<i64>() as f64).sqrt();
    let mut sorted: Vec<&HypercubeDimension> = dims.iter().collect();
    sorted.sort_by_key(|d| d.cost());
    let mut babies: BTreeSet<HypercubeCoordinate> = BTreeSet::from([(0, 0)]);
    let mut giants: BTreeSet<HypercubeCoordinate> = BTreeSet::from([(0, 0)]);
    for dim in sorted {
        let mut baby_size: i64 = dim.size;
        while baby_size > 1 && (babies.len() as i64 * baby_size) as f64 > target {
            baby_size /= 2;
        }
        let start_giant: i64 = if dim.is_bad() { -(dim.size / baby_size) } else { 0 };
        let direction: i64 = if dim.backward && !dim.forward { -1 } else { 1 };
        let step: HypercubeCoordinate = if dim.exp == -1 { (1, 0) } else { (0, dim.exp) };
        let add = |c: &HypercubeCoordinate, power: i64| -> HypercubeCoordinate {
            (
                (c.0 + power * step.0).rem_euclid(2),
                (c.1 + direction * power * step.1).rem_euclid(order),
            )
        };
        babies = babies
            .iter()
            .flat_map(|c| (0..baby_size).map(move |power| add(c, power)))
            .collect();
        giants = giants
            .iter()
            .flat_map(|c| (start_giant..dim.size / baby_size).map(move |power| add(c, baby_size * power)))
            .collect();
    }
    (babies, giants)
}

#[cfg(test)]
mod hypercube_tests {
    use std::collections::BTreeSet;

    use poulpy_hal::GALOIS_CONJUGATION;

    use super::{
        HypercubeCoordinate, HypercubeSchedule, LinearTransformationLayout, LinearTransformationStrategy, diagonal_parts,
        hypercube_bsgs_schedule,
    };
    use crate::layouts::{galois_compose, galois_from_parts};

    /// Every coordinate is covered once, by a baby and a giant that add up to it.
    fn assert_covers(coordinates: &BTreeSet<HypercubeCoordinate>, schedule: &HypercubeSchedule, order: i64) {
        let mut covered: BTreeSet<HypercubeCoordinate> = BTreeSet::new();
        for (giant, steps) in schedule {
            for &(baby, e) in steps {
                assert_eq!(e, ((baby.0 + giant.0) % 2, (baby.1 + giant.1) % order));
                assert!(covered.insert(e), "{e:?} covered twice");
            }
        }
        assert_eq!(&covered, coordinates);
    }

    /// The key switches of a schedule: its non-identity babies and giants.
    fn key_switches(schedule: &HypercubeSchedule) -> usize {
        let babies: BTreeSet<HypercubeCoordinate> = schedule.values().flatten().map(|&(b, _)| b).collect();
        babies.len() + schedule.len() - usize::from(babies.contains(&(0, 0))) - usize::from(schedule.contains_key(&(0, 0)))
    }

    /// The plan of the strategy maps the coordinates to Galois elements and covers each diagonal once.
    #[test]
    fn hypercube_strategy_covers_its_diagonals() {
        let slots = 8;
        let galois: Vec<i64> = [(false, 0), (false, 3), (true, 0), (true, 5)]
            .into_iter()
            .map(|(c, r)| galois_from_parts(c, r))
            .collect();
        let plan = LinearTransformationLayout {
            galois: galois.clone(),
            slots,
            strategy: LinearTransformationStrategy::Hypercube,
        }
        .index();
        let mut covered: Vec<(bool, i64)> = plan
            .giant_galois
            .iter()
            .zip(&plan.index)
            .flat_map(|(&giant, babies)| {
                babies
                    .iter()
                    .map(move |&baby| diagonal_parts(galois_compose(giant, baby), slots))
            })
            .collect();
        covered.sort_unstable();
        let mut want: Vec<(bool, i64)> = galois.iter().map(|&p| diagonal_parts(p, slots)).collect();
        want.sort_unstable();
        assert_eq!(covered, want);
        assert!(plan.giant_galois.contains(&GALOIS_CONJUGATION) || plan.baby_galois.contains(&GALOIS_CONJUGATION));
    }

    #[test]
    fn hypercube_schedule_covers_every_automorphism() {
        for order in [1i64, 2, 8, 16] {
            let all: BTreeSet<HypercubeCoordinate> = (0..2).flat_map(|i| (0..order).map(move |j| (i, j))).collect();
            let sets: [BTreeSet<HypercubeCoordinate>; 4] = [
                all.clone(),
                all.iter().copied().filter(|c| c.0 == 0).collect(),
                all.iter().copied().filter(|c| c.0 == 1).collect(),
                all.iter().copied().filter(|c| (c.1 * 7 + c.0) % 3 == 0).collect(),
            ];
            for coordinates in sets {
                assert_covers(&coordinates, &hypercube_bsgs_schedule(&coordinates, order), order);
            }
        }
    }

    #[test]
    fn hypercube_schedule_of_a_butterfly_takes_two_key_switches() {
        // The diagonals `0, s, -s` of one FFT factor: a dimension stepping both ways.
        for (order, s) in [(8i64, 1i64), (8, 2), (16, 4)] {
            let coordinates: BTreeSet<HypercubeCoordinate> = [(0, 0), (0, s), (0, order - s)].into_iter().collect();
            let schedule: HypercubeSchedule = hypercube_bsgs_schedule(&coordinates, order);
            assert_covers(&coordinates, &schedule, order);
            assert_eq!(key_switches(&schedule), 2, "order={order}, s={s}");
        }
    }

    #[test]
    fn hypercube_schedule_of_the_whole_group_is_balanced() {
        // 2 x 16 automorphisms: 4 x 8 or 8 x 4, so 10 key switches.
        let all: BTreeSet<HypercubeCoordinate> = (0..2).flat_map(|i| (0..16).map(move |j| (i, j))).collect();
        assert_eq!(key_switches(&hypercube_bsgs_schedule(&all, 16)), 10);
    }
}
