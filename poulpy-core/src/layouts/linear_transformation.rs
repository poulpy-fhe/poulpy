//! Unprepared GLWE linear transformation (BSGS) data and schedule.
//!
//! Holds the unprepared transform (`LinearTransformation`, encoded diagonals
//! bucketed by giant step) and the integer-level BSGS schedule types
//! (`LinearTransformationLayout`, `LinearTransformationPlan`,
//! `LinearTransformationStrategy`) plus their derivation, which is pure integer
//! math (cf. docs/linear_transformation.md).
//!
//! The *prepared* (convolution-domain) caches live in
//! [`crate::layouts::prepared`]; the HAL-dependent allocators and the
//! prepare/eval reference algorithms live in
//! [`crate::reference::linear_transformation`].

use std::collections::{BTreeMap, BTreeSet};

use poulpy_hal::layouts::galois_elements_from_rotations;

// ===================================================================
// Unprepared transform
// ===================================================================

/// One non-zero diagonal of the linear map, attached to a giant step.
///
/// `plaintext` is the pre-rotated diagonal `u~_{j,k} = rot(diag_{n1*j+k}, -n1*j)`,
/// already encoded as a plaintext polynomial. `baby` is the baby-step slot
/// rotation `k` whose prepared `rot(v, k)` it multiplies.
pub struct LinearTransformationDiagonal<P> {
    /// Baby-step slot rotation.
    pub baby: i64,
    /// Pre-rotated diagonal, encoded as a plaintext polynomial.
    pub plaintext: P,
}

/// A single giant step `j`: its inner sum is rotated by `rot` slots and added to
/// the output (`rot == 0` is the identity giant step).
pub struct LinearTransformationGiantStep<P> {
    /// Slot rotation amount `n1*j`.
    pub rot: i64,
    /// The non-zero diagonals contributing to this giant step.
    pub diagonals: Vec<LinearTransformationDiagonal<P>>,
}

/// A linear transformation in baby-step / giant-step form.
///
/// `P` is the encoded-plaintext container.
pub struct LinearTransformation<P> {
    /// Distinct baby-step slot rotations `k`; `baby_steps[0] == 0` (the identity).
    pub baby_steps: Vec<i64>,
    /// The giant steps.
    pub giant_steps: Vec<LinearTransformationGiantStep<P>>,
    /// How the evaluation reaches the baby and giant rotations.
    pub rotations: BsgsRotations,
}

impl<P> LinearTransformation<P> {
    /// The distinct baby-step slot rotations `k` (`baby_steps[0] == 0`, the
    /// identity); the set the prepared baby cache must hold. Accessor over the
    /// [`baby_steps`](Self::baby_steps) field for callers that prefer a method.
    pub fn baby_steps(&self) -> &[i64] {
        &self.baby_steps
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
    /// baby/giant rotations. Useful for one-shot allocation of the prepared
    /// cache directly from an unprepared transform.
    ///
    /// The output is canonical — giant steps sorted, per-giant baby rotations
    /// sorted and de-duplicated — so it matches the schedule
    /// [`LinearTransformationLayout::index`] would derive from the same diagonal
    /// set. Only baby rotations actually referenced by at least one non-empty
    /// giant step are included; the field `self.baby_steps` may declare more
    /// rotations than the transform's data populates, and those extras would
    /// otherwise force the caller to provide automorphism keys that the
    /// schedule does not actually need.
    pub fn index(&self) -> LinearTransformationPlan {
        let mut by_giant: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        let mut used_babies: BTreeSet<i64> = BTreeSet::new();
        for gs in &self.giant_steps {
            if gs.diagonals.is_empty() {
                continue;
            }
            let babies = by_giant.entry(gs.rot).or_default();
            for d in &gs.diagonals {
                used_babies.insert(d.baby);
                babies.insert(d.baby);
            }
        }
        let baby_steps = self.rotations.baby_closure(used_babies);
        let mut giant_steps = Vec::with_capacity(by_giant.len());
        let mut index = Vec::with_capacity(by_giant.len());
        for (rot, babies) in by_giant {
            giant_steps.push(rot);
            index.push(babies.into_iter().collect());
        }
        LinearTransformationPlan {
            baby_steps,
            giant_steps,
            index,
            rotations: self.rotations,
        }
    }

    /// The Galois elements whose automorphism keys are required to evaluate this
    /// transform: one per non-zero baby- and giant-step rotation.
    ///
    /// Automorphism keys are keyed by Galois element throughout the engine (cf.
    /// [`LinearTransformationPlan::galois_elements`]); pass the result here to
    /// index the key store the eval entry points look up.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let babies = self.giant_steps.iter().flat_map(|gs| gs.diagonals.iter()).map(|d| d.baby);
        let giants = self.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty()).map(|gs| gs.rot);
        self.rotations.galois_elements(babies, giants, cyclotomic_order)
    }
}

// ===================================================================
// Rotation schedule
// ===================================================================

/// How a BSGS evaluation reaches its baby and giant rotations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BsgsRotations {
    /// One key per baby rotation, applied to the input, and one per giant rotation.
    #[default]
    PerRotation,
    /// Keys of the power-of-two multiples of the two steps only, each applied to several
    /// ciphertexts at once. The baby rotation `k·baby_step` is the rotation of
    /// `(k - 2^i)·baby_step` by `2^i·baby_step`, `2^i` the top bit of `k`, so the rotations
    /// of one pass share a key. The giant steps are summed as a binary tree over the
    /// two's-complement digits of `j`, the giant rotation `j·giant_step` modulo `slots`:
    /// `T(u; G) = T(u_even; 2G) + rot(G)(T(u_odd; 2G))`, the top digit of weight `-2^(b-1)`,
    /// so the rotations of one level share a key. The number of key switches is that of
    /// [`Self::PerRotation`]; a product goes through more of them.
    PowersOfTwo {
        /// Every baby rotation is a multiple of it.
        baby_step: i64,
        /// Every giant rotation is a multiple of it.
        giant_step: i64,
        /// The slot count, the modulus of the rotations.
        slots: i64,
    },
}

/// One level of the giant-step tree of [`BsgsRotations::PowersOfTwo`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GiantTreeLevel {
    /// The rotation of the level's key, in `[0, slots)`.
    pub rot: i64,
    /// `(even, odd)` node pairs, parent `even / 2`: node `odd` rotated by `rot` is added to
    /// node `even`, absent when no giant step lies under it.
    pub pairs: Vec<(Option<usize>, usize)>,
}

impl BsgsRotations {
    /// The baby rotations the cache holds for the needed rotations `babies`: themselves and,
    /// with [`Self::PowersOfTwo`], every non-zero rotation a doubling reads.
    pub fn baby_closure(&self, babies: impl IntoIterator<Item = i64>) -> Vec<i64> {
        let mut closure: BTreeSet<i64> = babies.into_iter().collect();
        if let Self::PowersOfTwo { baby_step, .. } = *self {
            for baby in closure.clone() {
                let mut k = baby_multiple(baby, baby_step);
                while k != 0 {
                    closure.insert(k * baby_step);
                    k &= !(1 << k.ilog2());
                }
            }
        }
        closure.into_iter().collect()
    }

    /// The doubling passes of [`Self::PowersOfTwo`] over the cached rotations `babies` (a
    /// [`Self::baby_closure`]): `(key rotation, [(source, target)])`, pass `i` rotating by
    /// `2^i·baby_step` the targets whose top bit is `2^i`. Empty for [`Self::PerRotation`].
    pub fn baby_passes(&self, babies: impl IntoIterator<Item = i64>) -> Vec<(i64, Vec<(i64, i64)>)> {
        let Self::PowersOfTwo { baby_step, .. } = *self else {
            return Vec::new();
        };
        let mut passes: BTreeMap<u32, Vec<(i64, i64)>> = BTreeMap::new();
        for baby in babies {
            let k = baby_multiple(baby, baby_step);
            if k != 0 {
                let top = k.ilog2();
                passes.entry(top).or_default().push(((k - (1 << top)) * baby_step, baby));
            }
        }
        passes
            .into_iter()
            .map(|(top, pairs)| ((1i64 << top) * baby_step, pairs))
            .collect()
    }

    /// The giant-step tree of [`Self::PowersOfTwo`] over the giant rotations `giants`: the
    /// leaf of each giant step, and the levels from the leaves up; the root is node 0.
    /// `None` for [`Self::PerRotation`].
    pub fn giant_tree(&self, giants: impl IntoIterator<Item = i64>) -> Option<(Vec<usize>, Vec<GiantTreeLevel>)> {
        let Self::PowersOfTwo { giant_step, slots, .. } = *self else {
            return None;
        };
        let digits: Vec<i64> = giants.into_iter().map(|rot| giant_digit(rot, giant_step, slots)).collect();
        // The fewest digits whose two's complement holds every `j`.
        let bits = (1..63)
            .find(|&b| digits.iter().all(|&j| (-(1i64 << (b - 1))..(1i64 << (b - 1))).contains(&j)))
            .unwrap();
        let leaves: Vec<usize> = digits.iter().map(|&j| j.rem_euclid(1 << bits) as usize).collect();
        let mut nodes: BTreeSet<usize> = leaves.iter().copied().collect();
        let mut levels = Vec::with_capacity(bits);
        for i in 0..bits {
            // In `[0, slots)`: a negative generator is a conjugation, not the opposite rotation.
            let rot = (if i + 1 == bits { -(1i64 << i) } else { 1i64 << i } * giant_step).rem_euclid(slots);
            let pairs = nodes
                .iter()
                .filter(|&&node| node & 1 == 1)
                .map(|&odd| (nodes.contains(&(odd - 1)).then_some(odd - 1), odd))
                .collect();
            levels.push(GiantTreeLevel { rot, pairs });
            nodes = nodes.into_iter().map(|node| node >> 1).collect();
        }
        Some((leaves, levels))
    }

    /// The non-zero rotations whose keys the preparation of the baby rotations `babies`
    /// switches with.
    pub fn baby_key_rotations(&self, babies: impl IntoIterator<Item = i64>) -> Vec<i64> {
        match self {
            Self::PerRotation => babies.into_iter().filter(|&rot| rot != 0).collect(),
            Self::PowersOfTwo { .. } => self
                .baby_passes(self.baby_closure(babies))
                .into_iter()
                .map(|(rot, _)| rot)
                .collect(),
        }
    }

    /// The non-zero rotations whose keys the evaluation of the giant rotations `giants`
    /// switches with.
    pub fn giant_key_rotations(&self, giants: impl IntoIterator<Item = i64>) -> Vec<i64> {
        match self.giant_tree(giants) {
            None => Vec::new(),
            Some((_, levels)) => levels
                .into_iter()
                .filter(|level| !level.pairs.is_empty())
                .map(|level| level.rot)
                .collect(),
        }
    }

    /// The Galois elements of the keys the evaluation of the baby rotations `babies` and the
    /// giant rotations `giants` switches with.
    pub fn galois_elements(
        &self,
        babies: impl IntoIterator<Item = i64>,
        giants: impl IntoIterator<Item = i64>,
        cyclotomic_order: i64,
    ) -> Vec<i64> {
        let giants: Vec<i64> = match self {
            Self::PerRotation => giants.into_iter().collect(),
            Self::PowersOfTwo { .. } => self.giant_key_rotations(giants),
        };
        galois_elements_from_rotations(self.baby_key_rotations(babies).into_iter().chain(giants), cyclotomic_order)
    }
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
/// non-zero diagonal indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTransformationStrategy {
    /// Use an explicit BSGS giant-step width. For the cost-optimal width, call
    /// [`optimal_bsgs_giant_step`] and pass the result here.
    Bsgs { giant_step: usize },
    /// The BSGS schedule of [`Self::Bsgs`], evaluated with the keys of
    /// [`BsgsRotations::PowersOfTwo`].
    BsgsPowersOfTwo { giant_step: usize },
    /// Use one giant step per diagonal and no baby rotations.
    Direct,
}

/// Scheme-agnostic specification of a linear transformation.
///
/// Carries only the integer-level information needed to derive the BSGS
/// schedule: the non-zero diagonal indexes, the slot count, and the
/// schedule-selection strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTransformationLayout {
    /// Non-zero diagonal indexes of the matrix.
    pub indexes: Vec<i64>,
    /// Number of slots (typically `n / 2` for a CKKS plaintext over `C^{n/2}`).
    pub slots: usize,
    /// Strategy used to pick the BSGS schedule.
    pub strategy: LinearTransformationStrategy,
}

impl LinearTransformationLayout {
    /// Returns the BSGS index schedule implied by this layout.
    pub fn index(&self) -> LinearTransformationPlan {
        linear_transform_index(self.indexes.iter().copied(), self.slots, self.strategy)
    }

    /// Returns the BSGS schedule for an explicit `giant_step`, ignoring `strategy`.
    pub fn plan(&self, giant_step: usize) -> LinearTransformationPlan {
        linear_transformation_plan(self.indexes.iter().copied(), self.slots, giant_step)
    }

    /// Distinct baby-step rotations (`k`) used by the schedule.
    ///
    /// This is the set of rotations the prepared baby cache must hold; pass it
    /// to `LinearTransformationBabySteps::alloc` to size the cache up-front.
    pub fn baby_steps(&self) -> Vec<i64> {
        self.index().baby_steps
    }

    /// Galois elements required for all non-zero baby- and giant-step rotations.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.index().galois_elements(cyclotomic_order)
    }
}

/// BSGS index schedule for a linear transformation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTransformationPlan {
    /// Distinct baby-step rotations used by the schedule.
    pub baby_steps: Vec<i64>,
    /// Giant-step rotations used by the schedule.
    pub giant_steps: Vec<i64>,
    /// Baby-step rotations grouped by giant step.
    ///
    /// `index[g]` contains the real baby rotations `k` used with
    /// `giant_steps[g]`. The corresponding diagonal is
    /// `giant_steps[g] + k` modulo the slot count.
    pub index: Vec<Vec<i64>>,
    /// How the evaluation reaches the rotations.
    pub rotations: BsgsRotations,
}

impl LinearTransformationPlan {
    /// Galois elements required for all non-zero baby- and giant-step rotations.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.rotations.galois_elements(
            self.baby_steps.iter().copied(),
            self.giant_steps.iter().copied(),
            cyclotomic_order,
        )
    }
}

/// Normalizes a diagonal index modulo the number of slots.
///
/// Internal helper; external callers access schedule construction through
/// [`LinearTransformationLayout`] methods (`.index()` / `.plan(giant_step)`).
pub(crate) fn normalize_linear_transform_diagonal(diagonal: i64, slots: usize) -> usize {
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    diagonal.rem_euclid(slots as i64) as usize
}

/// Returns a BSGS schedule for the provided non-zero diagonal indexes.
///
/// Internal helper; external callers go through
/// [`LinearTransformationLayout::plan`].
pub(crate) fn linear_transformation_plan<I>(diagonal_indexes: I, slots: usize, giant_step: usize) -> LinearTransformationPlan
where
    I: IntoIterator<Item = i64>,
{
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    assert!(giant_step > 0, "linear transformation giant step must be non-zero");

    let mut by_giant: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    let mut baby_rots: BTreeSet<usize> = BTreeSet::from([0]);
    for diagonal in diagonal_indexes {
        let diagonal = normalize_linear_transform_diagonal(diagonal, slots);
        let baby_rot = diagonal % giant_step;
        let giant_rot = diagonal - baby_rot;
        baby_rots.insert(baby_rot);
        by_giant.entry(giant_rot).or_default().push((diagonal, baby_rot));
    }

    let baby_steps: Vec<i64> = baby_rots.iter().map(|&rot| rot as i64).collect();

    let mut giant_steps = Vec::with_capacity(by_giant.len());
    let mut index = Vec::with_capacity(by_giant.len());
    for (rot, mut diagonals) in by_giant {
        diagonals.sort_unstable();
        diagonals.dedup_by_key(|(diagonal, _)| *diagonal);
        giant_steps.push(rot as i64);
        index.push(diagonals.into_iter().map(|(_, baby_rot)| baby_rot as i64).collect());
    }

    LinearTransformationPlan {
        baby_steps,
        giant_steps,
        index,
        rotations: BsgsRotations::PerRotation,
    }
}

/// Returns the optimal BSGS giant-step width.
///
/// Only evaluates candidates that are multiples of the minimum gap between
/// consecutive sorted diagonal indices. This prunes the search space from
/// `O(slots)` to `O(slots / min_gap)`, which is significant for structured
/// sparse matrices (e.g. stride-k diagonals).
pub fn optimal_bsgs_giant_step<I>(diagonal_indexes: I, slots: usize) -> usize
where
    I: IntoIterator<Item = i64>,
{
    assert!(slots > 0, "linear transformation slot count must be non-zero");

    let diagonals: Vec<usize> = {
        let set: BTreeSet<usize> = diagonal_indexes
            .into_iter()
            .map(|diagonal| normalize_linear_transform_diagonal(diagonal, slots))
            .collect();
        set.into_iter().collect() // BTreeSet iteration is already sorted
    };

    if diagonals.len() <= 1 {
        return 1;
    }

    // Minimum gap between consecutive sorted normalized diagonal indices.
    // Candidate giant steps only need to be multiples of this value.
    let min_gap = diagonals.windows(2).map(|w| w[1] - w[0]).min().unwrap().max(1);

    let mut best_cost = usize::MAX;
    let mut best_step = min_gap;

    let mut step = min_gap;
    while step < slots {
        let mut baby_rots = BTreeSet::new();
        let mut giant_rots = BTreeSet::new();
        for &diagonal in &diagonals {
            let baby_rot = diagonal % step;
            baby_rots.insert(baby_rot);
            giant_rots.insert(diagonal - baby_rot);
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

/// Derives an index schedule from diagonal indexes and a strategy.
fn linear_transform_index<I>(
    diagonal_indexes: I,
    slots: usize,
    strategy: LinearTransformationStrategy,
) -> LinearTransformationPlan
where
    I: IntoIterator<Item = i64>,
{
    match strategy {
        LinearTransformationStrategy::Bsgs { giant_step } => linear_transformation_plan(diagonal_indexes, slots, giant_step),
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step } => {
            let plan = linear_transformation_plan(diagonal_indexes, slots, giant_step);
            let baby_step = plan.baby_steps.iter().fold(0, |gcd, &baby| gcd_i64(gcd, baby)).max(1);
            let rotations = BsgsRotations::PowersOfTwo {
                baby_step,
                giant_step: giant_step as i64,
                slots: slots as i64,
            };
            LinearTransformationPlan {
                baby_steps: rotations.baby_closure(plan.baby_steps),
                rotations,
                ..plan
            }
        }
        LinearTransformationStrategy::Direct => {
            let diagonals: BTreeSet<usize> = diagonal_indexes
                .into_iter()
                .map(|diagonal| normalize_linear_transform_diagonal(diagonal, slots))
                .collect();
            LinearTransformationPlan {
                baby_steps: vec![0],
                giant_steps: diagonals.iter().map(|&diagonal| diagonal as i64).collect(),
                index: diagonals.into_iter().map(|_| vec![0]).collect(),
                rotations: BsgsRotations::PerRotation,
            }
        }
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd_i64(b, a % b) }
}
