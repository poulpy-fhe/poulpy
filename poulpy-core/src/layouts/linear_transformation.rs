//! Unprepared GLWE linear transformation (BSGS) data and schedule.
//!
//! Holds the unprepared transform (`GLWELinearTransform`, encoded diagonals
//! bucketed by giant step) and the integer-level BSGS schedule types
//! (`LinearTransformationLayout`, `GLWELinearTransformationSchedule`,
//! `LinearTransformationStrategy`) plus their derivation, which is pure integer
//! math (cf. docs/lt_bsgs.md §3).
//!
//! The *prepared* (convolution-domain) caches live in
//! [`crate::layouts::prepared`]; the HAL-dependent allocators and the
//! prepare/eval reference algorithms live in
//! [`crate::default::linear_transformation`].

use std::collections::{BTreeMap, BTreeSet};

// ===================================================================
// Unprepared transform
// ===================================================================

/// One non-zero diagonal of the linear map, attached to a giant step.
///
/// `plaintext` is the pre-rotated diagonal `u~_{j,k} = rot(diag_{n1*j+k}, -n1*j)`,
/// already encoded as a plaintext polynomial. `baby` is the baby-step slot
/// rotation `k` whose prepared `rot(v, k)` it multiplies.
pub struct GLWELinearTransformDiagonal<P> {
    /// Baby-step slot rotation.
    pub baby: i64,
    /// Pre-rotated diagonal, encoded as a plaintext polynomial.
    pub plaintext: P,
}

/// A single giant step `j`: its inner sum is rotated by `rot` slots and added to
/// the output (`rot == 0` is the identity giant step).
pub struct GLWELinearTransformGiantStep<P> {
    /// Slot rotation amount `n1*j`.
    pub rot: i64,
    /// The non-zero diagonals contributing to this giant step.
    pub diagonals: Vec<GLWELinearTransformDiagonal<P>>,
}

/// A linear transformation in baby-step / giant-step form.
///
/// `P` is the encoded-plaintext container.
pub struct GLWELinearTransform<P> {
    /// Distinct baby-step slot rotations `k`; `baby_steps[0] == 0` (the identity).
    pub baby_steps: Vec<i64>,
    /// The giant steps.
    pub giant_steps: Vec<GLWELinearTransformGiantStep<P>>,
}

impl<P> GLWELinearTransform<P> {
    /// Derives the BSGS index schedule implied by this transform's actual
    /// baby/giant rotations. Useful for one-shot allocation of the prepared
    /// cache directly from an unprepared transform.
    ///
    /// Only baby rotations actually referenced by at least one non-empty
    /// giant step are included; the field `self.baby_steps` may declare more
    /// rotations than the transform's data populates, and those extras would
    /// otherwise force the caller to provide automorphism keys that the
    /// schedule does not actually need.
    pub fn index(&self) -> GLWELinearTransformationSchedule {
        let mut used_babies: BTreeSet<i64> = BTreeSet::new();
        for gs in &self.giant_steps {
            for d in &gs.diagonals {
                used_babies.insert(d.baby);
            }
        }
        let baby_steps: Vec<i64> = used_babies.into_iter().collect();
        let mut giant_steps = Vec::with_capacity(self.giant_steps.len());
        let mut index = Vec::with_capacity(self.giant_steps.len());
        for gs in &self.giant_steps {
            if gs.diagonals.is_empty() {
                continue;
            }
            giant_steps.push(gs.rot);
            index.push(gs.diagonals.iter().map(|d| d.baby).collect());
        }
        GLWELinearTransformationSchedule {
            baby_steps,
            giant_steps,
            index,
        }
    }

    /// The non-zero baby- and giant-step rotations whose automorphism keys are required.
    pub fn required_rotations(&self) -> Vec<i64> {
        let mut rots: Vec<i64> = self
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| d.baby)
            .filter(|&r| r != 0)
            .collect();
        rots.extend(
            self.giant_steps
                .iter()
                .filter(|gs| !gs.diagonals.is_empty())
                .map(|gs| gs.rot)
                .filter(|&r| r != 0),
        );
        rots.sort_unstable();
        rots.dedup();
        rots
    }
}

// ===================================================================
// Schedule (strategy / layout / index) + derivation
// ===================================================================

/// Strategy used to derive a linear-transformation evaluation schedule from
/// non-zero diagonal indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTransformationStrategy {
    /// Pick a BSGS giant step with the best cost rule.
    Auto,
    /// Use an explicit BSGS giant-step width.
    Bsgs { giant_step: usize },
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
    pub fn index(&self) -> GLWELinearTransformationSchedule {
        linear_transform_index(self.indexes.iter().copied(), self.slots, self.strategy)
    }

    /// Returns the BSGS schedule for an explicit `giant_step`, ignoring `strategy`.
    pub fn linear_transformation_schedule(&self, giant_step: usize) -> GLWELinearTransformationSchedule {
        linear_transformation_schedule(self.indexes.iter().copied(), self.slots, giant_step)
    }

    /// Distinct baby-step rotations (`k`) used by the schedule.
    ///
    /// This is the set of rotations the prepared baby cache must hold; pass it
    /// to `GLWEPreparedLinearTransformationLhs::alloc` to size the cache up-front.
    pub fn baby_steps(&self) -> Vec<i64> {
        self.index().baby_steps
    }

    /// Non-zero baby- and giant-step rotations whose automorphism keys the
    /// evaluator will request.
    pub fn required_rotations(&self) -> Vec<i64> {
        self.index().required_rotations()
    }
}

/// BSGS index schedule for a linear transformation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GLWELinearTransformationSchedule {
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
}

impl GLWELinearTransformationSchedule {
    /// The non-zero baby- and giant-step rotations required by this schedule.
    pub fn required_rotations(&self) -> Vec<i64> {
        let mut rots: Vec<i64> = self.baby_steps.iter().copied().filter(|&k| k != 0).collect();
        rots.extend(self.giant_steps.iter().copied().filter(|&r| r != 0));
        rots.sort_unstable();
        rots.dedup();
        rots
    }
}

/// Normalizes a diagonal index modulo the number of slots.
///
/// Internal helper; external callers access schedule construction through
/// [`LinearTransformationLayout`] methods (`.index()` / `.linear_transformation_schedule(giant_step)`).
pub(crate) fn normalize_linear_transform_diagonal(diagonal: i64, slots: usize) -> usize {
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    diagonal.rem_euclid(slots as i64) as usize
}

/// Returns a BSGS schedule for the provided non-zero diagonal indexes.
///
/// Internal helper; external callers go through
/// [`LinearTransformationLayout::linear_transformation_schedule`].
pub(crate) fn linear_transformation_schedule<I>(
    diagonal_indexes: I,
    slots: usize,
    giant_step: usize,
) -> GLWELinearTransformationSchedule
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

    GLWELinearTransformationSchedule {
        baby_steps,
        giant_steps,
        index,
    }
}

/// Returns the optimal BSGS giant-step width.
pub fn optimal_bsgs_giant_step<I>(diagonal_indexes: I, slots: usize) -> usize
where
    I: IntoIterator<Item = i64>,
{
    assert!(slots > 0, "linear transformation slot count must be non-zero");

    let diagonals: BTreeSet<usize> = diagonal_indexes
        .into_iter()
        .map(|diagonal| normalize_linear_transform_diagonal(diagonal, slots))
        .collect();
    if diagonals.is_empty() {
        return 1;
    }

    (1..=slots)
        .min_by_key(|&giant_step| {
            let mut baby_rots = BTreeSet::new();
            let mut giant_rots = BTreeSet::new();
            for &diagonal in &diagonals {
                let baby_rot = diagonal % giant_step;
                baby_rots.insert(baby_rot);
                giant_rots.insert(diagonal - baby_rot);
            }
            let n1 = baby_rots.len();
            let n2 = giant_rots.len();
            ((n1 + n2) + n1.abs_diff(n2), giant_step)
        })
        .unwrap()
}

/// Returns whether the direct diagonal schedule is preferable to BSGS before
/// backend-specific benchmarks are available.
fn should_use_direct_linear_transform(diagonal_count: usize) -> bool {
    diagonal_count <= 2
}

/// Derives an index schedule from diagonal indexes and a strategy.
pub fn linear_transform_index<I>(
    diagonal_indexes: I,
    slots: usize,
    strategy: LinearTransformationStrategy,
) -> GLWELinearTransformationSchedule
where
    I: IntoIterator<Item = i64>,
{
    let diagonal_indexes: Vec<i64> = diagonal_indexes.into_iter().collect();
    match strategy {
        LinearTransformationStrategy::Auto => {
            let diagonal_count = diagonal_indexes
                .iter()
                .map(|&diagonal| normalize_linear_transform_diagonal(diagonal, slots))
                .collect::<BTreeSet<_>>()
                .len();
            if should_use_direct_linear_transform(diagonal_count) {
                linear_transform_index(diagonal_indexes, slots, LinearTransformationStrategy::Direct)
            } else {
                let giant_step = optimal_bsgs_giant_step(diagonal_indexes.iter().copied(), slots);
                linear_transformation_schedule(diagonal_indexes, slots, giant_step)
            }
        }
        LinearTransformationStrategy::Bsgs { giant_step } => linear_transformation_schedule(diagonal_indexes, slots, giant_step),
        LinearTransformationStrategy::Direct => {
            let diagonals: BTreeSet<usize> = diagonal_indexes
                .into_iter()
                .map(|diagonal| normalize_linear_transform_diagonal(diagonal, slots))
                .collect();
            GLWELinearTransformationSchedule {
                baby_steps: vec![0],
                giant_steps: diagonals.iter().map(|&diagonal| diagonal as i64).collect(),
                index: diagonals.into_iter().map(|_| vec![0]).collect(),
            }
        }
    }
}
