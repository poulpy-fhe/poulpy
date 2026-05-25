//! BSGS schedule construction for linear transformations.
//!
//! This is the integer-index side of docs/lt_bsgs.md §3: each non-zero diagonal
//! `i` is factored as `i = n1*j + k`, producing one reusable baby rotation `k`
//! and one giant rotation `n1*j`.

use std::collections::{BTreeMap, BTreeSet};

use super::{GLWELinearTransformIndex, LinearTransformationStrategy};

/// Normalizes a diagonal index modulo the number of slots.
pub fn normalize_linear_transform_diagonal(diagonal: i64, slots: usize) -> usize {
    assert!(slots > 0, "linear transformation slot count must be non-zero");
    diagonal.rem_euclid(slots as i64) as usize
}

/// Returns a BSGS schedule for the provided non-zero diagonal indexes.
pub fn bsgs_index<I>(diagonal_indexes: I, slots: usize, giant_step: usize) -> GLWELinearTransformIndex
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

    GLWELinearTransformIndex {
        baby_steps,
        giant_steps,
        index,
    }
}

/// Returns the BSGS giant-step width selected by the Lattigo-compatible cost rule.
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
) -> GLWELinearTransformIndex
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
                bsgs_index(diagonal_indexes, slots, giant_step)
            }
        }
        LinearTransformationStrategy::Bsgs { giant_step } => bsgs_index(diagonal_indexes, slots, giant_step),
        LinearTransformationStrategy::Direct => {
            let diagonals: BTreeSet<usize> = diagonal_indexes
                .into_iter()
                .map(|diagonal| normalize_linear_transform_diagonal(diagonal, slots))
                .collect();
            GLWELinearTransformIndex {
                baby_steps: vec![0],
                giant_steps: diagonals.iter().map(|&diagonal| diagonal as i64).collect(),
                index: diagonals.into_iter().map(|_| vec![0]).collect(),
            }
        }
    }
}
