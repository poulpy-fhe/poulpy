#[cfg(test)]
use super::{LinearTransformationLayout, LinearTransformationStrategy, optimal_bsgs_giant_step};
#[cfg(test)]
use crate::layouts::{galois_of_rotation, linear_transformation_plan};

/// The Galois elements of the slot rotations `rotations`.
#[cfg(test)]
fn rot(rotations: impl IntoIterator<Item = i64>) -> Vec<i64> {
    rotations.into_iter().map(galois_of_rotation).collect()
}

#[test]
fn builds_stable_sparse_linear_transformation_schedule() {
    let index = linear_transformation_plan(rot([5, 3, 13, -1]), 8, 3);
    assert_eq!(index.baby_galois, rot([0, 1, 2]));
    assert_eq!(index.giant_galois, rot([3, 6]));
    assert_eq!(index.index, vec![rot([0, 2]), rot([1])]);
    assert_eq!(index.galois_elements(32), vec![5, 9, 25, 29]);
}

#[test]
fn builds_dense_linear_transformation_schedule() {
    let index = linear_transformation_plan(rot(0..8), 8, 3);
    assert_eq!(index.baby_galois, rot([0, 1, 2]));
    assert_eq!(index.giant_galois, rot([0, 3, 6]));
    assert_eq!(index.index, vec![rot([0, 1, 2]), rot([0, 1, 2]), rot([0, 1])]);
}

#[test]
fn builds_single_diagonal_linear_transformation_schedule() {
    let index = linear_transformation_plan(rot([6]), 8, 3);
    assert_eq!(index.baby_galois, rot([0]));
    assert_eq!(index.giant_galois, rot([6]));
    assert_eq!(index.index, vec![rot([0])]);
    assert_eq!(index.galois_elements(32), vec![9]);
}

#[test]
fn derives_direct_index() {
    let index = LinearTransformationLayout {
        galois: rot([2, 5]),
        slots: 8,
        strategy: LinearTransformationStrategy::Direct,
    }
    .index();
    assert_eq!(index.baby_galois, rot([0]));
    assert_eq!(index.giant_galois, rot([2, 5]));
    assert_eq!(index.index, vec![rot([0]), rot([0])]);
}

#[test]
fn optimal_bsgs_giant_step_matches_explicit_schedule() {
    // The heuristic width produces the same schedule as spelling it out.
    let diagonals = rot([0, 1, 2, 3, 6, 7]);
    let giant_step = optimal_bsgs_giant_step(diagonals.iter().copied(), 8);
    let from_width = LinearTransformationLayout {
        galois: diagonals.clone(),
        slots: 8,
        strategy: LinearTransformationStrategy::Bsgs { giant_step },
    }
    .index();
    let explicit = linear_transformation_plan(diagonals, 8, giant_step);
    assert_eq!(from_width, explicit);
}
