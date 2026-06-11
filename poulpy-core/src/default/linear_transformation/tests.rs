#[cfg(test)]
use super::{LinearTransformationLayout, LinearTransformationStrategy, optimal_bsgs_giant_step};
#[cfg(test)]
use crate::layouts::{linear_transformation_plan, normalize_linear_transform_diagonal};

#[test]
fn normalizes_diagonal_indexes() {
    assert_eq!(normalize_linear_transform_diagonal(-1, 8), 7);
    assert_eq!(normalize_linear_transform_diagonal(10, 8), 2);
}

#[test]
fn builds_stable_sparse_linear_transformation_schedule() {
    let index = linear_transformation_plan([5, 3, 13, -1], 8, 3);
    assert_eq!(index.baby_steps, vec![0, 1, 2]);
    assert_eq!(index.giant_steps.len(), 2);
    assert_eq!(index.giant_steps, vec![3, 6]);
    assert_eq!(index.index, vec![vec![0, 2], vec![1]]);
    assert_eq!(index.galois_elements(32), vec![5, 9, 25, 29]);
}

#[test]
fn builds_dense_linear_transformation_schedule() {
    let index = linear_transformation_plan(0..8, 8, 3);
    assert_eq!(index.baby_steps, vec![0, 1, 2]);
    assert_eq!(index.giant_steps, vec![0, 3, 6]);
    assert_eq!(index.index, vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1]]);
}

#[test]
fn builds_single_diagonal_linear_transformation_schedule() {
    let index = linear_transformation_plan([6], 8, 3);
    assert_eq!(index.baby_steps, vec![0]);
    assert_eq!(index.giant_steps.len(), 1);
    assert_eq!(index.giant_steps, vec![6]);
    assert_eq!(index.index, vec![vec![0]]);
    assert_eq!(index.galois_elements(32), vec![9]);
}

#[test]
fn derives_direct_index() {
    let index = LinearTransformationLayout {
        indexes: vec![2, 5],
        slots: 8,
        strategy: LinearTransformationStrategy::Direct,
    }
    .index();
    assert_eq!(index.baby_steps, vec![0]);
    assert_eq!(index.giant_steps, vec![2, 5]);
    assert_eq!(index.index, vec![vec![0], vec![0]]);
}

#[test]
fn auto_uses_direct_for_tiny_sparse_index() {
    let auto = LinearTransformationLayout {
        indexes: vec![2, 5],
        slots: 8,
        strategy: LinearTransformationStrategy::Auto,
    }
    .index();
    let direct = LinearTransformationLayout {
        indexes: vec![2, 5],
        slots: 8,
        strategy: LinearTransformationStrategy::Direct,
    }
    .index();
    assert_eq!(auto, direct);
}

#[test]
fn auto_uses_optimal_linear_transformation_schedule() {
    let diagonals = [0, 1, 2, 3, 6, 7];
    let giant_step = optimal_bsgs_giant_step(diagonals, 8);
    let auto = LinearTransformationLayout {
        indexes: diagonals.to_vec(),
        slots: 8,
        strategy: LinearTransformationStrategy::Auto,
    }
    .index();
    let explicit = linear_transformation_plan(diagonals, 8, giant_step);
    assert_eq!(auto, explicit);
}
