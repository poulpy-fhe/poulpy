#[cfg(test)]
use super::{
    LinearTransformationStrategy, bsgs_index, linear_transform_index, normalize_linear_transform_diagonal,
    optimal_bsgs_giant_step,
};

#[test]
fn normalizes_diagonal_indexes() {
    assert_eq!(normalize_linear_transform_diagonal(-1, 8), 7);
    assert_eq!(normalize_linear_transform_diagonal(10, 8), 2);
}

#[test]
fn builds_stable_sparse_bsgs_index() {
    let index = bsgs_index([5, 3, 13, -1], 8, 3);
    assert_eq!(index.baby_steps, vec![0, 1, 2]);
    assert_eq!(index.giant_steps.len(), 2);
    assert_eq!(index.giant_steps, vec![3, 6]);
    assert_eq!(index.index, vec![vec![0, 2], vec![1]]);
    assert_eq!(index.required_rotations(), vec![1, 2, 3, 6]);
}

#[test]
fn builds_dense_bsgs_index() {
    let index = bsgs_index(0..8, 8, 3);
    assert_eq!(index.baby_steps, vec![0, 1, 2]);
    assert_eq!(index.giant_steps, vec![0, 3, 6]);
    assert_eq!(index.index, vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1]]);
}

#[test]
fn builds_single_diagonal_bsgs_index() {
    let index = bsgs_index([6], 8, 3);
    assert_eq!(index.baby_steps, vec![0]);
    assert_eq!(index.giant_steps.len(), 1);
    assert_eq!(index.giant_steps, vec![6]);
    assert_eq!(index.index, vec![vec![0]]);
    assert_eq!(index.required_rotations(), vec![6]);
}

#[test]
fn derives_direct_index() {
    let index = linear_transform_index([2, 5], 8, LinearTransformationStrategy::Direct);
    assert_eq!(index.baby_steps, vec![0]);
    assert_eq!(index.giant_steps, vec![2, 5]);
    assert_eq!(index.index, vec![vec![0], vec![0]]);
}

#[test]
fn auto_uses_direct_for_tiny_sparse_index() {
    let auto = linear_transform_index([2, 5], 8, LinearTransformationStrategy::Auto);
    let direct = linear_transform_index([2, 5], 8, LinearTransformationStrategy::Direct);
    assert_eq!(auto, direct);
}

#[test]
fn auto_uses_optimal_bsgs_index() {
    let diagonals = [0, 1, 2, 3, 6, 7];
    let giant_step = optimal_bsgs_giant_step(diagonals, 8);
    let auto = linear_transform_index(diagonals, 8, LinearTransformationStrategy::Auto);
    let explicit = bsgs_index(diagonals, 8, giant_step);
    assert_eq!(auto, explicit);
}
