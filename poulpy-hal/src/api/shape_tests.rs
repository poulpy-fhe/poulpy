//! Compile-time shape tests for the preserved normalization API family.
//!
//! Per `docs/spec/normalization_typestate.md` (PR 0), the existing normalization entry
//! points must keep their exact call shapes through the typestate migration. The generic
//! functions below are never called; they exist so that any change to a preserved
//! signature fails to compile here before it reaches a caller. Update them only together
//! with an amendment to the spec's §5.2.

#![allow(dead_code)]

use crate::{
    api::{
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes,
        VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeCoeffAssignBackend, VecZnxNormalizeCoeffBackend,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, NormalizationState, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendRef, VecZnxDftBackendMut,
    },
};

fn shape_vec_znx_normalize_tmp_bytes<M: VecZnxNormalizeTmpBytes>(m: &M) -> usize {
    m.vec_znx_normalize_tmp_bytes()
}

fn shape_vec_znx_normalize<B: Backend, M: VecZnxNormalize<B>, S1: NormalizationState, S2: NormalizationState>(
    m: &M,
    res: &mut VecZnxBackendMut<'_, B, S1>,
    a: &VecZnxBackendRef<'_, B, S2>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_normalize(res, 17, -3, 0, a, 18, 1, scratch);
}

fn shape_vec_znx_normalize_assign<B: Backend, M: VecZnxNormalizeAssignBackend<B>, S: NormalizationState>(
    m: &M,
    a: &mut VecZnxBackendMut<'_, B, S>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_normalize_assign_backend(17, a, 0, scratch);
}

fn shape_vec_znx_normalize_coeff<
    B: Backend,
    M: VecZnxNormalizeCoeffBackend<B>,
    S1: NormalizationState,
    S2: NormalizationState,
>(
    m: &M,
    res: &mut VecZnxBackendMut<'_, B, S1>,
    a: &VecZnxBackendRef<'_, B, S2>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_normalize_coeff_backend(res, 17, -3, 0, a, 18, 1, 2, scratch);
}

fn shape_vec_znx_normalize_coeff_assign<B: Backend, M: VecZnxNormalizeCoeffAssignBackend<B>, S: NormalizationState>(
    m: &M,
    a: &mut VecZnxBackendMut<'_, B, S>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_normalize_coeff_assign_backend(17, a, 0, 2, scratch);
}

fn shape_vec_znx_big_normalize_tmp_bytes<M: VecZnxBigNormalizeTmpBytes>(m: &M) -> usize {
    m.vec_znx_big_normalize_tmp_bytes()
}

fn shape_vec_znx_big_normalize<B: Backend, M: VecZnxBigNormalize<B>, S: NormalizationState>(
    m: &M,
    res: &mut VecZnxBackendMut<'_, B, S>,
    a: &VecZnxBigBackendRef<'_, B>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_big_normalize(res, 17, -3, 0, a, 18, 1, scratch);
}

fn shape_vec_znx_idft_normalize_consume_tmp_bytes<M: VecZnxIdftNormalizeConsumeTmpBytes>(m: &M) -> usize {
    m.vec_znx_idft_normalize_consume_tmp_bytes(4, 5)
}

fn shape_vec_znx_idft_normalize_consume<
    B: Backend,
    M: VecZnxIdftNormalizeConsume<B>,
    S1: NormalizationState,
    S2: NormalizationState,
>(
    m: &M,
    res: &mut VecZnxBackendMut<'_, B, S1>,
    a: &mut VecZnxDftBackendMut<'_, B>,
    addend: &VecZnxBackendRef<'_, B, S2>,
    scratch: &mut ScratchArena<'_, B>,
) {
    m.vec_znx_idft_normalize_consume(res, 17, 0, a, 1, 18, Some((addend, 0)), scratch);
}
