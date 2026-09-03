//! Compile-time shape tests for the preserved GLWE normalization API.
//!
//! Per `docs/spec/normalization_typestate.md` (PR 0), `GLWENormalize` must keep its
//! exact call shape through the typestate migration, apart from new state bounds. These
//! generic functions are never called; any signature change fails to compile here first.
//! Update them only together with an amendment to the spec's §5.2.

#![allow(dead_code)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    api::GLWENormalize,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};

fn shape_glwe_normalize_tmp_bytes<BE: Backend, M: GLWENormalize<BE>>(m: &M) -> usize {
    m.glwe_normalize_tmp_bytes()
}

fn shape_glwe_normalize<BE: Backend, M: GLWENormalize<BE>, R, A>(m: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    R: GLWEToBackendMut<BE>,
    A: GLWEToBackendRef<BE>,
{
    m.glwe_normalize(res, a, scratch);
}

fn shape_glwe_normalize_assign<BE: Backend, M: GLWENormalize<BE>, R>(m: &M, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
where
    R: GLWEToBackendMut<BE>,
{
    m.glwe_normalize_assign(res, scratch);
}
