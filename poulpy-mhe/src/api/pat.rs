//! One trait per PAT type, each with the same operations:
//!
//! - Aggregation, `*_aggregate_assign`: `res += a`. `res` and `a` must share
//!   their layout and, for seeded PATs, their seeds. The sum is not normalized.
//!   The accumulator starts from the first share, a clone of it or a
//!   `read_from` into it: a freshly allocated PAT has a zero seed and
//!   aggregating into it panics on the seed check.
//! - Finalization, `*_finalize`: expansion of an aggregated PAT into the
//!   ciphertext it transcribes, the only normalization. `res` must have the
//!   PAT's layout. The PAT is left unchanged and `res` is canonical.
use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Operations on a [`GLWEPatCompressed`](crate::layouts::GLWEPatCompressed).
pub trait GLWEPatCompressedOps<BE: Backend> {
    fn glwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos;

    fn glwe_pat_compressed_finalize_tmp_bytes(&self) -> usize;

    fn glwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos;
}

/// Operations on a [`GGLWEPatCompressed`](crate::layouts::GGLWEPatCompressed).
pub trait GGLWEPatCompressedOps<BE: Backend> {
    fn gglwe_pat_compressed_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos;

    fn gglwe_pat_compressed_finalize_tmp_bytes(&self) -> usize;

    fn gglwe_pat_compressed_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<BE> + GGLWEInfos;
}

/// Operations on a [`GGLWEPat`](crate::layouts::GGLWEPat).
pub trait GGLWEPatOps<BE: Backend> {
    fn gglwe_pat_aggregate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn gglwe_pat_finalize_tmp_bytes(&self) -> usize;

    fn gglwe_pat_finalize<R, P>(&self, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        P: GGLWEToBackendRef<BE> + GGLWEInfos;
}
