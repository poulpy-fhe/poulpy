use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEToBackendMut};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressedOwned};

/// Aggregation of public aggregatable transcripts: `res += a`.
///
/// `res` and `a` must share their layout and, for seeded PATs, their seeds.
/// The sum is not normalized: `res` is flagged non-canonical.
///
/// The accumulator starts from the first share, a clone of it or a `read_from`
/// into it: a freshly allocated PAT has a zero seed and aggregating into it
/// panics on the seed check.
pub trait PatAggregate<BE: Backend> {
    fn glwe_pat_compressed_aggregate_assign(&self, res: &mut GLWEPatCompressedOwned<BE>, a: &GLWEPatCompressedOwned<BE>);

    fn gglwe_pat_compressed_aggregate_assign(&self, res: &mut GGLWEPatCompressedOwned<BE>, a: &GGLWEPatCompressedOwned<BE>);

    fn gglwe_pat_aggregate_assign(&self, res: &mut GGLWEPatOwned<BE>, a: &GGLWEPatOwned<BE>);
}

/// In-place normalization of a PAT, a no-op on a canonical one.
pub trait PatNormalize<BE: Backend> {
    fn pat_normalize_tmp_bytes(&self) -> usize;

    fn glwe_pat_compressed_normalize_assign(&self, res: &mut GLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>);

    fn gglwe_pat_compressed_normalize_assign(&self, res: &mut GGLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>);

    fn gglwe_pat_normalize_assign(&self, res: &mut GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>);
}

/// Expansion of an aggregated PAT into the ciphertext it transcribes.
///
/// `res` must have the PAT's layout. The PAT is left unchanged and `res` is
/// canonical.
pub trait PatFinalize<BE: Backend> {
    fn pat_finalize_tmp_bytes(&self) -> usize;

    fn glwe_pat_compressed_finalize<R>(&self, res: &mut R, pat: &GLWEPatCompressedOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn gglwe_pat_compressed_finalize<R>(
        &self,
        res: &mut R,
        pat: &GGLWEPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;

    fn gglwe_pat_finalize<R>(&self, res: &mut R, pat: &GGLWEPatOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}
