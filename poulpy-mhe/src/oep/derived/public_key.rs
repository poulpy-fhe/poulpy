use poulpy_core::{
    Distribution, GetDistributionMut,
    layouts::{GLWEInfos, GLWEToBackendMut},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{layouts::GLWEPatCompressedOwned, oep::PatFinalizeImpl};

pub(crate) fn glwe_public_key_finalize_tmp_bytes_derived<BE: PatFinalizeImpl>(module: &Module<BE>) -> usize {
    BE::pat_finalize_tmp_bytes(module)
}

pub(crate) fn glwe_public_key_finalize_derived<BE: PatFinalizeImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    pat: &GLWEPatCompressedOwned<BE>,
    dist: Distribution,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    assert!(
        !matches!(dist, Distribution::NONE | Distribution::ENCAPSULATED(_)),
        "invalid distribution: a public key needs a samplable distribution"
    );
    BE::glwe_pat_compressed_finalize(module, res, pat, scratch);
    *res.dist_mut() = dist;
}
