use poulpy_core::{
    Distribution, GetDistributionMut,
    layouts::{GLWECompressedSeed, GLWECompressedToBackendRef, GLWEInfos, GLWEPublicKeyAtViewMut},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::oep::GLWEPatCompressedImpl;

pub(crate) fn glwe_public_key_finalize_tmp_bytes_derived<BE: GLWEPatCompressedImpl>(module: &Module<BE>) -> usize {
    BE::glwe_pat_compressed_finalize_tmp_bytes(module)
}

pub(crate) fn glwe_public_key_finalize_derived<BE: GLWEPatCompressedImpl, R, P>(
    module: &Module<BE>,
    res: &mut R,
    pats: &[P],
    dist: Distribution,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos,
    P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
{
    assert!(
        !matches!(dist, Distribution::NONE | Distribution::ENCAPSULATED(_)),
        "invalid distribution: a public key needs a samplable distribution"
    );
    assert!(
        pats.iter()
            .enumerate()
            .all(|(i, pat)| pats[..i].iter().all(|other| other.seed() != pat.seed())),
        "invalid finalization: public key entries share a seed"
    );
    assert!(
        pats.len() == res.rank().as_usize(),
        "invalid finalization: one share per public key entry"
    );
    for (l, pat) in pats.iter().enumerate() {
        BE::glwe_pat_compressed_finalize(module, &mut res.at_view_mut(l), pat, scratch);
    }
    *res.dist_mut() = dist;
}
