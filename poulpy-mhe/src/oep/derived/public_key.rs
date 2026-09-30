use poulpy_core::{
    Distribution, GetDistributionMut,
    layouts::{GLWECompressedSeed, GLWEInfos, GLWEPublicKeyAtViewMut},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{layouts::GLWEPublicKeyShareOwned, oep::GLWEPatCompressedImpl};

pub(crate) fn mhe_glwe_public_key_share_aggregate_derived<BE: GLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut GLWEPublicKeyShareOwned<BE>,
    a: &GLWEPublicKeyShareOwned<BE>,
) {
    assert!(res.dist == a.dist, "invalid aggregation: secret distributions differ");
    assert!(res.entries.len() == a.entries.len(), "invalid aggregation: ranks differ");
    for (res, a) in res.entries.iter_mut().zip(&a.entries) {
        BE::glwe_pat_compressed_aggregate_assign(module, res, a);
    }
}

pub(crate) fn mhe_glwe_public_key_share_finalize_derived<BE: GLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &GLWEPublicKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos,
{
    let (entries, dist) = (&share.entries, share.dist);
    assert!(
        !matches!(dist, Distribution::NONE | Distribution::ENCAPSULATED(_)),
        "invalid distribution: a public key needs a samplable distribution"
    );
    assert!(
        entries
            .iter()
            .enumerate()
            .all(|(i, entry)| entries[..i].iter().all(|other| other.seed() != entry.seed())),
        "invalid finalization: public key entries share a seed"
    );
    assert!(
        entries.len() == res.rank().as_usize(),
        "invalid finalization: one share per public key entry"
    );
    for (l, entry) in entries.iter().enumerate() {
        BE::glwe_pat_compressed_finalize(module, &mut res.at_view_mut(l), entry, scratch);
    }
    *res.dist_mut() = dist;
}
