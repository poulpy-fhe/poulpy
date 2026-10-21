use poulpy_core::{
    Distribution, GetDistribution, GetDistributionMut,
    layouts::{
        GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyCompressedSeed, GLWEPublicKeyCompressedToBackendMut,
        GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut, LWEInfos,
    },
};
use poulpy_hal::{
    layouts::{MatZnxToBackendMut, MatZnxToBackendRef, Module, ScratchArena},
    oep::HalMatZnxImpl,
};

use crate::{layouts::GLWEPublicKeyShareOwned, oep::GLWEPatCompressedImpl};

pub(crate) fn mhe_glwe_public_key_share_aggregate_derived<BE: HalMatZnxImpl>(
    module: &Module<BE>,
    res: &mut GLWEPublicKeyShareOwned<BE>,
    a: &GLWEPublicKeyShareOwned<BE>,
) {
    assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
    assert!(res.dist() == a.dist(), "invalid aggregation: secret distributions differ");
    assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
    let metadata = crate::reference::aggregate_metadata(res.noise(), a.noise());
    BE::mat_znx_add_assign(
        module,
        &mut MatZnxToBackendMut::<BE>::to_backend_mut(res.key.data_mut()),
        &MatZnxToBackendRef::<BE>::to_backend_ref(a.key.data()),
    );
    GLWEPublicKeyCompressedToBackendMut::<BE>::set_noise(res, metadata);
}

pub(crate) fn mhe_glwe_public_key_share_finalize_derived<BE: GLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &GLWEPublicKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEPublicKeyAtViewMut<BE> + GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    let (seeds, dist) = (share.seed(), *share.dist());
    assert!(
        !matches!(dist, Distribution::NONE | Distribution::ENCAPSULATED(_)),
        "invalid distribution: a public key needs a samplable distribution"
    );
    assert!(
        seeds.iter().enumerate().all(|(i, seed)| !seeds[..i].contains(seed)),
        "invalid finalization: public key entries share a seed"
    );
    assert!(
        res.glwe_layout() == share.glwe_layout(),
        "invalid finalization: layouts differ"
    );
    let share = share.key.to_backend_ref();
    for l in 0..share.rank().as_usize() {
        BE::glwe_pat_compressed_finalize(module, &mut res.at_view_mut(l), &share.at_view(l), scratch);
    }
    *res.dist_mut() = dist;
    res.set_noise(share.noise());
}
