use poulpy_core::{
    Distribution, GetDistribution, GetDistributionMut,
    layouts::{
        GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyCompressedSeed, GLWEPublicKeyCompressedToBackendMut,
        GLWEPublicKeyToBackendMut, LWEInfos,
    },
    oep::EncryptionImpl,
};
use poulpy_hal::{
    layouts::{MatZnxToBackendMut, MatZnxToBackendRef, Module, ScratchArena},
    oep::HalMatZnxImpl,
};

use crate::layouts::MHEGLWEPublicKeyShareOwned;

pub(crate) fn mhe_glwe_public_key_share_aggregate_derived<BE: HalMatZnxImpl>(
    module: &Module<BE>,
    res: &mut MHEGLWEPublicKeyShareOwned<BE>,
    a: &MHEGLWEPublicKeyShareOwned<BE>,
) {
    assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
    assert!(res.dist() == a.dist(), "invalid aggregation: secret distributions differ");
    assert!(res.seed() == a.seed(), "invalid aggregation: seeds differ");
    let metadata = crate::metadata::aggregate_metadata(res.noise(), a.noise());
    BE::mat_znx_add_assign(
        module,
        &mut MatZnxToBackendMut::<BE>::to_backend_mut(res.key.data_mut()),
        &MatZnxToBackendRef::<BE>::to_backend_ref(a.key.data()),
    );
    GLWEPublicKeyCompressedToBackendMut::<BE>::set_noise(res, metadata);
}

pub(crate) fn mhe_glwe_public_key_share_finalize_derived<BE: HalMatZnxImpl + EncryptionImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &MHEGLWEPublicKeyShareOwned<BE>,
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
    assert!(
        scratch.available() >= BE::mat_znx_normalize_tmp_bytes(module),
        "invalid finalization: insufficient scratch"
    );
    let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().into());
    BE::mat_znx_normalize(
        module,
        res.to_backend_mut().data_mut(),
        base2k,
        k,
        0,
        0,
        &MatZnxToBackendRef::<BE>::to_backend_ref(share.key.data()),
        base2k,
        0,
        scratch,
    );
    // Seeded masks are uniform digits, already canonical.
    for (l, seed) in seeds.iter().enumerate() {
        BE::glwe_fill_mask_from_seed(module, &mut res.at_view_mut(l), *seed);
    }
    *res.dist_mut() = dist;
    res.set_noise(share.noise());
}
