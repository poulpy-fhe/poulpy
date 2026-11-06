use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWESwitchingKeyDegreesMut, SetGaloisElement};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    layouts::{MHEGLWEAutomorphismKeyShareOwned, MHEGLWESwitchingKeyShareOwned},
    oep::MHEGGLWEPatCompressedImpl,
};

pub(crate) fn mhe_glwe_switching_key_share_aggregate_derived<BE: MHEGGLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut MHEGLWESwitchingKeyShareOwned<BE>,
    a: &MHEGLWESwitchingKeyShareOwned<BE>,
) {
    assert!(
        res.input_degree == a.input_degree && res.output_degree == a.output_degree,
        "invalid aggregation: degrees differ"
    );
    BE::mhe_gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_switching_key_share_finalize_derived<BE: MHEGGLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &MHEGLWESwitchingKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
{
    BE::mhe_gglwe_pat_compressed_finalize(module, res, &share.key, scratch);
    *res.input_degree() = share.input_degree;
    *res.output_degree() = share.output_degree;
}

pub(crate) fn mhe_glwe_automorphism_key_share_aggregate_derived<BE: MHEGGLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut MHEGLWEAutomorphismKeyShareOwned<BE>,
    a: &MHEGLWEAutomorphismKeyShareOwned<BE>,
) {
    assert!(res.p == a.p, "invalid aggregation: Galois elements differ");
    BE::mhe_gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_automorphism_key_share_finalize_derived<BE: MHEGGLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &MHEGLWEAutomorphismKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
{
    BE::mhe_gglwe_pat_compressed_finalize(module, res, &share.key, scratch);
    res.set_p(share.p);
}
