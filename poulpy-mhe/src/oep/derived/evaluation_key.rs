use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWESwitchingKeyDegreesMut, SetGaloisElement};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    layouts::{GLWEAutomorphismKeyShareOwned, GLWESwitchingKeyShareOwned},
    oep::GGLWEPatCompressedImpl,
};

pub(crate) fn mhe_glwe_switching_key_aggregate_derived<BE: GGLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut GLWESwitchingKeyShareOwned<BE>,
    a: &GLWESwitchingKeyShareOwned<BE>,
) {
    assert!(
        res.input_degree == a.input_degree && res.output_degree == a.output_degree,
        "invalid aggregation: degrees differ"
    );
    BE::gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_switching_key_finalize_derived<BE: GGLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &GLWESwitchingKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
{
    BE::gglwe_pat_compressed_finalize(module, res, &share.key, scratch);
    *res.input_degree() = share.input_degree;
    *res.output_degree() = share.output_degree;
}

pub(crate) fn mhe_glwe_automorphism_key_aggregate_derived<BE: GGLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut GLWEAutomorphismKeyShareOwned<BE>,
    a: &GLWEAutomorphismKeyShareOwned<BE>,
) {
    assert!(res.p == a.p, "invalid aggregation: Galois elements differ");
    BE::gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_automorphism_key_finalize_derived<BE: GGLWEPatCompressedImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &GLWEAutomorphismKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
{
    BE::gglwe_pat_compressed_finalize(module, res, &share.key, scratch);
    res.set_p(share.p);
}
