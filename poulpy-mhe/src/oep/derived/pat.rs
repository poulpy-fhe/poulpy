use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GetGaloisElement, SetGaloisElement,
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::oep::GGLWEPatCompressedImpl;

pub(crate) fn glwe_switching_key_pat_compressed_aggregate_assign_derived<BE: GGLWEPatCompressedImpl, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
) where
    R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos + GLWESwitchingKeyDegrees,
    A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    assert!(
        res.input_degree() == a.input_degree() && res.output_degree() == a.output_degree(),
        "invalid aggregation: degrees differ"
    );
    BE::gglwe_pat_compressed_aggregate_assign(module, res, a);
}

pub(crate) fn glwe_switching_key_pat_compressed_finalize_derived<BE: GGLWEPatCompressedImpl, R, P>(
    module: &Module<BE>,
    res: &mut R,
    pat: &P,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    P: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::gglwe_pat_compressed_finalize(module, res, pat, scratch);
    *res.input_degree() = *pat.input_degree();
    *res.output_degree() = *pat.output_degree();
}

pub(crate) fn glwe_automorphism_key_pat_compressed_aggregate_assign_derived<BE: GGLWEPatCompressedImpl, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
) where
    R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeed + GGLWEInfos + GetGaloisElement,
    A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed + GGLWEInfos + GetGaloisElement,
{
    assert!(res.p() == a.p(), "invalid aggregation: Galois elements differ");
    BE::gglwe_pat_compressed_aggregate_assign(module, res, a);
}

pub(crate) fn glwe_automorphism_key_pat_compressed_finalize_derived<BE: GGLWEPatCompressedImpl, R, P>(
    module: &Module<BE>,
    res: &mut R,
    pat: &P,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    P: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
{
    BE::gglwe_pat_compressed_finalize(module, res, pat, scratch);
    res.set_p(pat.p());
}
