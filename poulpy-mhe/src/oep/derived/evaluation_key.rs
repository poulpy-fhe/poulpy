use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWESwitchingKeyDegreesMut, SetGaloisElement};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned},
    oep::{PatAggregateImpl, PatFinalizeImpl, PatNormalizeImpl},
};

pub(crate) fn glwe_switching_key_share_aggregate_assign_derived<BE: PatAggregateImpl>(
    module: &Module<BE>,
    res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
    a: &GLWESwitchingKeyPatCompressedOwned<BE>,
) {
    assert!(
        res.input_degree == a.input_degree && res.output_degree == a.output_degree,
        "invalid aggregation: degrees differ"
    );
    BE::gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn glwe_switching_key_share_normalize_tmp_bytes_derived<BE: PatNormalizeImpl>(module: &Module<BE>) -> usize {
    BE::pat_normalize_tmp_bytes(module)
}

pub(crate) fn glwe_switching_key_share_normalize_assign_derived<BE: PatNormalizeImpl>(
    module: &Module<BE>,
    res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) {
    BE::gglwe_pat_compressed_normalize_assign(module, &mut res.key, scratch);
}

pub(crate) fn glwe_switching_key_finalize_tmp_bytes_derived<BE: PatFinalizeImpl>(module: &Module<BE>) -> usize {
    BE::pat_finalize_tmp_bytes(module)
}

pub(crate) fn glwe_switching_key_finalize_derived<BE: PatFinalizeImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    pat: &GLWESwitchingKeyPatCompressedOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
{
    BE::gglwe_pat_compressed_finalize(module, res, &pat.key, scratch);
    *res.input_degree() = pat.input_degree;
    *res.output_degree() = pat.output_degree;
}

pub(crate) fn glwe_automorphism_key_share_aggregate_assign_derived<BE: PatAggregateImpl>(
    module: &Module<BE>,
    res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
    a: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
) {
    assert!(res.p == a.p, "invalid aggregation: Galois elements differ");
    BE::gglwe_pat_compressed_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn glwe_automorphism_key_share_normalize_tmp_bytes_derived<BE: PatNormalizeImpl>(module: &Module<BE>) -> usize {
    BE::pat_normalize_tmp_bytes(module)
}

pub(crate) fn glwe_automorphism_key_share_normalize_assign_derived<BE: PatNormalizeImpl>(
    module: &Module<BE>,
    res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) {
    BE::gglwe_pat_compressed_normalize_assign(module, &mut res.key, scratch);
}

pub(crate) fn glwe_automorphism_key_finalize_tmp_bytes_derived<BE: PatFinalizeImpl>(module: &Module<BE>) -> usize {
    BE::pat_finalize_tmp_bytes(module)
}

pub(crate) fn glwe_automorphism_key_finalize_derived<BE: PatFinalizeImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    pat: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
{
    BE::gglwe_pat_compressed_finalize(module, res, &pat.key, scratch);
    res.set_p(pat.p);
}
