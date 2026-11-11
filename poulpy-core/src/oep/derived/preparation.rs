//! Preparations of key wrappers composed from GGLWE and GLWE switching-key preparation, and the
//! zeroing of a prepared GGSW through the selected `vmp_zero`.
use poulpy_hal::layouts::{Module, Ring, ScratchArena};

use crate::{
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGSWPreparedToBackendMut, GLWESwitchingKeyDegrees,
        GLWESwitchingKeyDegreesMut, GetGaloisElement, SetGaloisElement,
    },
    oep::PreparationImpl,
};

pub(crate) fn glwe_automorphism_key_prepare_tmp_bytes_derived<BE, A, Rg: Ring>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl<Rg>,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_automorphism_key_prepare_derived<BE, R, O, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl<Rg>,
    R: GGLWEPreparedToBackendMut<BE, Rg> + SetGaloisElement,
    O: GGLWEToBackendRef<BE, Rg> + GetGaloisElement,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        glwe_automorphism_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEAutomorphismKeyPreparedFactory::glwe_automorphism_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::gglwe_prepare(module, res, other, scratch);
    res.set_p(other.p());
}

pub(crate) fn glwe_switching_key_prepare_tmp_bytes_derived<BE, A, Rg: Ring>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl<Rg>,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_switching_key_prepare_derived<BE, R, O, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl<Rg>,
    R: GGLWEPreparedToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut,
    O: GGLWEToBackendRef<BE, Rg> + GLWESwitchingKeyDegrees,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        glwe_switching_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWESwitchingKeyPreparedFactory::glwe_switching_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::gglwe_prepare(module, res, other, scratch);
    *res.input_degree() = *other.input_degree();
    *res.output_degree() = *other.output_degree();
}

pub(crate) fn glwe_tensor_key_prepare_tmp_bytes_derived<BE, A, Rg: Ring>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl<Rg>,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_tensor_key_prepare_derived<BE, R, O, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl<Rg>,
    R: GGLWEPreparedToBackendMut<BE, Rg>,
    O: GGLWEToBackendRef<BE, Rg>,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        glwe_tensor_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWETensorKeyPreparedFactory::glwe_tensor_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::gglwe_prepare(module, res, other, scratch);
}

pub(crate) fn glwe_to_lwe_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::glwe_switching_key_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_to_lwe_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
    O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        glwe_to_lwe_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEToLWEKeyPreparedFactory::glwe_to_lwe_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::glwe_switching_key_prepare(module, res, other, scratch);
}

pub(crate) fn lwe_switching_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::glwe_switching_key_prepare_tmp_bytes(module, infos)
}

pub(crate) fn lwe_switching_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
    O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        lwe_switching_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < LWESwitchingKeyPreparedFactory::lwe_switching_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::glwe_switching_key_prepare(module, res, other, scratch);
}

pub(crate) fn lwe_to_glwe_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::glwe_switching_key_prepare_tmp_bytes(module, infos)
}

pub(crate) fn lwe_to_glwe_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
    O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
{
    let tmp_bytes = {
        let res_infos = res.to_backend_mut();
        lwe_to_glwe_key_prepare_tmp_bytes_derived(module, &res_infos)
    };
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < LWEToGLWEKeyPreparedFactory::lwe_to_glwe_key_prepare_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    BE::glwe_switching_key_prepare(module, res, other, scratch);
}

pub(crate) fn ggsw_prepared_zero_derived<BE, R, Rg: Ring>(module: &Module<BE>, res: &mut R)
where
    BE: PreparationImpl<Rg>,
    R: GGSWPreparedToBackendMut<BE, Rg>,
{
    res.set_noise(None);
    let mut res = res.to_backend_mut();
    BE::vmp_zero(module, &mut res.data);
}
