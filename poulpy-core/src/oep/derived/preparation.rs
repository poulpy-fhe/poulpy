//! Preparations of key wrappers composed from GGLWE and GLWE switching-key preparation, and the
//! zeroing of a prepared GGSW through the selected `vmp_zero`.
use poulpy_hal::layouts::{Data, Module, ScratchArena};

use crate::{
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGSWPreparedToBackendMut, GLWECIKey, GLWECIKeyPrepared,
        GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GLWESwitchingKeyPrepared, GetGaloisElement,
        SetGaloisElement,
    },
    oep::PreparationImpl,
};

pub(crate) fn glwe_automorphism_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_automorphism_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE> + SetGaloisElement,
    O: GGLWEToBackendRef<BE> + GetGaloisElement,
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

pub(crate) fn glwe_switching_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_switching_key_prepare_derived<BE, R, O>(
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

pub(crate) fn glwe_tensor_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::gglwe_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_tensor_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE>,
    O: GGLWEToBackendRef<BE>,
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

pub(crate) fn glwe_ci_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::glwe_switching_key_prepare_tmp_bytes(module, infos)
}

pub(crate) fn glwe_ci_key_prepare_derived<BE, DR, DO, M>(
    module: &Module<BE>,
    res: &mut GLWECIKeyPrepared<DR, BE, M>,
    other: &GLWECIKey<DO, BE::ZnxWord, M>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    DR: Data,
    DO: Data,
    GLWESwitchingKeyPrepared<DR, BE>: GGLWEPreparedToBackendMut<BE>,
    GLWESwitchingKey<DO, BE::ZnxWord>: GGLWEToBackendRef<BE>,
{
    BE::glwe_switching_key_prepare(module, &mut res.0, &other.0, scratch);
}

pub(crate) fn ggsw_zero_derived<BE, R>(module: &Module<BE>, res: &mut R)
where
    BE: PreparationImpl,
    R: GGSWPreparedToBackendMut<BE>,
{
    res.set_noise(None);
    let mut res = res.to_backend_mut();
    BE::vmp_zero(module, &mut res.data);
}
