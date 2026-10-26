//! Preparations of stored matrices through the selected `vmp_prepare`, of key wrappers composed from GGLWE
//! and GLWE switching-key preparation, and the zeroing of a prepared GGSW through the selected `vmp_zero`.
use poulpy_hal::layouts::{Data, Module, ScratchArena};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef,
        GGSWInfos, GGSWPreparedToBackendMut, GGSWToBackendRef, GLWECIKey, GLWECIKeyPrepared, GLWEInfos,
        GLWEPublicKeyPreparedToBackendMut, GLWEPublicKeyToBackendRef, GLWESwitchingKey, GLWESwitchingKeyDegrees,
        GLWESwitchingKeyDegreesMut, GLWESwitchingKeyPrepared, GetGaloisElement, LWEInfos, SetGaloisElement, operand_degree,
    },
    oep::PreparationImpl,
};

pub(crate) fn gglwe_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    BE::vmp_prepare_tmp_bytes(
        module,
        infos.dnum().into(),
        infos.rank_in().into(),
        (infos.rank() + 1).into(),
        infos.size(),
    )
}

pub(crate) fn gglwe_prepare_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
where
    BE: PreparationImpl,
    R: GGLWEPreparedToBackendMut<BE>,
    O: GGLWEToBackendRef<BE>,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        operand_degree(module.n(), &[res.n(), other.n()]);
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.size(), other.size());
        assert_eq!(res.k_digit, other.k_digit);
        assert_eq!(res.k_offset, other.k_offset, "gglwe_prepare: row offsets differ");
        let tmp_bytes: usize = gglwe_prepare_tmp_bytes_derived(module, &res);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GGLWEPreparedFactory::gglwe_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        BE::vmp_prepare(module, &mut res.data, &other.data, scratch);
    }
    res.set_noise(noise);
}

pub(crate) fn ggsw_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGSWInfos,
{
    operand_degree(module.n(), &[infos.n()]);
    BE::vmp_prepare_tmp_bytes(
        module,
        infos.dnum().into(),
        (infos.rank() + 1).into(),
        (infos.rank() + 1).into(),
        infos.size(),
    )
}

pub(crate) fn ggsw_prepare_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
where
    BE: PreparationImpl,
    R: GGSWPreparedToBackendMut<BE>,
    O: GGSWToBackendRef<BE>,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        operand_degree(module.n(), &[res.n(), other.n()]);
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.k_digit, other.k_digit);
        let tmp_bytes: usize = ggsw_prepare_tmp_bytes_derived(module, &res);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GGSWPreparedFactory::ggsw_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        BE::vmp_prepare(module, &mut res.data, &other.data, scratch);
    }
    res.set_noise(noise);
}

pub(crate) fn glwe_public_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GLWEInfos,
{
    let rank: usize = infos.rank().into();
    BE::vmp_prepare_tmp_bytes(module, 1, rank, rank + 1, infos.size())
}

pub(crate) fn glwe_public_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GLWEPublicKeyPreparedToBackendMut<BE> + GetDistributionMut,
    O: GLWEPublicKeyToBackendRef<BE> + GetDistribution,
{
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert!(
            res.data.cols_in() == other.data.cols_in(),
            "glwe_public_key_prepare: public key and prepared public key have different entry counts"
        );
        operand_degree(module.n(), &[res.n(), other.n()]);
        assert_eq!(res.base2k(), other.base2k());
        assert_eq!(res.k(), other.k());
        assert_eq!(res.size(), other.size());
        BE::vmp_prepare(module, &mut res.data, &other.data, scratch);
    }
    res.set_noise(other.to_backend_ref().noise());
    *res.dist_mut() = *other.dist();
}

pub(crate) fn gglwe_to_ggsw_key_prepare_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: PreparationImpl,
    A: GGLWEInfos,
{
    gglwe_prepare_tmp_bytes_derived(module, infos)
}

pub(crate) fn gglwe_to_ggsw_key_prepare_derived<BE, R, O>(
    module: &Module<BE>,
    res: &mut R,
    other: &O,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: PreparationImpl,
    R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
    O: GGLWEToGGSWKeyToBackendRef<BE>,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert_eq!(res.keys.len(), other.keys.len());
        let tmp_bytes: usize = gglwe_to_ggsw_key_prepare_tmp_bytes_derived(module, &res);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GGLWEToGGSWKeyPreparedFactory::gglwe_to_ggsw_key_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            BE::vmp_prepare(module, &mut a.data, &b.data, &mut scratch.borrow());
        }
    }
    res.set_noise(noise);
}

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
