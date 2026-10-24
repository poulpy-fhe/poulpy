use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEPreparedFactory, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedFactory,
        GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef, GGSWInfos, GGSWPreparedFactory, GGSWPreparedToBackendMut,
        GGSWToBackendRef, GLWEAutomorphismKeyPreparedFactory, GLWECIKey, GLWECIKeyPrepared, GLWECIKeyPreparedFactory, GLWEInfos,
        GLWEPreparedFactory, GLWEPreparedToBackendMut, GLWEPublicKeyPreparedFactory, GLWEPublicKeyPreparedToBackendMut,
        GLWEPublicKeyToBackendRef, GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
        GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory, GLWEToBackendRef,
        GLWEToLWEKeyPreparedFactory, GetGaloisElement, LWESwitchingKeyPreparedFactory, LWEToGLWEKeyPreparedFactory,
        SetGaloisElement,
    },
    oep::PreparationImpl,
};

macro_rules! impl_preparation_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + PreparationImpl,
        {
            $($body)+
        }
    };
}

impl_preparation_delegate!(
    GLWEPreparedFactory<BE>,
    fn glwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_prepare_tmp_bytes(self, infos)
    },
    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPreparedToBackendMut<BE>,
        O: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GGLWEPreparedFactory<BE>,
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_prepare_tmp_bytes(self, infos)
    },
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>,
    {
        BE::gglwe_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GGSWPreparedFactory<BE>,
    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_prepare_tmp_bytes(self, infos)
    },
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWPreparedToBackendMut<BE>,
        O: GGSWToBackendRef<BE>,
    {
        BE::ggsw_prepare(self, res, other, scratch)
    },
    fn ggsw_zero<R>(&self, res: &mut R)
    where
        R: GGSWPreparedToBackendMut<BE>,
    {
        BE::ggsw_zero(self, res)
    }
);

impl_preparation_delegate!(
    GLWEPublicKeyPreparedFactory<BE>,
    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<BE> + GetDistribution,
    {
        BE::glwe_public_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWEAutomorphismKeyPreparedFactory<BE>,
    fn glwe_automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + SetGaloisElement,
        O: GGLWEToBackendRef<BE> + GetGaloisElement,
    {
        BE::glwe_automorphism_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWESwitchingKeyPreparedFactory<BE>,
    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWETensorKeyPreparedFactory<BE>,
    fn prepare_tensor_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::prepare_tensor_key_tmp_bytes(self, infos)
    },
    fn prepare_tensor_key<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>,
    {
        BE::prepare_tensor_key(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWEToLWEKeyPreparedFactory<BE>,
    fn glwe_to_lwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_to_lwe_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_to_lwe_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
    {
        BE::glwe_to_lwe_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    LWESwitchingKeyPreparedFactory<BE>,
    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_switching_key_prepare_tmp_bytes(self, infos)
    },
    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
    {
        BE::lwe_switching_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    LWEToGLWEKeyPreparedFactory<BE>,
    fn lwe_to_glwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_to_glwe_key_prepare_tmp_bytes(self, infos)
    },
    fn lwe_to_glwe_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees,
    {
        BE::lwe_to_glwe_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GGLWEToGGSWKeyPreparedFactory<BE>,
    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_prepare_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>,
    {
        BE::gglwe_to_ggsw_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWECIKeyPreparedFactory<BE>,
    fn glwe_ci_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_ci_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_ci_key_prepare<DR, DO, M>(
        &self,
        res: &mut GLWECIKeyPrepared<DR, BE, M>,
        other: &GLWECIKey<DO, BE::ZnxWord, M>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        DR: Data,
        DO: Data,
        GLWESwitchingKeyPrepared<DR, BE>: GGLWEPreparedToBackendMut<BE>,
        GLWESwitchingKey<DO, BE::ZnxWord>: GGLWEToBackendRef<BE>,
    {
        BE::glwe_ci_key_prepare(self, res, other, scratch)
    }
);
