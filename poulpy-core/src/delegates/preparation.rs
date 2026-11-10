use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GGLWEPreparedFactory, GGLWEToGGSWKeyPreparedFactory, GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory,
        GLWEPreparedFactory, GLWEPublicKeyPreparedFactory, GLWESwitchingKeyPreparedFactory, GLWETensorKeyPreparedFactory,
        GLWEToLWEKeyPreparedFactory, LWESwitchingKeyPreparedFactory, LWEToGLWEKeyPreparedFactory,
    },
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef,
        GGSWInfos, GGSWPreparedToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEPreparedToBackendMut,
        GLWEPublicKeyPreparedToBackendMut, GLWEPublicKeyToBackendRef, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
        GLWEToBackendRef, GetGaloisElement, SetGaloisElement,
    },
    oep::{LWEKeyPreparationImpl, PreparationImpl},
};

macro_rules! impl_preparation_delegate {
    (free $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + LWEKeyPreparationImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            BE: Backend + PreparationImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_preparation_delegate!(
    GLWEPreparedFactory<BE, Rg>,
    fn glwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_prepare_tmp_bytes(self, infos)
    },
    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPreparedToBackendMut<BE, Rg>,
        O: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GGLWEPreparedFactory<BE, Rg>,
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_prepare_tmp_bytes(self, infos)
    },
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE, Rg>,
        O: GGLWEToBackendRef<BE, Rg>,
    {
        BE::gglwe_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GGSWPreparedFactory<BE, Rg>,
    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_prepare_tmp_bytes(self, infos)
    },
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWPreparedToBackendMut<BE, Rg>,
        O: GGSWToBackendRef<BE, Rg>,
    {
        BE::ggsw_prepare(self, res, other, scratch)
    },
    fn ggsw_prepared_zero<R>(&self, res: &mut R)
    where
        R: GGSWPreparedToBackendMut<BE, Rg>,
    {
        BE::ggsw_prepared_zero(self, res)
    }
);

impl_preparation_delegate!(
    GLWEPublicKeyPreparedFactory<BE, Rg>,
    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<BE, Rg> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<BE, Rg> + GetDistribution,
    {
        BE::glwe_public_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWEAutomorphismKeyPreparedFactory<BE, Rg>,
    fn glwe_automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE, Rg> + SetGaloisElement,
        O: GGLWEToBackendRef<BE, Rg> + GetGaloisElement,
    {
        BE::glwe_automorphism_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWESwitchingKeyPreparedFactory<BE, Rg>,
    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE, Rg> + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    GLWETensorKeyPreparedFactory<BE, Rg>,
    fn glwe_tensor_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_tensor_key_prepare_tmp_bytes(self, infos)
    },
    fn glwe_tensor_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE, Rg>,
        O: GGLWEToBackendRef<BE, Rg>,
    {
        BE::glwe_tensor_key_prepare(self, res, other, scratch)
    }
);

impl_preparation_delegate!(
    free GLWEToLWEKeyPreparedFactory<BE>,
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
    free LWESwitchingKeyPreparedFactory<BE>,
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
    free LWEToGLWEKeyPreparedFactory<BE>,
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
    GGLWEToGGSWKeyPreparedFactory<BE, Rg>,
    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_prepare_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE, Rg>,
        O: GGLWEToGGSWKeyToBackendRef<BE, Rg>,
    {
        BE::gglwe_to_ggsw_key_prepare(self, res, other, scratch)
    }
);
