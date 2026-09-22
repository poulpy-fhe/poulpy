pub mod glwe;
pub mod lwe;

pub(crate) use glwe::*;

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::GGLWEPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEKeyswitchReference<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEKeyswitchReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalizeConsumeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalizeConsume<BE>
        + poulpy_hal::api::VecZnxNormalize<BE>
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxNormalizeAssign<BE>,
{
    fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_tmp_bytes_reference::<BE, _, _, _, _>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_reference::<BE, _, _, _>(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_assign_reference::<BE, _, _>(self, res, key, scratch)
    }
}

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait LWEKeyswitchReference<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;
}

impl<BE: Backend> LWEKeyswitchReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEKeyswitch<BE>
        + poulpy_hal::api::VecZnxCopy<BE>
        + poulpy_hal::api::VecZnxZero<BE>,
{
    fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::LWEInfos,
        A: crate::layouts::LWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::lwe::lwe_keyswitch_tmp_bytes_reference::<BE, _, _, _, _>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn lwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::LWEToBackendMut<BE> + crate::layouts::LWEInfos,
        A: crate::layouts::LWEToBackendRef<BE> + crate::layouts::LWEInfos,
    {
        crate::reference::keyswitching::lwe::lwe_keyswitch_reference::<BE, _, _, _>(self, res, a, ksk, scratch)
    }
}
