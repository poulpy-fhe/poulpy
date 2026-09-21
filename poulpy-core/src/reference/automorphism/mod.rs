pub mod gglwe;
pub mod glwe;

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    SetGaloisElement, prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEAutomorphismReference<BE: Backend> {
    fn glwe_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_add_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_add_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_sub_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_sub_negate_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_sub_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_sub_negate_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEKeyswitch<BE>
        + poulpy_hal::api::VecZnxAutomorphismAssignTmpBytes
        + poulpy_hal::api::VecZnxAutomorphismAssign<BE>
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::VecZnxBigAutomorphismAssign<BE>
        + poulpy_hal::api::VecZnxBigAutomorphismAssignTmpBytes
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxBigSubSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigSubSmallNegateAssign<BE>
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalizeConsumeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxNormalizeTmpBytes,
{
    fn glwe_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_tmp_bytes_reference::<BE, _, _, _, _>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_reference::<BE, _, _, _>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_assign_reference::<BE, _, _>(self, res, key, scratch)
    }

    fn glwe_automorphism_add_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_add_reference::<BE, _, _, _>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_add_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_add_assign_reference::<BE, _, _>(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_reference::<BE, _, _, _>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_negate_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_reference::<BE, _, _, _>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_assign_reference::<BE, _, _>(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_negate_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_assign_reference::<BE, _, _>(self, res, key, scratch)
    }
}

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GGLWEAutomorphismReference<BE: Backend> {
    fn glwe_automorphism_key_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos;
}

impl<BE: Backend> GGLWEAutomorphismReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEKeyswitch<BE>
        + poulpy_hal::api::VecZnxAutomorphismAssignTmpBytes
        + poulpy_hal::layouts::GaloisElement
        + poulpy_hal::api::VecZnxAutomorphism<BE>
        + poulpy_hal::api::VecZnxAutomorphismAssign<BE>
        + poulpy_hal::layouts::CyclotomicOrder,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GGLWEInfos,
        A: crate::layouts::GGLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_tmp_bytes_reference::<BE, _, _, _, _>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_key_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GGLWEToBackendMut<BE> + crate::layouts::SetGaloisElement + crate::layouts::GGLWEInfos,
        A: crate::layouts::GGLWEToBackendRef<BE> + crate::layouts::GetGaloisElement + crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_reference::<BE, _, _, _>(
            self, res, a, key, scratch,
        )
    }

    fn glwe_automorphism_key_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GGLWEToBackendMut<BE>
            + crate::layouts::SetGaloisElement
            + crate::layouts::GetGaloisElement
            + crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_assign_reference::<BE, _, _>(
            self, res, key, scratch,
        )
    }
}
