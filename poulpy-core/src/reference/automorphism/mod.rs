pub mod gglwe;
pub mod glwe;

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    SetGaloisElement, prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEAutomorphismReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_add_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_add_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_sub_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_sub_negate_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_sub_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_sub_negate_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_automorphism_many_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_many_reference<R, A>(
        &self,
        res: &mut [&mut R],
        a: &A,
        keys: &[&GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;
}

impl<BE: Backend, Rg: Ring> GLWEAutomorphismReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    BE: crate::oep::GGLWEProductDigitsStridedImpl<Rg>,
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEAutomorphism<BE, Rg>
        + crate::api::GLWEKeyswitch<BE, Rg>
        + poulpy_hal::api::VecZnxAutomorphismAssignTmpBytes
        + poulpy_hal::api::VecZnxAutomorphismAssign<BE, Rg>
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE, Rg>
        + crate::api::GLWENormalize<BE, Rg>
        + poulpy_hal::api::VecZnxBigAutomorphismAssign<BE, Rg>
        + poulpy_hal::api::VecZnxBigAutomorphismAssignTmpBytes
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE, Rg>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxIdftApply<BE, Rg>
        + poulpy_hal::api::VecZnxBigSubSmallAssign<BE, Rg>
        + poulpy_hal::api::VecZnxBigSubSmallNegateAssign<BE, Rg>
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes,
{
    fn glwe_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_tmp_bytes_reference::<BE, _, _, _, _, Rg>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_reference::<BE, _, _, _, Rg>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_assign_reference::<BE, _, _, Rg>(self, res, key, scratch)
    }

    fn glwe_automorphism_add_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_add_reference::<BE, _, _, _, Rg>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_add_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_add_assign_reference::<BE, _, _, Rg>(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_reference::<BE, _, _, _, Rg>(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_negate_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_reference::<BE, _, _, _, Rg>(
            self, res, a, key, scratch,
        )
    }

    fn glwe_automorphism_sub_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_assign_reference::<BE, _, _, Rg>(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_negate_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_assign_reference::<BE, _, _, Rg>(
            self, res, key, scratch,
        )
    }

    fn glwe_automorphism_many_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_many_tmp_bytes_reference::<BE, _, _, _, Rg>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_many_reference<R, A>(
        &self,
        res: &mut [&mut R],
        a: &A,
        keys: &[&crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>],
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::automorphism::glwe::glwe_automorphism_many_reference::<BE, _, _, Rg>(self, res, a, keys, scratch)
    }
}

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEAutomorphismKeyAutomorphismReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_automorphism_key_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE, Rg> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + SetGaloisElement + GetGaloisElement + GGLWEInfos;
}

impl<BE: Backend, Rg: Ring> GLWEAutomorphismKeyAutomorphismReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEKeyswitch<BE, Rg>
        + poulpy_hal::api::VecZnxAutomorphismAssignTmpBytes
        + poulpy_hal::layouts::GaloisElement
        + poulpy_hal::api::VecZnxAutomorphism<BE, Rg>
        + poulpy_hal::api::VecZnxAutomorphismAssign<BE, Rg>
        + poulpy_hal::layouts::CyclotomicOrder,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GGLWEInfos,
        A: crate::layouts::GGLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_tmp_bytes_reference::<BE, _, _, _, _, Rg>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_key_automorphism_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GGLWEToBackendMut<BE, Rg> + crate::layouts::SetGaloisElement + crate::layouts::GGLWEInfos,
        A: crate::layouts::GGLWEToBackendRef<BE, Rg> + crate::layouts::GetGaloisElement + crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_reference::<BE, _, _, _, Rg>(
            self, res, a, key, scratch,
        )
    }

    fn glwe_automorphism_key_automorphism_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GGLWEToBackendMut<BE, Rg>
            + crate::layouts::SetGaloisElement
            + crate::layouts::GetGaloisElement
            + crate::layouts::GGLWEInfos,
    {
        crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_assign_reference::<BE, _, _, Rg>(
            self, res, key, scratch,
        )
    }
}
