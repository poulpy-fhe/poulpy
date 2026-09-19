#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    SetGaloisElement, prepared::GLWEAutomorphismKeyPreparedBackendRef,
};

pub trait GLWEAutomorphism<BE: Backend> {
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_add<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_add_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_sub<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_sub_negate<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_automorphism_sub_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    fn glwe_automorphism_sub_negate_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

pub trait GLWEAutomorphismKeyAutomorphism<BE: Backend> {
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos;
}
