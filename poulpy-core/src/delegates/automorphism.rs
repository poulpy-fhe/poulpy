use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GGSWAutomorphism, GLWEAutomorphism, GLWEAutomorphismKeyAutomorphism},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos,
        GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, SetGaloisElement,
        prepared::{GGLWEToGGSWKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
    oep::AutomorphismImpl,
};

macro_rules! impl_automorphism_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_automorphism_delegate!(
    GLWEAutomorphism<BE>,
    [BE: Backend + AutomorphismImpl<BE>],
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism<R, A>(&self, res: &mut R, a: &A, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_automorphism(self, res, a, key, scratch)
    }

    fn glwe_automorphism_assign<R>(&self, res: &mut R, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_add<R, A>(&self, res: &mut R, a: &A, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_add(self, res, a, key, scratch)
    }

    fn glwe_automorphism_add_assign<R>(&self, res: &mut R, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_add_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_sub<R, A>(&self, res: &mut R, a: &A, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_sub(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_negate<R, A>(&self, res: &mut R, a: &A, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_sub_negate(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_assign<R>(&self, res: &mut R, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_sub_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_negate_assign<R>(&self, res: &mut R, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_automorphism_sub_negate_assign(self, res, key, scratch)
    }
);

impl_automorphism_delegate!(
    GGSWAutomorphism<BE>,
    [BE: Backend + AutomorphismImpl<BE> + crate::oep::ConversionImpl<BE>, Module<BE>: crate::oep::ConversionDefault<BE>],
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ggsw_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_automorphism<R, A>(&self, res: &mut R, a: &A, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
    {
        BE::ggsw_automorphism(self, res, a, key, tsk, scratch)
    }

    fn ggsw_automorphism_assign<R>(&self, res: &mut R, key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>, tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        BE::ggsw_automorphism_assign(self, res, key, tsk, scratch)
    }
);

impl_automorphism_delegate!(
    GLWEAutomorphismKeyAutomorphism<BE>,
    [BE: Backend + AutomorphismImpl<BE>],
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism_key_automorphism<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism(self, res, a, key, scratch)
    }

    fn glwe_automorphism_key_automorphism_assign<R>(
        &self,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism_assign(self, res, key, scratch)
    }
);
