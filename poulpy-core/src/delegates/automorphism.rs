use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    api::{GGSWAutomorphism, GLWEAutomorphism, GLWEAutomorphismKeyAutomorphism},
    automorphism::GGSWAutomorphismDefault,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToMut, GGSWToRef,
        GLWEInfos, GLWEToMut, GLWEToRef, GetGaloisElement, SetGaloisElement,
    },
    oep::CoreImpl,
};

impl<BE> GLWEAutomorphism<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism(self, res, a, key, scratch)
    }

    fn glwe_automorphism_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_add<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_add(self, res, a, key, scratch)
    }

    fn glwe_automorphism_add_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_add_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_sub<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_sub(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_negate<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_sub_negate(self, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_sub_assign(self, res, key, scratch)
    }

    fn glwe_automorphism_sub_negate_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_automorphism_sub_negate_assign(self, res, key, scratch)
    }
}

impl<BE> GGSWAutomorphism<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGSWAutomorphismDefault<BE>,
{
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ggsw_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_automorphism<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut + GGSWInfos,
        A: GGSWToRef + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_automorphism(self, res, a, key, tsk, scratch)
    }

    fn ggsw_automorphism_assign<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_automorphism_assign(self, res, key, tsk, scratch)
    }
}

impl<BE> GLWEAutomorphismKeyAutomorphism<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism_key_automorphism<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism(self, res, a, key, scratch)
    }

    fn glwe_automorphism_key_automorphism_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        BE::glwe_automorphism_key_automorphism_assign(self, res, key, scratch)
    }
}
