#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
};

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<'s, R, A, K>(&self, res: &mut R, a: &A, key: &K, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_keyswitch_assign<'s, R, K>(&self, res: &mut R, key: &K, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

pub trait GGLWEKeyswitch<BE: Backend> {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn gglwe_keyswitch_assign<'s, R, A>(&self, res: &mut R, a: &A, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

pub trait GGSWKeyswitch<BE: Backend> {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<'s, R, A, K, T>(
        &self,
        res: &mut R,
        a: &A,
        key: &K,
        key_size: usize,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_keyswitch_assign<'s, R, K, T>(
        &self,
        res: &mut R,
        key: &K,
        key_size: usize,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

pub trait LWEKeyswitch<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<'s, R, A, K>(&self, res: &mut R, a: &A, ksk: &K, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}
