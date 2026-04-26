use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    api::{GGLWEKeyswitch, GGSWKeyswitch, GLWEKeyswitch, LWEKeySwitch},
    keyswitching::{GGLWEKeyswitchDefault, GGSWKeyswitchDefault, LWEKeySwitchDefault},
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToMut, GGSWToRef,
        GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
    },
    oep::CoreImpl,
};

impl<BE> GLWEKeyswitch<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_keyswitch(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_keyswitch_assign(self, res, key, scratch)
    }
}

impl<BE> GGLWEKeyswitch<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGLWEKeyswitchDefault<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::gglwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::gglwe_keyswitch(self, res, a, b, scratch)
    }

    fn gglwe_keyswitch_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::gglwe_keyswitch_assign(self, res, a, scratch)
    }
}

impl<BE> GGSWKeyswitch<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGSWKeyswitchDefault<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ggsw_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_keyswitch<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_keyswitch(self, res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_assign<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_keyswitch_assign(self, res, key, tsk, scratch)
    }
}

impl<BE> LWEKeySwitch<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: LWEKeySwitchDefault<BE>,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::lwe_keyswitch(self, res, a, ksk, scratch)
    }
}
