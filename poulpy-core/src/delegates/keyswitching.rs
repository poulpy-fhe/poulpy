use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GLWEKeyswitch, LWEKeyswitch},
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
        prepared::GGLWEPreparedBackendRef,
    },
    oep::{GLWEKeyswitchImpl, LWEKeyswitchImpl},
};

macro_rules! impl_keyswitching_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_keyswitching_delegate!(
    GLWEKeyswitch<BE>,
    [BE: Backend + GLWEKeyswitchImpl],
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<R, A>(&self, res: &mut R, a: &A, key: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_keyswitch(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R>(&self, res: &mut R, key: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::glwe_keyswitch_assign(self, res, key, scratch)
    }
);

impl_keyswitching_delegate!(
    LWEKeyswitch<BE>,
    [BE: Backend + LWEKeyswitchImpl],
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A>(&self, res: &mut R, a: &A, ksk: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
    {
        BE::lwe_keyswitch(self, res, a, ksk, scratch)
    }
);
