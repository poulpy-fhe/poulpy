use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, VecZnxDftBackendMut};

use crate::{
    api::{GGLWEKeyswitch, GGSWKeyswitch, GLWEKeyswitch, LWEKeyswitch},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
        prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
    },
    oep::{GGLWEKeyswitchImpl, GGSWKeyswitchImpl, GLWEKeyswitchImpl, LWEKeyswitchImpl},
};

macro_rules! impl_keyswitching_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_keyswitching_delegate!(
    GLWEKeyswitch<BE, Rg>,
    [BE: Backend + GLWEKeyswitchImpl<Rg>],
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_keyswitch(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R>(&self, res: &mut R, key: &GGLWEPreparedBackendRef<'_, BE, Rg>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
    {
        BE::glwe_keyswitch_assign(self, res, key, scratch)
    }

    fn glwe_keyswitch_dft_tmp_bytes<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_keyswitch_dft_tmp_bytes(self, res_size, a_infos, key_infos)
    }

    fn glwe_keyswitch_dft<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_keyswitch_dft(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_dft_add_tmp_bytes<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_keyswitch_dft_add_tmp_bytes(self, res_size, a_infos, key_infos)
    }

    fn glwe_keyswitch_dft_add<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_keyswitch_dft_add(self, res, a, key, scratch)
    }
);

impl_keyswitching_delegate!(
    GGLWEKeyswitch<BE, Rg>,
    [BE: Backend + GGLWEKeyswitchImpl<Rg>],
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::gglwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        A: GGLWEToBackendRef<BE, Rg> + GGLWEInfos,
    {
        BE::gglwe_keyswitch(self, res, a, b, scratch)
    }

    fn gglwe_keyswitch_assign<R>(&self, res: &mut R, a: &GGLWEPreparedBackendRef<'_, BE, Rg>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
    {
        BE::gglwe_keyswitch_assign(self, res, a, scratch)
    }
);

impl_keyswitching_delegate!(
    GGSWKeyswitch<BE, Rg>,
    [BE: Backend + GGSWKeyswitchImpl<Rg>],
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        BE::ggsw_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE, Rg> + GGSWInfos,
        A: GGSWToBackendRef<BE, Rg> + GGSWInfos,
    {
        BE::ggsw_keyswitch(self, res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_assign<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE, Rg> + GGSWInfos,
    {
        BE::ggsw_keyswitch_assign(self, res, key, tsk, scratch)
    }
);

impl_keyswitching_delegate!(
    LWEKeyswitch<BE, Rg>,
    [BE: Backend + LWEKeyswitchImpl<Rg>],
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_keyswitch_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE, Rg> + LWEInfos,
        A: LWEToBackendRef<BE, Rg> + LWEInfos,
    {
        BE::lwe_keyswitch(self, res, a, ksk, scratch)
    }
);
