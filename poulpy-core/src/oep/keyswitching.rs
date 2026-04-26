use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    ScratchTakeCore,
    keyswitching::{GGLWEKeyswitchDefault, GGSWKeyswitchDefault, GLWEKeyswitchDefault, LWEKeySwitchDefault},
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToMut, GGSWToRef,
        GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
    },
};

#[doc(hidden)]
pub trait CoreKeyswitchDefaults<BE: Backend>: Backend {
    fn glwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn gglwe_keyswitch_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch_default<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_keyswitch_assign_default<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn lwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CoreKeyswitchDefaults<BE> for BE
where
    Module<BE>: GLWEKeyswitchDefault<BE> + GGLWEKeyswitchDefault<BE> + GGSWKeyswitchDefault<BE> + LWEKeySwitchDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_default(module, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_assign_default(module, res, key, scratch)
    }

    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_default(module, res, a, key, scratch)
    }

    fn gglwe_keyswitch_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_assign_default(module, res, key, scratch)
    }

    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_tmp_bytes_default(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    fn ggsw_keyswitch_default<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_default(module, res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_assign_default<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_assign_default(module, res, key, tsk, scratch)
    }

    fn lwe_keyswitch_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as LWEKeySwitchDefault<BE>>::lwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as LWEKeySwitchDefault<BE>>::lwe_keyswitch_default(module, res, a, ksk, scratch)
    }
}

#[macro_export]
macro_rules! impl_core_keyswitch_default_methods {
    ($be:ty) => {
        fn glwe_keyswitch_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::glwe_keyswitch_tmp_bytes_default(
                module, res_infos, a_infos, key_infos,
            )
        }

        fn glwe_keyswitch<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::glwe_keyswitch_default(module, res, a, key, scratch)
        }

        fn glwe_keyswitch_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::glwe_keyswitch_assign_default(module, res, key, scratch)
        }

        fn gglwe_keyswitch_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::GGLWEInfos,
            A: $crate::layouts::GGLWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::gglwe_keyswitch_tmp_bytes_default(
                module, res_infos, a_infos, key_infos,
            )
        }

        fn gglwe_keyswitch<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GGLWEInfos,
            A: $crate::layouts::GGLWEToRef + $crate::layouts::GGLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::gglwe_keyswitch_default(module, res, a, key, scratch)
        }

        fn gglwe_keyswitch_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::gglwe_keyswitch_assign_default(module, res, key, scratch)
        }

        fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            key_infos: &K,
            tsk_infos: &T,
        ) -> usize
        where
            R: $crate::layouts::GGSWInfos,
            A: $crate::layouts::GGSWInfos,
            K: $crate::layouts::GGLWEInfos,
            T: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::ggsw_keyswitch_tmp_bytes_default(
                module, res_infos, a_infos, key_infos, tsk_infos,
            )
        }

        fn ggsw_keyswitch<R, A, K, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            A: $crate::layouts::GGSWToRef,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::ggsw_keyswitch_default(module, res, a, key, tsk, scratch)
        }

        fn ggsw_keyswitch_assign<R, K, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::ggsw_keyswitch_assign_default(module, res, key, tsk, scratch)
        }

        fn lwe_keyswitch_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::LWEInfos,
            A: $crate::layouts::LWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::lwe_keyswitch_tmp_bytes_default(
                module, res_infos, a_infos, key_infos,
            )
        }

        fn lwe_keyswitch<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            ksk: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::LWEToMut,
            A: $crate::layouts::LWEToRef,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreKeyswitchDefaults<$be>>::lwe_keyswitch_default(module, res, a, ksk, scratch)
        }
    };
}
