use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    ScratchTakeCore,
    automorphism::{GGSWAutomorphismDefault, GLWEAutomorphismDefault, GLWEAutomorphismKeyAutomorphismDefault},
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToMut, GGSWToRef,
        GLWEInfos, GLWEToMut, GLWEToRef, GetGaloisElement, SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait CoreAutomorphismDefaults<BE: Backend>: Backend {
    fn glwe_automorphism_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn ggsw_automorphism_tmp_bytes_default<R, A, K, T>(
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

    fn ggsw_automorphism_default<R, A, K, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        A: GGSWToRef + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_automorphism_assign_default<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_automorphism_key_automorphism_tmp_bytes_default<R, A, K>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism_default<R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign_default<R, K>(
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;
}

impl<BE: Backend> CoreAutomorphismDefaults<BE> for BE
where
    Module<BE>: GLWEAutomorphismDefault<BE> + GGSWAutomorphismDefault<BE> + GLWEAutomorphismKeyAutomorphismDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_automorphism_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn glwe_automorphism_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_default(module, res, a, key, scratch)
    }

    fn glwe_automorphism_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_assign_default(module, res, key, scratch)
    }

    fn glwe_automorphism_add_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_add_default(module, res, a, key, scratch)
    }

    fn glwe_automorphism_add_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_add_assign_default(module, res, key, scratch)
    }

    fn glwe_automorphism_sub_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_sub_default(module, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_negate_default<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_sub_negate_default(module, res, a, key, scratch)
    }

    fn glwe_automorphism_sub_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_sub_assign_default(module, res, key, scratch)
    }

    fn glwe_automorphism_sub_negate_assign_default<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismDefault<BE>>::glwe_automorphism_sub_negate_assign_default(module, res, key, scratch)
    }

    fn ggsw_automorphism_tmp_bytes_default<R, A, K, T>(
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
        <Module<BE> as GGSWAutomorphismDefault<BE>>::ggsw_automorphism_tmp_bytes_default(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    fn ggsw_automorphism_default<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut + GGSWInfos,
        A: GGSWToRef + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWAutomorphismDefault<BE>>::ggsw_automorphism_default(module, res, a, key, tsk, scratch)
    }

    fn ggsw_automorphism_assign_default<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWAutomorphismDefault<BE>>::ggsw_automorphism_assign_default(module, res, key, tsk, scratch)
    }

    fn glwe_automorphism_key_automorphism_tmp_bytes_default<R, A, K>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyAutomorphismDefault<BE>>::glwe_automorphism_key_automorphism_tmp_bytes_default(
            module, res_infos, a_infos, key_infos,
        )
    }

    fn glwe_automorphism_key_automorphism_default<R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyAutomorphismDefault<BE>>::glwe_automorphism_key_automorphism_default(
            module, res, a, key, scratch,
        )
    }

    fn glwe_automorphism_key_automorphism_assign_default<R, K>(
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        <Module<BE> as GLWEAutomorphismKeyAutomorphismDefault<BE>>::glwe_automorphism_key_automorphism_assign_default(
            module, res, key, scratch,
        )
    }
}

#[macro_export]
macro_rules! impl_core_automorphism_default_methods {
    ($be:ty) => {
        fn glwe_automorphism_tmp_bytes<R, A, K>(
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
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_tmp_bytes_default(
                module, res_infos, a_infos, key_infos,
            )
        }

        fn glwe_automorphism<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_default(module, res, a, key, scratch)
        }

        fn glwe_automorphism_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_assign_default(module, res, key, scratch)
        }

        fn glwe_automorphism_add<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_add_default(module, res, a, key, scratch)
        }

        fn glwe_automorphism_add_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_add_assign_default(module, res, key, scratch)
        }

        fn glwe_automorphism_sub<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_sub_default(module, res, a, key, scratch)
        }

        fn glwe_automorphism_sub_negate<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_sub_negate_default(
                module, res, a, key, scratch,
            )
        }

        fn glwe_automorphism_sub_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_sub_assign_default(module, res, key, scratch)
        }

        fn glwe_automorphism_sub_negate_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_sub_negate_assign_default(
                module, res, key, scratch,
            )
        }

        fn ggsw_automorphism_tmp_bytes<R, A, K, T>(
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
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::ggsw_automorphism_tmp_bytes_default(
                module, res_infos, a_infos, key_infos, tsk_infos,
            )
        }

        fn ggsw_automorphism<R, A, K, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut + $crate::layouts::GGSWInfos,
            A: $crate::layouts::GGSWToRef + $crate::layouts::GGSWInfos,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::ggsw_automorphism_default(module, res, a, key, tsk, scratch)
        }

        fn ggsw_automorphism_assign<R, K, T>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            tsk: &T,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            K: $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GGLWEInfos,
            T: $crate::layouts::GGLWEToGGSWKeyPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::ggsw_automorphism_assign_default(module, res, key, tsk, scratch)
        }

        fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(
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
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_key_automorphism_tmp_bytes_default(
                module, res_infos, a_infos, key_infos,
            )
        }

        fn glwe_automorphism_key_automorphism<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
            A: $crate::layouts::GGLWEToRef + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_key_automorphism_default(
                module, res, a, key, scratch,
            )
        }

        fn glwe_automorphism_key_automorphism_assign<R, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            key: &K,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut
                + $crate::layouts::SetGaloisElement
                + $crate::layouts::GetGaloisElement
                + $crate::layouts::GGLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreAutomorphismDefaults<$be>>::glwe_automorphism_key_automorphism_assign_default(
                module, res, key, scratch,
            )
        }
    };
}
