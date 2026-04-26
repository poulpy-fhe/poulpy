use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    ScratchTakeCore,
    external_product::{GGLWEExternalProductDefault, GGSWExternalProductDefault, GLWEExternalProductDefault},
    layouts::{
        GGLWEInfos, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWPreparedToRef, GGSWToMut, GGSWToRef, GLWEInfos, GLWEToMut, GLWEToRef,
    },
};

#[doc(hidden)]
pub trait CoreExternalProductDefaults<BE: Backend>: Backend {
    fn glwe_external_product_tmp_bytes_default<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_default<R, A, G>(module: &Module<BE>, res: &mut R, a: &A, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos;

    fn glwe_external_product_assign_default<R, G>(module: &Module<BE>, res: &mut R, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos;

    fn gglwe_external_product_tmp_bytes_default<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product_default<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn gglwe_external_product_assign_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_external_product_tmp_bytes_default<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product_default<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_external_product_assign_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CoreExternalProductDefaults<BE> for BE
where
    Module<BE>: GLWEExternalProductDefault<BE> + GGLWEExternalProductDefault<BE> + GGSWExternalProductDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_external_product_tmp_bytes_default<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_tmp_bytes_default(
            module, res_infos, a_infos, ggsw_infos,
        )
    }

    fn glwe_external_product_default<R, A, G>(module: &Module<BE>, res: &mut R, a: &A, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_default(module, res, a, ggsw, scratch)
    }

    fn glwe_external_product_assign_default<R, G>(module: &Module<BE>, res: &mut R, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos,
    {
        <Module<BE> as GLWEExternalProductDefault<BE>>::glwe_external_product_assign_default(module, res, ggsw, scratch)
    }

    fn gglwe_external_product_tmp_bytes_default<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_tmp_bytes_default(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn gglwe_external_product_default<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_default(module, res, a, b, scratch)
    }

    fn gglwe_external_product_assign_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGLWEExternalProductDefault<BE>>::gglwe_external_product_assign_default(module, res, a, scratch)
    }

    fn ggsw_external_product_tmp_bytes_default<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_tmp_bytes_default(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn ggsw_external_product_default<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_default(module, res, a, b, scratch)
    }

    fn ggsw_external_product_assign_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GGSWExternalProductDefault<BE>>::ggsw_external_product_assign_default(module, res, a, scratch)
    }
}

#[macro_export]
macro_rules! impl_core_external_product_default_methods {
    ($be:ty) => {
        fn glwe_external_product_tmp_bytes<R, A, G>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            ggsw_infos: &G,
        ) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            G: $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::glwe_external_product_tmp_bytes_default(
                module, res_infos, a_infos, ggsw_infos,
            )
        }

        fn glwe_external_product<R, A, G>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            ggsw: &G,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            G: $crate::layouts::GGSWPreparedToRef<$be> + $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::glwe_external_product_default(module, res, a, ggsw, scratch)
        }

        fn glwe_external_product_assign<R, G>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            ggsw: &G,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            G: $crate::layouts::GGSWPreparedToRef<$be> + $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::glwe_external_product_assign_default(
                module, res, ggsw, scratch,
            )
        }

        fn gglwe_external_product_tmp_bytes<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            b_infos: &B,
        ) -> usize
        where
            R: $crate::layouts::GGLWEInfos,
            A: $crate::layouts::GGLWEInfos,
            B: $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::gglwe_external_product_tmp_bytes_default(
                module, res_infos, a_infos, b_infos,
            )
        }

        fn gglwe_external_product<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            b: &B,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut + $crate::layouts::GGLWEInfos,
            A: $crate::layouts::GGLWEToRef + $crate::layouts::GGLWEInfos,
            B: $crate::layouts::GGSWPreparedToRef<$be> + $crate::layouts::GGSWInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::gglwe_external_product_default(module, res, a, b, scratch)
        }

        fn gglwe_external_product_assign<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGLWEToMut,
            A: $crate::layouts::GGSWPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::gglwe_external_product_assign_default(module, res, a, scratch)
        }

        fn ggsw_external_product_tmp_bytes<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            b_infos: &B,
        ) -> usize
        where
            R: $crate::layouts::GGSWInfos,
            A: $crate::layouts::GGSWInfos,
            B: $crate::layouts::GGSWInfos,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::ggsw_external_product_tmp_bytes_default(
                module, res_infos, a_infos, b_infos,
            )
        }

        fn ggsw_external_product<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            b: &B,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            A: $crate::layouts::GGSWToRef,
            B: $crate::layouts::GGSWPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::ggsw_external_product_default(module, res, a, b, scratch)
        }

        fn ggsw_external_product_assign<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            A: $crate::layouts::GGSWPreparedToRef<$be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreExternalProductDefaults<$be>>::ggsw_external_product_assign_default(module, res, a, scratch)
        }
    };
}
