pub mod glwe;

use crate::layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEExternalProductReference<BE: Backend> {
    fn glwe_external_product_dft_fill_tmp_bytes_reference<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_tmp_bytes_reference<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_external_product_assign_reference<R>(
        &self,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEExternalProductReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VmpApplyDftToDftTmpBytes
        + crate::api::GLWEBytesOf<BE>
        + crate::api::GLWEExternalProductInternal<BE>
        + crate::api::GLWENormalize<BE>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxIdftApply<BE>
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftApply<BE>
        + poulpy_hal::api::VmpApplyDftToDft<BE>
        + poulpy_hal::api::VmpApplyDftToDftAdd<BE>,
{
    fn glwe_external_product_dft_fill_tmp_bytes_reference<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
    where
        A: crate::layouts::GLWEInfos,
        G: crate::layouts::GGSWInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_dft_fill_tmp_bytes_reference::<BE, _, _, _>(
            self, a_infos, ggsw_infos,
        )
    }

    fn glwe_external_product_tmp_bytes_reference<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        G: crate::layouts::GGSWInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<BE, _, _, _, _>(
            self, res_infos, a_infos, ggsw_infos,
        )
    }

    fn glwe_external_product_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &crate::layouts::prepared::GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_reference::<BE, _, _, _>(self, res, a, ggsw, scratch)
    }

    fn glwe_external_product_assign_reference<R>(
        &self,
        res: &mut R,
        ggsw: &crate::layouts::prepared::GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_assign_reference::<BE, _, _>(self, res, ggsw, scratch)
    }
}
