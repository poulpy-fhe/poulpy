pub mod glwe;

use crate::layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEExternalProductReference<BE: Backend, Rg: Ring = Standard> {
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
        ggsw: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_external_product_assign_reference<R>(
        &self,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;
}

impl<BE: Backend, Rg: Ring> GLWEExternalProductReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VmpApplyDftToDftTmpBytes
        + crate::api::GLWEBytesOf<BE>
        + crate::api::GLWEExternalProductDft<BE, Rg>
        + crate::api::GLWENormalize<BE, Rg>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxIdftApply<BE, Rg>
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftApply<BE, Rg>
        + poulpy_hal::api::VmpApplyDftToDft<BE, Rg>
        + poulpy_hal::api::VmpApplyDftToDftAdd<BE, Rg>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes,
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
        crate::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<BE, _, _, _, _, Rg>(
            self, res_infos, a_infos, ggsw_infos,
        )
    }

    fn glwe_external_product_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &crate::layouts::prepared::GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_reference::<BE, _, _, _, Rg>(self, res, a, ggsw, scratch)
    }

    fn glwe_external_product_assign_reference<R>(
        &self,
        res: &mut R,
        ggsw: &crate::layouts::prepared::GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::external_product::glwe::glwe_external_product_assign_reference::<BE, _, _, Rg>(self, res, ggsw, scratch)
    }
}
