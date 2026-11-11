use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard, VecZnxDftBackendMut};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef,
};

pub trait GLWEExternalProduct<BE: Backend, Rg: Ring = Standard> {
    fn glwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_external_product<R, A>(
        &self,
        res: &mut R,
        lhs: &A,
        rhs: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;
}

/// DFT-domain external product, dispatched through `GLWEExternalProductImpl`.
/// The input must use the key radix. Size the output with
/// [`crate::layouts::glwe_external_product_output_size`] and use the matching
/// workspace query. No contiguous DFT layout is required by this interface.
pub trait GLWEExternalProductDft<BE: Backend, Rg: Ring = Standard> {
    fn glwe_external_product_dft_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    fn glwe_external_product_dft<'r, A>(
        &self,
        res_dft: &mut VecZnxDftBackendMut<'r, BE, Rg>,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg>;
}

pub trait GGLWEExternalProduct<BE: Backend, Rg: Ring = Standard> {
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        A: GGLWEToBackendRef<BE, Rg> + GGLWEInfos;

    fn gglwe_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos;
}

pub trait GGSWExternalProduct<BE: Backend, Rg: Ring = Standard> {
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE, Rg> + GGSWAtViewMut<BE, Rg> + GGSWInfos,
        A: GGSWToBackendRef<BE, Rg> + GGSWAtViewRef<BE, Rg> + GGSWInfos;

    fn ggsw_external_product_assign<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE, Rg> + GGSWAtViewMut<BE, Rg> + GGSWInfos;
}
