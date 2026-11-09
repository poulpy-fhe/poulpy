pub mod glwe;
pub mod lwe;

pub(crate) use glwe::*;

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::GGLWEPreparedBackendRef,
};
use poulpy_hal::layouts::{
    Backend, Module, Ring, ScratchArena, Standard, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
    VmpPMatBackendRef,
};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait GLWEKeyswitchReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_keyswitch_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos;

    fn glwe_keyswitch_dft_tmp_bytes_reference<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_dft_add_tmp_bytes_reference<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_dft_add_reference<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_keyswitch_dft_reference<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos;
}

impl<BE: Backend, Rg: Ring> GLWEKeyswitchReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    BE: crate::oep::GGLWEProductDigitsStridedImpl<Rg>,
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::reference::keyswitching::GLWEKeyswitchInternal<BE, Rg>
        + crate::api::GLWENormalize<BE, Rg>
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE, Rg>
        + poulpy_hal::api::VecZnxBigNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxIdftApply<BE, Rg>
        + poulpy_hal::api::VecZnxNormalizeAssign<BE, Rg>,
{
    fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_tmp_bytes_reference(self, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_reference(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_assign_reference<R>(
        &self,
        res: &mut R,
        key: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_assign_reference::<BE, _, _, Rg>(self, res, key, scratch)
    }

    fn glwe_keyswitch_dft_tmp_bytes_reference<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_dft_tmp_bytes_reference::<BE, _, _, _, Rg>(
            self, res_size, a_infos, key_infos,
        )
    }

    fn glwe_keyswitch_dft_add_tmp_bytes_reference<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: crate::layouts::GLWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_dft_add_tmp_bytes_reference::<BE, _, _, Rg>(
            self, res_size, a_infos, key_infos,
        )
    }

    fn glwe_keyswitch_dft_add_reference<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_dft_add_reference(self, res, a, key, scratch)
    }

    fn glwe_keyswitch_dft_reference<A>(
        &self,
        res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, BE, Rg>,
        a: &A,
        key: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        A: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::keyswitching::glwe::glwe_keyswitch_dft_reference::<BE, _, _, Rg>(self, res, a, key, scratch)
    }
}

/// Reference bodies of the [`GGLWEProductDigitsStridedImpl`](crate::oep::GGLWEProductDigitsStridedImpl) methods, which
/// [`impl_gglwe_product_digits_strided_reference_full!`](crate::impl_gglwe_product_digits_strided_reference_full) and,
/// for the step hooks, [`impl_gglwe_product_digits_strided_reference!`](crate::impl_gglwe_product_digits_strided_reference)
/// forward to.
pub trait GGLWEProductDigitsStridedReference<BE: Backend, Rg: Ring = Standard> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_digits_strided_tmp_bytes_reference(
        &self,
        res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        pmat_cols_out: usize,
        pmat_size: usize,
    ) -> usize;

    #[doc(hidden)]
    fn gglwe_product_digits_strided_reference(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    #[doc(hidden)]
    fn gglwe_product_fused_tmp_bytes_reference<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_fused_reference(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE, Rg>, usize)>],
        p: i64,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, BE>,
    );

    #[doc(hidden)]
    fn gglwe_product_idft_normalize_tmp_bytes_reference<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_idft_normalize_reference(
        &self,
        res: &mut VecZnxBackendMut<'_, BE, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE, Rg>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, BE, Rg>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    #[doc(hidden)]
    fn gglwe_products_from_dft_tmp_bytes_reference<K>(
        &self,
        count: usize,
        rank: usize,
        source_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn gglwe_products_from_dft_reference(
        &self,
        p: i64,
        term_count: usize,
        res: &mut [VecZnxDftBackendMut<'_, BE, Rg>],
        output_size: usize,
        accumulate: &[bool],
        sources: &mut [VecZnxDftBackendMut<'_, BE, Rg>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend, Rg: Ring> GGLWEProductDigitsStridedReference<BE, Rg> for Module<BE>
where
    BE: crate::oep::GGLWEProductDigitsStridedImpl<Rg>,
    Module<BE>: GGLWEProductReference<BE, Rg>
        + poulpy_hal::api::VecZnxDftAddAssign<BE, Rg>
        + poulpy_hal::api::VecZnxDftAutomorphism<BE, Rg>
        + poulpy_hal::api::VecZnxDftAutomorphismAddWithPlanTmpBytes
        + poulpy_hal::api::VecZnxIdftNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftApply<BE, Rg>
        + poulpy_hal::api::VecZnxGadgetDecompose<BE, Rg>
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes,
{
    fn gglwe_product_digits_strided_tmp_bytes_reference(
        &self,
        res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        pmat_cols_out: usize,
        pmat_size: usize,
    ) -> usize {
        glwe::gglwe_product_digits_strided_tmp_bytes_reference(
            self,
            res_size,
            a_cols,
            a_size,
            dsize,
            pmat_rows,
            pmat_cols_in,
            pmat_cols_out,
            pmat_size,
        )
    }

    fn gglwe_product_digits_strided_reference(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        glwe::gglwe_product_digits_strided_reference(self, res, a, dsize, product_limbs, pmat, scratch)
    }

    fn gglwe_product_fused_tmp_bytes_reference<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        glwe::gglwe_product_fused_tmp_bytes_reference(self, res_size, product_size, a_size, key_infos)
    }

    fn gglwe_product_fused_reference(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE, Rg>, usize)>],
        p: i64,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        glwe::gglwe_product_fused_reference(self, res, product_size, a, key, term_count, addends, p, accumulate, scratch)
    }

    fn gglwe_product_idft_normalize_tmp_bytes_reference<K>(
        &self,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        glwe::gglwe_product_idft_normalize_tmp_bytes_reference(self, res_size, product_size, a_size, key_infos)
    }

    fn gglwe_product_idft_normalize_reference(
        &self,
        res: &mut VecZnxBackendMut<'_, BE, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &VecZnxDftBackendRef<'_, BE, Rg>,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, BE, Rg>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, BE, Rg>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        glwe::gglwe_product_idft_normalize_reference(
            self, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, scratch,
        )
    }

    fn gglwe_products_from_dft_tmp_bytes_reference<K>(
        &self,
        count: usize,
        rank: usize,
        source_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        glwe::gglwe_products_from_dft_tmp_bytes_reference(self, count, rank, source_size, key_infos)
    }

    fn gglwe_products_from_dft_reference(
        &self,
        p: i64,
        term_count: usize,
        res: &mut [VecZnxDftBackendMut<'_, BE, Rg>],
        output_size: usize,
        accumulate: &[bool],
        sources: &mut [VecZnxDftBackendMut<'_, BE, Rg>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        glwe::gglwe_products_from_dft_reference(
            self,
            p,
            term_count,
            res,
            output_size,
            accumulate,
            sources,
            source_sizes,
            source_base2k,
            key,
            scratch,
        )
    }
}

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait LWEKeyswitchReference<BE: Backend, Rg: Ring = Standard> {
    fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE, Rg> + LWEInfos,
        A: LWEToBackendRef<BE, Rg> + LWEInfos;
}

impl<BE: Backend, Rg: Ring> LWEKeyswitchReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: crate::api::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleN
        + crate::api::GLWEKeyswitch<BE, Rg>
        + poulpy_hal::api::VecZnxCopy<BE, Rg>
        + poulpy_hal::api::VecZnxZero<BE, Rg>,
{
    fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::LWEInfos,
        A: crate::layouts::LWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::keyswitching::lwe::lwe_keyswitch_tmp_bytes_reference::<BE, _, _, _, _, Rg>(
            self, res_infos, a_infos, key_infos,
        )
    }

    fn lwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: crate::layouts::LWEToBackendMut<BE, Rg> + crate::layouts::LWEInfos,
        A: crate::layouts::LWEToBackendRef<BE, Rg> + crate::layouts::LWEInfos,
    {
        crate::reference::keyswitching::lwe::lwe_keyswitch_reference::<BE, _, _, _, Rg>(self, res, a, ksk, scratch)
    }
}
