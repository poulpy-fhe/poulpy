use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GGSWExpandRows, GGSWFromGGLWE, GLWEFromLWE, LWEFromGLWE, LWESampleExtract},
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut,
        LWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{ConversionDefault, ConversionImpl},
};

macro_rules! impl_conversion_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_conversion_delegate!(
    LWESampleExtract<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn lwe_sample_extract<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::lwe_sample_extract(self, res, a)
    }
);

impl_conversion_delegate!(
    GLWEFromLWE<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_from_lwe_tmp_bytes(self, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<'s, R, A, K>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::glwe_from_lwe(self, res, lwe, ksk, key_size, scratch)
    }
);

impl_conversion_delegate!(
    LWEFromGLWE<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_from_glwe_tmp_bytes(self, lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<'s, R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::lwe_from_glwe(self, res, a, a_idx, key, key_size, scratch)
    }
);

impl_conversion_delegate!(
    GGSWFromGGLWE<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_from_gglwe_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<'s, R, A, T>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: crate::layouts::GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::ggsw_from_gglwe(self, res, a, tsk, tsk_size, scratch)
    }
);

impl_conversion_delegate!(
    GGSWExpandRows<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_expand_rows_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_expand_row<'s, R, T>(
        &self,
        res: &mut R,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        BE::ggsw_expand_row(self, res, tsk, tsk_size, scratch)
    }
);
