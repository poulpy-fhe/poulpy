use poulpy_hal::{
    api::{CoeffGemmPrepare, ModuleN, VecZnxMatMulPrepared, VecZnxMatMulTmpBytes, VecZnxZeroBackend},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef},
};

use crate::{
    api::{
        CoeffMatrixPrepare, GGSWExpandRows, GGSWFromGGLWE, GLWEExpandLWE, GLWEExpandLWEMatrix, GLWEFromLWE, LWEFromGLWE,
        LWEMatrixMul, LWESampleExtract,
    },
    layouts::{
        CoeffBound, CoeffMatrixInfos, CoeffMatrixPreparedOwned, CoeffMatrixToBackendRef, GGLWEInfos, GGSWInfos, GGSWToBackendMut,
        GLWECompressedSeed,
        GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEMatrixInfos,
        LWEMatrixToBackendMut, LWEMatrixToBackendRef, LWEToBackendMut, LWEToBackendRef,
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

    fn glwe_from_lwe<R, A, K>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
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

    fn lwe_from_glwe<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        BE::lwe_from_glwe(self, res, a, a_idx, key, key_size, scratch)
    }
);

impl_conversion_delegate!(
    GLWEExpandLWE<BE>,
    [BE: Backend + ConversionImpl<BE>, Module<BE>: ConversionDefault<BE>],
    fn glwe_expand_lwe_tmp_bytes<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_expand_lwe_tmp_bytes(self, lwe_infos, a_infos)
    }

    fn glwe_expand_lwe<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_expand_lwe(self, res, a, scratch)
    }
);

impl_conversion_delegate!(
    GLWEExpandLWEMatrix<BE>,
    [
        BE: Backend + ConversionImpl<BE>,
        Module<BE>: ConversionDefault<BE>
    ],
    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos,
    {
        BE::glwe_expand_lwe_matrix_tmp_bytes(self, res_infos, a_infos)
    }

    fn glwe_expand_lwe_matrix<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_expand_lwe_matrix(self, res, a, scratch)
    }
);

impl_conversion_delegate!(
    LWEMatrixMul<BE>,
    [
        BE: Backend + ConversionImpl<BE>,
        Module<BE>: ConversionDefault<BE>
    ],
    fn lwe_matrix_mul_tmp_bytes<R, U, A>(&self, res_infos: &R, u_infos: &U, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        U: CoeffMatrixInfos,
        A: LWEMatrixInfos,
    {
        BE::lwe_matrix_mul_tmp_bytes(self, res_infos, u_infos, a_infos)
    }

    fn lwe_matrix_mul<R, U, A>(&self, res: &mut R, u: &U, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        U: CoeffMatrixToBackendRef<BE> + CoeffMatrixInfos,
        A: LWEMatrixToBackendRef<BE> + LWEMatrixInfos,
    {
        BE::lwe_matrix_mul(self, res, u, a, scratch)
    }

    fn lwe_matrix_mul_mask_tmp_bytes<R, U, A>(&self, res_infos: &R, u_infos: &U, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        U: CoeffMatrixInfos,
        A: GLWEInfos,
    {
        BE::lwe_matrix_mul_mask_tmp_bytes(self, res_infos, u_infos, a_infos)
    }

    fn lwe_matrix_mul_mask<R, U, A>(&self, res: &mut R, u: &U, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        U: CoeffMatrixToBackendRef<BE> + CoeffMatrixInfos,
        A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::lwe_matrix_mul_mask(self, res, u, a, scratch)
    }

    fn lwe_matrix_mul_body_tmp_bytes<R, U, A>(&self, res_infos: &R, u_infos: &U, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        U: CoeffMatrixInfos,
        A: GLWEInfos,
    {
        BE::lwe_matrix_mul_body_tmp_bytes(self, res_infos, u_infos, a_infos)
    }

    fn lwe_matrix_mul_body<R, U, A>(&self, res: &mut R, u: &U, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        U: CoeffMatrixToBackendRef<BE> + CoeffMatrixInfos,
        A: GLWECompressedToBackendRef<BE> + GLWEInfos,
    {
        BE::lwe_matrix_mul_body(self, res, u, a, scratch)
    }

    fn lwe_matrix_mul_bodies_tmp_bytes<U>(&self, u_infos: &U, num_bodies: usize, res_size: usize, a_size: usize) -> usize
    where
        U: CoeffMatrixInfos,
    {
        BE::lwe_matrix_mul_bodies_tmp_bytes(self, u_infos, num_bodies, res_size, a_size)
    }

    fn lwe_matrix_mul_bodies<R, U, A>(
        &self,
        res_bodies: &mut R,
        res_base2k: usize,
        u: &U,
        bodies: &A,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE>,
        U: CoeffMatrixToBackendRef<BE> + CoeffMatrixInfos,
        A: VecZnxToBackendRef<BE>,
    {
        BE::lwe_matrix_mul_bodies(self, res_bodies, res_base2k, u, bodies, a_base2k, scratch)
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

    fn ggsw_from_gglwe<R, A, T>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: crate::layouts::GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
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

    fn ggsw_expand_row<R, T>(
        &self,
        res: &mut R,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    )
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        BE::ggsw_expand_row(self, res, tsk, tsk_size, scratch)
    }
);

impl<BE> CoeffMatrixPrepare<BE> for Module<BE>
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN + CoeffGemmPrepare<BE> + VecZnxZeroBackend<BE> + VecZnxMatMulPrepared<BE> + VecZnxMatMulTmpBytes,
{
    fn coeff_matrix_prepare<U>(&self, u: &U) -> CoeffMatrixPreparedOwned<BE, <U as CoeffMatrixInfos>::Bound>
    where
        U: CoeffMatrixToBackendRef<BE> + CoeffMatrixInfos,
    {
        crate::default::conversion::coeff_matrix_prepare_default::<BE, _, _>(self, u)
    }

    fn lwe_matrix_mul_bodies_prepared_tmp_bytes<BU>(
        &self,
        prepared: &CoeffMatrixPreparedOwned<BE, BU>,
        num_bodies: usize,
        res_size: usize,
        a_size: usize,
    ) -> usize
    where
        BU: CoeffBound,
    {
        crate::default::conversion::lwe_matrix_mul_bodies_prepared_tmp_bytes_default::<BE, _, BU>(
            self, prepared, num_bodies, res_size, a_size,
        )
    }

    fn lwe_matrix_mul_bodies_prepared<R, A, BU>(
        &self,
        res_bodies: &mut R,
        res_base2k: usize,
        prepared: &CoeffMatrixPreparedOwned<BE, BU>,
        bodies: &A,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE>,
        A: VecZnxToBackendRef<BE>,
        BU: CoeffBound,
    {
        crate::default::conversion::lwe_matrix_mul_bodies_prepared_default::<BE, _, _, _, BU>(
            self, res_bodies, res_base2k, prepared, bodies, a_base2k, scratch,
        )
    }
}
