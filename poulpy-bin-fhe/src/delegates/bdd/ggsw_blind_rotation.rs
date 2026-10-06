use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<T: UnsignedInteger, BE: Backend> GGSWBlindRotation<T, BE> for Module<BE>
where
    BE: crate::oep::GGSWBlindRotationImpl<T>,
{
    #[allow(clippy::too_many_arguments)]
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        BE::ggsw_to_ggsw_blind_rotation_tmp_bytes::<R, A, K>(self, res_infos, a_infos, k_infos)
    }
    fn ggsw_blind_rotation_assign_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        BE::ggsw_blind_rotation_assign_tmp_bytes::<R, K>(self, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        BE::ggsw_blind_rotation_assign::<R, K>(self, res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        BE::ggsw_blind_rotation::<R, A, K>(self, res, a, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        BE::scalar_to_ggsw_blind_rotation_tmp_bytes::<R, K>(self, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: ScalarZnxToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        BE::scalar_to_ggsw_blind_rotation::<R, A, K>(self, res, test_vector, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch)
    }
}
