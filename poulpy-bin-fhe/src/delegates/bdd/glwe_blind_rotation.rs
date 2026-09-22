use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> GLWEBlindRotation<BE> for Module<BE>
where
    BE: crate::oep::GLWEBlindRotationImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        BE::glwe_blind_rotation_tmp_bytes::<R, A, K>(self, res_infos, a_infos, k_infos)
    }
    fn glwe_blind_rotation_assign_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        BE::glwe_blind_rotation_assign_tmp_bytes::<R, K>(self, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        BE::glwe_blind_rotation_assign::<R, K>(self, res, value, sign, bit_rsh, bit_mask, bit_lsh, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation<R, A, K>(
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        BE::glwe_blind_rotation::<R, A, K>(self, res, a, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch)
    }
}
