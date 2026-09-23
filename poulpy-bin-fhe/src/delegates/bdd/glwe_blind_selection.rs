use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
use std::collections::HashMap;
impl<T: UnsignedInteger, BE: Backend> GLWEBlindSelection<T, BE> for Module<BE>
where
    BE: crate::oep::GLWEBlindSelectionImpl<T>,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection_tmp_bytes<R, A, K>(&self, res_infos: &R, input_infos: &[A], k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        BE::glwe_blind_selection_tmp_bytes::<R, A, K>(self, res_infos, input_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
    {
        BE::glwe_blind_selection::<R, A, K>(self, res, a, fhe_uint, bit_rsh, bit_mask, scratch)
    }
}
