use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> GLWEBlindRetrieval<BE> for Module<BE>
where
    BE: crate::oep::GLWEBlindRetrievalImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        BE::glwe_blind_retrieval_tmp_bytes::<R, K>(self, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_statefull<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
    {
        BE::glwe_blind_retrieval_statefull::<R, K>(self, res, bits, bit_rsh, bit_mask, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_statefull_rev<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
    {
        BE::glwe_blind_retrieval_statefull_rev::<R, K>(self, res, bits, bit_rsh, bit_mask, scratch)
    }
}
