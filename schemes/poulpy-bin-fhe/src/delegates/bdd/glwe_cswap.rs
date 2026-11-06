use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> GLWECswap<BE> for Module<BE>
where
    BE: crate::oep::GLWECswapImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_cswap_assign_tmp_bytes<R, A, S>(&self, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos,
    {
        BE::glwe_cswap_assign_tmp_bytes::<R, A, S>(self, res_a_infos, res_b_infos, s_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_cswap_assign<A, B>(
        &self,
        res_a: &mut A,
        res_b: &mut B,
        s: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_cswap_assign::<A, B>(self, res_a, res_b, s, scratch)
    }
}
