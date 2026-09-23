use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> Cmux<BE> for Module<BE>
where
    BE: crate::oep::CmuxImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        BE::cmux_tmp_bytes::<R, A, B>(self, res_infos, a_infos, selector_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn cmux<R, T, F>(&self, res: &mut R, t: &T, f: &F, s: &GGSWPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        T: GLWEToBackendRef<BE>,
        F: GLWEToBackendRef<BE>,
    {
        BE::cmux::<R, T, F>(self, res, t, f, s, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign_neg<R, A>(&self, res: &mut R, a: &A, s: &GGSWPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
    {
        BE::cmux_assign_neg::<R, A>(self, res, a, s, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign<R, A>(&self, res: &mut R, a: &A, s: &GGSWPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
    {
        BE::cmux_assign::<R, A>(self, res, a, s, scratch)
    }
}
