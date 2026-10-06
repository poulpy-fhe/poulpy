use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GLWECIEmbed, GLWECITrace},
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
    oep::GLWECIConversionImpl,
};

impl<BE: Backend + GLWECIConversionImpl> GLWECIEmbed<BE> for Module<BE> {
    fn glwe_ci_embed<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_embed(self, res, a);
        res.set_noise(None);
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWECITrace<BE> for Module<BE> {
    fn glwe_ci_trace_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_ci_trace_tmp_bytes(self, res_infos, a_infos)
    }

    fn glwe_ci_trace<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_trace(self, res, a, scratch);
        res.set_noise(None);
    }
}
