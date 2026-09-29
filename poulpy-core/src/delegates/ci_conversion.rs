use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{GLWECIFold, GLWECIUnfold},
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
    oep::GLWECIConversionImpl,
};

impl<BE: Backend + GLWECIConversionImpl> GLWECIUnfold<BE> for Module<BE> {
    fn glwe_ci_unfold<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_unfold(self, res, a)
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWECIFold<BE> for Module<BE> {
    fn glwe_ci_fold_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_ci_fold_tmp_bytes(self, res_infos, a_infos)
    }

    fn glwe_ci_fold<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_fold(self, res, a, scratch)
    }
}
