use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{layouts::GLWETensorKeyShareOwned, oep::GGLWEPatImpl};

pub(crate) fn mhe_glwe_tensor_key_share_aggregate_derived<BE: GGLWEPatImpl>(
    module: &Module<BE>,
    res: &mut GLWETensorKeyShareOwned<BE>,
    a: &GLWETensorKeyShareOwned<BE>,
) {
    BE::gglwe_pat_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_tensor_key_share_finalize_derived<BE: GGLWEPatImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &GLWETensorKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
{
    BE::gglwe_pat_finalize(module, res, &share.key, scratch);
}
