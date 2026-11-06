use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{layouts::MHEGLWETensorKeyShareOwned, oep::MHEGGLWEPatImpl};

pub(crate) fn mhe_glwe_tensor_key_share_aggregate_derived<BE: MHEGGLWEPatImpl>(
    module: &Module<BE>,
    res: &mut MHEGLWETensorKeyShareOwned<BE>,
    a: &MHEGLWETensorKeyShareOwned<BE>,
) {
    BE::mhe_gglwe_pat_aggregate_assign(module, &mut res.key, &a.key);
}

pub(crate) fn mhe_glwe_tensor_key_share_finalize_derived<BE: MHEGGLWEPatImpl, R>(
    module: &Module<BE>,
    res: &mut R,
    share: &MHEGLWETensorKeyShareOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
{
    BE::mhe_gglwe_pat_finalize(module, res, &share.key, scratch);
}
