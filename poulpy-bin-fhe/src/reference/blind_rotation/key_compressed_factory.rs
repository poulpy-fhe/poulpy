use std::marker::PhantomData;

use crate::blind_rotation::{BlindRotationKeyCompressed, BlindRotationKeyInfos, CGGI};
use poulpy_core::{
    Distribution,
    layouts::{GGSWCompressed, ModuleCoreCompressedAlloc},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module},
};

/// Canonical compressed CGGI key allocation using core storage operations.
pub fn blind_rotation_key_compressed_alloc_ref<BE, A>(
    module: &Module<BE>,
    infos: &A,
) -> BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>
where
    BE: Backend,
    A: BlindRotationKeyInfos,
    Module<BE>: ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + ModuleN,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(module.n(), infos.n_glwe().as_usize());
    }
    let mut data: Vec<GGSWCompressed<BE::OwnedBuf, BE::ZnxWord>> = Vec::with_capacity(infos.n_lwe().into());
    (0..infos.n_lwe().as_usize()).for_each(|_| data.push(module.ggsw_compressed_alloc_from_infos(infos)));
    BlindRotationKeyCompressed {
        keys: data,
        dist: Distribution::NONE,
        _phantom: PhantomData,
    }
}
