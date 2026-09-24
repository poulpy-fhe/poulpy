use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed, BlindRotationKeyInfos};
use poulpy_hal::layouts::{Backend, Module};

/// Backend allocation of compressed blind-rotation keys.
///
/// # Safety
/// Allocate zero-filled key elements with the requested dimensions and backend
/// storage. Initialize the secret distribution to `Distribution::NONE`.
pub unsafe trait BlindRotationKeyCompressedFactoryImpl<BRA: BlindRotationAlgo>: Backend {
    fn blind_rotation_key_compressed_alloc<A>(
        module: &Module<Self>,
        infos: &A,
    ) -> BlindRotationKeyCompressed<Self::OwnedBuf, BRA, Self::ZnxWord>
    where
        A: BlindRotationKeyInfos;
}

/// Selects compressed CGGI key allocation built from core storage operations.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_key_compressed_factory_reference {
    ($backend:ty) => {
        unsafe impl $crate::oep::BlindRotationKeyCompressedFactoryImpl<$crate::blind_rotation::CGGI> for $backend {
            fn blind_rotation_key_compressed_alloc<A: $crate::blind_rotation::BlindRotationKeyInfos>(
                module: &poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> $crate::blind_rotation::BlindRotationKeyCompressed<Self::OwnedBuf, $crate::blind_rotation::CGGI, Self::ZnxWord>
            {
                $crate::reference::blind_rotation::blind_rotation_key_compressed_alloc_ref(module, infos)
            }
        }
    };
}
