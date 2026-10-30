use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed, BlindRotationKeyInfos, CGGI};
use poulpy_core::{
    Distribution,
    layouts::{GGSWCompressedAlloc, GGSWCompressedBytesOf, ModuleCoreCompressedAlloc},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module},
};
use std::marker::PhantomData;

/// Allocates and sizes compressed blind-rotation keys, directly on the module,
/// which implements it for each algorithm with a compressed key.
pub trait BlindRotationKeyCompressedFactory<BRA: BlindRotationAlgo, BE: Backend>:
    ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GGSWCompressedAlloc + GGSWCompressedBytesOf + ModuleN
{
    /// Allocates a zero-filled compressed key of `infos`, one compressed GGSW
    /// per LWE coefficient, with `Distribution::NONE`, which compressed key
    /// encryption fills.
    ///
    /// # Panics
    ///
    /// When the degree of `infos` differs from the module degree.
    fn blind_rotation_key_compressed_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>
    where
        A: BlindRotationKeyInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "blind_rotation_key_compressed_alloc_from_infos: key degree differs from the module degree"
        );
        BlindRotationKeyCompressed {
            keys: (0..infos.n_lwe().as_usize())
                .map(|_| self.ggsw_compressed_alloc_from_infos(infos))
                .collect(),
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }

    /// Bytes of the compressed GGSW bodies [`Self::blind_rotation_key_compressed_alloc_from_infos`]
    /// allocates for `infos`, without their host-side mask seeds.
    ///
    /// # Panics
    ///
    /// As [`Self::blind_rotation_key_compressed_alloc_from_infos`], and when
    /// the byte count overflows `usize`.
    fn blind_rotation_key_compressed_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "blind_rotation_key_compressed_bytes_of_from_infos: key degree differs from the module degree"
        );
        self.ggsw_compressed_bytes_of_from_infos(infos)
            .checked_mul(infos.n_lwe().as_usize())
            .expect("blind_rotation_key_compressed_bytes_of_from_infos: the byte count overflows usize")
    }
}

impl<BE: Backend> BlindRotationKeyCompressedFactory<CGGI, BE> for Module<BE> {}
