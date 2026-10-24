use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared};
use poulpy_core::{Distribution, api::GGSWPreparedFactory};
use poulpy_hal::layouts::{Backend, ScratchArena};
use std::marker::PhantomData;

/// Allocates and prepares backend-owned blind-rotation key representations.
///
/// Allocation and sizing are direct on the module; the preparation and its
/// query dispatch through [`BlindRotationKeyPreparedImpl`](crate::oep::BlindRotationKeyPreparedImpl).
pub trait BlindRotationKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    /// Allocates a zero-filled prepared key of `infos`, one prepared GGSW per
    /// LWE coefficient, with `Distribution::NONE` and no monomial table, which
    /// [`Self::blind_rotation_key_prepare`] fills.
    ///
    /// # Panics
    ///
    /// When the degree of `infos` differs from the module degree.
    fn blind_rotation_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BlindRotationKeyInfos,
        Self: GGSWPreparedFactory<BE>,
    {
        assert!(
            infos.n() == self.ring_degree(),
            "blind_rotation_key_prepared_alloc_from_infos: key degree differs from the module degree"
        );
        BlindRotationKeyPrepared {
            data: (0..infos.n_lwe().as_usize())
                .map(|_| self.ggsw_prepared_alloc_from_infos(infos))
                .collect(),
            dist: Distribution::NONE,
            x_pow_a: None,
            _phantom: PhantomData,
        }
    }

    /// Bytes of the prepared GGSWs [`Self::blind_rotation_key_prepared_alloc_from_infos`]
    /// allocates for `infos`, without the monomial table a block-binary
    /// preparation adds.
    ///
    /// # Panics
    ///
    /// As [`Self::blind_rotation_key_prepared_alloc_from_infos`], and when the
    /// byte count overflows `usize`.
    fn blind_rotation_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
        Self: GGSWPreparedFactory<BE>,
    {
        assert!(
            infos.n() == self.ring_degree(),
            "blind_rotation_key_prepared_bytes_of_from_infos: key degree differs from the module degree"
        );
        infos
            .n_lwe()
            .as_usize()
            .checked_mul(self.ggsw_prepared_bytes_of_from_infos(infos))
            .expect("blind_rotation_key_prepared_bytes_of_from_infos: the byte count overflows usize")
    }

    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_prepare`][Self::blind_rotation_key_prepare].
    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos;

    /// Transforms the standard key `other` into the DFT-domain prepared form
    /// `res`, ready for use in `BlindRotationExecute::blind_rotation_execute`.
    ///
    /// For the `BinaryBlock` distribution this also pre-computes the
    /// `X^{a_i}` scalar polynomial products used in the batched CMux loop.
    fn blind_rotation_key_prepare(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
