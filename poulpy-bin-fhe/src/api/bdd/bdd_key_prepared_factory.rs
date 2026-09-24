use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
/// Backend-level factory for allocating and preparing [`BDDKeyPrepared`] values.
///
/// Implemented for `Module<BE>` when the backend supports preparation of all
/// three constituent sub-keys.  Default method implementations delegate to
/// the corresponding sub-key factories.
pub trait BDDKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn alloc_bdd_key_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
