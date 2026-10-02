use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`BDDKeyPreparedFactory`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait BDDKeyPreparedFactoryImpl<BRA: BlindRotationAlgo>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn alloc_bdd_key_from_infos<A>(module: &Module<Self>, infos: &A) -> BDDKeyPrepared<Self::OwnedBuf, BRA, Self>
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key(
        module: &Module<Self>,
        res: &mut BDDKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &BDDKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
