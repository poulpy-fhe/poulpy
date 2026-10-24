use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`BDDKeyPreparedFactory`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait BDDKeyPreparedFactoryImpl<BRA: BlindRotationAlgo>: Backend {
    fn bdd_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    fn bdd_key_prepare(
        module: &Module<Self>,
        res: &mut BDDKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &BDDKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
