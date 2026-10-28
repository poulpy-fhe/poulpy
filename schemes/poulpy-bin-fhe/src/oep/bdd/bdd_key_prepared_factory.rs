use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`BDDKeyPreparedFactory`](crate::api::BDDKeyPreparedFactory).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BDDKeyPreparedFactoryImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::bdd_key_prepare`]; may report less than the reference, never less than the body takes.
    fn bdd_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    /// Reproduces [`bdd_key_prepare_reference`](crate::reference::bdd::bdd_key_prepare_reference) bit for bit, within
    /// [`Self::bdd_key_prepare_tmp_bytes`]; free in kernels and scratch use.
    fn bdd_key_prepare(
        module: &Module<Self>,
        res: &mut BDDKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &BDDKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
