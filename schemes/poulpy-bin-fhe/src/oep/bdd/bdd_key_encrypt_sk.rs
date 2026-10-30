use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
/// Backend implementation contract for [`BDDKeyEncryptSk`](crate::api::BDDKeyEncryptSk).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BDDKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::bdd_key_encrypt_sk`]; may report less than the reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    /// Reproduces [`bdd_key_encrypt_sk_reference`](crate::reference::bdd::bdd_key_encrypt_sk_reference) bit for bit,
    /// within [`Self::bdd_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut BDDKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;
}
