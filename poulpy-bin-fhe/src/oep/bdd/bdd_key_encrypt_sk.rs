use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
/// Backend implementation contract for [`BDDKeyEncryptSk`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait BDDKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut BDDKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;
}
