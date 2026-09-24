use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
/// Backend implementation contract for [`FheUintPreparedEncryptSk`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait FheUintPreparedEncryptSkImpl<T: UnsignedInteger + ToBits>: Backend<ZnxWord = i64> {
    fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<Self>, infos: &A) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepared_encrypt_sk<S, E>(
        module: &Module<Self>,
        res: &mut FheUintPrepared<Self::OwnedBuf, T, Self>,
        value: T,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;
}
