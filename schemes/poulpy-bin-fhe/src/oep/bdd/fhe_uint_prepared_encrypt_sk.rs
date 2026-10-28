use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::{layouts::*, source::Source};
/// Backend implementation contract for [`FheUintPreparedEncryptSk`](crate::api::FheUintPreparedEncryptSk).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait FheUintPreparedEncryptSkImpl<T: UnsignedInteger + ToBits>: Backend<ZnxWord = i64> {
    /// Scratch bound of [`Self::fhe_uint_prepared_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<Self>, infos: &A) -> usize;
    /// Reproduces
    /// [`fhe_uint_prepared_encrypt_sk_reference`](crate::reference::bdd::fhe_uint_prepared_encrypt_sk_reference) bit for
    /// bit, within [`Self::fhe_uint_prepared_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepared_encrypt_sk<S>(
        module: &Module<Self>,
        res: &mut FheUintPrepared<Self::OwnedBuf, T, Self>,
        value: T,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;
}
