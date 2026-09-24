use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
/// Backend-level factory for directly encrypting a plaintext value into a
/// [`FheUintPrepared`] without first creating an [`FheUint`].
///
/// Useful in testing and debugging scenarios where the packed-GLWE intermediate
/// form is not needed.  Each bit is encrypted independently as a constant GGSW
/// and then immediately DFT-prepared in place.
pub trait FheUintPreparedEncryptSk<T: UnsignedInteger + ToBits, BE: Backend<ZnxWord = i64>> {
    /// Workspace for encrypting and preparing each bit sequentially.
    fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(&self, infos: &A) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepared_encrypt_sk<S, E>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        value: T,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;
}
