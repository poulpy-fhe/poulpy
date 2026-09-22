use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
impl<T: UnsignedInteger + ToBits, BE: Backend<ZnxWord = i64>> FheUintPreparedEncryptSk<T, BE> for Module<BE>
where
    BE: crate::oep::FheUintPreparedEncryptSkImpl<T>,
{
    fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(&self, infos: &A) -> usize {
        BE::fhe_uint_prepared_encrypt_sk_tmp_bytes(self, infos)
    }
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
        E: EncryptionInfos,
    {
        BE::fhe_uint_prepared_encrypt_sk::<S, E>(self, res, value, sk, enc_infos, source_xe, source_xa, scratch)
    }
}
