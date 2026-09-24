use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
impl<BRA: BlindRotationAlgo, BE: Backend> BDDKeyEncryptSk<BRA, BE> for Module<BE>
where
    BE: crate::oep::BDDKeyEncryptSkImpl<BRA>,
{
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        BE::bdd_key_encrypt_sk_tmp_bytes::<A>(self, infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::bdd_key_encrypt_sk::<S0, S1>(self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch)
    }
}
