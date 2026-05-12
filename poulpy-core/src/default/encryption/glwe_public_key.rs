use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use crate::{
    Distribution, EncryptionInfos, GLWEEncryptSk, GetDistribution, GetDistributionMut, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, prepared::GLWESecretPreparedToBackendRef},
};

#[doc(hidden)]
pub trait GLWEPublicKeyGenerateDefault<BE: Backend> {
    fn glwe_public_key_generate_default<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution;
}

impl<BE: Backend> GLWEPublicKeyGenerateDefault<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_public_key_generate_default<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        {
            let sk_ref = sk.to_backend_ref();

            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(sk_ref.n(), self.n() as u32);

            if sk_ref.dist == Distribution::NONE {
                panic!("invalid sk: SecretDistribution::NONE")
            }

            // Its ok to allocate scratch space here since pk is usually generated only once.
            let mut scratch: ScratchOwned<BE> =
                ScratchOwned::alloc(<Module<BE> as GLWEEncryptSk<BE>>::glwe_encrypt_sk_tmp_bytes(self, res));
            self.glwe_encrypt_zero_sk(res, sk, enc_infos, source_xe, source_xa, &mut scratch.borrow());
        }
        *res.dist_mut() = *sk.dist();
    }
}
