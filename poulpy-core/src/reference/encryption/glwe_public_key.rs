use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    Distribution, EncryptionInfos, GLWEEncryptSk, GetDistribution, GetDistributionMut,
    layouts::{GLWEInfos, GLWEToBackendMut, prepared::GLWESecretPreparedToBackendRef},
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GLWEPublicKeyGenerateReference<BE: Backend> {
    fn glwe_public_key_generate_reference<R, S, E>(
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

/// Independently callable portable composition for [`GLWEPublicKeyGenerateReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GLWEPublicKeyGenerateComposition<BE: Backend> {
    fn glwe_public_key_generate_composition<R, S, E>(
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

impl<BE: Backend> GLWEPublicKeyGenerateComposition<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn glwe_public_key_generate_composition<R, S, E>(
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

            match sk_ref.dist {
                Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                Distribution::ENCAPSULATED(name) => {
                    panic!("invalid sk: {name} is tagged for encapsulation and cannot back a public key")
                }
                _ => {}
            }

            // Its ok to allocate scratch space here since pk is usually generated only once.
            let mut scratch: ScratchOwned<BE> =
                ScratchOwned::alloc(<Module<BE> as GLWEEncryptSk<BE>>::glwe_encrypt_sk_tmp_bytes(self, res));
            self.glwe_encrypt_zero_sk(res, sk, enc_infos, source_xe, source_xa, &mut scratch.borrow());
        }
        *res.dist_mut() = *sk.dist();
    }
}

/// Forwards every method of [`GLWEPublicKeyGenerateReference`] to its portable composition.
#[macro_export]
macro_rules! impl_glwe_public_key_generate_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GLWEPublicKeyGenerateReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn glwe_public_key_generate_reference<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::GetDistributionMut + $crate::layouts::GLWEInfos,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> + $crate::GetDistribution {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEPublicKeyGenerateComposition<$be>>::glwe_public_key_generate_composition::<R, S, E>(self, res, sk, enc_infos, source_xe, source_xa)
        }
        }
    };
}
