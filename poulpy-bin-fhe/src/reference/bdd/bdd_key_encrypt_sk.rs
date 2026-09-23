use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo, circuit_bootstrapping::*};
use poulpy_core::{layouts::*, *};
use poulpy_hal::{layouts::*, source::Source};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyEncryptSk::bdd_key_encrypt_sk_tmp_bytes`].
pub fn bdd_key_encrypt_sk_tmp_bytes_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>, A>(
    module: &Module<BE>,
    infos: &A,
) -> usize
where
    A: BDDKeyInfos,
    Module<BE>: CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretSampling<BE>,
{
    module
        .circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&infos.cbt_infos())
        .max(module.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&infos.ks_lwe_infos()))
        .max(
            infos
                .ks_glwe_infos()
                .map_or(0, |key| module.glwe_switching_key_encrypt_sk_tmp_bytes(&key)),
        )
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyEncryptSk::bdd_key_encrypt_sk`].
pub fn bdd_key_encrypt_sk_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>, S0, S1>(
    module: &Module<BE>,
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
    Module<BE>: CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretSampling<BE>,
{
    if let Some(key) = &mut res.ks_glwe {
        let ks_glwe_infos = enc_infos
            .ks_glwe
            .as_ref()
            .expect("ks_glwe enc_infos missing when ks_glwe key exists");
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(key.rank_out());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, source_xe);
        module.glwe_switching_key_encrypt_sk(key, sk_glwe, &sk_out, ks_glwe_infos, source_xe, source_xa, scratch);
        module.glwe_to_lwe_key_encrypt_sk(
            &mut res.ks_lwe,
            sk_lwe,
            &sk_out,
            &enc_infos.ks_lwe,
            source_xe,
            source_xa,
            scratch,
        );
    } else {
        module.glwe_to_lwe_key_encrypt_sk(
            &mut res.ks_lwe,
            sk_lwe,
            sk_glwe,
            &enc_infos.ks_lwe,
            source_xe,
            source_xa,
            scratch,
        );
    }

    module.circuit_bootstrapping_key_encrypt_sk(&mut res.cbt, sk_lwe, sk_glwe, &enc_infos.cbt, source_xe, source_xa, scratch);
}
