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
        + GLWEToLWEKeyEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretSampling<BE>,
{
    // With a GLWE switch, its output secret is held beside both switching-key encryptions.
    let switched = infos.ks_glwe_infos().map_or(0, |key| {
        BE::scratch_aligned(module.glwe_secret_bytes_of(key.n(), key.rank_out()))
            + module
                .glwe_switching_key_encrypt_sk_tmp_bytes(&key)
                .max(module.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&infos.ks_lwe_infos()))
    });
    module
        .circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&infos.cbt_infos())
        .max(module.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&infos.ks_lwe_infos()))
        .max(switched)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyEncryptSk::bdd_key_encrypt_sk`].
pub fn bdd_key_encrypt_sk_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>, S0, S1>(
    module: &Module<BE>,
    res: &mut BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
    sk_lwe: &S0,
    sk_glwe: &S1,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
    S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    Module<BE>: CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GLWEToLWEKeyEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretSampling<BE>,
{
    if let Some(key) = &mut res.ks_glwe {
        // On scratch, so that the wipe below erases it.
        let (mut sk_out, mut scratch_1) = scratch.borrow().take_glwe_secret_scratch(key.n(), key.rank_out());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, source_xe);
        module.glwe_switching_key_encrypt_sk(key, sk_glwe, &sk_out, source_xe, source_xa, &mut scratch_1);
        module.glwe_to_lwe_key_encrypt_sk(&mut res.ks_lwe, sk_lwe, &sk_out, source_xe, source_xa, &mut scratch_1);
    } else {
        module.glwe_to_lwe_key_encrypt_sk(&mut res.ks_lwe, sk_lwe, sk_glwe, source_xe, source_xa, scratch);
    }

    module.circuit_bootstrapping_key_encrypt_sk(&mut res.cbt, sk_lwe, sk_glwe, source_xe, source_xa, scratch);

    scratch.wipe(bdd_key_encrypt_sk_tmp_bytes_reference::<BRA, BE, _>(module, &*res));
}
