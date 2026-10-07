//! Encryption operations expressed entirely through other core operations.
//!
//! Defaults dispatch through the selected backend, so overriding a primitive
//! also changes every derived operation that uses it.

#![allow(clippy::too_many_arguments)]

use crate::layouts::operand_degree;
use crate::{
    Distribution, GLWENormalize, GetDistribution, GetDistributionMut, Noise, ScratchArenaTakeCore, VecZnxAddNoise,
    api::GLWEBytesOf,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GGLWEToBackendMut, GGSWAtViewMut, GGSWInfos,
        GGSWToBackendMut, GLWEInfos, GLWEPlaintext, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut, GLWESecretPreparedFactory,
        GLWESecretTensorFactory, GLWESecretToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEToBackendMut,
        compressed::{GLWEPublicKeyCompressedSeedMut, GLWEPublicKeyCompressedToBackendMut},
        prepared::{GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
    oep::EncryptionImpl,
};
use poulpy_hal::{
    api::{VecZnxAddScalarAssign, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxZero},
    layouts::{Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

/// Reconstruct a mask using the backend's source-driven mask sampler.
pub(crate) fn fill_glwe_mask_from_seed_derived<BE: EncryptionImpl, R: GLWEToBackendMut<BE>>(
    module: &Module<BE>,
    res: &mut R,
    seed_xa: [u8; 32],
) {
    BE::fill_glwe_mask_from_source(module, res, &mut Source::new(seed_xa));
}

/// Reconstruct an LWE mask using the backend's source-driven mask sampler.
pub(crate) fn fill_lwe_mask_from_seed_derived<BE: EncryptionImpl, R: LWEToBackendMut<BE>>(
    module: &Module<BE>,
    base2k: usize,
    res: &mut R,
    seed_xa: [u8; 32],
) {
    BE::fill_lwe_mask_from_source(module, base2k, res, &mut Source::new(seed_xa));
}

pub(crate) fn glwe_public_key_generate_tmp_bytes_derived<BE: EncryptionImpl, A: GLWEInfos>(
    module: &Module<BE>,
    infos: &A,
) -> usize
where
    Module<BE>: GLWENormalize<BE>,
{
    operand_degree(module.n(), &[infos.n()]);
    BE::glwe_encrypt_sk_tmp_bytes(module, infos).max(module.glwe_normalize_tmp_bytes())
}

pub(crate) fn glwe_public_key_generate_derived<BE, R, S>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    Module<BE>: GLWENormalize<BE>,
{
    {
        let sk_ref = sk.to_backend_ref();

        operand_degree(module.n(), &[res.n(), sk_ref.n()]);

        match sk_ref.dist {
            Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
            Distribution::ENCAPSULATED(_) => {
                panic!("invalid sk: encapsulated secrets cannot back a public key")
            }
            _ => {}
        }

        assert!(
            scratch.available() >= glwe_public_key_generate_tmp_bytes_derived(module, res),
            "insufficient scratch for GLWE public key generation"
        );
        let mut pk = res.to_backend_mut();
        for l in 0..pk.rank().as_usize() {
            let mut entry = GLWEPublicKeyAtViewMut::<BE>::at_view_mut(&mut pk, l);
            BE::glwe_encrypt_zero_sk(module, &mut entry, sk, source_xe, source_xa, scratch);
            module.glwe_normalize_assign(&mut entry, scratch);
        }
    }
    res.set_noise(Some(crate::ComponentNoise::from_secret_at(
        *sk.dist(),
        res.k(),
        res.rank().as_usize(),
    )));
    *res.dist_mut() = *sk.dist();
}

pub(crate) fn glwe_public_key_compressed_generate_tmp_bytes_derived<BE: EncryptionImpl, A: GLWEInfos>(
    module: &Module<BE>,
    infos: &A,
) -> usize {
    operand_degree(module.n(), &[infos.n()]);
    BE::glwe_compressed_encrypt_sk_tmp_bytes(module, infos)
}

pub(crate) fn glwe_public_key_compressed_generate_derived<BE, R, S>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    seed: [u8; 32],
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GLWEPublicKeyCompressedToBackendMut<BE> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
    S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
{
    {
        let sk_ref = sk.to_backend_ref();
        operand_degree(module.n(), &[res.n(), sk_ref.n()]);
        match sk_ref.dist {
            Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
            Distribution::ENCAPSULATED(_) => {
                panic!("invalid sk: encapsulated secrets cannot back a public key")
            }
            _ => {}
        }
    }
    assert!(
        scratch.available() >= glwe_public_key_compressed_generate_tmp_bytes_derived(module, res),
        "insufficient scratch for compressed GLWE public key generation"
    );
    let mut seeds = Source::new(seed);
    let entry_seeds: Vec<[u8; 32]> = (0..res.rank().as_usize()).map(|_| seeds.new_seed()).collect();
    {
        let mut pk = res.to_backend_mut();
        for (l, entry_seed) in entry_seeds.iter().enumerate() {
            BE::glwe_compressed_encrypt_zero_sk(module, &mut pk.at_view_mut(l), sk, *entry_seed, source_xe, scratch);
        }
    }
    res.seed_mut().copy_from_slice(&entry_seeds);
    res.set_noise(Some(crate::ComponentNoise::from_secret_at(
        *sk.dist(),
        res.k(),
        res.rank().as_usize(),
    )));
    *res.dist_mut() = *sk.dist();
}

pub(crate) fn glwe_encrypt_pk_derived<BE, R, P, K>(
    module: &Module<BE>,
    res: &mut R,
    pt: &P,
    pk: &K,
    source_xu: &mut Source,
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: GLWEToBackendRef<BE> + GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    BE::glwe_encrypt_pk_at_col(module, res, Some((pt, 0)), true, pk, source_xu, source_xe, scratch);
}

pub(crate) fn glwe_encrypt_pk_smudged_tmp_bytes_derived<BE: EncryptionImpl, R: GLWEInfos, K: GLWEInfos>(
    module: &Module<BE>,
    res_infos: &R,
    pk_infos: &K,
) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    BE::glwe_encrypt_pk_tmp_bytes(module, res_infos, pk_infos).max(module.vec_znx_normalize_tmp_bytes())
}

/// Public-key encryption without a body error, then the flood on the canonical
/// body at the output's precision, normalized once more.
pub(crate) fn glwe_encrypt_pk_smudged_derived<BE, R, P, K>(
    module: &Module<BE>,
    res: &mut R,
    pt: &P,
    pk: &K,
    flood: Noise,
    source_xu: &mut Source,
    source_xe: &mut Source,
    source_smudge: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: GLWEToBackendRef<BE> + GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    Module<BE>: VecZnxAddNoise<BE> + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes,
{
    let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().as_usize());
    flood.assert_valid_for(base2k, k);
    assert!(
        scratch.available() >= glwe_encrypt_pk_smudged_tmp_bytes_derived(module, res, pk),
        "insufficient scratch for smudged GLWE public-key encryption"
    );
    let metadata = crate::fresh_noise_model::public_key_encryption_plan::<BE, _, _>(
        res,
        pk,
        crate::fresh_noise_model::PublicKeyBodyNoise::Flood(flood),
    )
    .noise;
    BE::glwe_encrypt_pk_at_col(module, res, Some((pt, 0)), false, pk, source_xu, source_xe, scratch);
    res.set_noise(metadata);
    let mut res = res.to_backend_mut();
    module.vec_znx_add_noise(base2k, k, &mut res.data, 0, flood, source_smudge);
    module.vec_znx_normalize_assign(base2k, k, 0, &mut res.data, 0, scratch);
}

pub(crate) fn glwe_encrypt_zero_pk_derived<BE, R, K>(
    module: &Module<BE>,
    res: &mut R,
    pk: &K,
    source_xu: &mut Source,
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    BE::glwe_encrypt_pk_at_col::<R, GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>, K>(
        module, res, None, true, pk, source_xu, source_xe, scratch,
    );
}

pub(crate) fn ggsw_encrypt_pk_tmp_bytes_derived<BE: EncryptionImpl, R: GGSWInfos, K: GLWEInfos>(
    module: &Module<BE>,
    res_infos: &R,
    pk_infos: &K,
) -> usize
where
    Module<BE>: VecZnxNormalizeTmpBytes,
{
    operand_degree(module.n(), &[res_infos.n()]);
    BE::scratch_aligned(module.glwe_plaintext_bytes_of_from_infos(res_infos))
        + BE::glwe_encrypt_pk_tmp_bytes(module, res_infos, pk_infos).max(module.vec_znx_normalize_tmp_bytes())
}

pub(crate) fn ggsw_encrypt_pk_derived<BE, R, P, K>(
    module: &Module<BE>,
    res: &mut R,
    pt: &P,
    pk: &K,
    source_xu: &mut Source,
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: EncryptionImpl,
    R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
    P: ScalarZnxToBackendRef<BE> + ZnxInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    Module<BE>: VecZnxZero<BE> + VecZnxAddScalarAssign<BE> + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes,
{
    let metadata = crate::fresh_noise_model::public_key_encryption_plan::<BE, _, _>(
        res,
        pk,
        crate::fresh_noise_model::PublicKeyBodyNoise::Sampled,
    )
    .noise;
    operand_degree(module.n(), &[res.n(), pt.n().into(), pk.n()]);
    assert!(
        scratch.available() >= ggsw_encrypt_pk_tmp_bytes_derived(module, res, pk),
        "insufficient scratch for GGSW public-key encryption"
    );
    let tmp_bytes: usize = ggsw_encrypt_pk_tmp_bytes_derived(module, res, pk);
    {
        let (base2k, k): (usize, usize) = (res.base2k().into(), res.k().as_usize());
        let dsize: usize = res.dsize().into();
        let rank: usize = res.rank().into();
        let (mut tmp_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res);
        for row in 0..res.dnum().into() {
            module.vec_znx_zero(&mut tmp_pt.data, 0);
            module.vec_znx_add_scalar_assign(
                &mut tmp_pt.to_backend_mut().data,
                0,
                (dsize - 1) + row * dsize,
                &pt.to_backend_ref(),
                0,
            );
            // The encryption adds the plaintext's limbs to its accumulator as they are.
            module.vec_znx_normalize_assign(base2k, k, 0, &mut tmp_pt.data, 0, &mut scratch_1.borrow());
            for col in 0..rank + 1 {
                BE::glwe_encrypt_pk_at_col(
                    module,
                    &mut res.at_view_mut(row, col),
                    Some((&tmp_pt, col)),
                    true,
                    pk,
                    source_xu,
                    source_xe,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
    scratch.wipe(tmp_bytes);
    res.set_noise(metadata);
}

pub(crate) fn glwe_tensor_key_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    A: GGLWEInfos,
{
    operand_degree(module.n(), &[infos.n()]);

    let sk_prepared: usize = module.glwe_secret_prepared_bytes_of(infos.rank_out());
    let sk_tensor: usize = module.glwe_secret_tensor_bytes_of_from_infos(infos);

    let lvl_0: usize = sk_prepared;
    let lvl_1: usize = sk_tensor;
    let lvl_2_prepare: usize = module.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
    let lvl_2: usize = lvl_2_prepare;
    let lvl_3_encrypt: usize = BE::gglwe_encrypt_sk_tmp_bytes(module, infos);

    BE::scratch_aligned(lvl_0) + BE::scratch_aligned(lvl_1) + BE::scratch_aligned(lvl_2) + lvl_3_encrypt
}

pub(crate) fn glwe_tensor_key_encrypt_sk_derived<BE, R, S>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_eq!(res.rank_out(), sk.rank());
    assert_eq!(res.n(), sk.n());
    assert!(
        scratch.available() >= glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, res),
        "insufficient scratch for GLWE tensor key encryption"
    );
    let tmp_bytes: usize = glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, res);
    {
        let scratch = scratch.borrow();
        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(res.n(), res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(res.n().as_usize().into(), res.rank());
        let (mut tensor_scratch, scratch_3) = scratch_2.split_at(module.glwe_secret_tensor_prepare_tmp_bytes(res.rank()));
        module.glwe_secret_prepare(&mut sk_prepared, sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, sk, &mut tensor_scratch);

        let (mut enc_scratch, _scratch_4) = scratch_3.split_at(BE::gglwe_encrypt_sk_tmp_bytes(module, res));
        let sk_tensor_data = sk_tensor.data_mut();
        BE::gglwe_encrypt_sk(
            module,
            res,
            &sk_tensor_data,
            &sk_prepared,
            source_xe,
            source_xa,
            &mut enc_scratch,
        );
    }
    scratch.wipe(tmp_bytes);
}

pub(crate) fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    A: GGLWEInfos,
{
    operand_degree(module.n(), &[infos.n()]);

    let sk_prepared: usize = module.glwe_secret_prepared_bytes_of(infos.rank_out());
    let sk_tensor: usize = module.glwe_secret_tensor_bytes_of_from_infos(infos);

    let lvl_0: usize = sk_prepared;
    let lvl_1: usize = sk_tensor;
    let lvl_2_prepare: usize = module.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
    let lvl_2: usize = lvl_2_prepare;
    let lvl_3_encrypt: usize = BE::gglwe_compressed_encrypt_sk_tmp_bytes(module, infos);

    BE::scratch_aligned(lvl_0) + BE::scratch_aligned(lvl_1) + BE::scratch_aligned(lvl_2) + lvl_3_encrypt
}

pub(crate) fn glwe_tensor_key_compressed_encrypt_sk_derived<BE, R, S>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    seed_xa: [u8; 32],
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    R: GGLWEInfos + GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
    S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_eq!(res.rank_out(), sk.rank());
    assert_eq!(res.n(), sk.n());
    assert!(
        scratch.available() >= glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, res),
        "insufficient scratch for compressed GLWE tensor key encryption"
    );
    let tmp_bytes: usize = glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, res);
    {
        let scratch = scratch.borrow();
        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(res.n(), res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(res.n().as_usize().into(), res.rank());
        let (mut tensor_scratch, scratch_3) = scratch_2.split_at(module.glwe_secret_tensor_prepare_tmp_bytes(res.rank()));
        module.glwe_secret_prepare(&mut sk_prepared, sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, sk, &mut tensor_scratch);

        let (mut enc_scratch, _scratch_4) = scratch_3.split_at(BE::gglwe_compressed_encrypt_sk_tmp_bytes(module, res));
        let sk_tensor_data = sk_tensor.data_mut();
        BE::gglwe_compressed_encrypt_sk(
            module,
            res,
            &sk_tensor_data,
            &sk_prepared,
            seed_xa,
            source_xe,
            &mut enc_scratch,
        );
    }
    scratch.wipe(tmp_bytes);
}
