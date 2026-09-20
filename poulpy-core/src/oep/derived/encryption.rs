//! Encryption operations expressed entirely through other core operations.
//!
//! Defaults dispatch through the selected backend, so overriding a primitive
//! also changes every derived operation that uses it.

#![allow(clippy::too_many_arguments)]

use crate::{
    Distribution, EncryptionInfos, GetDistribution, GetDistributionMut, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretPreparedFactory,
        GLWESecretTensorFactory, GLWESecretToBackendRef, GLWEToBackendMut, LWEToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
    oep::EncryptionImpl,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchArena, ScratchOwned},
    source::Source,
};

/// Reconstruct a mask using the backend's source-driven mask sampler.
pub fn fill_glwe_mask_from_seed_derived<BE: EncryptionImpl, R: GLWEToBackendMut<BE>>(
    module: &Module<BE>,
    base2k: usize,
    res: &mut R,
    res_col: usize,
    rank: usize,
    seed_xa: [u8; 32],
) {
    BE::fill_glwe_mask_from_source(module, base2k, res, res_col, rank, &mut Source::new(seed_xa));
}

/// Reconstruct an LWE mask using the backend's source-driven mask sampler.
pub fn fill_lwe_mask_from_seed_derived<BE: EncryptionImpl, R: LWEToBackendMut<BE>>(
    module: &Module<BE>,
    base2k: usize,
    res: &mut R,
    seed_xa: [u8; 32],
) {
    BE::fill_lwe_mask_from_source(module, base2k, res, &mut Source::new(seed_xa));
}

pub fn glwe_public_key_generate_derived<BE, R, S, E>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    enc_infos: &E,
    source_xe: &mut Source,
    source_xa: &mut Source,
) where
    BE: EncryptionImpl,
    R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    E: EncryptionInfos,
    S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
{
    {
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.n(), module.n() as u32);
        assert_eq!(sk_ref.n(), module.n() as u32);

        match sk_ref.dist {
            Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
            Distribution::ENCAPSULATED(name) => {
                panic!("invalid sk: {name} is tagged for encapsulation and cannot back a public key")
            }
            _ => {}
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(BE::glwe_encrypt_sk_tmp_bytes(module, res));
        BE::glwe_encrypt_zero_sk(module, res, sk, enc_infos, source_xe, source_xa, &mut scratch.borrow());
    }
    *res.dist_mut() = *sk.dist();
}

pub fn glwe_tensor_key_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    A: GGLWEInfos,
{
    assert_eq!(module.n() as u32, infos.n());

    let sk_prepared: usize = module.glwe_secret_prepared_bytes_of(infos.rank_out());
    let sk_tensor: usize = module.glwe_secret_tensor_bytes_of_from_infos(infos);

    let lvl_0: usize = sk_prepared;
    let lvl_1: usize = sk_tensor;
    let lvl_2_prepare: usize = module.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
    let lvl_2: usize = lvl_2_prepare;
    let lvl_3_encrypt: usize = BE::gglwe_encrypt_sk_tmp_bytes(module, infos);

    BE::scratch_aligned(lvl_0) + BE::scratch_aligned(lvl_1) + BE::scratch_aligned(lvl_2) + lvl_3_encrypt
}

pub fn glwe_tensor_key_encrypt_sk_derived<BE, R, S, E>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    enc_infos: &E,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    E: EncryptionInfos,
    S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_eq!(res.rank_out(), sk.rank());
    assert_eq!(res.n(), sk.n());
    assert!(
        scratch.available() >= glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, res),
        "scratch.available(): {} < GLWETensorKeyEncryptSk::glwe_tensor_key_encrypt_sk_tmp_bytes: {}",
        scratch.available(),
        glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, res)
    );

    let scratch = scratch.borrow();
    let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(module, res.rank());
    let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(module.n().into(), res.rank());
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
        enc_infos,
        source_xe,
        source_xa,
        &mut enc_scratch,
    );
}

pub fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    A: GGLWEInfos,
{
    assert_eq!(module.n() as u32, infos.n());

    let sk_prepared: usize = module.glwe_secret_prepared_bytes_of(infos.rank_out());
    let sk_tensor: usize = module.glwe_secret_tensor_bytes_of_from_infos(infos);

    let lvl_0: usize = sk_prepared;
    let lvl_1: usize = sk_tensor;
    let lvl_2_prepare: usize = module.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
    let lvl_2: usize = lvl_2_prepare;
    let lvl_3_encrypt: usize = BE::gglwe_compressed_encrypt_sk_tmp_bytes(module, infos);

    BE::scratch_aligned(lvl_0) + BE::scratch_aligned(lvl_1) + BE::scratch_aligned(lvl_2) + lvl_3_encrypt
}

pub fn glwe_tensor_key_compressed_encrypt_sk_derived<BE, R, S, E>(
    module: &Module<BE>,
    res: &mut R,
    sk: &S,
    seed_xa: [u8; 32],
    enc_infos: &E,
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    BE: EncryptionImpl,
    R: GGLWEInfos + GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
    E: EncryptionInfos,
    S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_eq!(res.rank_out(), sk.rank());
    assert_eq!(res.n(), sk.n());
    assert!(
        scratch.available() >= glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, res),
        "scratch.available(): {} < GLWETensorKeyCompressedEncryptSk::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes: {}",
        scratch.available(),
        glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, res)
    );

    let scratch = scratch.borrow();
    let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(module, res.rank());
    let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(module.n().into(), res.rank());
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
        enc_infos,
        source_xe,
        &mut enc_scratch,
    );
}
