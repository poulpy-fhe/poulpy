use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWENoise, GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk, ScratchArenaTakeCore,
    decryption::GLWEDecrypt,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dsize, GGLWEDecompress, GGLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory,
        GLWETensorKey, GLWETensorKeyCompressed, GLWETensorKeyDecompress, GLWETensorKeyLayout, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, prepared::GLWESecretPrepared,
    },
};

pub fn test_gglwe_tensor_key_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensorKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWESecretTensorFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 2_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos = EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        })
        .unwrap();

        let mut tensor_key: GLWETensorKey<Vec<u8>> = module.glwe_tensor_key_alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_tensor_key_encrypt_sk_tmp_bytes(&tensor_key_infos)
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.gglwe_noise_tmp_bytes(&tensor_key_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&tensor_key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.glwe_tensor_key_encrypt_sk(
            &mut tensor_key,
            &sk,
            &tensor_key_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = module.glwe_secret_tensor_alloc_from_infos(&sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, &mut crate::test_suite::scratch_host_arena(&mut scratch));

        let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k as f64) + 0.5;

        for row in 0..tensor_key.dnum().as_usize() {
            for col in 0..tensor_key.rank_in().as_usize() {
                let noise_have = tensor_key
                    .0
                    .noise(
                        module,
                        row,
                        col,
                        &sk_tensor.data.to_ref(),
                        &sk_prepared,
                        &mut scratch.borrow(),
                    )
                    .std()
                    .log2();
                assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
            }
        }
    }
}

pub fn test_gglwe_tensor_key_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensorKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensorKeyCompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretTensorFactory<BE>
        + GGLWENoise<BE>
        + GGLWEDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let tensor_key_infos = EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            dnum: dnum.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        })
        .unwrap();

        let mut tensor_key_compressed: GLWETensorKeyCompressed<Vec<u8>> =
            module.glwe_tensor_key_compressed_alloc_from_infos(&tensor_key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(&tensor_key_infos)
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.gglwe_noise_tmp_bytes(&tensor_key_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&tensor_key_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let seed_xa: [u8; 32] = [1u8; 32];

        module.glwe_tensor_key_compressed_encrypt_sk(
            &mut tensor_key_compressed,
            &sk,
            seed_xa,
            &tensor_key_infos,
            &mut source_xe,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );

        let mut tensor_key: GLWETensorKey<Vec<u8>> = module.glwe_tensor_key_alloc_from_infos(&tensor_key_infos);
        module.decompress_tensor_key(&mut tensor_key, &tensor_key_compressed);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = module.glwe_secret_tensor_alloc_from_infos(&sk);
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, &mut crate::test_suite::scratch_host_arena(&mut scratch));

        let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k as f64) + 0.5;

        for row in 0..tensor_key.dnum().as_usize() {
            for col in 0..tensor_key.rank_in().as_usize() {
                let noise_have = tensor_key
                    .0
                    .noise(
                        module,
                        row,
                        col,
                        &sk_tensor.data.to_ref(),
                        &sk_prepared,
                        &mut scratch.borrow(),
                    )
                    .std()
                    .log2();
                assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
            }
        }
    }
}
