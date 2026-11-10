use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy},
    layouts::{Module, Ring, ScalarZnx, ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::api::GLWESecretSampling;
use crate::test_suite::poisoned_scratch;
use crate::{
    GGLWENoise, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyDecompress, GGLWEToGGSWKeyEncryptSk,
    api::{GLWESecretPreparedFactory, GLWESecretTensorFromGLWESecret},
    decryption::GLWEDecrypt,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dsize, GGLWE, GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyAlloc, GGLWEToGGSWKeyCompressed, GGLWEToGGSWKeyCompressedAlloc,
        GGLWEToGGSWKeyLayout, GLWESecret, GLWESecretAlloc, GLWESecretTensor, GLWESecretTensorAlloc, LWEInfos,
        prepared::GLWESecretPrepared,
    },
};

pub fn test_gglwe_to_ggsw_key_encrypt_sk<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEToGGSWKeyEncryptSk<BE, Rg>
        + GLWESecretTensorFromGLWESecret<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>
        + GLWEDecrypt<BE, Rg>
        + GGLWENoise<BE, Rg>
        + VecZnxCopy<BE, Rg>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 2_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let key_infos = GGLWEToGGSWKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, Dsize(1)),
            rank: rank.into(),
        };

        let mut key: GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord, Rg> = module.gglwe_to_ggsw_key_alloc_from_infos(&key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_secret_tensor_from_glwe_secret_tmp_bytes(rank.into())
                .max(module.gglwe_noise_tmp_bytes(&key_infos)),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc_from_infos(&key_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.gglwe_to_ggsw_key_encrypt_sk(
            &mut key,
            &sk,
            &mut source_xe,
            &mut source_xa,
            &mut poisoned_scratch::<BE>(module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&key_infos)).borrow(),
        );

        let mut sk_tensor: GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_tensor_alloc_from_infos(&sk);
        module.glwe_secret_tensor_from_glwe_secret(
            &mut sk_tensor,
            &sk,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );

        let max_noise = DEFAULT_SIGMA_XE.log2() + 0.5 - (key_infos.k().as_usize() as f64);

        let mut pt_want: ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> = module.scalar_znx_alloc(module.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                let (row, col) = if i > j { (j, i) } else { (i, j) };
                let sk_tensor_col = row * rank + col - (row * (row + 1) / 2);
                let mut pt_want_backend =
                    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendMut<BE, Rg>>::as_vec_znx_backend_mut(
                        &mut pt_want,
                    );
                let sk_tensor_backend =
                    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendRef<BE, Rg>>::as_vec_znx_backend(
                        &sk_tensor.data,
                    );
                module.vec_znx_copy(&mut pt_want_backend, j, &sk_tensor_backend, sk_tensor_col);
            }

            let ksk: &GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> = key.at(i);
            for row in 0..ksk.dnum().as_usize() {
                for col in 0..ksk.rank_in().as_usize() {
                    let noise_have = ksk
                        .noise_stats(module, row, col, &pt_want.to_ref(), &sk_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                }
            }
        }
    }
}

pub fn test_gglwe_to_ggsw_key_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEToGGSWKeyCompressedEncryptSk<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>
        + GLWEDecrypt<BE, Rg>
        + GLWESecretTensorFromGLWESecret<BE, Rg>
        + GGLWENoise<BE, Rg>
        + GGLWEToGGSWKeyDecompress<BE, Rg>
        + VecZnxCopy<BE, Rg>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let dnum: usize = k / base2k;

        let key_infos = GGLWEToGGSWKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, Dsize(1)),
            rank: rank.into(),
        };

        let mut key_compressed: GGLWEToGGSWKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg> =
            module.gglwe_to_ggsw_key_compressed_alloc_from_infos(&key_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_secret_tensor_from_glwe_secret_tmp_bytes(rank.into())
                .max(module.gglwe_noise_tmp_bytes(&key_infos)),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc_from_infos(&key_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let seed_xa: [u8; 32] = [1u8; 32];

        module.gglwe_to_ggsw_key_compressed_encrypt_sk(
            &mut key_compressed,
            &sk,
            seed_xa,
            &mut source_xe,
            &mut poisoned_scratch::<BE>(module.gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes(&key_infos)).borrow(),
        );

        let mut key: GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord, Rg> = module.gglwe_to_ggsw_key_alloc_from_infos(&key_infos);
        module.gglwe_to_ggsw_key_decompress(&mut key, &key_compressed);

        let mut sk_tensor: GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_tensor_alloc_from_infos(&sk);
        module.glwe_secret_tensor_from_glwe_secret(
            &mut sk_tensor,
            &sk,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );

        let max_noise = DEFAULT_SIGMA_XE.log2() + 0.5 - (key_infos.k().as_usize() as f64);

        let mut pt_want: ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> = module.scalar_znx_alloc(module.n(), rank);

        for i in 0..rank {
            for j in 0..rank {
                let (row, col) = if i > j { (j, i) } else { (i, j) };
                let sk_tensor_col = row * rank + col - (row * (row + 1) / 2);
                let mut pt_want_backend =
                    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendMut<BE, Rg>>::as_vec_znx_backend_mut(
                        &mut pt_want,
                    );
                let sk_tensor_backend =
                    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendRef<BE, Rg>>::as_vec_znx_backend(
                        &sk_tensor.data,
                    );
                module.vec_znx_copy(&mut pt_want_backend, j, &sk_tensor_backend, sk_tensor_col);
            }

            let ksk: &GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> = key.at(i);
            for row in 0..ksk.dnum().as_usize() {
                for col in 0..ksk.rank_in().as_usize() {
                    let noise_have = ksk
                        .noise_stats(module, row, col, &pt_want.to_ref(), &sk_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                }
            }
        }
    }
}
