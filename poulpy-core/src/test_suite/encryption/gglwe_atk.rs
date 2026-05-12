use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismBackend, VecZnxFillUniformSourceBackend},
    layouts::{Backend, GaloisElement, Module, ScalarZnx, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEKeyswitch, GLWEAutomorphismKeyCompressedEncryptSk, GLWEAutomorphismKeyEncryptSk,
    GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptSk, ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyDecompress, GLWEAutomorphismKeyLayout, GLWEInfos, GLWESecret,
        GLWESecretPreparedFactory, GLWESwitchingKeyDecompress, ModuleCoreAlloc, ModuleCoreCompressedAlloc,
        compressed::GLWEAutomorphismKeyCompressed, prepared::GLWESecretPrepared,
    },
    noise::GGLWENoise,
};

pub fn test_gglwe_automorphism_key_encrypt_sk<BE: crate::test_suite::TestBackend + Backend<OwnedBuf = Vec<u8>>>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + GLWESwitchingKeyDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>
        + GGLWENoise<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + VecZnxAutomorphismBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let dsize: usize = k_ksk.div_ceil(base2k) - 1;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

            let atk_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut atk: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
                    .max(module.gglwe_noise_tmp_bytes(&atk_infos)),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&atk_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let p = -5;

            module.glwe_automorphism_key_encrypt_sk(
                &mut atk,
                p,
                &sk,
                &atk_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data.as_ref()), sk.data.n(), sk.data.cols());
            let mut sk_out_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_out.data.data.as_ref()),
                sk_out.data.n(),
                sk_out.data.cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_out_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_out_backend);
                for i in 0..atk.rank().into() {
                    module.vec_znx_automorphism_backend(
                        module.galois_element_inv(p),
                        &mut sk_out_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_out_backend.data, sk_out.data.data.as_mut());
            let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(sk_out.rank());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k_ksk as f64) + 0.5;

            for row in 0..atk.dnum().as_usize() {
                for col in 0..atk.rank().as_usize() {
                    let noise_have = atk
                        .key
                        .noise(module, row, col, &sk.data.to_ref(), &sk_out_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    assert!(
                        noise_have <= max_noise,
                        "row:{row} col:{col} noise_have:{noise_have} > max_noise:{max_noise}",
                    );
                }
            }
        }
    }
}

pub fn test_gglwe_automorphism_key_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyCompressedEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + GLWEAutomorphismKeyDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let max_dsize: usize = k_ksk.div_ceil(base2k) - 1;
    for rank in 2_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k_ksk - dsize * base2k) / (dsize * base2k);

            let atk_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut atk_compressed: GLWEAutomorphismKeyCompressed<Vec<u8>> =
                module.glwe_automorphism_key_compressed_alloc_from_infos(&atk_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(&atk_infos)
                    .max(module.gglwe_noise_tmp_bytes(&atk_infos)),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&atk_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let p: i64 = -5;

            let seed_xa: [u8; 32] = [1u8; 32];

            module.glwe_automorphism_key_compressed_encrypt_sk(
                &mut atk_compressed,
                p,
                &sk,
                seed_xa,
                &atk_infos,
                &mut source_xe,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            let mut sk_out: GLWESecret<Vec<u8>> = sk.clone();
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data.as_ref()), sk.data.n(), sk.data.cols());
            let mut sk_out_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_out.data.data.as_ref()),
                sk_out.data.n(),
                sk_out.data.cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_out_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_out_backend);
                for i in 0..atk_compressed.rank().into() {
                    module.vec_znx_automorphism_backend(
                        module.galois_element_inv(p),
                        &mut sk_out_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_out_backend.data, sk_out.data.data.as_mut());
            let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(sk_out.rank());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            let mut atk: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
            module.decompress_automorphism_key(&mut atk, &atk_compressed);

            let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k_ksk as f64) + 0.5;

            for row in 0..atk.dnum().as_usize() {
                for col in 0..atk.rank().as_usize() {
                    let noise_have = atk
                        .key
                        .noise(module, row, col, &sk.data.to_ref(), &sk_out_prepared, &mut scratch.borrow())
                        .std()
                        .log2();

                    assert!(
                        noise_have <= max_noise,
                        "row:{row} col:{col} noise_have:{noise_have} > max_noise:{max_noise}",
                    );
                }
            }
        }
    }
}
