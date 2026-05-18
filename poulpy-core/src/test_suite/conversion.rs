use dashu_float::{FBig, round::mode::HalfEven};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize},
    layouts::{FillUniform, Module, ReaderFrom, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};

use byteorder::{LittleEndian, WriteBytesExt};

use crate::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEExpandLWE, GLWEExpandLWEMatrix, GLWEFromLWE, GLWENoise,
    GLWENormalize, GLWEToLWESwitchingKeyEncryptSk, LWEDecrypt, LWEEncryptSk, LWEFromGLWE, LWEMatrixDecrypt, LWEMatrixMul,
    LWEToGLWESwitchingKeyEncryptSk, ScratchArenaTakeCore,
    layouts::{
        Base2K, CoeffMatrixLayout, Degree, Dnum, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        GLWEToLWEKey, GLWEToLWEKeyLayout, GLWEToLWEKeyPrepared, GLWEToLWEKeyPreparedFactory, LWE, LWEInfos, LWELayout,
        LWEMatrixLayout, LWEPlaintext, LWESecret, LWEToGLWEKey, LWEToGLWEKeyLayout, LWEToGLWEKeyPrepared,
        LWEToGLWEKeyPreparedFactory, ModuleCoreAlloc, Rank, SecretConversion, TorusPrecision, prepared::GLWESecretPrepared,
    },
};

fn write_vec_znx_bytes(out: &mut Vec<u8>, n: u64, cols: u64, size: u64, max_size: u64, coeffs: &[i64]) {
    out.write_u64::<LittleEndian>(n).unwrap();
    out.write_u64::<LittleEndian>(cols).unwrap();
    out.write_u64::<LittleEndian>(size).unwrap();
    out.write_u64::<LittleEndian>(max_size).unwrap();

    let mut raw = Vec::with_capacity(std::mem::size_of_val(coeffs));
    for coeff in coeffs {
        raw.write_i64::<LittleEndian>(*coeff).unwrap();
    }

    out.write_u64::<LittleEndian>(raw.len() as u64).unwrap();
    out.extend_from_slice(&raw);
}

pub fn test_lwe_read_from_rejects_malformed_shape<BE: crate::test_suite::TestBackend>(_params: &TestParams, _module: &Module<BE>)
where
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    let infos = LWELayout {
        n: Degree(2),
        base2k: Base2K(32),
        k: TorusPrecision(64),
    };

    let mut lwe = LWE::<Vec<u8>>::alloc_from_infos(&infos);
    let mut bytes = Vec::new();

    bytes.write_u32::<LittleEndian>(32).unwrap();
    write_vec_znx_bytes(&mut bytes, 1, 1, 1, 1, &[123]);
    write_vec_znx_bytes(&mut bytes, 2, 1, 2, 2, &[1, 2, 3, 4]);

    let err = lwe
        .read_from(&mut &bytes[..])
        .expect_err("malformed LWE body/mask shape must be rejected");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

pub fn test_lwe_secret_from_glwe_secret_flattens_rank_and_preserves_metadata<BE: crate::test_suite::TestBackend>(
    _params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + SecretConversion<BE>,
{
    let rank = Rank(2);
    let mut source = Source::new([9u8; 32]);
    let mut sk_glwe = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_hw(3, &mut source);

    let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

    assert_eq!(sk_lwe.n(), Degree((module.n() * rank.as_usize()) as u32));
    assert_eq!(sk_lwe.dist(), crate::dist::Distribution::TernaryFixed(3));
}

pub fn test_glwe_base2k_conversion<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWENormalize<BE> + GLWESecretPreparedFactory<BE> + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let base2k: usize = params.base2k;

    for rank in 1_usize..3 {
        for bases in [[base2k, base2k - 3], [base2k - 3, base2k]] {
            let k_in = 4 * bases[0] + 1;
            let k_out = 4 * bases[0] + 1;

            let glwe_infos_in = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[0] as u32),
                k: TorusPrecision(k_in as u32),
                rank: Rank(rank as u32),
            })
            .unwrap();

            let glwe_infos_out: GLWELayout = GLWELayout {
                n: n_glwe,
                base2k: Base2K(bases[1] as u32),
                k: TorusPrecision(k_out as u32),
                rank: Rank(rank as u32),
            };

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prep: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prep, &sk);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .glwe_encrypt_sk_tmp_bytes(&glwe_infos_in)
                    .max(module.glwe_noise_tmp_bytes(&glwe_infos_out)),
            );

            let mut ct_in: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos_in);
            let mut ct_out: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos_out);

            let pt_in: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos_in);
            let pt_out: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos_out);

            module.glwe_encrypt_sk(
                &mut ct_in,
                &pt_in,
                &sk_prep,
                &glwe_infos_in,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut data: Vec<FBig<HalfEven>> = (0..module.n()).map(|_| FBig::ZERO).collect();
            ct_in.data().decode_vec_float(ct_in.base2k().into(), 0, &mut data);

            ct_out.fill_uniform(ct_out.base2k().into(), &mut source_xa);
            module.glwe_normalize(&mut ct_out, &ct_in, &mut scratch.borrow());

            let mut data_conv: Vec<FBig<HalfEven>> = (0..module.n()).map(|_| FBig::ZERO).collect();
            ct_out.data().decode_vec_float(ct_out.base2k().into(), 0, &mut data_conv);

            let noise_have = module
                .glwe_noise(&ct_out, &pt_out, &sk_prep, &mut scratch.borrow())
                .std()
                .log2();
            let noise_max = -(k_out as f64) + DEFAULT_SIGMA_XE.log2() + 0.50;

            assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}")
        }
    }
}

pub fn test_lwe_to_glwe<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEFromLWE<BE>
        + LWEToGLWESwitchingKeyEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + LWEEncryptSk<BE>
        + LWEToGLWEKeyPreparedFactory<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);
    let base2k: usize = params.base2k;

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);
    let k_ksk = 5 * base2k + 1;
    let k_glwe = 4 * base2k + 1;
    let k_lwe = 4 * base2k + 1;

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let lwe_to_glwe_infos = EncryptionLayout::new_from_default_sigma(LWEToGLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        dnum: Dnum(2),
        rank_out: rank,
    })
    .unwrap();

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32 - 1),
        k: TorusPrecision(k_glwe as u32),
        rank,
    };

    let lwe_infos = EncryptionLayout::new_from_default_sigma(LWELayout {
        n: n_lwe,
        base2k: Base2K(base2k as u32 - 2),
        k: TorusPrecision(k_lwe as u32),
    })
    .unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).lwe_to_glwe_key_encrypt_sk_tmp_bytes(&lwe_to_glwe_infos)
            | (module).glwe_from_lwe_tmp_bytes(&glwe_infos, &lwe_infos, &lwe_to_glwe_infos)
            | (module).glwe_decrypt_tmp_bytes(&glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_infos);
    lwe_pt.encode_i64(data, k_lwe_pt);

    let mut lwe_ct: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut lwe_ct,
        &lwe_pt,
        &sk_lwe,
        &lwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut ksk: LWEToGLWEKey<Vec<u8>> = module.lwe_to_glwe_key_alloc_from_infos(&lwe_to_glwe_infos);

    module.lwe_to_glwe_key_encrypt_sk(
        &mut ksk,
        &sk_lwe,
        &sk_glwe_prepared,
        &lwe_to_glwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut crate::test_suite::scratch_host_arena(&mut scratch),
    );

    let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);

    let mut ksk_prepared: LWEToGLWEKeyPrepared<BE::OwnedBuf, BE> = module.lwe_to_glwe_key_prepared_alloc_from_infos(&ksk);
    module.lwe_to_glwe_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

    module.glwe_from_lwe(
        &mut glwe_ct,
        &lwe_ct,
        &ksk_prepared,
        ksk_prepared.size(),
        &mut scratch.borrow(),
    );

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    module.glwe_decrypt(&glwe_ct, &mut glwe_pt, &sk_glwe_prepared, &mut scratch.borrow());

    let mut lwe_pt_conv = module.lwe_plaintext_alloc(glwe_pt.base2k(), lwe_pt.max_k());

    module.vec_znx_normalize(
        &mut vec_znx_backend_mut::<BE>(&mut lwe_pt_conv.data),
        glwe_pt.base2k().as_usize(),
        0,
        0,
        &vec_znx_backend_ref::<BE>(&lwe_pt.data),
        lwe_pt.base2k().as_usize(),
        0,
        &mut scratch.borrow(),
    );

    assert_eq!(glwe_pt.data.at(0, 0)[0], lwe_pt_conv.data.at(0, 0)[0]);
}

pub fn test_glwe_to_lwe<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEFromLWE<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + LWEDecrypt<BE>
        + LWEFromGLWE<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEToLWESwitchingKeyEncryptSk<BE>
        + GLWEToLWEKeyPreparedFactory<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n_glwe: Degree = Degree(module.n() as u32);
    let n_lwe: Degree = Degree(22);
    let base2k: usize = params.base2k;
    let k_ksk = 5 * base2k + 1;
    let k_glwe = 4 * base2k + 1;
    let k_lwe = 4 * base2k + 1;

    let rank: Rank = Rank(2);
    let k_lwe_pt: TorusPrecision = TorusPrecision(8);

    let glwe_to_lwe_infos = EncryptionLayout::new_from_default_sigma(GLWEToLWEKeyLayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        dnum: Dnum(2),
        rank_in: rank,
    })
    .unwrap();

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n_glwe,
        base2k: Base2K(base2k as u32 - 1),
        k: TorusPrecision(k_glwe as u32),
        rank,
    })
    .unwrap();

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe,
        base2k: Base2K(base2k as u32 - 2),
        k: TorusPrecision(k_lwe as u32),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).glwe_to_lwe_key_encrypt_sk_tmp_bytes(&glwe_to_lwe_infos)
            | (module).lwe_from_glwe_tmp_bytes(&lwe_infos, &glwe_infos, &glwe_to_lwe_infos)
            | (module).glwe_decrypt_tmp_bytes(&glwe_infos),
    );

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe);
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);

    let a_idx: usize = 1;

    let mut data: Vec<i64> = vec![0i64; module.n()];
    data[a_idx] = 17;
    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    glwe_pt.encode_vec_i64(&data, k_lwe_pt);

    let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
    module.glwe_encrypt_sk(
        &mut glwe_ct,
        &glwe_pt,
        &sk_glwe_prepared,
        &glwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut ksk: GLWEToLWEKey<Vec<u8>> = module.glwe_to_lwe_key_alloc_from_infos(&glwe_to_lwe_infos);

    module.glwe_to_lwe_key_encrypt_sk(
        &mut ksk,
        &sk_lwe,
        &sk_glwe,
        &glwe_to_lwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.arena(),
    );

    let mut lwe_ct: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);

    let mut ksk_prepared: GLWEToLWEKeyPrepared<BE::OwnedBuf, BE> = module.glwe_to_lwe_key_prepared_alloc_from_infos(&ksk);
    module.glwe_to_lwe_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

    module.lwe_from_glwe(
        &mut lwe_ct,
        &glwe_ct,
        a_idx,
        &ksk_prepared,
        ksk_prepared.size(),
        &mut scratch.borrow(),
    );

    let mut lwe_pt: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_infos);
    module.lwe_decrypt(&lwe_ct, &mut lwe_pt, &sk_lwe, &mut scratch.borrow());

    let mut glwe_pt_conv = GLWEPlaintext::<Vec<u8>>::alloc(glwe_ct.n(), lwe_pt.base2k(), lwe_pt.max_k());

    module.vec_znx_normalize(
        &mut vec_znx_backend_mut::<BE>(&mut glwe_pt_conv.data),
        lwe_pt.base2k().as_usize(),
        0,
        0,
        &vec_znx_backend_ref::<BE>(&glwe_pt.data),
        glwe_ct.base2k().as_usize(),
        0,
        &mut scratch.borrow(),
    );

    assert_eq!(glwe_pt_conv.data.at(0, 0)[a_idx], lwe_pt.data.at(0, 0)[0]);
}

pub fn test_glwe_expand_lwe<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + LWEDecrypt<BE>
        + GLWEExpandLWE<BE>
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k = 4 * base2k + 1;
    let k_pt = TorusPrecision(8);

    for rank in [Rank(1), Rank(2)] {
        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(k as u32),
            rank,
        })
        .unwrap();

        let lwe_infos = LWELayout {
            n: Degree(n as u32 * rank.0),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(k as u32),
        };

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_glwe_prep: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
        module.glwe_secret_prepare(&mut sk_glwe_prep, &sk_glwe);

        let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

        let a_idx: usize = 3;
        let mut data: Vec<i64> = vec![0i64; n];
        data[a_idx] = 17;

        let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        glwe_pt.encode_vec_i64(&data, k_pt);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module.glwe_encrypt_sk_tmp_bytes(&glwe_infos)
                | module.lwe_decrypt_tmp_bytes(&lwe_infos)
                | module.glwe_expand_lwe_tmp_bytes(&lwe_infos, &glwe_infos),
        );

        let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut glwe_ct,
            &glwe_pt,
            &sk_glwe_prep,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut lwe_cts: Vec<LWE<Vec<u8>>> = (0..n).map(|_| module.lwe_alloc_from_infos(&lwe_infos)).collect();
        module.glwe_expand_lwe(lwe_cts.as_mut_slice(), &glwe_ct, &mut scratch.borrow());

        let mut lwe_pt: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_infos);
        module.lwe_decrypt(&lwe_cts[a_idx], &mut lwe_pt, &sk_lwe, &mut scratch.borrow());

        let mut glwe_pt_conv = GLWEPlaintext::<Vec<u8>>::alloc(glwe_ct.n(), lwe_pt.base2k(), lwe_pt.max_k());
        module.vec_znx_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut glwe_pt_conv.data),
            lwe_pt.base2k().as_usize(),
            0,
            0,
            &vec_znx_backend_ref::<BE>(&glwe_pt.data),
            glwe_ct.base2k().as_usize(),
            0,
            &mut scratch.borrow(),
        );

        assert_eq!(
            glwe_pt_conv.data.at(0, 0)[a_idx],
            lwe_pt.data.at(0, 0)[0],
            "rank={} failed",
            rank.0
        );
    }
}

pub fn test_glwe_expand_lwe_rejects_incompatible_lwe_layout<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEExpandLWE<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let k = 2 * base2k;

    let glwe_infos = GLWELayout {
        n: Degree(module.n() as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k as u32),
        rank: Rank(1),
    };

    let bad_lwe_infos = LWELayout {
        n: Degree(module.n() as u32),
        base2k: Base2K((base2k - 1) as u32),
        k: TorusPrecision(k as u32),
    };

    let glwe_ct = module.glwe_alloc_from_infos(&glwe_infos);
    let mut lwe_out = module.lwe_alloc_from_infos(&bad_lwe_infos);

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.glwe_expand_lwe_tmp_bytes(&bad_lwe_infos, &glwe_infos);
        }))
        .is_err(),
        "glwe_expand_lwe_tmp_bytes must reject incompatible LWE layout"
    );

    let mut scratch = ScratchOwned::<BE>::alloc(0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.glwe_expand_lwe(std::slice::from_mut(&mut lwe_out), &glwe_ct, &mut scratch.borrow());
        }))
        .is_err(),
        "glwe_expand_lwe must reject incompatible LWE layout"
    );
}

pub fn test_glwe_expand_lwe_matrix_decrypt<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEExpandLWEMatrix<BE>
        + LWEMatrixDecrypt<BE>
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k = 4 * base2k + 1;
    let k_pt = TorusPrecision(8);

    for rank in [Rank(1), Rank(2)] {
        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(k as u32),
            rank,
        })
        .unwrap();

        let matrix_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32 * rank.0),
            base2k: glwe_infos.base2k(),
            k: glwe_infos.max_k(),
        };

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);

        let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_glwe_prep: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
        module.glwe_secret_prepare(&mut sk_glwe_prep, &sk_glwe);
        let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

        let mut data: Vec<i64> = vec![0i64; n];
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i as i64 % 7) - 3;
        }

        let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        glwe_pt.encode_vec_i64(&data, k_pt);

        let scratch_bytes = module
            .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
            .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &glwe_infos))
            .max(module.lwe_matrix_decrypt_tmp_bytes(&matrix_infos));
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

        let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut glwe_ct,
            &glwe_pt,
            &sk_glwe_prep,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut glwe_pt_dec: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.glwe_decrypt(&glwe_ct, &mut glwe_pt_dec, &sk_glwe_prep, &mut scratch.borrow());

        let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
        module.glwe_expand_lwe_matrix(&mut lwe_matrix, &glwe_ct, &mut scratch.borrow());

        let mut matrix_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.lwe_matrix_decrypt(&lwe_matrix, &mut matrix_pt, &sk_lwe, &mut scratch.borrow());

        for limb in 0..glwe_pt_dec.data.size() {
            assert_eq!(
                &glwe_pt_dec.data.at(0, limb)[..n],
                &matrix_pt.data().at(0, limb)[..n],
                "rank={} limb={} failed",
                rank.0,
                limb
            );
        }
    }
}

pub fn test_lwe_matrix_mul_identity<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEExpandLWEMatrix<BE>
        + LWEMatrixDecrypt<BE>
        + LWEMatrixMul<BE>
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let rows: usize = n.min(8);
    let base2k: usize = params.base2k;
    let k = 4 * base2k + 1;
    let k_pt = TorusPrecision(8);
    let rank = Rank(1);

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k as u32),
        rank,
    })
    .unwrap();

    let matrix_infos = LWEMatrixLayout {
        rows,
        n: Degree(n as u32 * rank.0),
        base2k: glwe_infos.base2k(),
        k: glwe_infos.max_k(),
    };

    let u_infos = CoeffMatrixLayout {
        n: Degree(rows as u32),
        rows_out: rows,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision((2 * base2k) as u32),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prep: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_glwe_prep, &sk_glwe);
    let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

    let mut data: Vec<i64> = vec![0i64; n];
    for (i, x) in data.iter_mut().enumerate() {
        *x = (i as i64 % 5) - 2;
    }

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    glwe_pt.encode_vec_i64(&data, k_pt);

    let scratch_bytes = module
        .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
        .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &glwe_infos))
        .max(module.lwe_matrix_mul_tmp_bytes(&matrix_infos, &u_infos, &matrix_infos))
        .max(module.lwe_matrix_decrypt_tmp_bytes(&matrix_infos));
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
    module.glwe_encrypt_sk(
        &mut glwe_ct,
        &glwe_pt,
        &sk_glwe_prep,
        &glwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &glwe_ct, &mut scratch.borrow());

    let mut u = module.coeff_matrix_alloc_from_infos(&u_infos);
    for row in 0..rows {
        u.data_mut().at_mut(row, 0)[row] = 1;
    }

    let mut res_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.lwe_matrix_mul(&mut res_matrix, &u, &lwe_matrix, &mut scratch.borrow());

    let mut matrix_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut res_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    module.lwe_matrix_decrypt(&lwe_matrix, &mut matrix_pt, &sk_lwe, &mut scratch.borrow());
    module.lwe_matrix_decrypt(&res_matrix, &mut res_pt, &sk_lwe, &mut scratch.borrow());

    for limb in 0..matrix_pt.data().size() {
        assert_eq!(
            &matrix_pt.data().at(0, limb)[..rows],
            &res_pt.data().at(0, limb)[..rows],
            "limb={limb} failed"
        );
    }
}

pub fn test_lwe_matrix_mul_decrypts_to_plain_product<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEExpandLWEMatrix<BE> + LWEMatrixDecrypt<BE> + LWEMatrixMul<BE> + SecretConversion<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let rows: usize = n.min(8);
    let base2k: usize = params.base2k;
    let k = 4 * base2k + 1;
    let k_pt = TorusPrecision(8);
    let rank = Rank(1);

    let glwe_infos = GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k as u32),
        rank,
    };

    let matrix_infos = LWEMatrixLayout {
        rows,
        n: Degree(n as u32 * rank.0),
        base2k: glwe_infos.base2k(),
        k: glwe_infos.max_k(),
    };

    let u_infos = CoeffMatrixLayout {
        n: Degree(rows as u32),
        rows_out: rows,
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(base2k as u32),
    };

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 3) - 1).collect();
    let mut want_data: Vec<i64> = vec![0; n];
    for row in 0..rows {
        want_data[row] += data[row];
        if row > 0 {
            want_data[row] -= data[row - 1];
        }
        if row + 1 < rows {
            want_data[row] += 2 * data[row + 1];
        }
    }

    let mut glwe_pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    glwe_pt.encode_vec_i64(&data, k_pt);

    let mut glwe_ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
    for limb in 0..glwe_pt.data.size() {
        glwe_ct.data.at_mut(0, limb).copy_from_slice(glwe_pt.data.at(0, limb));
    }

    let mut u = module.coeff_matrix_alloc_from_infos(&u_infos);
    for row in 0..rows {
        u.data_mut().at_mut(row, 0)[row] = 1;
        if row > 0 {
            u.data_mut().at_mut(row, 0)[row - 1] = -1;
        }
        if row + 1 < rows {
            u.data_mut().at_mut(row, 0)[row + 1] = 2;
        }
    }

    let scratch_bytes = module
        .glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &glwe_infos)
        .max(module.lwe_matrix_mul_tmp_bytes(&matrix_infos, &u_infos, &matrix_infos))
        .max(module.lwe_matrix_decrypt_tmp_bytes(&matrix_infos));
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &glwe_ct, &mut scratch.borrow());

    let mut res_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.lwe_matrix_mul(&mut res_matrix, &u, &lwe_matrix, &mut scratch.borrow());

    let mut res_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    module.lwe_matrix_decrypt(&res_matrix, &mut res_pt, &sk_lwe, &mut scratch.borrow());

    let mut want_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    want_pt.data_mut().encode_vec_i64(base2k, 0, k_pt.as_usize(), &want_data);

    for limb in 0..want_pt.data().size() {
        assert_eq!(
            &want_pt.data().at(0, limb)[..rows],
            &res_pt.data().at(0, limb)[..rows],
            "limb={limb} failed"
        );
    }
}
