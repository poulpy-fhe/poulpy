use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend, VecZnxAutomorphismAssignBackend,
        VecZnxAutomorphismAssignTmpBytes, VecZnxFillUniformSourceBackend,
    },
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
    test_suite::vec_znx_backend_mut,
};

use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWENoise, GLWENormalize,
    api::GLWEBytesOf,
    default::{keyswitching::GLWEKeyswitchInternal, operations::GLWENormalizeDefault},
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dsize, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout,
        GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc, TorusPrecision, WithEffectiveDsize,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
        resolve_gglwe_key_use,
    },
    noise::GGLWENoiseModel,
    oep::{GLWEAutomorphismDefault, GLWEKeyswitchDefault},
};
use poulpy_hal::api::{
    VecZnxBigAddSmallAssign, VecZnxBigAutomorphismAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftBytesOf, VecZnxIdftApply,
};

pub fn test_glwe_automorphism<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = base2k - 2;
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_out: usize = k_in + key_base2k * dsize; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(key_base2k * dsize);

            let ct_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                rank: rank.into(),
                dsize: dsize.into(),
            })
            .unwrap();

            let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
            let mut ct_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
            let mut ct_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
            let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
            let mut pt_out: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source_backend(
                in_base2k,
                &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                0,
                &mut source_xa,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct_out)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct_in)
                    | module.glwe_automorphism_tmp_bytes(&ct_out, &ct_in, &autokey),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_out);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &autokey_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.glwe_encrypt_sk(
                &mut ct_in,
                &pt_in,
                &sk_prepared,
                &ct_in_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, &mut scratch.borrow());

            module.glwe_automorphism(&mut ct_out, &ct_in, &autokey_prepared, &mut scratch.borrow());

            let max_noise: f64 = autokey_infos.log2_std_noise_keyswitch(
                &ct_in_infos,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            );

            module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());
            module.vec_znx_automorphism_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut pt_out.data),
                0,
                &mut scratch.borrow(),
            );

            assert!(
                module
                    .glwe_noise(&ct_out, &pt_out, &sk_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
                    <= max_noise + 1.0
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism_assign<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(key_base2k * dsize);

            let ct_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            })
            .unwrap();

            let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                rank: rank.into(),
                dsize: dsize.into(),
            })
            .unwrap();

            let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
            let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source_backend(
                out_base2k,
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                0,
                &mut source_xa,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct)
                    | module.glwe_automorphism_tmp_bytes(&ct, &ct, &autokey),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &autokey_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.glwe_encrypt_sk(
                &mut ct,
                &pt_want,
                &sk_prepared,
                &ct_out_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, &mut scratch.borrow());

            module.glwe_automorphism_assign(&mut ct, &autokey_prepared, &mut scratch.borrow());

            let max_noise: f64 = autokey_infos.log2_std_noise_keyswitch(
                &ct_out_infos,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            );

            module.vec_znx_automorphism_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                0,
                &mut scratch.borrow(),
            );

            assert!(
                module
                    .glwe_noise(&ct, &pt_want, &sk_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
                    <= max_noise + 1.0
            )
        }
    }
}

/// A fine automorphism key used through a coarsening decrypts, and is as quiet
/// as a natively generated key of the decomposition it stands in for.
pub fn test_glwe_automorphism_selected<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEAutomorphismDefault<BE>
        + VecZnxAddAssignBackend<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + GLWEBytesOf<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let p: i64 = -5;
    let n: usize = module.n();
    let rank: usize = 1;
    // (parent dnum at dsize 1, coarsening factor, active coarse digits)
    for (dnum, s, r_active) in [(8usize, 2usize, 2usize), (8, 2, 3), (12, 4, 2)] {
        let effective_dsize: Dsize = Dsize(s as u32);
        let k_in: usize = r_active * s * base2k;
        let k_out: usize = k_in + base2k * s;

        let ct_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_in.into(),
            rank: rank.into(),
        })
        .unwrap();
        let ct_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_out.into(),
            rank: rank.into(),
        };
        // The physical parent: a fine decomposition the policy never asks for directly.
        let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        })
        .unwrap();

        let use_ = *resolve_gglwe_key_use(&autokey_infos, TorusPrecision(k_in as u32), effective_dsize)
            .expect("valid layout")
            .expect("parent realizes the coarsening")
            .active()
            .expect("positive precision");
        assert_eq!(use_.logical_layout().dnum.as_usize(), r_active);
        assert_eq!(use_.physical_row_step().get(), s);

        let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
        let mut ct_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
        let mut ct_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
        let mut pt_out: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_out_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt_in.data), 0, &mut source_xa);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                | (module).glwe_decrypt_tmp_bytes(&ct_out)
                | (module).glwe_encrypt_sk_tmp_bytes(&ct_in)
                | module.glwe_automorphism_tmp_bytes_default(&ct_out, &ct_in, &autokey_infos.with_dsize(effective_dsize))
                | module.glwe_automorphism_tmp_bytes_default(
                    &ct_in_infos,
                    &ct_in_infos,
                    &autokey_infos.with_dsize(effective_dsize),
                )
                | module.vec_znx_automorphism_assign_tmp_bytes(),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_out);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.glwe_automorphism_key_encrypt_sk(
            &mut autokey,
            p,
            &sk,
            &autokey_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );
        module.glwe_encrypt_sk(
            &mut ct_in,
            &pt_in,
            &sk_prepared,
            &ct_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut autokey_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
        module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, &mut scratch.borrow());

        crate::default::automorphism::glwe::glwe_automorphism_default(
            module,
            &mut ct_out,
            &ct_in,
            &autokey_prepared.with_dsize(effective_dsize),
            &mut scratch.borrow(),
        );

        // The bound of the decomposition the parent stands in for, not its own.
        let max_noise: f64 = use_.logical_layout().log2_std_noise_keyswitch(
            &ct_in_infos,
            0.5,
            0.5,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            0f64,
        );

        module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());
        module.vec_znx_automorphism_assign_backend(p, &mut vec_znx_backend_mut::<BE>(&mut pt_out.data), 0, &mut scratch.borrow());

        let noise = module
            .glwe_noise(&ct_out, &pt_out, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(
            noise <= max_noise + 1.0,
            "dnum={dnum} s={s} r_active={r_active}: noise {noise} exceeds {max_noise}"
        );

        // `res += phi(res)`, the variant the trace loop runs, over the same coarsening.
        let mut ct_acc: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
        module.glwe_encrypt_sk(
            &mut ct_acc,
            &pt_in,
            &sk_prepared,
            &ct_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        crate::default::automorphism::glwe::glwe_automorphism_add_assign_default(
            module,
            &mut ct_acc,
            &autokey_prepared.with_dsize(effective_dsize),
            &mut scratch.borrow(),
        );

        let mut pt_acc: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
        module.glwe_normalize(&mut pt_acc, &pt_in, &mut scratch.borrow());
        module.vec_znx_automorphism_assign_backend(p, &mut vec_znx_backend_mut::<BE>(&mut pt_acc.data), 0, &mut scratch.borrow());
        module.vec_znx_add_assign_backend(
            &mut vec_znx_backend_mut::<BE>(&mut pt_acc.data),
            0,
            &poulpy_hal::test_suite::vec_znx_backend_ref::<BE>(&pt_in.data),
            0,
        );

        let noise = module
            .glwe_noise(&ct_acc, &pt_acc, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(
            noise <= max_noise + 1.0,
            "add_assign dnum={dnum} s={s} r_active={r_active}: noise {noise} exceeds {max_noise}"
        );
    }
}

/// **A bound reads only the rows it resolves, end to end.**
///
/// Runs every key-consuming GLWE variant twice under the same coarsened key:
/// once against the stored parent, once against a twin identical on the rows
/// the bound selects and zeroed everywhere else. Byte equality is the
/// address-level statement that no unselected row reached a kernel, and it
/// holds for the assign forms and the add/sub forms, not only the plain one.
pub fn test_glwe_bound_reads_only_selected_rows<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + GLWEKeyswitchDefault<BE>
        + GLWEAutomorphismDefault<BE>
        + VecZnxAddAssignBackend<BE>
        + GLWEKeyswitchInternal<BE>
        + GLWENormalizeDefault<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigSubSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigSubSmallNegateAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + GLWEBytesOf<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let p: i64 = -5;
    let n: usize = module.n();
    let rank: usize = 1;

    for (dnum, s, r_active) in [(8usize, 2usize, 2usize), (8, 2, 3), (12, 4, 2), (8, 1, 3)] {
        let effective_dsize: Dsize = Dsize(s as u32);
        let k_in: usize = r_active * s * base2k;
        let k_out: usize = k_in + base2k * s;

        let ct_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_in.into(),
            rank: rank.into(),
        })
        .unwrap();
        let ct_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_out.into(),
            rank: rank.into(),
        };
        let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        })
        .unwrap();

        let use_ = *resolve_gglwe_key_use(&autokey_infos, TorusPrecision(k_in as u32), effective_dsize)
            .expect("valid layout")
            .expect("parent realizes the coarsening")
            .active()
            .expect("positive precision");
        let selected: Vec<usize> = (0..use_.logical_layout().dnum.as_usize())
            .map(|i| use_.first_physical_row() + i * use_.physical_row_step().get())
            .collect();
        assert!(
            selected.len() < dnum || !use_.is_dense(),
            "the case must leave at least one row or limb outside the bound"
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt_in.data), 0, &mut source_xa);

        let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
        let mut twin: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
        let mut ct_a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                | (module).glwe_encrypt_sk_tmp_bytes(&ct_a)
                | module.glwe_automorphism_tmp_bytes_default(
                    &ct_out_infos,
                    &ct_in_infos,
                    &autokey_infos.with_dsize(effective_dsize),
                )
                | module.glwe_automorphism_tmp_bytes_default(
                    &ct_in_infos,
                    &ct_in_infos,
                    &autokey_infos.with_dsize(effective_dsize),
                )
                | module.glwe_keyswitch_tmp_bytes_default(
                    &ct_out_infos,
                    &ct_in_infos,
                    &autokey_infos.with_dsize(effective_dsize),
                )
                | module.glwe_keyswitch_tmp_bytes_default(&ct_in_infos, &ct_in_infos, &autokey_infos.with_dsize(effective_dsize))
                | module.vec_znx_automorphism_assign_tmp_bytes(),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_out_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        // Two encryptions from the same seeds: identical key material.
        for key in [&mut autokey, &mut twin] {
            let (mut xe, mut xa) = (Source::new([1u8; 32]), Source::new([2u8; 32]));
            module.glwe_automorphism_key_encrypt_sk(
                key,
                p,
                &sk,
                &autokey_infos,
                &mut xe,
                &mut xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );
        }
        // Everything the bound does not select is overwritten in the twin with
        // fresh uniform material. Not zeroes: zero is the neutral element of a
        // multiply-accumulate, so a kernel that did read a zeroed row would
        // still land on the same output and the check would pass vacuously.
        let mut source_poison: Source = Source::new([7u8; 32]);
        for row in (0..dnum).filter(|row| !selected.contains(row)) {
            for col in 0..rank {
                poulpy_hal::layouts::FillUniform::fill_uniform(&mut twin.key.at_mut(row, col).data, base2k, &mut source_poison);
            }
        }

        module.glwe_encrypt_sk(
            &mut ct_a,
            &pt_in,
            &sk_prepared,
            &ct_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut parent_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
        module.glwe_automorphism_key_prepare(&mut parent_prepared, &autokey, &mut scratch.borrow());
        let mut twin_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
        module.glwe_automorphism_key_prepare(&mut twin_prepared, &twin, &mut scratch.borrow());

        let label = format!("dnum={dnum} s={s} r_active={r_active}");
        macro_rules! same_under_bound {
            ($name:literal, out, $op:path) => {{
                let mut run = |key: &GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>| {
                    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
                    $op(
                        module,
                        &mut res,
                        &ct_a,
                        &key.with_dsize(effective_dsize),
                        &mut scratch.borrow(),
                    );
                    BE::to_host_bytes(&res.data.data)
                };
                assert_eq!(
                    run(&parent_prepared),
                    run(&twin_prepared),
                    "{label}: {} read key material outside its bound",
                    $name
                );
            }};
            ($name:literal, assign, $op:path) => {{
                let mut run = |key: &GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>| {
                    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
                    module.glwe_normalize(&mut res, &ct_a, &mut scratch.borrow());
                    $op(
                        module,
                        &mut res,
                        &key.with_dsize(effective_dsize),
                        &mut scratch.borrow(),
                    );
                    BE::to_host_bytes(&res.data.data)
                };
                assert_eq!(
                    run(&parent_prepared),
                    run(&twin_prepared),
                    "{label}: {} read key material outside its bound",
                    $name
                );
            }};
        }

        use crate::default::automorphism::glwe as auto;
        use crate::default::keyswitching::glwe as ks;
        same_under_bound!("automorphism", out, auto::glwe_automorphism_default);
        same_under_bound!("automorphism_add", out, auto::glwe_automorphism_add_default);
        same_under_bound!("automorphism_sub", out, auto::glwe_automorphism_sub_default);
        same_under_bound!("automorphism_sub_negate", out, auto::glwe_automorphism_sub_negate_default);
        same_under_bound!("keyswitch", out, ks::glwe_keyswitch_default);
        same_under_bound!("automorphism_assign", assign, auto::glwe_automorphism_assign_default);
        same_under_bound!("automorphism_add_assign", assign, auto::glwe_automorphism_add_assign_default);
        same_under_bound!("automorphism_sub_assign", assign, auto::glwe_automorphism_sub_assign_default);
        same_under_bound!(
            "automorphism_sub_negate_assign",
            assign,
            auto::glwe_automorphism_sub_negate_assign_default
        );
        same_under_bound!("keyswitch_assign", assign, ks::glwe_keyswitch_assign_default);
    }
}
