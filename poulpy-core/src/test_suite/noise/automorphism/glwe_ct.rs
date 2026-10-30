use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismAssign, VecZnxFillUniformSource},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
    test_suite::vec_znx_backend_mut,
};

use crate::api::GLWESecretSampling;
use crate::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef;
use crate::test_suite::noise::{assert_refuses_uncovered_input, glwe_noise_checked};
use crate::test_suite::poisoned_scratch;
use crate::{
    GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWENoise, GLWENormalize,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory},
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::GGLWENoiseModel,
};

pub fn test_glwe_automorphism<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssign<BE>
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

            let ct_in_infos = GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            };

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                rank: rank.into(),
                k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
            };

            let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
            let mut ct_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
            let mut ct_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
            let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
            let mut pt_out: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source(
                in_base2k,
                pt_in.k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                0,
                &mut source_xa,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct_out)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct_in)
                    | module.glwe_automorphism_key_prepare_tmp_bytes(&autokey_infos)
                    | module.glwe_noise_tmp_bytes(&ct_out),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_out);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.glwe_encrypt_sk(
                &mut ct_in,
                &pt_in,
                &sk_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, &mut scratch.borrow());

            let automorphism_bytes = module.glwe_automorphism_tmp_bytes(&ct_out, &ct_in, &autokey);
            module.glwe_automorphism(
                &mut ct_out,
                &ct_in,
                &autokey_prepared.to_backend_ref(),
                &mut poisoned_scratch::<BE>(automorphism_bytes).borrow(),
            );

            let max_noise: f64 = autokey_infos.log2_std_noise_keyswitch(
                &ct_out_infos,
                &ct_in_infos,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            );

            module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());
            module.vec_znx_automorphism_assign(p, &mut vec_znx_backend_mut::<BE>(&mut pt_out.data), 0, &mut scratch.borrow());

            assert!(
                glwe_noise_checked(module, &ct_out, &pt_out, &sk_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
                    <= max_noise + 1.0
            );

            let wide: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&GLWELayout {
                k: TorusPrecision(autokey_infos.gglwe_layout().max_input_k().0 + 1),
                ..ct_in_infos
            });
            assert_refuses_uncovered_input("glwe_automorphism took an input one bit past its key", || {
                module.glwe_automorphism(&mut ct_out, &wide, &autokey_prepared.to_backend_ref(), &mut scratch.borrow())
            });
        }
    }
}

/// `glwe_automorphism_many` writes, output by output, the bits `glwe_automorphism` writes for the same input and key:
/// keys at the input's radix in two digit layouts, whose decompositions are shared, and a key at another radix, on an
/// input that is canonical and on the flag-clear sum of two ciphertexts.
pub fn test_glwe_automorphism_many<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAdd<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k_in: usize = 4 * base2k + 1;
    let n: usize = module.n();
    // Galois element, key radix and digit width: two whole-limb layouts sharing one decomposition, one off the limbs,
    // and one at another radix.
    let shapes: [(i64, usize, usize); 4] = [
        (5, base2k, base2k),
        (25, base2k, 2 * base2k),
        (-5, base2k, base2k + 3),
        (125, base2k - 1, base2k - 1),
    ];
    for rank in 1_usize..3 {
        let ct_in_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_in.into(),
            rank: rank.into(),
        };
        let ct_out_infos = GLWELayout {
            n: n.into(),
            base2k: (base2k - 1).into(),
            k: (k_in + base2k).into(),
            rank: rank.into(),
        };
        let key_layouts: Vec<GLWEAutomorphismKeyLayout> = shapes
            .iter()
            .map(|&(_, key_base2k, k_digit)| GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: k_in.div_ceil(k_digit).into(),
                k_aux: (k_digit + module.log_n()).into(),
                rank: rank.into(),
                k_digit: k_digit.into(),
            })
            .collect();

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([2u8; 32]);
        let mut source_xa: Source = Source::new([3u8; 32]);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            key_layouts
                .iter()
                .map(|layout| {
                    module.glwe_automorphism_key_encrypt_sk_tmp_bytes(layout)
                        | module.glwe_automorphism_key_prepare_tmp_bytes(layout)
                })
                .fold(module.glwe_encrypt_sk_tmp_bytes(&ct_in_infos), |a, b| a | b),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_in_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let keys: Vec<GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = shapes
            .iter()
            .zip(&key_layouts)
            .map(|(&(p, _, _), layout)| {
                let mut key: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_automorphism_key_alloc_from_infos(layout);
                module.glwe_automorphism_key_encrypt_sk(
                    &mut key,
                    p,
                    &sk,
                    &mut source_xe,
                    &mut source_xa,
                    &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
                );
                let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                    module.glwe_automorphism_key_prepared_alloc_from_infos(layout);
                module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut scratch.borrow());
                prepared
            })
            .collect();
        let key_views: Vec<_> = keys.iter().map(|key| key.to_backend_ref()).collect();
        let key_refs: Vec<_> = key_views.iter().collect();

        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_in_infos);
        let mut cts: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = Vec::new();
        for _ in 0..2 {
            module.vec_znx_fill_uniform_source(base2k, k_in, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
            let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
            module.glwe_encrypt_sk(
                &mut ct,
                &pt,
                &sk_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            cts.push(ct);
        }
        let mut sum: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_in_infos);
        module.glwe_add_into(&mut sum, &cts[0], &cts[1]);
        assert!(!sum.is_canonical(), "the sum of two ciphertexts is expected flag-clear");

        for a in [&cts[0], &sum] {
            let want: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = key_refs
                .iter()
                .map(|key| {
                    let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
                    let bytes = module.glwe_automorphism_tmp_bytes(&want, a, *key);
                    module.glwe_automorphism(&mut want, a, key, &mut poisoned_scratch::<BE>(bytes).borrow());
                    want
                })
                .collect();
            let mut got: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = (0..key_refs.len())
                .map(|_| module.glwe_alloc_from_infos(&ct_out_infos))
                .collect();
            let bytes = key_layouts
                .iter()
                .map(|layout| module.glwe_automorphism_many_tmp_bytes(&ct_out_infos, a, layout))
                .max()
                .unwrap();
            let mut got_refs: Vec<_> = got.iter_mut().collect();
            module.glwe_automorphism_many(&mut got_refs, a, &key_refs, &mut poisoned_scratch::<BE>(bytes).borrow());
            for (t, (got, want)) in got.iter().zip(&want).enumerate() {
                assert!(
                    got == want,
                    "rank {rank}, key {t}: glwe_automorphism_many differs from glwe_automorphism"
                );
            }

            let mut short: Vec<_> = got.iter_mut().take(1).collect();
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                module.glwe_automorphism_many(&mut short, a, &key_refs, &mut scratch.borrow())
            }))
            .expect_err("glwe_automorphism_many took fewer outputs than keys");
            let message = crate::test_suite::decryption::panic_message(&*err);
            assert!(
                message.contains("res and keys differ in length"),
                "glwe_automorphism_many panicked with {message:?}"
            );
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
        + VecZnxFillUniformSource<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssign<BE>,
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

            let ct_out_infos = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                rank: rank.into(),
                k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
            };

            let mut autokey: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&autokey_infos);
            let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source(
                out_base2k,
                pt_want.k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                0,
                &mut source_xa,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct)
                    | module.glwe_automorphism_key_prepare_tmp_bytes(&autokey)
                    | module.glwe_noise_tmp_bytes(&ct),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.glwe_encrypt_sk(
                &mut ct,
                &pt_want,
                &sk_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, &mut scratch.borrow());

            let automorphism_bytes = module.glwe_automorphism_tmp_bytes(&ct, &ct, &autokey);
            module.glwe_automorphism_assign(
                &mut ct,
                &autokey_prepared.to_backend_ref(),
                &mut poisoned_scratch::<BE>(automorphism_bytes).borrow(),
            );

            let max_noise: f64 = autokey_infos.log2_std_noise_keyswitch(
                &ct_out_infos,
                &ct_out_infos,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            );

            module.vec_znx_automorphism_assign(p, &mut vec_znx_backend_mut::<BE>(&mut pt_want.data), 0, &mut scratch.borrow());

            assert!(
                glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
                    <= max_noise + 1.0
            )
        }
    }
}
