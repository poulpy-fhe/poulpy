//! Collective GGSW over three parties: every entry of the finalized GGSW
//! decrypts to the sum of the messages times a component of the ideal
//! secret, and the GGSW drives external products under the ideal secret.

use poulpy_core::{
    ComponentNoise, DEFAULT_SIGMA_XE, Distribution, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct,
    layouts::{
        Dnum, Dsize, GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendMut, GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout,
        GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, GLWESwitchingKey, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{GGSWPrepared, GGSWPreparedToBackendRef},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ModuleNew, ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ReaderFrom, ScalarZnx, ScratchOwned, WriterTo, ZnxView,
        ZnxViewMut,
    },
    source::Source,
};

use super::fixtures::{
    BASE2K, K, LOG_MESSAGE, PARTIES, SEEDS, Secret, assert_decrypts_to, bounded_integers, encrypt_integers, ggsw_layout,
    glwe_layout_at, ideal_secret, secret_from_seed_at,
};
use crate::{
    api::{GGSWMHEProtocol, GLWESwitchingKeyMHEProtocol},
    layouts::MHEModuleAlloc,
};

fn trial_seed(role: u8, trial: u8) -> [u8; 32] {
    let mut seed = [role; 32];
    seed[31] ^= trial;
    seed
}

/// The ephemeral key layout for `layout`: the GGSW rank to itself, covering the
/// GGSW precision with one guard digit.
fn ephemeral_key_layout<BE: Backend>(module: &Module<BE>, layout: &GGSWLayout) -> GGLWELayout {
    GGLWELayout {
        n: layout.n,
        base2k: layout.base2k,
        dnum: Dnum(layout.k().as_u32().div_ceil(layout.base2k.as_u32() * layout.dsize.as_u32())),
        k_aux: TorusPrecision(layout.dsize.as_u32() * layout.base2k.as_u32() + module.log_n() as u32),
        rank_in: layout.rank,
        rank_out: layout.rank,
        dsize: layout.dsize,
        stride: 1,
    }
}

fn parties<BE>(module: &Module<BE>, rank: Rank, trial: u8) -> (Vec<Secret<BE>>, Vec<Secret<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets = (0..PARTIES)
        .map(|i| secret_from_seed_at(module, rank, trial_seed(100 + i as u8, trial)))
        .collect();
    let ephemerals = (0..PARTIES)
        .map(|i| secret_from_seed_at(module, rank, trial_seed(170 + i as u8, trial)))
        .collect();
    (secrets, ephemerals)
}

/// The collective switching key from the sum of `ephemerals` to the sum of
/// `secrets`, prepared.
fn ephemeral_key<BE>(
    module: &Module<BE>,
    ephemerals: &[Secret<BE>],
    secrets: &[Secret<BE>],
    layout: &GGLWELayout,
    trial: u8,
) -> GLWESwitchingKeyPrepared<AlignedBuf, BE>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyMHEProtocol<BE> + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_switching_key_share_gen_tmp_bytes(layout)
            .max(module.mhe_glwe_switching_key_share_finalize_tmp_bytes())
            .max(module.glwe_switching_key_prepare_tmp_bytes(layout)),
    );
    let mut acc = module.glwe_switching_key_share_alloc_from_infos(layout);
    let mut share = module.glwe_switching_key_share_alloc_from_infos(layout);
    for (i, ((u, _), (sk, _))) in ephemerals.iter().zip(secrets).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new(trial_seed(60 + i as u8, trial));
        module.mhe_glwe_switching_key_share_gen(dst, u, sk, trial_seed(2, trial), &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.mhe_glwe_switching_key_share_aggregate(&mut acc, &share);
        }
    }
    let mut key: GLWESwitchingKey<AlignedBuf, i64> = module.glwe_switching_key_alloc_from_infos(layout);
    module.mhe_glwe_switching_key_share_finalize(&mut key, &acc, &mut scratch.borrow());
    let mut key_prepared: GLWESwitchingKeyPrepared<AlignedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(layout);
    module.glwe_switching_key_prepare(&mut key_prepared, &key, &mut scratch.borrow());
    key_prepared
}

/// One-column plaintexts of small signed integers, one per party.
fn messages<BE>(module: &Module<BE>) -> Vec<ScalarZnx<AlignedBuf, i64>>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ScalarZnxAlloc<BE>,
{
    (0..PARTIES)
        .map(|i| {
            let mut m: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
            m.at_mut(0, 0)
                .copy_from_slice(&bounded_integers(module.n(), 2, [80 + i as u8; 32]));
            m
        })
        .collect()
}

/// The collective GGSW at `layout` of the sum of `messages`, one per party,
/// and the ideal secret it is encrypted under.
fn collective_ggsw<BE>(
    module: &Module<BE>,
    layout: &GGSWLayout,
    messages: &[ScalarZnx<AlignedBuf, i64>],
    binary: bool,
    trial: u8,
) -> (GGSW<AlignedBuf, i64>, GLWESecretPrepared<AlignedBuf, BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (mut secrets, mut ephemerals) = parties(module, layout.rank, trial);
    if binary {
        for (i, (secret, prepared)) in secrets.iter_mut().chain(&mut ephemerals).enumerate() {
            module.glwe_secret_fill_binary_prob(secret, 0.5, &mut Source::new(trial_seed(180 + i as u8, trial)));
            module.glwe_secret_prepare(prepared, secret);
        }
    }
    let key_layout = ephemeral_key_layout(module, layout);
    let mut key = ephemeral_key(module, &ephemerals, &secrets, &key_layout, trial);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_ggsw_share_gen_tmp_bytes(layout)
            .max(module.mhe_ggsw_share_finalize_tmp_bytes(layout, &key_layout)),
    );

    let mut acc = module.ggsw_share_alloc_from_infos(layout);
    let mut share = module.ggsw_share_alloc_from_infos(layout);
    for (i, (((_, sk), (_, u)), m)) in secrets.iter().zip(&ephemerals).zip(messages).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new(trial_seed(10 + i as u8, trial));
        // The message and the secret-derived products would be left in the scratch.
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(module.mhe_ggsw_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_ggsw_share_gen(dst, m, sk, u, trial_seed(1, trial), &mut source_xe, scratch)
        });
        if i > 0 {
            module.mhe_ggsw_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(layout);
    if trial == 0 {
        let saved_noise = key.noise();
        GGLWEPreparedToBackendMut::<BE>::set_noise(&mut key, None);
        module.mhe_ggsw_share_finalize(&mut res, &acc, &key, &mut scratch.borrow());
        assert!(res.noise().unwrap().phase_noise(module.n()).variance().is_infinite());
        let wrong = ComponentNoise::from_secret_at(Distribution::BinaryProb(0.25), key.k(), key.rank_out().as_usize());
        GGLWEPreparedToBackendMut::<BE>::set_noise(&mut key, Some(wrong));
        super::fixtures::assert_panics_with("invalid finalization: output key provenance differs", || {
            module.mhe_ggsw_share_finalize(&mut res, &acc, &key, &mut scratch.borrow());
        });
        GGLWEPreparedToBackendMut::<BE>::set_noise(&mut key, saved_noise);
    }
    module.mhe_ggsw_share_finalize(&mut res, &acc, &key, &mut scratch.borrow());
    assert_eq!(res.noise().unwrap().parties(), PARTIES as u64);
    assert_eq!(
        res.noise().unwrap().secret_distribution().base(),
        if binary {
            Distribution::BinaryProb(0.5)
        } else {
            Distribution::TernaryProb(0.5)
        }
    );
    (res, ideal_secret(module, &secrets))
}

fn check_collective_ggsw<BE>(module: &Module<BE>, layout: &GGSWLayout, binary: bool)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWNoise<BE>
        + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let messages = messages(module);
    let mut pt_want: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    for m in &messages {
        for (x, y) in pt_want.at_mut(0, 0).iter_mut().zip(m.at(0, 0)) {
            *x += y;
        }
    }
    let k = layout.k().as_usize();
    let key_layout = ephemeral_key_layout(module, layout);
    let rank_n = layout.rank.as_usize() as f64 * module.n() as f64;
    let sigma2 = DEFAULT_SIGMA_XE.powi(2);
    let parties = PARTIES as f64;
    let secret_second = parties * 0.5 + if binary { parties * (parties - 1.0) * 0.25 } else { 0.0 };
    let circular_variance = 2.0 * rank_n * parties * secret_second * sigma2;
    let digit_bits = key_layout.dsize.as_usize() * key_layout.base2k.as_usize();
    let digits = k.div_ceil(digit_bits).min(key_layout.dnum.as_usize());
    let digit_factor = (1.0 - (-(digit_bits as f64)).exp2()) / (1.0 - (-(key_layout.base2k.as_usize() as f64)).exp2());
    let switching_variance = digit_factor.powi(2)
        * rank_n
        * digits as f64
        * (2.0 * digit_bits as f64 - 2.0).exp2()
        * parties
        * sigma2
        * (2.0 * (k as f64 - key_layout.k().as_usize() as f64)).exp2();
    let rounding_variance = (1.0 + rank_n * secret_second) / 4.0;
    let mut components = vec![parties * sigma2 + 0.25; layout.rank.as_usize() + 1];
    components[0] = circular_variance / 2.0 + switching_variance + 0.25;
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_noise_tmp_bytes(layout));
    // Binary secrets correlate the coefficients of each circular error
    // polynomial. Pool independent secrets and errors, rather than treating
    // the coefficients of one transcript as independent measurements.
    const TRIALS: u8 = 16;
    let mut measured = vec![0.0; layout.rank.as_usize() + 1];
    let mut modeled = vec![0.0; measured.len()];
    for trial in 0..TRIALS {
        let (ggsw, sk) = collective_ggsw(module, layout, &messages, binary, trial);
        super::fixtures::assert_fresh_noise(&ggsw, circular_variance + switching_variance + rounding_variance, layout.k());
        super::fixtures::assert_noise_components(&ggsw, &components);
        let noise = ggsw.noise().unwrap().phase_noise(module.n());
        assert!(noise.variance() > parties * sigma2);
        for row in 0..layout.dnum.as_usize() {
            for col in 0..measured.len() {
                let stats = module.ggsw_noise(&ggsw, row, col, &pt_want.to_ref(), &sk, &mut scratch.borrow());
                measured[col] += stats.second_moment();
                modeled[col] += if col == 0 {
                    parties * sigma2 * (-2.0 * k as f64).exp2()
                } else {
                    noise.variance_at(0u32.into())
                };
            }
        }
    }
    for (col, (measured, modeled)) in measured.into_iter().zip(modeled).enumerate() {
        let ratio = measured / modeled;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "GGSW binary={binary} col={col}: pooled residual/model ratio={ratio} across {TRIALS} independent transcripts"
        );
    }
}

pub fn test_ggsw_share<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWNoise<BE>
        + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    for binary in [false, true] {
        check_collective_ggsw(module, &ggsw_layout(module), binary);
    }
}

/// Rank 1, two base digits per row.
pub fn test_ggsw_share_rank_one<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWNoise<BE>
        + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GGSWLayout {
        dnum: Dnum(2),
        dsize: Dsize(2),
        k_aux: TorusPrecision(2 * BASE2K.0 + module.log_n() as u32),
        rank: Rank(1),
        ..ggsw_layout(module)
    };
    for binary in [false, true] {
        check_collective_ggsw(module, &layout, binary);
    }
}

/// A GGSW of `X^3` contributed by one party alone rotates a GLWE by three
/// coefficients through the external product.
pub fn test_ggsw_share_external_product<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    const SHIFT: usize = 3;
    let n = module.n();
    let layout = ggsw_layout(module);
    let mut messages: Vec<ScalarZnx<AlignedBuf, i64>> = (0..PARTIES).map(|_| module.scalar_znx_alloc(n, 1)).collect();
    messages[1].at_mut(0, 0)[SHIFT] = 1;
    let (ggsw, sk) = collective_ggsw(module, &layout, &messages, false, 0);

    let glwe_layout: GLWELayout = glwe_layout_at(module, K);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ggsw_prepare_tmp_bytes(&layout)
            .max(module.glwe_external_product_tmp_bytes(&glwe_layout, &glwe_layout, &layout))
            .max(module.glwe_encrypt_sk_tmp_bytes(&glwe_layout))
            .max(module.glwe_decrypt_tmp_bytes(&glwe_layout)),
    );
    let mut ggsw_prepared: GGSWPrepared<AlignedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&layout);
    module.ggsw_prepare(&mut ggsw_prepared, &ggsw, &mut scratch.borrow());

    let data = bounded_integers(n, LOG_MESSAGE, [90u8; 32]);
    let ct = encrypt_integers(module, &glwe_layout, &data, &sk, &mut scratch);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&glwe_layout);
    module.glwe_external_product(&mut res, &ct, &ggsw_prepared.to_backend_ref(), &mut scratch.borrow());
    assert_eq!(res.noise(), None);
    let want: Vec<i64> = (0..n)
        .map(|i| if i >= SHIFT { data[i - SHIFT] } else { -data[n + i - SHIFT] })
        .collect();
    // The product itself adds noise of about 8 units at this layout, as with a core GGSW.
    assert_decrypts_to(module, &res, &want, &sk, 1 << 6, &mut scratch);
}

pub fn test_ggsw_share_seed_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>:
        MHEModuleAlloc<BE> + GGSWMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE> + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(module);

    let (secrets, ephemerals) = parties(module, layout.rank, 0);
    let m: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ggsw_share_gen_tmp_bytes(&layout));
    let [mut a, b] = [0, 1].map(|i| {
        let mut share = module.ggsw_share_alloc_from_infos(&layout);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.mhe_ggsw_share_gen(
            &mut share,
            &m,
            &secrets[i].1,
            &ephemerals[i].1,
            SEEDS[i],
            &mut source_xe,
            &mut scratch.borrow(),
        );
        share
    });
    module.mhe_ggsw_share_aggregate(&mut a, &b);
}

pub fn test_ggsw_share_ephemeral_rank_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>:
        MHEModuleAlloc<BE> + GGSWMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE> + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(module);

    let (secrets, _) = parties(module, layout.rank, 0);
    let u = secret_from_seed_at(module, Rank(1), [9u8; 32]);
    let m: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    let mut share = module.ggsw_share_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ggsw_share_gen_tmp_bytes(&layout));
    module.mhe_ggsw_share_gen(
        &mut share,
        &m,
        &secrets[0].1,
        &u.1,
        SEEDS[0],
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

pub fn test_ggsw_share_key_rank_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GGSWMHEProtocol<BE> + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(module);
    let key_layout = GGLWELayout {
        rank_out: Rank(1),
        ..ephemeral_key_layout(module, &layout)
    };
    let pat = module.ggsw_share_alloc_from_infos(&layout);
    let key: GLWESwitchingKeyPrepared<AlignedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(&key_layout);
    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ggsw_share_finalize_tmp_bytes(&layout, &key_layout));
    module.mhe_ggsw_share_finalize(&mut res, &pat, &key, &mut scratch.borrow());
}

pub fn test_ggsw_share_key_precision_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GGSWMHEProtocol<BE> + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(module);
    let key_layout = GGLWELayout {
        dnum: Dnum(3),
        ..ephemeral_key_layout(module, &layout)
    };
    let pat = module.ggsw_share_alloc_from_infos(&layout);
    let key: GLWESwitchingKeyPrepared<AlignedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(&key_layout);
    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ggsw_share_finalize_tmp_bytes(&layout, &key_layout));
    module.mhe_ggsw_share_finalize(&mut res, &pat, &key, &mut scratch.borrow());
}

/// Shape errors are rejected before encryption or key switching can print operands.
pub fn test_ggsw_share_shape_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ModuleNew<BE>
        + MHEModuleAlloc<BE>
        + GGSWMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(module);

    let (secrets, ephemerals) = parties(module, layout.rank, 0);
    let small_module = Module::<BE>::new((module.n() / 2) as u64);
    let expected = [
        "invalid share: secret degree differs from the transcript's",
        "invalid share: ephemeral secret degree differs from the transcript's",
        "invalid share: message degree differs from the transcript's",
        "invalid share: message must have one column",
        "invalid finalization: key and GGSW degrees differ",
    ];
    for (case, expected) in expected.into_iter().enumerate() {
        super::fixtures::assert_panics_with(expected, || {
            let mut pat = module.ggsw_share_alloc_from_infos(&layout);
            if case == 4 {
                let key_layout = ephemeral_key_layout(module, &layout);
                let key = module.glwe_switching_key_prepared_alloc_from_infos(&GGLWELayout {
                    n: (module.n() / 2).into(),
                    ..key_layout
                });
                let mut res = module.ggsw_alloc_from_infos(&layout);
                let mut scratch: ScratchOwned<BE> =
                    ScratchOwned::alloc(module.mhe_ggsw_share_finalize_tmp_bytes(&layout, &key_layout));
                module.mhe_ggsw_share_finalize(&mut res, &pat, &key, &mut scratch.borrow());
                return;
            }
            let small_sk = small_module.glwe_secret_prepared_alloc(layout.rank);
            let small_u = small_module.glwe_secret_prepared_alloc(layout.rank);
            let pt = module.scalar_znx_alloc(
                if case == 2 { module.n() / 2 } else { module.n() },
                if case == 3 { 2 } else { 1 },
            );
            let sk = if case == 0 { &small_sk } else { &secrets[0].1 };
            let u = if case == 1 { &small_u } else { &ephemerals[0].1 };
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ggsw_share_gen_tmp_bytes(&layout));
            module.mhe_ggsw_share_gen(
                &mut pat,
                &pt,
                sk,
                u,
                SEEDS[0],
                &mut Source::new([10u8; 32]),
                &mut scratch.borrow(),
            );
        });
    }
}

/// Reading a share of another rank fails: its parts do not form one GGSW of the allocation's rank.
pub fn test_ggsw_share_read_rank_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>,
{
    let layout = ggsw_layout(module);
    let share = module.ggsw_share_alloc_from_infos(&layout);
    let mut bytes = Vec::new();
    share.write_to(&mut bytes).unwrap();
    let mut res = module.ggsw_share_alloc_from_infos(&GGSWLayout { rank: Rank(1), ..layout });
    assert!(res.read_from(&mut bytes.as_slice()).is_err());
}
