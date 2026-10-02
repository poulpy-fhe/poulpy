//! Collective GGSW over three parties: every entry of the finalized GGSW
//! decrypts to the sum of the messages times a component of the ideal
//! secret, and the GGSW drives external products under the ideal secret.

use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct,
    layouts::{
        Dnum, Dsize, GGLWELayout, GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWESecretPrepared,
        GLWESecretPreparedFactory, GLWESecretSampling, GLWESwitchingKey, GLWESwitchingKeyPrepared,
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

use super::{
    fixtures::{
        BASE2K, K, LOG_MESSAGE, PARTIES, SEEDS, Secret, assert_decrypts_to, bounded_integers, encrypt_integers, ggsw_layout,
        glwe_layout_at, ideal_secret, secret_from_seed_at,
    },
    pat::aggregate_noise_bound,
};
use crate::{
    api::{GGSWMHEProtocol, GLWESwitchingKeyMHEProtocol},
    layouts::MHEModuleAlloc,
};

/// Log2 noise bound of a column `j >= 1`: `Sum_l e1_l s_l - Sum_i e2_i u_i`, `rank`
/// terms each, summed over `PARTIES` parties with ternary secrets of variance
/// 1/2 has variance `rank * n * PARTIES^2 * sigma^2`; the key switching noise
/// lies far below.
fn circular_noise_bound(n: usize, rank: usize, k: usize) -> f64 {
    (DEFAULT_SIGMA_XE * PARTIES as f64 * ((rank * n) as f64).sqrt()).log2() - k as f64 + 0.5
}

/// The ephemeral key layout for `layout`: the GGSW rank to itself, covering the
/// GGSW precision with one guard digit.
fn ephemeral_key_layout<BE: Backend>(module: &Module<BE>, layout: &GGSWLayout) -> GGLWELayout {
    GGLWELayout {
        n: layout.n,
        base2k: layout.base2k,
        dnum: Dnum(layout.k().as_u32().div_ceil(layout.base2k.as_u32())),
        k_aux: TorusPrecision(layout.base2k.as_u32() + module.log_n() as u32),
        rank_in: layout.rank,
        rank_out: layout.rank,
        dsize: Dsize(1),
        stride: 1,
    }
}

fn parties<BE>(module: &Module<BE>, rank: Rank) -> (Vec<Secret<BE>>, Vec<Secret<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets = (0..PARTIES)
        .map(|i| secret_from_seed_at(module, rank, [100 + i as u8; 32]))
        .collect();
    let ephemerals = (0..PARTIES)
        .map(|i| secret_from_seed_at(module, rank, [170 + i as u8; 32]))
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
) -> GLWESwitchingKeyPrepared<AlignedBuf, BE>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyMHEProtocol<BE> + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let enc_infos = EncryptionLayout::new_from_default_sigma(*layout).unwrap();
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
        let mut source_xe = Source::new([60 + i as u8; 32]);
        module.mhe_glwe_switching_key_share_gen(dst, u, sk, SEEDS[1], &enc_infos, &mut source_xe, &mut scratch.borrow());
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
    let (secrets, ephemerals) = parties(module, layout.rank);
    let key_layout = ephemeral_key_layout(module, layout);
    let key = ephemeral_key(module, &ephemerals, &secrets, &key_layout);
    let enc_infos = EncryptionLayout::new_from_default_sigma(*layout).unwrap();
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_ggsw_share_gen_tmp_bytes(layout)
            .max(module.mhe_ggsw_share_finalize_tmp_bytes(layout, &key_layout)),
    );

    let mut acc = module.ggsw_share_alloc_from_infos(layout);
    let mut share = module.ggsw_share_alloc_from_infos(layout);
    for (i, (((_, sk), (_, u)), m)) in secrets.iter().zip(&ephemerals).zip(messages).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        // The message and the secret-derived products would be left in the scratch.
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(module.mhe_ggsw_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_ggsw_share_gen(dst, m, sk, u, SEEDS[0], &enc_infos, &mut source_xe, scratch)
        });
        if i > 0 {
            module.mhe_ggsw_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(layout);
    module.mhe_ggsw_share_finalize(&mut res, &acc, &key, &mut scratch.borrow());
    (res, ideal_secret(module, &secrets))
}

fn check_collective_ggsw<BE>(module: &Module<BE>, layout: &GGSWLayout)
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
    let (ggsw, sk) = collective_ggsw(module, layout, &messages);
    let mut pt_want: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    for m in &messages {
        for (x, y) in pt_want.at_mut(0, 0).iter_mut().zip(m.at(0, 0)) {
            *x += y;
        }
    }
    let k = layout.k().as_usize();
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_noise_tmp_bytes(layout));
    for row in 0..layout.dnum.as_usize() {
        for col in 0..=layout.rank.as_usize() {
            let noise = module
                .ggsw_noise(&ggsw, row, col, &pt_want.to_ref(), &sk, &mut scratch.borrow())
                .std()
                .log2();
            let bound = if col == 0 {
                aggregate_noise_bound(k)
            } else {
                circular_noise_bound(module.n(), layout.rank.as_usize(), k)
            };
            assert!(noise <= bound, "row {row} col {col}: noise {noise} above bound {bound}");
        }
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
    check_collective_ggsw(module, &ggsw_layout(module));
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
    check_collective_ggsw(module, &layout);
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
    let (ggsw, sk) = collective_ggsw(module, &layout, &messages);

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
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (secrets, ephemerals) = parties(module, layout.rank);
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
            &enc_infos,
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
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (secrets, _) = parties(module, layout.rank);
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
        &enc_infos,
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
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (secrets, ephemerals) = parties(module, layout.rank);
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
                let key = small_module.glwe_switching_key_prepared_alloc_from_infos(&key_layout);
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
                &enc_infos,
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
