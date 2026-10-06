use poulpy_core::{
    Distribution, GLWEDecrypt, GLWEEncryptSk, GetDistribution, GetDistributionMut, Noise,
    layouts::{
        Base2K, Dnum, Dsize, GGLWELayout, GGSWLayout, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPublicKey,
        GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory,
        GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ReaderFrom, ScalarZnxAsVecZnxBackendMut, ScalarZnxToBackendRef,
        ScratchOwned, WriterTo,
    },
    source::Source,
};

use crate::{api::GLWEPublicKeyMHEProtocol, layouts::MHEModuleAlloc};

pub(crate) const BASE2K: Base2K = Base2K(12);
pub(crate) const K: TorusPrecision = TorusPrecision(33);
pub(crate) const RANK: Rank = Rank(2);
pub(crate) const DNUM: Dnum = Dnum(3);
pub(crate) const DSIZE: Dsize = Dsize(1);
pub(crate) const SEEDS: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
pub(crate) const SEED_XE: [u8; 32] = [3u8; 32];
pub(crate) const PARTIES: usize = 3;

/// Galois element of the automorphism key tests.
pub(crate) const P: i64 = -5;

/// Bits of the integer plaintexts of the sharing tests.
pub(crate) const LOG_MESSAGE: usize = 10;
/// Output precision of the shares-to-encryption tests.
pub(crate) const K_OUT: TorusPrecision = TorusPrecision(K.0 + 2 * BASE2K.0);

pub(crate) type Secret<BE> = (GLWESecret<AlignedBuf, i64>, GLWESecretPrepared<AlignedBuf, BE>);

pub(crate) fn gglwe_layout<BE: Backend>(module: &Module<BE>) -> GGLWELayout {
    GGLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        dnum: DNUM,
        k_aux: TorusPrecision(DSIZE.0 * BASE2K.0 + module.log_n() as u32),
        rank_in: RANK,
        rank_out: RANK,
        dsize: DSIZE,
        stride: 1,
    }
}

/// The GGSW at the rows, digits and precision of [`gglwe_layout`].
pub(crate) fn ggsw_layout<BE: Backend>(module: &Module<BE>) -> GGSWLayout {
    let gglwe = gglwe_layout(module);
    GGSWLayout {
        n: gglwe.n,
        base2k: gglwe.base2k,
        dnum: gglwe.dnum,
        k_aux: gglwe.k_aux,
        rank: RANK,
        dsize: gglwe.dsize,
    }
}

pub(crate) fn secret_from_seed<BE>(module: &Module<BE>, seed: [u8; 32]) -> Secret<BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    secret_from_seed_at(module, RANK, seed)
}

pub(crate) fn secret_from_seed_at<BE>(module: &Module<BE>, rank: Rank, seed: [u8; 32]) -> Secret<BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    let mut sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(rank);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new(seed));
    let mut sk_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(rank);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);
    (sk, sk_prepared)
}

/// One secret per party, each from its own seed.
pub(crate) fn party_secrets<BE>(module: &Module<BE>) -> Vec<Secret<BE>>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    (0..PARTIES).map(|i| secret_from_seed(module, [100 + i as u8; 32])).collect()
}

/// One secret-shaped GGLWE message per party, distinct from every party secret.
pub(crate) fn party_messages<BE>(module: &Module<BE>) -> Vec<Secret<BE>>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    (0..PARTIES).map(|i| secret_from_seed(module, [70 + i as u8; 32])).collect()
}

/// The coefficient-wise sum of the secrets of `parties`.
pub(crate) fn secret_sum<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecret<AlignedBuf, i64>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: VecZnxAddScalarAssign<BE>,
{
    let rank = parties[0].0.rank();
    let mut sum: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(rank);
    for (sk, _) in parties {
        for col in 0..rank.as_usize() {
            module.vec_znx_add_scalar_assign(
                &mut ScalarZnxAsVecZnxBackendMut::<BE>::as_vec_znx_backend_mut(sum.data_mut()),
                col,
                0,
                &ScalarZnxToBackendRef::<BE>::to_backend_ref(sk.data()),
                col,
            );
        }
    }
    sum
}

/// The ideal secret: the sum of the parties' secrets. It is not ternary; the
/// tag only has to be valid, since decryption ignores it.
pub(crate) fn ideal_secret<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecretPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretPreparedFactory<BE> + VecZnxAddScalarAssign<BE>,
{
    let mut sum: GLWESecret<AlignedBuf, i64> = secret_sum(module, parties);
    *sum.dist_mut() = Distribution::TernaryProb(0.5);
    let mut sum_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(sum.rank());
    module.glwe_secret_prepare(&mut sum_prepared, &sum);
    sum_prepared
}

/// The collective public key of `parties` at `layout`: the public key protocol
/// under `SEEDS[0]`, finalized and prepared.
pub(crate) fn collective_public_key<BE>(
    module: &Module<BE>,
    parties: &[Secret<BE>],
    layout: &GLWELayout,
) -> GLWEPublicKeyPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_public_key_share_gen_tmp_bytes(layout)
            .max(module.mhe_glwe_public_key_share_finalize_tmp_bytes())
            .max(module.glwe_public_key_prepare_tmp_bytes(layout)),
    );
    let mut acc = module.glwe_public_key_share_alloc_from_infos(layout);
    let mut share = module.glwe_public_key_share_alloc_from_infos(layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([40 + i as u8; 32]);
        module.mhe_glwe_public_key_share_gen(dst, sk, SEEDS[0], &mut source_xe, &mut scratch.borrow());
        assert_collective_metadata(dst, 1);
        if i > 0 {
            module.mhe_glwe_public_key_share_aggregate(&mut acc, &share);
        }
        assert_collective_metadata(&acc, i + 1);
        assert_fresh_noise(&acc, (i + 1) as f64 * poulpy_core::DEFAULT_SIGMA_XE.powi(2), layout.k);
    }
    let mut encoded = Vec::new();
    acc.write_to(&mut encoded).unwrap();
    let mut decoded = module.glwe_public_key_share_alloc_from_infos(layout);
    decoded.read_from(&mut encoded.as_slice()).unwrap();
    assert!(decoded == acc);
    assert_collective_metadata(&decoded, parties.len());

    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(layout);
    module.mhe_glwe_public_key_share_finalize(&mut pk, &acc, &mut scratch.borrow());
    assert_collective_metadata(&pk, parties.len());
    assert!(
        pk.dist() == parties[0].0.dist(),
        "the key takes the parties' secret distribution"
    );
    let mut pk_prepared: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(layout);
    module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());
    assert_collective_metadata(&pk_prepared, parties.len());
    assert_fresh_noise(
        &pk_prepared,
        parties.len() as f64 * poulpy_core::DEFAULT_SIGMA_XE.powi(2),
        layout.k,
    );
    pk_prepared
}

pub(crate) fn glwe_layout_at<BE: Backend>(module: &Module<BE>, k: TorusPrecision) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k,
        rank: RANK,
    }
}

/// Signed integers of `log` bits, one per coefficient.
pub(crate) fn bounded_integers(n: usize, log: usize, seed: [u8; 32]) -> Vec<i64> {
    let mut source = Source::new(seed);
    (0..n).map(|_| source.next_i64() >> (64 - log)).collect()
}

/// `data` as integers in the top-`k` window of a plaintext at `layout`.
pub(crate) fn integer_plaintext<BE>(module: &Module<BE>, layout: &GLWELayout, data: &[i64]) -> GLWEPlaintext<AlignedBuf, i64>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
{
    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(layout);
    pt.encode_vec_i64(data, layout.k);
    pt
}

/// The integers in the top-`k` window of `pt`.
pub(crate) fn plaintext_integers(pt: &GLWEPlaintext<AlignedBuf, i64>) -> Vec<i64> {
    let mut data = vec![0i64; pt.n().as_usize()];
    pt.decode_vec_i64(&mut data, pt.k());
    data
}

/// The encryption under `sk` of `data` as integers at `layout`.
pub(crate) fn encrypt_integers<BE>(
    module: &Module<BE>,
    layout: &GLWELayout,
    data: &[i64],
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> GLWE<AlignedBuf, i64>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(layout);
    module.glwe_encrypt_sk(
        &mut ct,
        &integer_plaintext(module, layout, data),
        sk,
        &mut Source::new([31u8; 32]),
        &mut Source::new([32u8; 32]),
        &mut scratch.borrow(),
    );
    ct
}

/// `ct` decrypts under `sk` to `want`, as integers in its top-`k` window,
/// within `bound`.
pub(crate) fn assert_decrypts_to<BE>(
    module: &Module<BE>,
    ct: &GLWE<AlignedBuf, i64>,
    want: &[i64],
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    bound: i64,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(ct);
    module.glwe_decrypt(ct, &mut pt, sk, &mut scratch.borrow());
    for (got, want) in plaintext_integers(&pt).iter().zip(want) {
        assert!((got - want).abs() <= bound, "decrypted {got}, want {want} within {bound}");
    }
}

/// Small functional-test flooding parameters, not a production security margin.
pub(crate) fn integer_flood_infos(sigma: f64) -> Noise {
    Noise::Gaussian { sigma, cutoff_factor: 6 }
}

/// Checks both correctness and the presence of caller-sized, per-party flooding.
pub(crate) fn assert_flooded_integers(got: &[i64], want: &[i64], sigma: f64, other_variance: f64, bound: i64) {
    assert_eq!(got.len(), want.len());
    let errors: Vec<f64> = got
        .iter()
        .zip(want)
        .map(|(&got, &want)| {
            let error = got - want;
            assert!(error.abs() <= bound, "integer error {error} exceeds {bound}");
            error as f64
        })
        .collect();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let variance = errors.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / errors.len() as f64;
    let flood_variance = PARTIES as f64 * sigma * sigma;
    assert!(
        variance >= 0.5 * flood_variance && variance <= 2.0 * (flood_variance + other_variance),
        "integer noise variance {variance} outside the expected flooding interval"
    );
}

/// Assert a protocol boundary rejects a layout with the exact static message.
pub(crate) fn assert_panics_with(expected: &str, f: impl FnOnce()) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a layout rejection");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some(expected));
}

/// The aggregate preserves the base law and records the number of independent
/// secret contributions, including after conversion to a prepared key.
pub(crate) fn assert_collective_metadata<A: poulpy_core::layouts::LWEInfos>(infos: &A, parties: usize) {
    let metadata = infos.encryption_metadata().expect("derived encryption provenance");
    assert_eq!(metadata.parties(), parties as u64);
    assert_eq!(metadata.secret_distribution().parties(), parties as u64);
    assert_eq!(metadata.secret_distribution().base(), Distribution::TernaryProb(0.5));
}

/// Checks a fresh phase estimate independently of the secret's party count.
pub(crate) fn assert_fresh_noise<A: poulpy_core::layouts::LWEInfos>(
    infos: &A,
    expected_variance: f64,
    precision: TorusPrecision,
) {
    let estimate = infos.encryption_metadata().expect("derived fresh noise").fresh_noise();
    assert_eq!(estimate.precision(), precision);
    assert!(
        (estimate.variance() - expected_variance).abs() <= 1e-12 * expected_variance.max(1.0),
        "fresh variance differs from expected variance"
    );
}

/// Independently enumerate the small test precisions. Sampling error quarters
/// per bit, while the omitted balanced PK tail changes at limb boundaries.
pub(crate) fn expected_pk_variance(
    inherited: f64,
    mut fresh: f64,
    phase_fold: f64,
    prefix_amplification: f64,
    base2k: usize,
    k: usize,
    k_pk: usize,
) -> (usize, f64) {
    let rounding = phase_fold / 4.0;
    let tail = 0.5 / (1.0 - (-(base2k as f64)).exp2());
    for extra_bits in 0..=k_pk - k {
        let work_limbs = (k + extra_bits).div_ceil(base2k);
        let work_precision = (work_limbs * base2k).min(k_pk);
        let truncation = if work_limbs < k_pk.div_ceil(base2k) {
            prefix_amplification * tail * tail * (-2.0 * (work_precision - k) as f64).exp2()
        } else {
            0.0
        };
        let inherited_and_truncation = if truncation == 0.0 {
            inherited
        } else {
            (inherited.sqrt() + truncation.sqrt()).powi(2)
        };
        let pre_round = inherited_and_truncation + fresh;
        if pre_round <= rounding || k + extra_bits == k_pk {
            return (extra_bits, pre_round + if work_precision > k { rounding } else { 0.0 });
        }
        fresh *= 0.25;
    }
    unreachable!()
}
