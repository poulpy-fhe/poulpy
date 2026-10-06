//! Encryption parity between caller-selected backends receiving identical draws.
//! Backends with matching random streams can be compared directly. Otherwise,
//! the caller supplies a sampling adapter using [`super::controlled_sampling`];
//! seed equality alone does not imply identical samples across backends.
use super::{ParityBackend, ParityShapes, poisoned_scratch, unnormalized_twin};
use crate::{
    Distribution, GetDistribution, GetDistributionMut,
    api::*,
    layouts::*,
    oep::{ConversionImpl, DecryptionImpl, EncryptionImpl, GLWENormalizeImpl, SamplingImpl},
};
use poulpy_hal::{
    AlignedBuf,
    api::{VecZnxAddAssign, VecZnxCopy, VecZnxFillUniformSource},
    layouts::*,
    oep::*,
    source::Source,
    test_suite::{TestParams, download_vec_znx},
};

/// Capabilities needed by the complete encryption/decryption composition suite.
pub trait EncryptionParityBackend:
    ParityBackend
    + EncryptionImpl
    + DecryptionImpl
    + GLWENormalizeImpl
    + SamplingImpl
    + ConversionImpl
    + HalModuleImpl
    + HalVecZnxImpl
    + HalVecZnxBigImpl
    + HalVecZnxDftImpl
    + HalSvpImpl
    + HalVmpImpl
    + HalConvolutionImpl
{
}
impl<B> EncryptionParityBackend for B where
    B: ParityBackend
        + EncryptionImpl
        + DecryptionImpl
        + GLWENormalizeImpl
        + SamplingImpl
        + ConversionImpl
        + HalModuleImpl
        + HalVecZnxImpl
        + HalVecZnxBigImpl
        + HalVecZnxDftImpl
        + HalSvpImpl
        + HalVmpImpl
        + HalConvolutionImpl
{
}

#[derive(PartialEq, Eq)]
pub(crate) enum EncryptionValue {
    Glwe(GLWE<AlignedBuf, i64>),
    GlwePlaintext(GLWEPlaintext<AlignedBuf, i64>),
    Lwe(LWE<AlignedBuf, i64>),
    LwePlaintext(LWEPlaintext<AlignedBuf, i64>),
    Gglwe(GGLWE<AlignedBuf, i64>),
    Ggsw(GGSW<AlignedBuf, i64>),
    Sources([[u8; 32]; 2]),
    Seeds(Vec<[u8; 32]>),
    SwitchingDegrees(Degree, Degree),
}

#[derive(PartialEq, Eq)]
pub(crate) struct EncryptionObservation {
    pub(crate) operation: &'static str,
    pub(crate) value: EncryptionValue,
}

pub(crate) fn assert_observations_eq(
    want: Vec<EncryptionObservation>,
    have: Vec<EncryptionObservation>,
    context: std::fmt::Arguments<'_>,
) {
    assert!(want.len() == have.len(), "{context}: observation count");
    for (want, have) in want.iter().zip(&have) {
        assert!(want.operation == have.operation, "{context}: observation order");
        assert!(
            want.value == have.value,
            "{context}: {} differs between backends",
            want.operation
        );
        if let (EncryptionValue::Glwe(want), EncryptionValue::Glwe(have)) = (&want.value, &have.value) {
            assert!(want.is_canonical() == have.is_canonical(), "{context}: GLWE canonical flag");
        }
    }
}

// Backend views are not required to expose host slices. Transfer their storage
// into a typed host container while retaining the complete view geometry.
fn host_vec<B: Backend>(value: &VecZnxBackendRef<'_, B>) -> VecZnx<AlignedBuf, B::ZnxWord> {
    let mut data = poulpy_hal::alloc_aligned::<u8>(B::len_bytes_ref(value.data()));
    B::copy_view_to_host(value.data(), &mut data);
    VecZnx::from_shape(data, value.shape())
}

pub(crate) fn host_mat<B: Backend>(value: &MatZnx<B::BufRef<'_>, B::ZnxWord>) -> MatZnx<AlignedBuf, B::ZnxWord> {
    let mut data = poulpy_hal::alloc_aligned::<u8>(B::len_bytes_ref(value.data()));
    B::copy_view_to_host(value.data(), &mut data);
    MatZnx::from_data(data, value.n(), value.rows(), value.cols_in(), value.cols_out(), value.size())
}

pub(crate) fn assert_fresh_noise(value: &impl LWEInfos) {
    assert_noise(value, crate::DEFAULT_SIGMA_XE.powi(2));
    let noise = value.noise().unwrap();
    assert!(
        noise.body().variance() == crate::DEFAULT_SIGMA_XE.powi(2),
        "fresh body variance"
    );
    assert!(
        noise.masks().iter().all(|component| component.variance() == 0.0),
        "fresh mask variance"
    );
}

fn assert_noise(value: &impl LWEInfos, variance: f64) {
    let metadata = value.noise().expect("encryption must record its provenance");
    assert!(metadata.parties() == 1, "encryption party count");
    assert!(
        metadata.secret_distribution().base() == Distribution::TernaryProb(2.0 / 3.0),
        "encryption secret distribution"
    );
    assert!(
        metadata.secret_distribution().parties() == 1,
        "secret distribution party count"
    );
    assert!(metadata.precision() == value.k(), "fresh noise precision");
    assert!((metadata.phase_noise(value.n().as_usize()).variance() - variance).abs() <= variance * 1e-12);
}

fn observe_glwe<B: Backend<ZnxWord = i64>, G: GLWEToBackendRef<B>>(operation: &'static str, value: &G) -> EncryptionObservation {
    let view = value.to_backend_ref();
    if operation == "encrypt_pk" || operation == "encrypt_zero_pk" {
        let variance = (2.0 * view.rank().as_usize() as f64 * view.n().as_usize() as f64 * (2.0 / 3.0) + 1.0)
            * crate::DEFAULT_SIGMA_XE.powi(2);
        assert_noise(&view, variance);
        let noise = LWEInfos::noise(&view).unwrap();
        assert!(noise.rank() == view.rank().as_usize(), "noise rank");
        let sigma2 = crate::DEFAULT_SIGMA_XE.powi(2);
        let body = (view.rank().as_usize() as f64 * view.n().as_usize() as f64 * (2.0 / 3.0) + 1.0) * sigma2;
        assert!((noise.body().variance() - body).abs() <= body * 1e-12);
        assert!(
            noise.masks().iter().all(|component| component.variance() == sigma2),
            "mask variance"
        );
    } else if operation.contains("encrypt") || operation == "public_key_generate" {
        assert_fresh_noise(&view);
        assert!(LWEInfos::noise(&view).unwrap().rank() == view.rank().as_usize(), "noise rank");
    }
    EncryptionObservation {
        operation,
        value: EncryptionValue::Glwe(GLWE {
            noise: view.noise(),
            data: host_vec::<B>(&view.data),
            k: view.k,
            base2k: view.base2k,
            canonical: view.canonical,
        }),
    }
}

fn observe_plaintext<B: Backend<ZnxWord = i64>>(
    operation: &'static str,
    value: &GLWEPlaintext<B::OwnedBuf, i64>,
) -> EncryptionObservation {
    EncryptionObservation {
        operation,
        value: EncryptionValue::GlwePlaintext(value.to_host_owned::<B>()),
    }
}

fn poison_glwe<B: Backend, G: GLWEToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let bytes = B::len_bytes_mut(view.data.data_mut());
    B::copy_host_to_view(view.data.data_mut(), &vec![0xA5; bytes]);
}
fn poison_glwe_compressed<B: Backend, G: GLWECompressedToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let bytes = B::len_bytes_mut(view.data.data_mut());
    B::copy_host_to_view(view.data.data_mut(), &vec![0xA5; bytes]);
}
fn poison_lwe<B: Backend, G: LWEToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let body_bytes = B::len_bytes_mut(view.body.data_mut());
    B::copy_host_to_view(view.body.data_mut(), &vec![0xA5; body_bytes]);
    let mask_bytes = B::len_bytes_mut(view.mask.data_mut());
    B::copy_host_to_view(view.mask.data_mut(), &vec![0xA5; mask_bytes]);
}
pub(crate) fn observe_sources(operation: &'static str, e: &mut Source, a: &mut Source) -> EncryptionObservation {
    EncryptionObservation {
        operation,
        value: EncryptionValue::Sources([e.new_seed(), a.new_seed()]),
    }
}
pub(crate) fn secret<B: EncryptionParityBackend>(module: &Module<B>, n: usize, rank: usize) -> BackendGLWESecret<B> {
    let mut secret = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: n.into(),
        rank: (rank as u32).into(),
    });
    let data: Vec<i64> = (0..n * rank).map(|i| (i % 3) as i64 - 1).collect();
    let mut view = GLWESecretToBackendMut::<B>::to_backend_mut(&mut secret);
    B::copy_host_to_view(view.data.data_mut(), bytemuck::cast_slice(&data));
    drop(view);
    *secret.dist_mut() = Distribution::TernaryProb(2.0 / 3.0);
    secret
}

/// GLWE masks, secret/public-key encryption, compressed encryption and decryption.
pub fn test_glwe_encryption_parity<BR: EncryptionParityBackend, BT: EncryptionParityBackend>(
    params: &TestParams,
    shapes: &ParityShapes,
    reference: &Module<BR>,
    tested: &Module<BT>,
) where
    Module<BR>: GLWEExpandLWEMatrix<BR>,
    Module<BT>: GLWEExpandLWEMatrix<BT>,
{
    fn run<B: EncryptionParityBackend>(
        module: &Module<B>,
        n: usize,
        base2k: usize,
        rank: usize,
        k: usize,
    ) -> Vec<EncryptionObservation>
    where
        Module<B>: GLWEExpandLWEMatrix<B>,
    {
        let infos = GLWELayout {
            n: (n as u32).into(),
            base2k: (base2k as u32).into(),
            k: (k as u32).into(),
            rank: (rank as u32).into(),
        };

        let sk = secret(module, n, rank);
        let mut skp = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut skp, &sk);
        let mut pt = module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
            n: infos.n,
            base2k: infos.base2k,
            k: (k as u32 - 1).into(),
        });
        module.vec_znx_fill_uniform_source(
            base2k,
            pt.k().as_usize(),
            &mut poulpy_hal::test_suite::vec_znx_backend_mut::<B>(&mut pt.data),
            0,
            &mut Source::new([67; 32]),
        );
        let mut out = module.glwe_alloc_from_infos(&infos);
        let mut results = Vec::new();
        let mut e = Source::new([71; 32]);
        let mut a = Source::new([73; 32]);
        poison_glwe::<B, _>(&mut out);
        module.glwe_encrypt_sk(
            &mut out,
            &pt,
            &skp,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        results.push(observe_glwe::<B, _>("encrypt_sk", &out));
        results.push(observe_sources("encrypt_sk_sources", &mut e, &mut a));
        let mut with_mask = module.glwe_alloc_from_infos(&infos);
        poison_glwe::<B, _>(&mut with_mask);
        module.fill_glwe_mask_from_seed(&mut with_mask, [73; 32]);
        module.glwe_encrypt_sk_with_mask(
            &mut with_mask,
            &pt,
            &skp,
            &mut Source::new([71; 32]),
            &mut poisoned_scratch::<B>(module.glwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        assert!(
            with_mask.to_host_owned::<B>() == out.to_host_owned::<B>(),
            "caller-filled masks must produce the same ciphertext"
        );
        let mut have = module.glwe_plaintext_alloc_from_infos(&infos);
        poison_glwe::<B, _>(&mut have);
        module.glwe_decrypt(
            &out,
            &mut have,
            &skp,
            &mut poisoned_scratch::<B>(module.glwe_decrypt_tmp_bytes(&infos)).arena(),
        );
        results.push(observe_plaintext::<B>("decrypt", &have));
        let mut twin = module.glwe_alloc_from_infos(&infos);
        unnormalized_twin::<B, B>(&out, &mut twin);
        let mut have_twin = module.glwe_plaintext_alloc_from_infos(&infos);
        module.glwe_decrypt(
            &twin,
            &mut have_twin,
            &skp,
            &mut poisoned_scratch::<B>(module.glwe_decrypt_tmp_bytes(&infos)).arena(),
        );
        assert!(
            have_twin.to_host_owned::<B>() == have.to_host_owned::<B>(),
            "glwe_decrypt, unnormalized operand"
        );
        // The mask phase is the decryption without the body: of the ciphertext's
        // mask in place and of an allocated copy; the unnormalized twin's is rejected.
        let mut phase = module.glwe_alloc_from_infos(&GLWELayout { rank: Rank(0), ..infos });
        GLWEToBackendMut::<B>::set_noise(
            &mut phase,
            Some(crate::ComponentNoise::from_secret_at(*sk.dist(), infos.k, 0)),
        );
        poison_glwe::<B, _>(&mut phase);
        let mask_scratch = || poisoned_scratch::<B>(module.glwe_mask_inner_product_tmp_bytes(&infos));
        module.glwe_mask_inner_product(&mut phase, &out, &skp, &mut mask_scratch().arena());
        assert!(phase.noise().is_none(), "mask inner product metadata");
        results.push(observe_glwe::<B, _>("mask_inner_product", &phase));
        let mut mask = module.glwe_mask_alloc_from_infos(&infos);
        for j in 0..rank {
            module.vec_znx_copy(
                &mut poulpy_hal::test_suite::vec_znx_backend_mut::<B>(mask.data_mut()),
                j,
                &poulpy_hal::test_suite::vec_znx_backend_ref::<B>(out.data()),
                j + 1,
            );
        }
        let mut other = module.glwe_plaintext_alloc_from_infos(&infos);
        module.glwe_mask_inner_product(&mut other, &mask, &skp, &mut mask_scratch().arena());
        assert!(
            other.to_host_owned::<B>().data == phase.to_host_owned::<B>().data,
            "glwe_mask_inner_product, allocated mask"
        );
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.glwe_mask_inner_product(&mut other, &twin, &skp, &mut mask_scratch().arena());
        }));
        assert!(rejected.is_err(), "glwe_mask_inner_product accepted an unnormalized mask");
        module.vec_znx_add_assign(
            &mut poulpy_hal::test_suite::vec_znx_backend_mut::<B>(&mut phase.data),
            0,
            &poulpy_hal::test_suite::vec_znx_backend_ref::<B>(out.data()),
            0,
        );
        module.glwe_normalize_assign(
            &mut phase,
            &mut poisoned_scratch::<B>(module.glwe_normalize_tmp_bytes()).arena(),
        );
        assert!(
            phase.to_host_owned::<B>().data == have.to_host_owned::<B>().data,
            "glwe_mask_inner_product plus the body"
        );
        poison_glwe::<B, _>(&mut out);
        module.glwe_encrypt_zero_sk(
            &mut out,
            &skp,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        results.push(observe_glwe::<B, _>("encrypt_zero_sk", &out));
        results.push(observe_sources("encrypt_zero_sk_sources", &mut e, &mut a));

        for gap in [0, 1, base2k, 3 * base2k] {
            let pk_infos = GLWELayout {
                k: TorusPrecision(infos.k.0 + gap as u32),
                ..infos
            };
            let mut wide_pt = module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
                n: infos.n,
                base2k: infos.base2k,
                k: TorusPrecision(pk_infos.k.0 + base2k as u32),
            });
            module.vec_znx_fill_uniform_source(
                base2k,
                wide_pt.k().as_usize(),
                &mut poulpy_hal::test_suite::vec_znx_backend_mut::<B>(&mut wide_pt.data),
                0,
                &mut Source::new([68; 32]),
            );
            for compressed in [false, true] {
                let mut pk = module.glwe_public_key_alloc_from_infos(&pk_infos);
                for l in 0..pk.rank().as_usize() {
                    poison_glwe::<B, _>(&mut GLWEPublicKeyAtViewMut::<B>::at_view_mut(&mut pk, l));
                }
                if compressed {
                    let mut pk_compressed = module.glwe_public_key_compressed_alloc_from_infos(&pk_infos);
                    module.glwe_public_key_compressed_generate(
                        &mut pk_compressed,
                        &skp,
                        [89; 32],
                        &mut e,
                        &mut poisoned_scratch::<B>(module.glwe_public_key_compressed_generate_tmp_bytes(&pk_infos)).arena(),
                    );
                    module.decompress_glwe_public_key(&mut pk, &pk_compressed);
                } else {
                    module.glwe_public_key_generate(
                        &mut pk,
                        &skp,
                        &mut e,
                        &mut a,
                        &mut poisoned_scratch::<B>(module.glwe_public_key_generate_tmp_bytes(&pk_infos)).arena(),
                    );
                }
                for l in 0..pk.rank().as_usize() {
                    results.push(observe_glwe::<B, _>(
                        "public_key_generate",
                        &GLWEPublicKeyAtViewRef::<B>::at_view(&pk, l),
                    ));
                }
                results.push(observe_sources("public_key_generate_sources", &mut e, &mut a));
                assert!(pk.dist() == sk.dist(), "public key distribution");
                let mut pkp = module.glwe_public_key_prepared_alloc_from_infos(&pk);
                module.glwe_public_key_prepare(
                    &mut pkp,
                    &pk,
                    &mut poisoned_scratch::<B>(module.glwe_public_key_prepare_tmp_bytes(&pk_infos)).arena(),
                );
                assert!(pkp.dist() == pk.dist(), "prepared public key distribution");
                assert!(pkp.noise() == pk.noise(), "prepared public key metadata");
                poison_glwe::<B, _>(&mut out);
                module.glwe_encrypt_pk(
                    &mut out,
                    &wide_pt,
                    &pkp,
                    &mut e,
                    &mut a,
                    &mut poisoned_scratch::<B>(module.glwe_encrypt_pk_tmp_bytes(&infos, &pkp)).arena(),
                );
                results.push(observe_glwe::<B, _>(if gap == 0 { "encrypt_pk" } else { "pk_reduced" }, &out));
                results.push(observe_sources("encrypt_pk_sources", &mut e, &mut a));
                poison_glwe::<B, _>(&mut out);
                module.glwe_encrypt_zero_pk(
                    &mut out,
                    &pkp,
                    &mut e,
                    &mut a,
                    &mut poisoned_scratch::<B>(module.glwe_encrypt_pk_tmp_bytes(&infos, &pkp)).arena(),
                );
                results.push(observe_glwe::<B, _>(
                    if gap == 0 { "encrypt_zero_pk" } else { "pk_zero_reduced" },
                    &out,
                ));
                results.push(observe_sources("encrypt_zero_pk_sources", &mut e, &mut a));
            }
        }
        let mut compressed = module.glwe_compressed_alloc_from_infos(&infos);
        poison_glwe_compressed::<B, _>(&mut compressed);
        module.glwe_compressed_encrypt_sk(
            &mut compressed,
            &pt,
            &skp,
            [79; 32],
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_compressed_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        poison_glwe::<B, _>(&mut out);
        module.decompress_glwe(&mut out, &compressed);
        results.push(observe_glwe::<B, _>("compressed_encrypt_sk", &out));
        results.push(observe_sources("compressed_sources", &mut e, &mut a));
        poison_glwe_compressed::<B, _>(&mut compressed);
        module.glwe_compressed_encrypt_zero_sk(
            &mut compressed,
            &skp,
            [79; 32],
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_compressed_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        let mut zero = module.glwe_alloc_from_infos(&infos);
        poison_glwe::<B, _>(&mut zero);
        module.decompress_glwe(&mut zero, &compressed);
        results.push(observe_glwe::<B, _>("compressed_encrypt_zero_sk", &zero));
        results.push(observe_sources("compressed_zero_sources", &mut e, &mut a));
        assert!(out.noise().is_some(), "encryption metadata");
        let previous_metadata = out.noise();
        module.fill_glwe_mask_from_seed(&mut out, [83; 32]);
        assert!(out.noise().is_none(), "mask fill metadata");
        results.push(observe_glwe::<B, _>("mask_seed", &out));
        GLWEToBackendMut::<B>::set_noise(&mut out, previous_metadata);
        module.fill_glwe_mask_from_source(&mut out, &mut a);
        assert!(out.noise().is_none(), "mask fill metadata");
        results.push(observe_glwe::<B, _>("mask_source", &out));
        results.push(observe_sources("mask_sources", &mut e, &mut a));
        let mut lsk = module.lwe_secret_alloc((n * rank).into());
        let data: Vec<i64> = (0..n * rank).map(|i| (i % 3) as i64 - 1).collect();
        {
            let mut view = LWESecretToBackendMut::<B>::to_backend_mut(&mut lsk);
            B::copy_host_to_view(view.data.data_mut(), bytemuck::cast_slice(&data));
        }
        *lsk.dist_mut() = *sk.dist();
        for rows in [1, n] {
            let layout = LWEMatrixLayout {
                rows,
                n: (n * rank).into(),
                base2k: infos.base2k,
                k: infos.k,
            };
            let mut matrix = module.lwe_matrix_alloc_from_infos(&layout);
            module.glwe_expand_lwe_matrix(
                &mut matrix,
                &out,
                &mut poisoned_scratch::<B>(module.glwe_expand_lwe_matrix_tmp_bytes(&layout, &infos)).arena(),
            );
            let mut plain = module.glwe_plaintext_alloc_from_infos(&infos);
            poison_glwe::<B, _>(&mut plain);
            module.lwe_matrix_decrypt(
                &matrix,
                &mut plain,
                &lsk,
                &mut poisoned_scratch::<B>(module.lwe_matrix_decrypt_tmp_bytes(&layout)).arena(),
            );
            results.push(observe_plaintext::<B>("matrix_decrypt", &plain));
        }
        results
    }
    let base2k = params.base2k.min(12);
    for &rank in &shapes.ranks {
        for k in [3 * base2k, 4 * base2k - 1, 4 * base2k + 1] {
            assert_observations_eq(
                run(reference, params.n, base2k, rank, k),
                run(tested, params.n, base2k, rank, k),
                format_args!("GLWE encryption rank={rank} k={k}"),
            );
        }
    }
}

fn observe_lwe<B: Backend<ZnxWord = i64>, G: LWEToBackendRef<B>>(operation: &'static str, value: &G) -> EncryptionObservation {
    let view = LWEToBackendRef::<B>::to_backend_ref(value);
    if operation == "lwe_encrypt_sk" {
        assert_fresh_noise(&view);
        assert!(
            LWEInfos::noise(&view).unwrap().rank() == view.n().as_usize(),
            "LWE noise rank"
        );
    }
    EncryptionObservation {
        operation,
        value: EncryptionValue::Lwe(LWE {
            noise: view.noise(),
            body: host_vec::<B>(&view.body),
            mask: host_vec::<B>(&view.mask),
            k: view.k,
            base2k: view.base2k,
        }),
    }
}

/// LWE encryption/decryption and masks use explicit source-consumption checks.
pub fn test_lwe_encryption_parity<BR: EncryptionParityBackend, BT: EncryptionParityBackend>(
    params: &TestParams,
    _shapes: &ParityShapes,
    reference: &Module<BR>,
    tested: &Module<BT>,
) {
    fn run<B: EncryptionParityBackend>(module: &Module<B>, n: usize, base2k: usize, k: usize) -> Vec<EncryptionObservation> {
        let infos = LWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
        };
        let mut sk = module.lwe_secret_alloc(n.into());
        {
            let mut view = LWESecretToBackendMut::<B>::to_backend_mut(&mut sk);
            let data: Vec<i64> = (0..n).map(|i| (i % 3) as i64 - 1).collect();
            B::copy_host_to_view(view.data.data_mut(), bytemuck::cast_slice(&data));
        }
        *sk.dist_mut() = Distribution::TernaryProb(2.0 / 3.0);
        let mut pt = module.lwe_plaintext_alloc(base2k.into(), k.into());
        {
            let mut view = LWEPlaintextToBackendMut::<B>::to_backend_mut(&mut pt);
            module.vec_znx_fill_uniform_source(base2k, k, &mut view.data, 0, &mut Source::new([89; 32]));
        }
        let mut out = module.lwe_alloc_from_infos(&infos);
        let mut e = Source::new([97; 32]);
        let mut a = Source::new([101; 32]);
        poison_lwe::<B, _>(&mut out);
        module.lwe_encrypt_sk(
            &mut out,
            &pt,
            &sk,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.lwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        let mut results = vec![
            observe_lwe::<B, _>("lwe_encrypt_sk", &out),
            observe_sources("lwe_sources", &mut e, &mut a),
        ];
        let mut have = module.lwe_plaintext_alloc(base2k.into(), k.into());
        {
            let mut view = LWEPlaintextToBackendMut::<B>::to_backend_mut(&mut have);
            let bytes = B::len_bytes_mut(view.data.data_mut());
            B::copy_host_to_view(view.data.data_mut(), &vec![0xA5; bytes]);
        }
        module.lwe_decrypt(
            &out,
            &mut have,
            &sk,
            &mut poisoned_scratch::<B>(module.lwe_decrypt_tmp_bytes(&infos)).arena(),
        );
        results.push(EncryptionObservation {
            operation: "lwe_decrypt",
            value: EncryptionValue::LwePlaintext(LWEPlaintext {
                data: download_vec_znx::<B>(&have.data),
                base2k: have.base2k,
                k: have.k,
            }),
        });
        assert!(out.noise().is_some(), "LWE encryption metadata");
        let previous_metadata = out.noise();
        module.fill_lwe_mask_from_seed(base2k, &mut out, [103; 32]);
        assert!(out.noise().is_none(), "LWE mask fill metadata");
        results.push(observe_lwe::<B, _>("lwe_mask_seed", &out));
        LWEToBackendMut::<B>::set_noise(&mut out, previous_metadata);
        module.fill_lwe_mask_from_source(base2k, &mut out, &mut a);
        assert!(out.noise().is_none(), "LWE mask fill metadata");
        results.push(observe_lwe::<B, _>("lwe_mask_source", &out));
        results.push(observe_sources("lwe_mask_sources", &mut e, &mut a));
        // LWECompressed has a public decompressor but no compressed encrypt
        // operation. Stage its canonical body and seed explicitly on each backend.
        let mut compact = module.lwe_compressed_alloc_from_infos(&infos);
        compact.seed = [107; 32];
        let mut body: Vec<i64> = (0..infos.size()).map(|i| i as i64 + 1).collect();
        let pad = (base2k - k % base2k) % base2k;
        *body.last_mut().unwrap() &= !0i64 << pad;
        B::copy_from_host(compact.data.data_mut(), bytemuck::cast_slice(&body));
        let before = LWECompressed {
            data: download_vec_znx::<B>(&compact.data),
            base2k: compact.base2k,
            k: compact.k,
            seed: compact.seed,
        };
        module.decompress_lwe(&mut out, &compact);
        results.push(observe_lwe::<B, _>("lwe_decompress", &out));
        let after = LWECompressed {
            data: download_vec_znx::<B>(&compact.data),
            base2k: compact.base2k,
            k: compact.k,
            seed: compact.seed,
        };
        assert!(after == before, "LWE decompression changed its source");
        assert!(compact.seed == [107; 32], "LWE decompression changed the seed");
        results
    }
    let b = params.base2k.min(12);
    for n in [8, reference.n()] {
        for k in [3 * b, 4 * b - 1, 4 * b + 1] {
            assert_observations_eq(
                run(reference, n, b, k),
                run(tested, n, b, k),
                format_args!("LWE encryption n={n} k={k}"),
            );
        }
    }
}
