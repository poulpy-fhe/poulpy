//! Encryption parity between caller-selected backends receiving identical draws.
//! Backends with matching random streams can be compared directly. Otherwise,
//! the caller supplies a sampling adapter using [`super::controlled_sampling`];
//! seed equality alone does not imply identical samples across backends.
use super::{ParityBackend, ParityShapes, poisoned_scratch};
use crate::{
    Distribution, EncryptionLayout, GetDistribution, GetDistributionMut,
    api::*,
    layouts::*,
    oep::{ConversionImpl, DecryptionImpl, EncryptionImpl, SamplingImpl},
};
use poulpy_hal::{api::VecZnxFillUniformSource, layouts::*, oep::*, source::Source, test_suite::TestParams};

/// Capabilities needed by the complete encryption/decryption composition suite.
pub trait EncryptionParityBackend:
    ParityBackend
    + EncryptionImpl
    + DecryptionImpl
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

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub(crate) label: &'static str,
    pub(crate) metadata: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
}
pub(crate) fn snapshot_glwe<B: Backend, G: GLWEToBackendRef<B>>(label: &'static str, value: &G) -> Snapshot {
    let view = value.to_backend_ref();
    let mut bytes = vec![0; view.data.n() * view.data.cols() * view.data.size() * size_of::<i64>()];
    B::copy_view_to_host(view.data.data(), &mut bytes);
    Snapshot {
        label,
        metadata: vec![
            view.n().as_usize(),
            view.base2k().as_usize(),
            view.k().as_usize(),
            view.rank().as_usize(),
        ],
        bytes,
    }
}
fn poison_glwe<B: Backend, G: GLWEToBackendMut<B>>(value: &mut G) {
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
pub(crate) fn source_snapshot(label: &'static str, e: &mut Source, a: &mut Source) -> Snapshot {
    Snapshot {
        label,
        metadata: vec![],
        bytes: [e.new_seed().as_slice(), a.new_seed().as_slice()].concat(),
    }
}
pub(crate) fn secret<B: EncryptionParityBackend>(module: &Module<B>, rank: usize) -> BackendGLWESecret<B> {
    let mut secret = module.glwe_secret_alloc((rank as u32).into());
    let data: Vec<i64> = (0..module.n() * rank).map(|i| (i % 3) as i64 - 1).collect();
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
    fn run<B: EncryptionParityBackend>(module: &Module<B>, base2k: usize, rank: usize, k: usize) -> Vec<Snapshot>
    where
        Module<B>: GLWEExpandLWEMatrix<B>,
    {
        let infos = GLWELayout {
            n: (module.n() as u32).into(),
            base2k: (base2k as u32).into(),
            k: (k as u32).into(),
            rank: (rank as u32).into(),
        };
        let enc = EncryptionLayout::new_from_default_sigma(infos).unwrap();
        let sk = secret(module, rank);
        let mut skp = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut skp, &sk);
        let mut pt = module.glwe_plaintext_alloc(infos.base2k, (k as u32 - 1).into());
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
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("encrypt_sk", &out));
        results.push(source_snapshot("encrypt_sk_sources", &mut e, &mut a));
        let mut have = module.glwe_plaintext_alloc(infos.base2k, infos.k);
        poison_glwe::<B, _>(&mut have);
        module.glwe_decrypt(
            &out,
            &mut have,
            &skp,
            &mut poisoned_scratch::<B>(module.glwe_decrypt_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("decrypt", &have));
        poison_glwe::<B, _>(&mut out);
        module.glwe_encrypt_zero_sk(
            &mut out,
            &skp,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("encrypt_zero_sk", &out));
        results.push(source_snapshot("encrypt_zero_sk_sources", &mut e, &mut a));

        let mut pk = module.glwe_public_key_alloc_from_infos(&infos);
        poison_glwe::<B, _>(&mut pk);
        module.glwe_public_key_generate(
            &mut pk,
            &skp,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_public_key_generate_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("public_key_generate", &pk));
        results.push(source_snapshot("public_key_generate_sources", &mut e, &mut a));
        assert_eq!(pk.dist(), sk.dist());
        let mut pkp = module.glwe_public_key_prepared_alloc_from_infos(&pk);
        module.glwe_public_key_prepare(&mut pkp, &pk);
        assert_eq!(pkp.dist(), pk.dist());
        poison_glwe::<B, _>(&mut out);
        module.glwe_encrypt_pk(
            &mut out,
            &pt,
            &pkp,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_pk_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("encrypt_pk", &out));
        results.push(source_snapshot("encrypt_pk_sources", &mut e, &mut a));
        poison_glwe::<B, _>(&mut out);
        module.glwe_encrypt_zero_pk(
            &mut out,
            &pkp,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_encrypt_pk_tmp_bytes(&infos)).arena(),
        );
        results.push(snapshot_glwe::<B, _>("encrypt_zero_pk", &out));
        results.push(source_snapshot("encrypt_zero_pk_sources", &mut e, &mut a));

        let mut compressed = module.glwe_compressed_alloc_from_infos(&infos);
        {
            let mut view = GLWECompressedToBackendMut::<B>::to_backend_mut(&mut compressed);
            let bytes = B::len_bytes_mut(view.data.data_mut());
            B::copy_host_to_view(view.data.data_mut(), &vec![0xA5; bytes]);
        }
        module.glwe_compressed_encrypt_sk(
            &mut compressed,
            &pt,
            &skp,
            [79; 32],
            &enc,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_compressed_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        poison_glwe::<B, _>(&mut out);
        module.decompress_glwe(&mut out, &compressed);
        results.push(snapshot_glwe::<B, _>("compressed_encrypt_sk", &out));
        results.push(source_snapshot("compressed_sources", &mut e, &mut a));
        module.fill_glwe_mask_from_seed(&mut out, [83; 32]);
        results.push(snapshot_glwe::<B, _>("mask_seed", &out));
        module.fill_glwe_mask_from_source(&mut out, &mut a);
        results.push(snapshot_glwe::<B, _>("mask_source", &out));
        results.push(source_snapshot("mask_sources", &mut e, &mut a));
        let mut lsk = module.lwe_secret_alloc((module.n() * rank).into());
        let data: Vec<i64> = (0..module.n() * rank).map(|i| (i % 3) as i64 - 1).collect();
        {
            let mut view = LWESecretToBackendMut::<B>::to_backend_mut(&mut lsk);
            B::copy_host_to_view(view.data.data_mut(), bytemuck::cast_slice(&data));
        }
        *lsk.dist_mut() = *sk.dist();
        for rows in [1, module.n()] {
            let layout = LWEMatrixLayout {
                rows,
                n: (module.n() * rank).into(),
                base2k: infos.base2k,
                k: infos.k,
            };
            let mut matrix = module.lwe_matrix_alloc_from_infos(&layout);
            module.glwe_expand_lwe_matrix(
                &mut matrix,
                &out,
                &mut poisoned_scratch::<B>(module.glwe_expand_lwe_matrix_tmp_bytes(&layout, &infos)).arena(),
            );
            let mut plain = module.glwe_plaintext_alloc(infos.base2k, infos.k);
            poison_glwe::<B, _>(&mut plain);
            module.lwe_matrix_decrypt(
                &matrix,
                &mut plain,
                &lsk,
                &mut poisoned_scratch::<B>(module.lwe_matrix_decrypt_tmp_bytes(&layout)).arena(),
            );
            results.push(snapshot_glwe::<B, _>("matrix_decrypt", &plain));
        }
        results
    }
    let base2k = params.base2k.min(12);
    for &rank in &shapes.ranks {
        for k in [3 * base2k, 4 * base2k - 1, 4 * base2k + 1] {
            assert_eq!(
                run(reference, base2k, rank, k),
                run(tested, base2k, rank, k),
                "GLWE encryption rank={rank} k={k}"
            );
        }
    }
}

fn snapshot_lwe<B: Backend, G: LWEToBackendRef<B>>(label: &'static str, value: &G) -> Snapshot {
    let view = LWEToBackendRef::<B>::to_backend_ref(value);
    let mut bytes = vec![0; view.body.n() * view.body.cols() * view.body.size() * size_of::<i64>()];
    B::copy_view_to_host(view.body.data(), &mut bytes);
    let mut mask = vec![0; view.mask.n() * view.mask.cols() * view.mask.size() * size_of::<i64>()];
    B::copy_view_to_host(view.mask.data(), &mut mask);
    bytes.extend(mask);
    Snapshot {
        label,
        metadata: vec![view.n().as_usize(), view.base2k().as_usize(), view.k().as_usize()],
        bytes,
    }
}

/// LWE encryption/decryption and masks use explicit source-consumption checks.
pub fn test_lwe_encryption_parity<BR: EncryptionParityBackend, BT: EncryptionParityBackend>(
    params: &TestParams,
    _shapes: &ParityShapes,
    reference: &Module<BR>,
    tested: &Module<BT>,
) {
    fn run<B: EncryptionParityBackend>(module: &Module<B>, n: usize, base2k: usize, k: usize) -> Vec<Snapshot> {
        let infos = LWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
        };
        let enc = EncryptionLayout::new_from_default_sigma(infos).unwrap();
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
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.lwe_encrypt_sk_tmp_bytes(&infos)).arena(),
        );
        let mut results = vec![
            snapshot_lwe::<B, _>("lwe_encrypt_sk", &out),
            source_snapshot("lwe_sources", &mut e, &mut a),
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
        let view = LWEPlaintextToBackendRef::<B>::to_backend_ref(&have);
        let mut bytes = vec![0; view.data.n() * view.data.cols() * view.data.size() * size_of::<i64>()];
        B::copy_view_to_host(view.data.data(), &mut bytes);
        results.push(Snapshot {
            label: "lwe_decrypt",
            metadata: vec![view.base2k().as_usize(), view.k().as_usize()],
            bytes,
        });
        module.fill_lwe_mask_from_seed(base2k, &mut out, [103; 32]);
        results.push(snapshot_lwe::<B, _>("lwe_mask_seed", &out));
        module.fill_lwe_mask_from_source(base2k, &mut out, &mut a);
        results.push(snapshot_lwe::<B, _>("lwe_mask_source", &out));
        results.push(source_snapshot("lwe_mask_sources", &mut e, &mut a));
        // LWECompressed has a public decompressor but no compressed encrypt
        // operation. Stage its canonical body and seed explicitly on each backend.
        let mut compact = module.lwe_compressed_alloc_from_infos(&infos);
        compact.seed = [107; 32];
        let mut body: Vec<i64> = (0..infos.size()).map(|i| i as i64 + 1).collect();
        let pad = (base2k - k % base2k) % base2k;
        *body.last_mut().unwrap() &= !0i64 << pad;
        B::copy_from_host(compact.data.data_mut(), bytemuck::cast_slice(&body));
        let mut before = vec![0; B::len_bytes(compact.data.data())];
        B::copy_to_host(compact.data.data(), &mut before);
        module.decompress_lwe(&mut out, &compact);
        results.push(snapshot_lwe::<B, _>("lwe_decompress", &out));
        let mut after = vec![0; B::len_bytes(compact.data.data())];
        B::copy_to_host(compact.data.data(), &mut after);
        assert_eq!(after, before);
        assert_eq!(compact.seed, [107; 32]);
        results
    }
    let b = params.base2k.min(12);
    for n in [8, reference.n()] {
        for k in [3 * b, 4 * b - 1, 4 * b + 1] {
            assert_eq!(run(reference, n, b, k), run(tested, n, b, k), "LWE encryption n={n} k={k}");
        }
    }
}
