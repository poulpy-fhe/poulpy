//! Evaluation-key and gadget encryption compositions with controlled sampling.
use super::encryption::{
    EncryptionObservation, EncryptionParityBackend, EncryptionValue, assert_fresh_encryption_metadata, assert_observations_eq,
    host_mat, observe_sources, secret,
};
use super::{ParityShapes, poisoned_scratch};
use crate::{api::*, layouts::*};
use poulpy_hal::{
    layouts::{Backend, DataViewMut, Module, ScalarZnx},
    source::Source,
    test_suite::TestParams,
};

fn observe_gglwe<B: Backend<ZnxWord = i64>, G: GGLWEToBackendRef<B> + GGLWEInfos>(
    operation: &'static str,
    value: &G,
) -> EncryptionObservation {
    let view = value.to_backend_ref();
    assert!(value.gglwe_layout() == view.gglwe_layout(), "GGLWE view layout");
    if operation.contains("encrypt") {
        assert_fresh_encryption_metadata(value);
        assert!(
            view.encryption_metadata() == value.encryption_metadata(),
            "GGLWE view metadata"
        );
    }
    EncryptionObservation {
        operation,
        value: EncryptionValue::Gglwe(GGLWE {
            encryption_metadata: view.encryption_metadata(),
            data: host_mat::<B>(&view.data),
            base2k: view.base2k,
            dsize: view.dsize,
            k_aux: view.k_aux,
        }),
    }
}
fn observe_ggsw<B: Backend<ZnxWord = i64>, G: GGSWToBackendRef<B> + GGSWInfos>(
    operation: &'static str,
    value: &G,
) -> EncryptionObservation {
    let view = value.to_backend_ref();
    assert!(value.ggsw_layout() == view.ggsw_layout(), "GGSW view layout");
    if operation.contains("encrypt") {
        assert_fresh_encryption_metadata(value);
        assert!(
            view.encryption_metadata() == value.encryption_metadata(),
            "GGSW view metadata"
        );
    }
    EncryptionObservation {
        operation,
        value: EncryptionValue::Ggsw(GGSW {
            encryption_metadata: view.encryption_metadata(),
            data: host_mat::<B>(&view.data),
            base2k: view.base2k,
            dsize: view.dsize,
            k_aux: view.k_aux,
        }),
    }
}
fn observe_seeds(operation: &'static str, seeds: &[[u8; 32]]) -> EncryptionObservation {
    EncryptionObservation {
        operation,
        value: EncryptionValue::Seeds(seeds.to_vec()),
    }
}
fn scalar<B: EncryptionParityBackend>(module: &Module<B>, n: usize, cols: usize) -> ScalarZnx<B::OwnedBuf, i64> {
    let mut value = module.scalar_znx_alloc(n, cols);
    let coefficients: Vec<i64> = (0..n * cols).map(|i| (i % 3) as i64 - 1).collect();
    B::copy_from_host(value.data_mut(), bytemuck::cast_slice(&coefficients));
    value
}

fn poison_gglwe<B: Backend, G: GGLWEToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let data = view.data.data_mut();
    let bytes = B::len_bytes_mut(data);
    B::copy_host_to_view(data, &vec![0xa5; bytes]);
}
fn poison_ggsw<B: Backend, G: GGSWToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let data = view.data.data_mut();
    let bytes = B::len_bytes_mut(data);
    B::copy_host_to_view(data, &vec![0xa5; bytes]);
}
fn poison_compressed<B: Backend, G: GGLWECompressedToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let data = view.data.data_mut();
    let bytes = B::len_bytes_mut(data);
    B::copy_host_to_view(data, &vec![0xa5; bytes]);
}
fn poison_compressed_ggsw<B: Backend, G: GGSWCompressedToBackendMut<B>>(value: &mut G) {
    let mut view = value.to_backend_mut();
    let data = view.data.data_mut();
    let bytes = B::len_bytes_mut(data);
    B::copy_host_to_view(data, &vec![0xa5; bytes]);
}

/// Covers gadget encryption and every evaluation-key encryption method, including
/// compressed forms. The caller-selected pair receives identical realised
/// samples, while all preparation and arithmetic execute independently.
pub fn test_key_encryption_parity<BR: EncryptionParityBackend, BT: EncryptionParityBackend>(
    params: &TestParams,
    shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) {
    fn run<B: EncryptionParityBackend>(
        module: &Module<B>,
        n: usize,
        b: usize,
        rank: usize,
        dsize: usize,
    ) -> Vec<EncryptionObservation> {
        let key = GGLWELayout {
            n: (n as u32).into(),
            base2k: (b as u32).into(),
            dnum: Dnum(3),
            dsize: Dsize(dsize as u32),
            k_aux: TorusPrecision((dsize * b + (n.ilog2() as usize) + 1) as u32),
            rank_in: Rank(rank as u32),
            rank_out: Rank(rank as u32),
            stride: 1,
        };

        let sk = secret(module, n, rank);
        let mut skp = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut skp, &sk);
        let mut lwe = module.lwe_secret_alloc(Degree((n / 2) as u32));
        let coeffs: Vec<i64> = (0..n / 2).map(|i| (i % 3) as i64 - 1).collect();
        B::copy_from_host(lwe.data.data_mut(), bytemuck::cast_slice(&coeffs));
        lwe.dist = sk.dist;
        let mut e = Source::new([149; 32]);
        let mut a = Source::new([151; 32]);
        let seed = [157; 32];
        let mut result = Vec::new();
        let pt = scalar(module, n, rank);
        let mut out = module.gglwe_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut out);
        module.gglwe_encrypt_sk(
            &mut out,
            &pt,
            &skp,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.gglwe_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(observe_gglwe::<B, _>("gglwe_encrypt_sk", &out));
        let mut prepared = module.gglwe_prepared_alloc_from_infos(&key);
        module.gglwe_prepare(
            &mut prepared,
            &out,
            &mut poisoned_scratch::<B>(module.gglwe_prepare_tmp_bytes(&key)).arena(),
        );
        assert!(
            prepared.encryption_metadata() == out.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_sources("gglwe_encrypt_sk_sources", &mut e, &mut a));
        let mut compact = module.gglwe_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.gglwe_compressed_encrypt_sk(
            &mut compact,
            &pt,
            &skp,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.gglwe_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(observe_seeds("gglwe_compressed_seeds", &compact.seed));
        GGLWEToBackendMut::<B>::set_encryption_metadata(&mut out, None);
        module.decompress_gglwe(&mut out, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            out.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_gglwe::<B, _>("gglwe_compressed_encrypt_sk", &out));
        result.push(observe_sources("gglwe_compressed_sources", &mut e, &mut a));
        let g = GGSWLayout {
            n: key.n,
            base2k: key.base2k,
            dnum: key.dnum,
            dsize: key.dsize,
            k_aux: key.k_aux,
            rank: key.rank_out,
        };

        let pt = scalar(module, n, 1);
        let mut out = module.ggsw_alloc_from_infos(&g);
        poison_ggsw::<B, _>(&mut out);
        module.ggsw_encrypt_sk(
            &mut out,
            &pt,
            &skp,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.ggsw_encrypt_sk_tmp_bytes(&g)).arena(),
        );
        result.push(observe_ggsw::<B, _>("ggsw_encrypt_sk", &out));
        let mut prepared = module.ggsw_prepared_alloc_from_infos(&g);
        module.ggsw_prepare(
            &mut prepared,
            &out,
            &mut poisoned_scratch::<B>(module.ggsw_prepare_tmp_bytes(&g)).arena(),
        );
        assert!(
            prepared.encryption_metadata() == out.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_sources("ggsw_encrypt_sk_sources", &mut e, &mut a));
        let mut compact = module.ggsw_compressed_alloc_from_infos(&g);
        poison_compressed_ggsw::<B, _>(&mut compact);
        module.ggsw_compressed_encrypt_sk(
            &mut compact,
            &pt,
            &skp,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.ggsw_compressed_encrypt_sk_tmp_bytes(&g)).arena(),
        );
        result.push(observe_seeds("ggsw_compressed_seeds", &compact.seed));
        GGSWToBackendMut::<B>::set_encryption_metadata(&mut out, None);
        module.decompress_ggsw(&mut out, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            out.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_ggsw::<B, _>("ggsw_compressed_encrypt_sk", &out));
        result.push(observe_sources("ggsw_compressed_sources", &mut e, &mut a));
        let mut switching = module.glwe_switching_key_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut switching);
        module.glwe_switching_key_encrypt_sk(
            &mut switching,
            &sk,
            &sk,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(observe_gglwe::<B, _>("glwe_switching_key_encrypt_sk", &switching));
        result.push(observe_sources("switching_sources", &mut e, &mut a));
        result.push(EncryptionObservation {
            operation: "switching_degrees",
            value: EncryptionValue::SwitchingDegrees(
                *GLWESwitchingKeyDegrees::input_degree(&switching),
                *GLWESwitchingKeyDegrees::output_degree(&switching),
            ),
        });
        let mut compact = module.glwe_switching_key_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_switching_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            &sk,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(observe_seeds("switching_compressed_seeds", &compact.key.seed));
        GGLWEToBackendMut::<B>::set_encryption_metadata(&mut switching, None);
        module.decompress_glwe_switching_key(&mut switching, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            switching.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_gglwe::<B, _>("glwe_switching_key_compressed_encrypt_sk", &switching));
        result.push(observe_sources("switching_compressed_sources", &mut e, &mut a));
        let mut auto = module.glwe_automorphism_key_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut auto);
        module.glwe_automorphism_key_encrypt_sk(
            &mut auto,
            -5,
            &sk,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        assert!(auto.p == -5, "automorphism key exponent");
        result.push(observe_gglwe::<B, _>("glwe_automorphism_key_encrypt_sk", &auto));
        result.push(observe_sources("automorphism_sources", &mut e, &mut a));
        let mut compact = module.glwe_automorphism_key_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_automorphism_key_compressed_encrypt_sk(
            &mut compact,
            -5,
            &sk,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(observe_seeds("automorphism_compressed_seeds", &compact.key.seed));
        GGLWEToBackendMut::<B>::set_encryption_metadata(&mut auto, None);
        module.decompress_automorphism_key(&mut auto, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            auto.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        assert!(auto.p == -5, "automorphism key exponent");
        result.push(observe_gglwe::<B, _>("glwe_automorphism_key_compressed_encrypt_sk", &auto));
        result.push(observe_sources("automorphism_compressed_sources", &mut e, &mut a));
        let tk = GLWETensorKeyLayout {
            n: key.n,
            base2k: key.base2k,
            dnum: key.dnum,
            dsize: key.dsize,
            k_aux: key.k_aux,
            rank: key.rank_out,
        };
        let mut tensor = module.glwe_tensor_key_alloc_from_infos(&tk);
        poison_gglwe::<B, _>(&mut tensor);
        module.glwe_tensor_key_encrypt_sk(
            &mut tensor,
            &sk,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tk)).arena(),
        );
        result.push(observe_gglwe::<B, _>("glwe_tensor_key_encrypt_sk", &tensor));
        result.push(observe_sources("tensor_sources", &mut e, &mut a));
        let mut compact = module.glwe_tensor_key_compressed_alloc_from_infos(&tk);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_tensor_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(&tk)).arena(),
        );
        result.push(observe_seeds("tensor_compressed_seeds", &compact.0.seed));
        GGLWEToBackendMut::<B>::set_encryption_metadata(&mut tensor, None);
        module.decompress_tensor_key(&mut tensor, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            tensor.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        result.push(observe_gglwe::<B, _>("glwe_tensor_key_compressed_encrypt_sk", &tensor));
        result.push(observe_sources("tensor_compressed_sources", &mut e, &mut a));
        let mut rows = module.gglwe_to_ggsw_key_alloc_from_infos(&key);
        for row in &mut rows.keys {
            poison_gglwe::<B, _>(row);
        }
        module.gglwe_to_ggsw_key_encrypt_sk(
            &mut rows,
            &sk,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        for row in &rows.keys {
            result.push(observe_gglwe::<B, _>("gglwe_to_ggsw_key_encrypt_sk", row));
        }
        result.push(observe_sources("row_key_sources", &mut e, &mut a));
        let mut compact = module.gglwe_to_ggsw_key_compressed_alloc_from_infos(&key);
        for row in &mut compact.keys {
            poison_compressed::<B, _>(row);
        }
        module.gglwe_to_ggsw_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            seed,
            &mut e,
            &mut poisoned_scratch::<B>(module.gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        for row in &compact.keys {
            result.push(observe_seeds("row_key_compressed_seeds", &row.seed));
        }
        GGLWEToGGSWKeyToBackendMut::<B>::set_encryption_metadata(&mut rows, None);
        module.decompress_gglwe_to_ggsw_key(&mut rows, &compact);
        assert!(compact.encryption_metadata().is_some());
        assert!(
            rows.encryption_metadata() == compact.encryption_metadata(),
            "key metadata or layout mismatch"
        );
        let mut prepared_rows = module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&rows);
        module.gglwe_to_ggsw_key_prepare(
            &mut prepared_rows,
            &rows,
            &mut poisoned_scratch::<B>(module.gglwe_to_ggsw_key_prepare_tmp_bytes(&rows)).arena(),
        );
        for (prepared, plain) in prepared_rows.keys.iter().zip(&rows.keys) {
            assert!(
                prepared.encryption_metadata() == plain.encryption_metadata(),
                "key metadata or layout mismatch"
            );
        }
        for row in &rows.keys {
            result.push(observe_gglwe::<B, _>("gglwe_to_ggsw_key_compressed_encrypt_sk", row));
        }
        result.push(observe_sources("row_key_compressed_sources", &mut e, &mut a));
        // These three wrappers expose decompression without a corresponding
        // compressed-encryption entry point. Stage the body and seeds explicitly.
        macro_rules! decompress_wrapper {
            ($alloc:ident,$plain:ident,$decompress:ident,$infos:ident) => {{
                let mut compact = module.$alloc(&$infos);
                {
                    let mut view = GGLWECompressedToBackendMut::<B>::to_backend_mut(&mut compact);
                    let data = view.data.data_mut();
                    let words: Vec<i64> = (0..B::len_bytes_mut(data) / 8).map(|i| (i % 7) as i64 - 3).collect();
                    B::copy_host_to_view(data, bytemuck::cast_slice(&words));
                }
                for (i, seed) in compact.0.key.seed.iter_mut().enumerate() {
                    *seed = [(i + 17) as u8; 32];
                }
                let seeds = compact.0.key.seed.clone();
                let mut out = module.$plain(&$infos);
                poison_gglwe::<B, _>(&mut out);
                module.$decompress(&mut out, &compact);
                assert!(
                    GLWESwitchingKeyDegrees::input_degree(&out) == GLWESwitchingKeyDegrees::input_degree(&compact),
                    "key metadata or layout mismatch"
                );
                assert!(
                    GLWESwitchingKeyDegrees::output_degree(&out) == GLWESwitchingKeyDegrees::output_degree(&compact),
                    "key metadata or layout mismatch"
                );
                assert!(compact.0.key.seed == seeds, "decompression changed seeds");
                result.push(observe_gglwe::<B, _>(stringify!($decompress), &out));
                result.push(observe_seeds(stringify!($decompress), &seeds));
            }};
        }
        // Scalar LWE switching/conversion keys are defined only for dsize=1.
        if dsize == 1 {
            let kl = GLWEToLWEKeyLayout {
                n: key.n,
                base2k: key.base2k,
                dnum: key.dnum,
                k_aux: key.k_aux,
                rank_in: key.rank_in,
            };
            decompress_wrapper!(
                glwe_to_lwe_key_compressed_alloc_from_infos,
                glwe_to_lwe_key_alloc_from_infos,
                decompress_glwe_to_lwe_key,
                kl
            );
            let mut out = module.glwe_to_lwe_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.glwe_to_lwe_key_encrypt_sk(
                &mut out,
                &lwe,
                &sk,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(observe_gglwe::<B, _>("glwe_to_lwe_key_encrypt_sk", &out));
            result.push(observe_sources("glwe_to_lwe_sources", &mut e, &mut a));
            let kl = LWEToGLWEKeyLayout {
                n: key.n,
                base2k: key.base2k,
                dnum: key.dnum,
                k_aux: key.k_aux,
                rank_out: key.rank_out,
            };
            decompress_wrapper!(
                lwe_to_glwe_key_compressed_alloc_from_infos,
                lwe_to_glwe_key_alloc_from_infos,
                decompress_lwe_to_glwe_key,
                kl
            );
            let mut out = module.lwe_to_glwe_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.lwe_to_glwe_key_encrypt_sk(
                &mut out,
                &lwe,
                &skp,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.lwe_to_glwe_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(observe_gglwe::<B, _>("lwe_to_glwe_key_encrypt_sk", &out));
            result.push(observe_sources("lwe_to_glwe_sources", &mut e, &mut a));
            let kl = LWESwitchingKeyLayout {
                n: key.n,
                base2k: key.base2k,
                dnum: key.dnum,
                k_aux: key.k_aux,
            };
            decompress_wrapper!(
                lwe_switching_key_compressed_alloc_from_infos,
                lwe_switching_key_alloc_from_infos,
                decompress_lwe_switching_key,
                kl
            );
            let mut out = module.lwe_switching_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.lwe_switching_key_encrypt_sk(
                &mut out,
                &lwe,
                &lwe,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.lwe_switching_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(observe_gglwe::<B, _>("lwe_switching_key_encrypt_sk", &out));
            result.push(observe_sources("lwe_switching_sources", &mut e, &mut a));
        }
        result
    }
    let b = params.base2k.min(12);
    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(2 * b, b) {
            assert_observations_eq(
                run(r, params.n, b, rank, dsize),
                run(t, params.n, b, rank, dsize),
                format_args!("key encryption rank={rank} dsize={dsize}"),
            );
        }
    }
}
