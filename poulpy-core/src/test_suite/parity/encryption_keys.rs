//! Evaluation-key and gadget encryption compositions with controlled sampling.
use super::encryption::{EncryptionParityBackend, Snapshot, secret, source_snapshot};
use super::{ParityShapes, poisoned_scratch};
use crate::{EncryptionLayout, api::*, layouts::*};
use poulpy_hal::{
    layouts::{Backend, DataView, DataViewMut, Module, ScalarZnx},
    source::Source,
    test_suite::TestParams,
};

fn gglwe_snapshot<B: Backend, G: GGLWEToBackendRef<B> + GGLWEInfos>(label: &'static str, value: &G) -> Snapshot {
    let view = value.to_backend_ref();
    let data = view.data.data();
    let mut bytes = vec![0; B::len_bytes_ref(data)];
    B::copy_view_to_host(data, &mut bytes);
    Snapshot {
        label,
        metadata: vec![
            value.n().as_usize(),
            value.base2k().as_usize(),
            value.dnum().as_usize(),
            value.dsize().as_usize(),
            value.k_aux().as_usize(),
            value.rank_in().as_usize(),
            value.rank_out().as_usize(),
        ],
        bytes,
    }
}
fn ggsw_snapshot<B: Backend, G: GGSWToBackendRef<B> + GGSWInfos>(label: &'static str, value: &G) -> Snapshot {
    let view = value.to_backend_ref();
    let data = view.data.data();
    let mut bytes = vec![0; B::len_bytes_ref(data)];
    B::copy_view_to_host(data, &mut bytes);
    Snapshot {
        label,
        metadata: vec![
            value.n().as_usize(),
            value.base2k().as_usize(),
            value.dnum().as_usize(),
            value.dsize().as_usize(),
            value.k_aux().as_usize(),
            value.rank().as_usize(),
        ],
        bytes,
    }
}
fn seeds_snapshot(label: &'static str, seeds: &[[u8; 32]]) -> Snapshot {
    Snapshot {
        label,
        metadata: vec![seeds.len()],
        bytes: seeds.concat(),
    }
}
fn scalar<B: EncryptionParityBackend>(module: &Module<B>, cols: usize) -> ScalarZnx<B::OwnedBuf, i64> {
    let mut value = module.scalar_znx_alloc(module.n(), cols);
    let coefficients: Vec<i64> = (0..module.n() * cols).map(|i| (i % 3) as i64 - 1).collect();
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
/// compressed forms. The reference backend replays the tested backend's realized
/// samples, while all preparation and arithmetic execute independently.
pub fn test_key_encryption_parity<BR: EncryptionParityBackend, BT: EncryptionParityBackend>(
    params: &TestParams,
    shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) {
    fn run<B: EncryptionParityBackend>(module: &Module<B>, b: usize, rank: usize, dsize: usize) -> Vec<Snapshot> {
        let key = GGLWELayout {
            n: (module.n() as u32).into(),
            base2k: (b as u32).into(),
            dnum: Dnum(3),
            dsize: Dsize(dsize as u32),
            k_aux: TorusPrecision((dsize * b + module.log_n() + 1) as u32),
            rank_in: Rank(rank as u32),
            rank_out: Rank(rank as u32),
            stride: 1,
        };
        let enc = EncryptionLayout::new_from_default_sigma(key).unwrap();
        let sk = secret(module, rank);
        let mut skp = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut skp, &sk);
        let mut lwe = module.lwe_secret_alloc(Degree((module.n() / 2) as u32));
        let coeffs: Vec<i64> = (0..module.n() / 2).map(|i| (i % 3) as i64 - 1).collect();
        B::copy_from_host(lwe.data.data_mut(), bytemuck::cast_slice(&coeffs));
        lwe.dist = sk.dist;
        let mut e = Source::new([149; 32]);
        let mut a = Source::new([151; 32]);
        let seed = [157; 32];
        let mut result = Vec::new();
        let pt = scalar(module, rank);
        let mut out = module.gglwe_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut out);
        module.gglwe_encrypt_sk(
            &mut out,
            &pt,
            &skp,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.gglwe_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(gglwe_snapshot::<B, _>("gglwe_encrypt_sk", &out));
        result.push(source_snapshot("gglwe_encrypt_sk_sources", &mut e, &mut a));
        let mut compact = module.gglwe_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.gglwe_compressed_encrypt_sk(
            &mut compact,
            &pt,
            &skp,
            seed,
            &enc,
            &mut e,
            &mut poisoned_scratch::<B>(module.gglwe_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(seeds_snapshot("gglwe_compressed_seeds", &compact.seed));
        module.decompress_gglwe(&mut out, &compact);
        result.push(gglwe_snapshot::<B, _>("gglwe_compressed_encrypt_sk", &out));
        result.push(source_snapshot("gglwe_compressed_sources", &mut e, &mut a));
        let g = GGSWLayout {
            n: key.n,
            base2k: key.base2k,
            dnum: key.dnum,
            dsize: key.dsize,
            k_aux: key.k_aux,
            rank: key.rank_out,
        };
        let genc = EncryptionLayout::new_from_default_sigma(g).unwrap();
        let pt = scalar(module, 1);
        let mut out = module.ggsw_alloc_from_infos(&g);
        poison_ggsw::<B, _>(&mut out);
        module.ggsw_encrypt_sk(
            &mut out,
            &pt,
            &skp,
            &genc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.ggsw_encrypt_sk_tmp_bytes(&g)).arena(),
        );
        result.push(ggsw_snapshot::<B, _>("ggsw_encrypt_sk", &out));
        result.push(source_snapshot("ggsw_encrypt_sk_sources", &mut e, &mut a));
        let mut compact = module.ggsw_compressed_alloc_from_infos(&g);
        poison_compressed_ggsw::<B, _>(&mut compact);
        module.ggsw_compressed_encrypt_sk(
            &mut compact,
            &pt,
            &skp,
            seed,
            &genc,
            &mut e,
            &mut poisoned_scratch::<B>(module.ggsw_compressed_encrypt_sk_tmp_bytes(&g)).arena(),
        );
        result.push(seeds_snapshot("ggsw_compressed_seeds", &compact.seed));
        module.decompress_ggsw(&mut out, &compact);
        result.push(ggsw_snapshot::<B, _>("ggsw_compressed_encrypt_sk", &out));
        result.push(source_snapshot("ggsw_compressed_sources", &mut e, &mut a));
        let mut switching = module.glwe_switching_key_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut switching);
        module.glwe_switching_key_encrypt_sk(
            &mut switching,
            &sk,
            &sk,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(gglwe_snapshot::<B, _>("glwe_switching_key_encrypt_sk", &switching));
        result.push(source_snapshot("switching_sources", &mut e, &mut a));
        result.push(Snapshot {
            label: "switching_degrees",
            metadata: vec![
                GLWESwitchingKeyDegrees::input_degree(&switching).as_usize(),
                GLWESwitchingKeyDegrees::output_degree(&switching).as_usize(),
            ],
            bytes: vec![],
        });
        let mut compact = module.glwe_switching_key_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_switching_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            &sk,
            seed,
            &enc,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(seeds_snapshot("switching_compressed_seeds", &compact.key.seed));
        module.decompress_glwe_switching_key(&mut switching, &compact);
        result.push(gglwe_snapshot::<B, _>("glwe_switching_key_compressed_encrypt_sk", &switching));
        result.push(source_snapshot("switching_compressed_sources", &mut e, &mut a));
        let mut auto = module.glwe_automorphism_key_alloc_from_infos(&key);
        poison_gglwe::<B, _>(&mut auto);
        module.glwe_automorphism_key_encrypt_sk(
            &mut auto,
            -5,
            &sk,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        assert_eq!(auto.p, -5);
        result.push(gglwe_snapshot::<B, _>("glwe_automorphism_key_encrypt_sk", &auto));
        result.push(source_snapshot("automorphism_sources", &mut e, &mut a));
        let mut compact = module.glwe_automorphism_key_compressed_alloc_from_infos(&key);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_automorphism_key_compressed_encrypt_sk(
            &mut compact,
            -5,
            &sk,
            seed,
            &enc,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        result.push(seeds_snapshot("automorphism_compressed_seeds", &compact.key.seed));
        module.decompress_automorphism_key(&mut auto, &compact);
        assert_eq!(auto.p, -5);
        result.push(gglwe_snapshot::<B, _>("glwe_automorphism_key_compressed_encrypt_sk", &auto));
        result.push(source_snapshot("automorphism_compressed_sources", &mut e, &mut a));
        let tk = GLWETensorKeyLayout {
            n: key.n,
            base2k: key.base2k,
            dnum: key.dnum,
            dsize: key.dsize,
            k_aux: key.k_aux,
            rank: key.rank_out,
        };
        let tenc = EncryptionLayout::new_from_default_sigma(tk).unwrap();
        let mut tensor = module.glwe_tensor_key_alloc_from_infos(&tk);
        poison_gglwe::<B, _>(&mut tensor);
        module.glwe_tensor_key_encrypt_sk(
            &mut tensor,
            &sk,
            &tenc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tk)).arena(),
        );
        result.push(gglwe_snapshot::<B, _>("glwe_tensor_key_encrypt_sk", &tensor));
        result.push(source_snapshot("tensor_sources", &mut e, &mut a));
        let mut compact = module.glwe_tensor_key_compressed_alloc_from_infos(&tk);
        poison_compressed::<B, _>(&mut compact);
        module.glwe_tensor_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            seed,
            &tenc,
            &mut e,
            &mut poisoned_scratch::<B>(module.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(&tk)).arena(),
        );
        result.push(seeds_snapshot("tensor_compressed_seeds", &compact.0.seed));
        module.decompress_tensor_key(&mut tensor, &compact);
        result.push(gglwe_snapshot::<B, _>("glwe_tensor_key_compressed_encrypt_sk", &tensor));
        result.push(source_snapshot("tensor_compressed_sources", &mut e, &mut a));
        let mut rows = module.gglwe_to_ggsw_key_alloc_from_infos(&key);
        for row in &mut rows.keys {
            poison_gglwe::<B, _>(row);
        }
        module.gglwe_to_ggsw_key_encrypt_sk(
            &mut rows,
            &sk,
            &enc,
            &mut e,
            &mut a,
            &mut poisoned_scratch::<B>(module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        for row in &rows.keys {
            result.push(gglwe_snapshot::<B, _>("gglwe_to_ggsw_key_encrypt_sk", row));
        }
        result.push(source_snapshot("row_key_sources", &mut e, &mut a));
        let mut compact = module.gglwe_to_ggsw_key_compressed_alloc_from_infos(&key);
        for row in &mut compact.keys {
            poison_compressed::<B, _>(row);
        }
        module.gglwe_to_ggsw_key_compressed_encrypt_sk(
            &mut compact,
            &sk,
            seed,
            &enc,
            &mut e,
            &mut poisoned_scratch::<B>(module.gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes(&key)).arena(),
        );
        for row in &compact.keys {
            result.push(seeds_snapshot("row_key_compressed_seeds", &row.seed));
        }
        module.decompress_gglwe_to_ggsw_key(&mut rows, &compact);
        for row in &rows.keys {
            result.push(gglwe_snapshot::<B, _>("gglwe_to_ggsw_key_compressed_encrypt_sk", row));
        }
        result.push(source_snapshot("row_key_compressed_sources", &mut e, &mut a));
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
                assert_eq!(
                    GLWESwitchingKeyDegrees::input_degree(&out),
                    GLWESwitchingKeyDegrees::input_degree(&compact)
                );
                assert_eq!(
                    GLWESwitchingKeyDegrees::output_degree(&out),
                    GLWESwitchingKeyDegrees::output_degree(&compact)
                );
                assert_eq!(compact.0.key.seed, seeds, "decompression changed seeds");
                result.push(gglwe_snapshot::<B, _>(stringify!($decompress), &out));
                result.push(seeds_snapshot(stringify!($decompress), &seeds));
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
            let enc = EncryptionLayout::new_from_default_sigma(kl).unwrap();
            let mut out = module.glwe_to_lwe_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.glwe_to_lwe_key_encrypt_sk(
                &mut out,
                &lwe,
                &sk,
                &enc,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(gglwe_snapshot::<B, _>("glwe_to_lwe_key_encrypt_sk", &out));
            result.push(source_snapshot("glwe_to_lwe_sources", &mut e, &mut a));
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
            let enc = EncryptionLayout::new_from_default_sigma(kl).unwrap();
            let mut out = module.lwe_to_glwe_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.lwe_to_glwe_key_encrypt_sk(
                &mut out,
                &lwe,
                &skp,
                &enc,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.lwe_to_glwe_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(gglwe_snapshot::<B, _>("lwe_to_glwe_key_encrypt_sk", &out));
            result.push(source_snapshot("lwe_to_glwe_sources", &mut e, &mut a));
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
            let enc = EncryptionLayout::new_from_default_sigma(kl).unwrap();
            let mut out = module.lwe_switching_key_alloc_from_infos(&kl);
            poison_gglwe::<B, _>(&mut out);
            module.lwe_switching_key_encrypt_sk(
                &mut out,
                &lwe,
                &lwe,
                &enc,
                &mut e,
                &mut a,
                &mut poisoned_scratch::<B>(module.lwe_switching_key_encrypt_sk_tmp_bytes(&kl)).arena(),
            );
            result.push(gglwe_snapshot::<B, _>("lwe_switching_key_encrypt_sk", &out));
            result.push(source_snapshot("lwe_switching_sources", &mut e, &mut a));
        }
        result
    }
    let b = params.base2k.min(12);
    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(2 * b, b) {
            assert_eq!(
                run(r, b, rank, dsize),
                run(t, b, rank, dsize),
                "key encryption rank={rank} dsize={dsize}"
            );
        }
    }
}
