//! Frozen fixtures for CKKS encoding and setup.
//!
//! Every backend must produce the same bytes for the same inputs, on every
//! platform. These tests hash the float coefficients of the slot transforms,
//! the quantized plaintexts, the decoded slots and the setup constants, and
//! compare the hashes with `determinism.txt`, whose keys do not depend on the
//! backend. A backend that encodes differently from the others, or a platform
//! whose math differs, fails here.
//!
//! Running the tests with `POULPY_UPDATE_FIXTURES=1` records the hashes they
//! compute in place of checking them. Regenerate only after an intended change
//! to the encoding or the setup math, on one backend, then run every backend
//! without the variable to check that they agree, and review the diff.

use crate::Scale;
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{Read, Seek, Write},
};

use poulpy_core::layouts::{Base2K, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, Ring, ScratchOwned, Standard, ZnxView},
};

use crate::{
    CKKSModuleInfos, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps},
    approximation::{Parity, RemezOptions, minimax, sign_composite_coeffs},
    layouts::{
        CKKSEncodingBuffer, CKKSModuleAlloc, CKKSPlaintextOwned, DFTOutputFormat, DFTPlan, DFTType,
        eval_mod::{EvalModPlan, EvalModPoly, EvalModType, compile_eval_mod},
        slot_coeff_count,
    },
    polynomial::SplitStrategy,
    reference::gen_dft_matrices,
    test_suite::{CKKSTestParams, helpers::TestContextBackend, helpers::TestScalar},
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/test_suite/determinism.txt");

fn hash(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Checks the hash of `bytes` against the fixture `key`, or records it when
/// `POULPY_UPDATE_FIXTURES` is set.
pub fn check_fixture(key: &str, bytes: impl IntoIterator<Item = u8>) {
    let got = hash(bytes);
    if std::env::var_os("POULPY_UPDATE_FIXTURES").is_some() {
        record_fixture(key, got);
        return;
    }
    let expected = include_str!("determinism.txt").lines().find_map(|line| {
        let (name, value) = line.split_once(' ')?;
        (name == key).then(|| u64::from_str_radix(value, 16).expect("fixture hashes are hexadecimal"))
    });
    match expected {
        Some(expected) => assert_eq!(
            format!("{got:016x}"),
            format!("{expected:016x}"),
            "fixture {key} differs, the encoding or the setup math is not reproduced"
        ),
        None => panic!("missing fixture {key} ({got:016x}), record it with POULPY_UPDATE_FIXTURES=1"),
    }
}

/// Inserts or replaces one fixture, holding a file lock so that concurrent
/// tests and backends merge their entries.
fn record_fixture(key: &str, value: u64) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(FIXTURES)
        .expect("fixture file is writable");
    file.lock().expect("fixture file lock");
    let mut text = String::new();
    file.read_to_string(&mut text).expect("fixture file is UTF-8");
    let mut entries: BTreeMap<String, String> = text
        .lines()
        .filter_map(|line| line.split_once(' ').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    entries.insert(key.to_string(), format!("{value:016x}"));
    let text: String = entries.iter().map(|(k, v)| format!("{k} {v}\n")).collect();
    file.set_len(0).expect("fixture file truncation");
    file.rewind().expect("fixture file rewind");
    file.write_all(text.as_bytes()).expect("fixture file write");
    file.unlock().expect("fixture file unlock");
}

fn scalar_bytes<F: TestScalar>(values: &[F]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let mut bytes = bytemuck::bytes_of(value).to_vec();
            if cfg!(target_endian = "big") {
                bytes.reverse();
            }
            bytes
        })
        .collect()
}

fn check_scalars<F: TestScalar>(key: &str, values: &[F]) {
    check_fixture(key, scalar_bytes(values));
}

fn check_plaintext<BE: Backend<ZnxWord = i64>>(key: &str, pt: &CKKSPlaintextOwned<BE>) {
    let host = pt.to_host_owned::<BE>();
    let mut bytes = Vec::new();
    for limb in 0..host.size() {
        for value in host.data().at(0, limb) {
            bytes.extend(value.to_le_bytes());
        }
    }
    check_fixture(key, bytes);
}

/// Dyadic test values with 20 significant bits, exact in every precision.
fn dyadic<F: TestScalar>(len: usize, seed: u64) -> Vec<F> {
    (0..len)
        .map(|i| {
            let bits = (i as u64 ^ seed).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23) >> 44;
            F::from_i64(bits as i64 - (1 << 19)).unwrap() / F::from_i64(1 << 19).unwrap()
        })
        .collect()
}

fn precision<F>() -> usize {
    8 * size_of::<F>()
}

/// Encoding fixtures of `module`, keyed by ring and precision only: the slot
/// transforms, then encoding, decoding and dequantization at three scales,
/// for dense and sparse slot counts.
pub fn encoding_fixtures<BE, F>(module: &Module<BE>)
where
    BE: Backend<ZnxWord = i64>,
    F: TestScalar,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
{
    let ring = if BE::Ring::CYCLOTOMIC_ORDER_FACTOR == Standard::CYCLOTOMIC_ORDER_FACTOR {
        "standard"
    } else {
        "ci"
    };
    let bits = precision::<F>();
    let max_slots = module.ckks_max_slots();
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(max_slots));
    let mut log_slots_set = vec![0, 1, 3, 6];
    log_slots_set.push(max_slots.ilog2() as usize);
    for log_slots in log_slots_set {
        let slots = 1usize << log_slots;
        let key = format!("encoding-{ring}-f{bits}-n{}-slots{slots}", module.n());
        let re = dyadic::<F>(slots, 1);
        let im = dyadic::<F>(slots, 2);
        // The invariant ring leaves workspace past its coefficients.
        let coeffs = slot_coeff_count(module, 2 * slots);

        let mut values = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&[re.as_slice(), im.as_slice()].concat());
        module.ckks_slots_to_coeffs_assign(&mut values).unwrap();
        check_scalars(&format!("{key}-coeffs"), &values.to_host::<BE>()[..coeffs]);
        module.ckks_coeffs_to_slots_assign(&mut values).unwrap();
        check_scalars(&format!("{key}-slots"), &values.to_host::<BE>());

        for log_delta in [20, 40, 100] {
            let mut pt = module.ckks_pt_vec_alloc(Base2K(19), TorusPrecision(log_delta as u32 + 20));
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            });
            module
                .ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch.borrow())
                .unwrap();
            check_plaintext::<BE>(&format!("{key}-delta{log_delta}-plaintext"), &pt);

            let mut got_re = vec![F::zero(); slots];
            let mut got_im = vec![F::zero(); slots];
            module
                .ckks_decode_reim_into(&pt, &mut got_re, &mut got_im, &mut scratch.borrow())
                .unwrap();
            check_scalars(&format!("{key}-delta{log_delta}-decoded"), &[got_re, got_im].concat());

            let mut coefficients = vec![F::zero(); coeffs];
            module
                .ckks_decode_coeffs_host_into(&pt, &mut coefficients, &mut scratch.borrow())
                .unwrap();
            check_scalars(&format!("{key}-delta{log_delta}-dequantized"), &coefficients);
        }
    }
}

/// Setup fixtures at precision `F`: the platform-independent math, the DFT
/// matrices, the sign and minimax approximations, and the EvalMod polynomials
/// with their encoded plaintexts.
pub fn setup_fixtures<BE, F>(module: &Module<BE>)
where
    BE: Backend<ZnxWord = i64>,
    F: TestScalar,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
{
    let bits = precision::<F>();

    let mut math = Vec::new();
    for i in 1..=31 {
        let x = F::from_i32(i).unwrap() / F::from_i32(17).unwrap();
        let (cos, sin) = F::ckks_root_of_unity(i as u64, 11);
        math.extend([
            x.ckks_sin(),
            x.ckks_cos(),
            x.ckks_sqrt(),
            x.ckks_exp2(),
            x.ckks_log2(),
            x.ckks_powf(F::one() / F::from_i32(7).unwrap()),
            x.ckks_powi(-5),
            cos,
            sin,
        ]);
    }
    check_scalars(&format!("setup-f{bits}-math"), &math);

    for kind in [DFTType::Encode, DFTType::Decode] {
        let plan = DFTPlan::new(
            kind,
            vec![(3, 4), (3, 4)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(100, 10),
        )
        .unwrap()
        .with_scaling(0.37)
        .unwrap();
        let mut scalars = Vec::new();
        for factor in gen_dft_matrices::<F>(&plan, 7) {
            for index in factor.indexes() {
                for part in [&factor.re, &factor.im] {
                    if let Some(values) = part.get(index) {
                        scalars.extend_from_slice(values);
                    }
                }
            }
        }
        check_scalars(&format!("setup-f{bits}-dft-{kind:?}"), &scalars);
    }

    let sign = sign_composite_coeffs(F::from_f64(0.1).unwrap(), 15.0, &[15], 8, RemezOptions::default()).unwrap();
    check_scalars(&format!("setup-f{bits}-sign"), &sign.concat());
    let fit = minimax(
        |x: F| (x * F::from_i32(3).unwrap()).ckks_sin(),
        -F::one(),
        F::one(),
        21,
        Parity::Odd,
    )
    .unwrap();
    check_scalars(&format!("setup-f{bits}-minimax"), &fit.poly.coeffs);

    let mut scratch = ScratchOwned::<BE>::alloc(1 << 22);
    for kind in [
        EvalModType::SinCheby,
        EvalModType::CosCheby,
        EvalModType::CosHK,
        EvalModType::CosHKEven,
        EvalModType::ExpCmplx,
    ] {
        let plan = EvalModPlan {
            eval_mod_type: kind,
            log_msg_ratio: 6,
            f_mod_degree: 30,
            f_mod_interval: 4,
            f_mod_log_interval_reduction: if kind == EvalModType::SinCheby { 0 } else { 2 },
            f_mod_inv_degree: None,
            scaling: Some(0.37),
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(100, 10),
            f_mod_log_delta: 110,
        };
        let compiled = compile_eval_mod::<BE, F>(Base2K(20), plan, module, &mut scratch.borrow()).unwrap();
        let scalars = match &compiled.f_mod_poly {
            EvalModPoly::Real(p) => p.coeffs.clone(),
            EvalModPoly::Complex(p) => p.re.iter().chain(&p.im).copied().collect(),
        };
        check_scalars(&format!("setup-f{bits}-evalmod-{kind:?}-polynomial"), &scalars);
        let mut bytes = Vec::new();
        compiled.map_plaintexts(|pt| {
            let host = pt.to_host_owned::<BE>();
            for limb in 0..host.size() {
                for value in host.data().at(0, limb) {
                    bytes.extend(value.to_le_bytes());
                }
            }
        });
        check_fixture(&format!("setup-f{bits}-evalmod-{kind:?}-plaintexts"), bytes);
    }
}

/// Suite entry point for standard-ring backends: encoding fixtures at degree
/// 4096 and setup fixtures.
pub fn test_encoding_determinism<BE, F, E>(_: CKKSTestParams, _: &Module<BE>, _: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    F: TestScalar,
    Module<BE>: ModuleNew<BE> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
{
    let module = Module::<BE>::new(4096);
    encoding_fixtures::<BE, F>(&module);
    setup_fixtures::<BE, F>(&module);
}
