use poulpy_core::layouts::{Base2K, Degree, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ZnxView};
use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};

use super::{
    CKKSTestParams,
    helpers::{TestContextBackend, TestContextModule, TestScalar},
};
use crate::{
    CKKSMeta, SetCKKSInfos, SlotsKind,
    api::CKKSEncodingOps,
    layouts::{CKKSCiphertextOwned, CKKSEncodingBuffer, CKKSModuleAlloc},
};

fn hash(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes
        .into_iter()
        .fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}

pub(crate) fn check_fixture(key: &str, bytes: impl IntoIterator<Item = u8>) {
    let got = hash(bytes);
    let expected = include_str!("determinism.txt").lines().find_map(|line| {
        let (name, value) = line.split_once(' ')?;
        (name == key).then(|| u64::from_str_radix(value, 16).unwrap())
    });
    assert_eq!(
        got,
        expected.unwrap_or_else(|| panic!("missing fixture {key}: {got:016x}")),
        "{key}"
    );
}

pub(crate) fn bootstrap_snapshot<BE: Backend<ZnxWord = i64>, F>(stage: &str, ct: &CKKSCiphertextOwned<BE>) {
    let host = ct.to_host_owned::<BE>();
    let mut bytes = Vec::new();
    for col in 0..host.data().cols() {
        for limb in 0..host.size() {
            for value in host.data().at(col, limb) {
                bytes.extend(value.to_le_bytes());
            }
        }
    }
    check_fixture(&format!("bootstrap-{}-{}-{stage}", size_of::<F>(), host.base2k()), bytes);
}

fn check_scalars<F: TestScalar>(key: &str, values: &[F]) {
    check_fixture(
        key,
        values.iter().flat_map(|value| {
            let mut bytes = bytemuck::bytes_of(value).to_vec();
            if cfg!(target_endian = "big") {
                bytes.reverse();
            }
            bytes
        }),
    );
}

pub fn test_encoding_determinism<BE, F, E>(_: CKKSTestParams, _: &Module<BE>, _: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    E: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let module = Module::<BE>::new(8192);
    for n in [8usize, 16, 32, 64, 2048, 8192] {
        let input: Vec<F> = (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17);
                F::from_i64(x as i64 >> 12).unwrap() / F::from_u64(1 << 51).unwrap()
                    + F::from_u64(x >> 11).unwrap() / F::from_u128(1 << 110).unwrap()
            })
            .collect();
        let mut values = CKKSEncodingBuffer::from_host::<BE>(&input);
        module.ckks_slots_to_coeffs_assign(&mut values).unwrap();
        let coeffs = values.to_host::<BE>();
        let key = format!("encoding-{}-{n}", size_of::<F>());
        check_scalars(&format!("{key}-inverse"), &coeffs);
        module.ckks_coeffs_to_slots_assign(&mut values).unwrap();
        check_scalars(&format!("{key}-forward"), &values.to_host::<BE>());
        for delta in [30, 60, 110] {
            let mut pt = module.ckks_plaintext_alloc(Degree(8192), Base2K(20), TorusPrecision(delta + 10));
            pt.set_meta(CKKSMeta {
                log_sparsity: (8192 / n).ilog2() as usize,
                log_delta: delta as usize,
                slots: SlotsKind::Complex,
            });
            values.copy_from_host::<BE>(&coeffs);
            module.ckks_encode_coeffs_into(&mut pt, &values).unwrap();
            let host = pt.to_host_owned::<BE>();
            let mut bytes = Vec::new();
            for limb in 0..host.size() {
                for value in host.data().at(0, limb) {
                    bytes.extend(value.to_le_bytes());
                }
            }
            check_fixture(&format!("{key}-{delta}-plaintext"), bytes);
            module.ckks_decode_coeffs_into(&pt, &mut values).unwrap();
            check_scalars(&format!("{key}-{delta}-decoded"), &values.to_host::<BE>());
        }
    }
    setup_fixtures::<BE, F>(&module);
}

fn setup_fixtures<BE, F>(module: &Module<BE>)
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    use crate::layouts::eval_mod::{EvalModPlan, EvalModPoly, EvalModType, compile_eval_mod};
    use crate::polynomial::SplitStrategy;
    use crate::{
        CoeffsMeta,
        default::dft::gen_dft_matrices,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
    };

    let mut math = Vec::new();
    for i in 1..=31 {
        let x = F::from_i32(i).unwrap() / F::from_i32(17).unwrap();
        math.extend([
            x.ckks_sin(),
            x.ckks_cos(),
            x.ckks_sqrt(),
            x.ckks_exp2(),
            x.ckks_log2(),
            x.ckks_powf(F::one() / F::from_i32(7).unwrap()),
            x.ckks_powi(-5),
        ]);
    }
    check_scalars(&format!("math-{}", size_of::<F>()), &math);
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
                if let Some(values) = factor.re.get(index) {
                    scalars.extend_from_slice(values);
                }
                if let Some(values) = factor.im.get(index) {
                    scalars.extend_from_slice(values);
                }
            }
        }
        check_scalars(&format!("matrices-{}-{kind:?}", size_of::<F>()), &scalars);
    }
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 20);
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
        let compiled = compile_eval_mod::<BE, F>(Base2K(20), plan, module, &mut scratch.arena()).unwrap();
        let scalars = match &compiled.f_mod_poly {
            EvalModPoly::Real(p) => p.coeffs.clone(),
            EvalModPoly::Complex(p) => p.re.iter().chain(&p.im).copied().collect(),
        };
        check_scalars(&format!("evalmod-{}-{kind:?}-polynomial", size_of::<F>()), &scalars);
        let mut bytes = Vec::new();
        compiled.map_plaintexts(|pt| {
            let host = pt.to_host_owned::<BE>();
            for limb in 0..host.size() {
                for value in host.data().at(0, limb) {
                    bytes.extend(value.to_le_bytes());
                }
            }
        });
        check_fixture(&format!("evalmod-{}-{kind:?}-plaintexts", size_of::<F>()), bytes);
    }
}

pub fn assert_transform_matches<F, Oracle, Candidate>()
where
    F: TestScalar,
    Oracle: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
    Candidate: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
{
    for log_m in 0..=15 {
        let m = 1 << log_m;
        let oracle = Oracle::new(m);
        let portable = Candidate::new(m);
        for sample in 0..10 {
            let input: Vec<F> = (0..2 * m)
                .map(|i| {
                    let bits = (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17);
                    match sample {
                        0 => F::zero(),
                        1 => {
                            if i == m / 2 {
                                F::one()
                            } else {
                                F::zero()
                            }
                        }
                        2 => F::from_f64(if i & 1 == 0 { 1.0 } else { -1.0 }).unwrap(),
                        3 => {
                            if i & 1 == 0 {
                                F::zero()
                            } else {
                                -F::zero()
                            }
                        }
                        4 => F::min_positive_value() * F::epsilon() * F::from_usize(i % 7).unwrap(),
                        5 => F::max_value() / F::from_usize(4 * m).unwrap(),
                        _ => {
                            F::from_i64(bits as i64 >> 12).unwrap() * F::from_f64(1.0 / (1u64 << 51) as f64).unwrap()
                                + F::from_f64((i as f64 + 1.0) / (1u128 << (60 + sample)) as f64).unwrap()
                        }
                    }
                })
                .collect();
            for inverse in [false, true] {
                let mut got = input.clone();
                let mut expected = input.clone();
                if inverse {
                    portable.ifft(&mut got);
                    oracle.ifft(&mut expected);
                } else {
                    portable.fft(&mut got);
                    oracle.fft(&mut expected);
                }
                for (i, (got, expected)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        bytemuck::bytes_of(got),
                        bytemuck::bytes_of(expected),
                        "m={m} sample={sample} inverse={inverse} scalar={i}: {got:?} != {expected:?}"
                    );
                }
            }
        }
    }
}
