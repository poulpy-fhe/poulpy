use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion};
use poulpy_ckks::{
    CKKSMeta, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSEncodingOps, CKKSEncodingScalar},
    default::dft::gen_dft_matrices,
    layouts::{CKKSEncodingBuffer, CKKSModuleAlloc, DFTOutputFormat, DFTPlan, DFTType},
};
use poulpy_core::layouts::{Base2K, Degree, TorusPrecision};
use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module},
};

fn dense<F: CKKSEncodingScalar>(n: usize) -> Vec<F> {
    let mut state = 0x853c49e6748fea9bu64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            F::from_i64(state as i64 >> 12).unwrap() / F::from_u64(1 << 51).unwrap()
                + F::from_u64(state >> 11).unwrap() / F::from_u128(1 << 110).unwrap()
        })
        .collect()
}

fn diagonals<F: CKKSEncodingScalar>(n: usize) -> Vec<Vec<F>> {
    let log_slots = n.ilog2() as usize - 1;
    let mut schedule = vec![(3, 4); log_slots / 3];
    if !log_slots.is_multiple_of(3) {
        schedule.push((log_slots % 3, 4));
    }
    let mut inputs = Vec::new();
    for kind in [DFTType::Encode, DFTType::Decode] {
        let plan = DFTPlan::new(
            kind,
            schedule.clone(),
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(58, 10),
        )
        .unwrap();
        for factor in gen_dft_matrices::<F>(&plan, n.ilog2() as usize) {
            for index in factor.indexes() {
                let mut input = factor
                    .re
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| vec![F::zero(); factor.slots()]);
                input.extend(
                    factor
                        .im
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| vec![F::zero(); factor.slots()]),
                );
                inputs.push(input);
            }
        }
    }
    inputs
}

/// Cached encoding with fixed dense inputs and real bootstrap matrix diagonals.
pub fn bench_encoding<BE, F>(c: &mut Criterion)
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<BE>: ModuleNew<BE> + CKKSEncodingOps<BE, F>,
{
    let backend = std::any::type_name::<BE>().rsplit("::").next().unwrap();
    let mut group = c.benchmark_group(format!("ckks_encoding/{backend}/f{}", size_of::<F>() * 8));
    for n in [2048, 65536] {
        let module = Module::<BE>::new(n as u64);
        let dense = vec![dense::<F>(n)];
        let diagonals = diagonals::<F>(n);
        group.bench_function(BenchmarkId::new("first_encoding", format!("n{n}/delta58")), |b| {
            b.iter_batched(
                || {
                    let module = Module::<BE>::new(n as u64);
                    let mut pt = module.ckks_plaintext_alloc(Degree(n as u32), Base2K(20), TorusPrecision(68));
                    pt.set_meta(CKKSMeta {
                        log_sparsity: 0,
                        log_delta: 58,
                        slots: SlotsKind::Complex,
                    });
                    let values = CKKSEncodingBuffer::from_host::<BE>(&dense[0]);
                    (module, pt, values)
                },
                |(module, mut pt, mut values)| {
                    module.ckks_encode_slots_assign_into(&mut pt, &mut values).unwrap();
                    black_box((module, pt, values))
                },
                BatchSize::PerIteration,
            );
        });
        for delta in [30usize, 58, 110] {
            let mut pt = module.ckks_plaintext_alloc(Degree(n as u32), Base2K(20), TorusPrecision((delta + 10) as u32));
            pt.set_meta(CKKSMeta {
                log_sparsity: 0,
                log_delta: delta,
                slots: SlotsKind::Complex,
            });
            let mut values = CKKSEncodingBuffer::from_host::<BE>(&dense[0]);
            module.ckks_encode_slots_assign_into(&mut pt, &mut values).unwrap();
            for (name, inputs, transform) in [
                ("slots_dense", &dense, true),
                ("slots_diagonals", &diagonals, true),
                ("coefficients_dense", &dense, false),
            ] {
                group.bench_function(BenchmarkId::new(name, format!("n{n}/delta{delta}")), |b| {
                    let mut index = 0;
                    b.iter(|| {
                        values.copy_from_host::<BE>(black_box(&inputs[index]));
                        if transform {
                            module.ckks_encode_slots_assign_into(&mut pt, &mut values).unwrap();
                        } else {
                            module.ckks_encode_coeffs_into(&mut pt, &values).unwrap();
                        }
                        index = (index + 1) % inputs.len();
                        black_box(&pt);
                    });
                });
            }
        }
    }
    group.finish();
}
