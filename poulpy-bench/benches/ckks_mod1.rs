//! Times the homomorphic `x mod 1` evaluation across the three Mod1Type
//! variants. Per case, prints the level consumption (log_budget delta)
//! and the number of CT-CT mul rounds (BSGS depth + r double-angle rounds
//! + arcsine depth).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_ckks::{
    CKKSInfos, CKKSMeta,
    api::CKKSMod1Ops,
    default::mod1::{Mod1Parameters, Mod1ParametersLiteral, Mod1Type},
    layouts::CKKSModuleAlloc,
    leveled::api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

const N: usize = 4096;
const BASE2K: usize = 52;
const CT_K: usize = 520;
const LOG_DELTA: usize = 30;
const DSIZE: usize = 1;

const COEFF_META: CKKSMeta = CKKSMeta {
    log_delta: LOG_DELTA,
    log_budget: BASE2K,
};

struct Case {
    label: &'static str,
    lit: Mod1ParametersLiteral,
}

const CASES: &[Case] = &[
    Case {
        label: "sin_continuous/d15",
        lit: Mod1ParametersLiteral {
            mod1_type: Mod1Type::SinContinuous,
            log_message_ratio: 4,
            mod1_degree: 15,
            mod1_interval: 1,
            double_angle: 0,
            mod1_inv_degree: 0,
            scaling: 1.0,
        },
    },
    Case {
        label: "sin_continuous_arcsine/d31_inv7",
        lit: Mod1ParametersLiteral {
            mod1_type: Mod1Type::SinContinuous,
            log_message_ratio: 4,
            mod1_degree: 31,
            mod1_interval: 1,
            double_angle: 0,
            mod1_inv_degree: 7,
            scaling: 1.0,
        },
    },
    Case {
        label: "cos_discrete/d30_K12_r3",
        lit: Mod1ParametersLiteral {
            mod1_type: Mod1Type::CosDiscrete,
            log_message_ratio: 8,
            mod1_degree: 30,
            mod1_interval: 12,
            double_angle: 3,
            mod1_inv_degree: 0,
            scaling: 1.0,
        },
    },
    Case {
        label: "cos_continuous/d31_K4_r2",
        lit: Mod1ParametersLiteral {
            mod1_type: Mod1Type::CosContinuous,
            log_message_ratio: 4,
            mod1_degree: 31,
            mod1_interval: 4,
            double_angle: 2,
            mod1_inv_degree: 0,
            scaling: 1.0,
        },
    },
];

fn glwe_layout() -> GLWELayout {
    GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(CT_K as u32),
        rank: Rank(1),
    }
}

fn tsk_layout() -> GLWETensorKeyLayout {
    let k = CT_K + DSIZE * BASE2K;
    GLWETensorKeyLayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(k as u32),
        rank: Rank(1),
        dsize: Dsize(DSIZE as u32),
        dnum: Dnum(k.div_ceil(DSIZE * BASE2K) as u32),
    }
}

fn depth(lit: &Mod1ParametersLiteral) -> usize {
    let d = lit.mod1_degree.next_power_of_two().trailing_zeros() as usize;
    let r = match lit.mod1_type {
        Mod1Type::SinContinuous => 0,
        _ => lit.double_angle,
    };
    let inv = if lit.mod1_inv_degree > 0 {
        lit.mod1_inv_degree.next_power_of_two().trailing_zeros() as usize
    } else {
        0
    };
    d + r + inv
}

fn bench_ntt120_ref(c: &mut Criterion) {
    type BE = poulpy_cpu_ref::NTT120Ref;
    let label = "ntt120-ref";

    let module = Module::<BE>::new(N as u64);
    let host_module = Module::<HostBytesBackend>::new(N as u64);
    let glwe_layout = glwe_layout();
    let tsk_layout = tsk_layout();
    let input_meta = CKKSMeta {
        log_delta: LOG_DELTA,
        log_budget: CT_K - LOG_DELTA,
    };

    let ct_template = module.ckks_ciphertext_alloc_from_infos(&glwe_layout);
    let mul_bytes = module.ckks_mul_tmp_bytes(&ct_template, &tsk_layout);
    let mul_pt_bytes = module.ckks_mul_pt_const_tmp_bytes(&ct_template, &ct_template, &COEFF_META);
    let add_bytes = module.ckks_add_tmp_bytes();
    let copy_bytes = module.ckks_copy_tmp_bytes();
    let ct_block = poulpy_core::layouts::GLWE::<Vec<u8>>::bytes_of_from_infos(&ct_template);
    let scratch_bytes = mul_bytes
        .max(mul_pt_bytes)
        .max(add_bytes)
        .max(copy_bytes)
        .max(mul_bytes + 3 * ct_block);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let mut ct_x = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
    ct_x.set_meta_checked(input_meta).unwrap();

    let mut group = c.benchmark_group(format!("ckks_mod1::{label}"));
    for case in CASES {
        let params = Mod1Parameters::from_literal(COEFF_META, Base2K(BASE2K as u32), case.lit, &host_module)
            .expect("Mod1Parameters::from_literal");

        let (levels, log_budget_in, log_budget_out) = {
            let mut ct_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
            {
                let mut sc = scratch.borrow();
                module.ckks_copy(&mut ct_run, &ct_x, &mut sc).unwrap();
            }
            let lb_in = ct_run.log_budget();
            let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
            {
                let mut sc = scratch.borrow();
                module
                    .ckks_eval_mod1(&mut ct_res, &ct_run, &params, &tsk_prepared, &mut sc)
                    .unwrap();
            }
            let lb_out = ct_res.log_budget();
            ((lb_in - lb_out) / LOG_DELTA, lb_in, lb_out)
        };
        eprintln!(
            "[mod1/{label} {case_label}] L={levels} depth_pred={depth} ({lb_in}→{lb_out} budget bits)",
            case_label = case.label,
            depth = depth(&case.lit),
            lb_in = log_budget_in,
            lb_out = log_budget_out,
        );

        group.bench_function(case.label, |b| {
            b.iter(|| {
                let mut ct_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module.ckks_copy(&mut ct_run, &ct_x, &mut sc).unwrap();
                }
                let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module
                        .ckks_eval_mod1(black_box(&mut ct_res), &ct_run, black_box(&params), &tsk_prepared, &mut sc)
                        .unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_ckks_mod1(c: &mut Criterion) {
    bench_ntt120_ref(c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::ckks_criterion_config();
    targets = bench_ckks_mod1
}
criterion_main!(benches);
