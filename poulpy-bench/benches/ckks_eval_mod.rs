//! Times the homomorphic `x mod 1` evaluation across the EvalModType variants,
//! including `CosHK` under both the `MinDepth` and `MinMult` BSGS split
//! strategies. Per case, prints the level consumption (log_budget delta) and
//! the number of CT-CT mul rounds (BSGS depth + r range-extension rounds +
//! inverse depth).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSMeta,
    api::CKKSEvalModOps,
    layouts::{
        CKKSModuleAlloc,
        eval_mod::{EvalMod, EvalModPlan, EvalModType},
    },
    leveled::api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
    polynomial::SplitStrategy,
};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

const N: usize = 4096;
const BASE2K: usize = 52;
const CT_K: usize = 520;
const LOG_DELTA: usize = 30;
const EVAL_MOD_LOG_DELTA: usize = 60;
const DSIZE: usize = 1;

const COEFF_META: CKKSLayout = CKKSLayout {
    glwe_layout: GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision((EVAL_MOD_LOG_DELTA + BASE2K) as u32),
        rank: Rank(1),
    },
    meta: CKKSMeta {
        log_delta: EVAL_MOD_LOG_DELTA,
        log_sparsity: 0,
    },
};

struct Case {
    label: &'static str,
    lit: EvalModPlan,
}

const CASES: &[Case] = &[
    Case {
        label: "sin_continuous/d15",
        lit: EvalModPlan {
            eval_mod_type: EvalModType::SinCheby,
            log_msg_ratio: 4,
            f_mod_degree: 15,
            f_mod_interval: 1,
            f_mod_log_interval_reduction: 0,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: COEFF_META,
            f_mod_log_delta: EVAL_MOD_LOG_DELTA,
        },
    },
    Case {
        label: "sin_continuous_arcsine/d31_inv7",
        lit: EvalModPlan {
            eval_mod_type: EvalModType::SinCheby,
            log_msg_ratio: 4,
            f_mod_degree: 31,
            f_mod_interval: 1,
            f_mod_log_interval_reduction: 0,
            f_mod_inv_degree: Some(7),
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: COEFF_META,
            f_mod_log_delta: EVAL_MOD_LOG_DELTA,
        },
    },
    Case {
        label: "cos_discrete/d30_K12_r3",
        lit: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: 8,
            f_mod_degree: 30,
            f_mod_interval: 12,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: COEFF_META,
            f_mod_log_delta: EVAL_MOD_LOG_DELTA,
        },
    },
    Case {
        label: "cos_discrete/d30_K12_r3_minmult",
        lit: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: 8,
            f_mod_degree: 30,
            f_mod_interval: 12,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinMult,
            coeffs_meta: COEFF_META,
            f_mod_log_delta: EVAL_MOD_LOG_DELTA,
        },
    },
    Case {
        label: "cos_continuous/d31_K4_r2",
        lit: EvalModPlan {
            eval_mod_type: EvalModType::CosCheby,
            log_msg_ratio: 4,
            f_mod_degree: 31,
            f_mod_interval: 4,
            f_mod_log_interval_reduction: 2,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: COEFF_META,
            f_mod_log_delta: EVAL_MOD_LOG_DELTA,
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

fn bench_ntt120_ref(c: &mut Criterion) {
    type BE = poulpy_cpu_ref::NTT120Ref;
    let label = "ntt120-ref";

    let module = Module::<BE>::new(N as u64);
    let host_module = Module::<HostBytesBackend>::new(N as u64);
    let glwe_layout = glwe_layout();
    let tsk_layout = tsk_layout();
    let input_meta = CKKSMeta {
        log_delta: LOG_DELTA,
        log_sparsity: 0,
    };

    let ct_template = module.ckks_ciphertext_alloc_from_infos(&glwe_layout);
    let mul_bytes = module.ckks_mul_tmp_bytes(&ct_template, &tsk_layout);
    let mul_pt_bytes = module.ckks_mul_pt_const_tmp_bytes(&ct_template, &ct_template, &COEFF_META);
    let add_bytes = module.ckks_add_tmp_bytes();
    let copy_bytes = module.ckks_copy_tmp_bytes();
    let ct_block = poulpy_core::layouts::GLWE::<Vec<u8>>::bytes_of_from_infos(&ct_template);
    // The giant step keeps the prepared `X^{gsp}` right operand alive across relinearization.
    let hoisted_right = module.bytes_of_cnv_pvec_right(2, CT_K.div_ceil(BASE2K));
    let scratch_bytes = mul_bytes
        .max(mul_pt_bytes)
        .max(add_bytes)
        .max(copy_bytes)
        .max(mul_bytes + 3 * ct_block + hoisted_right);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let mut ct_x = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
    ct_x.set_meta_checked(input_meta).unwrap();

    let mut group = c.benchmark_group(format!("ckks_eval_mod::{label}"));
    for case in CASES {
        let params =
            EvalMod::<f64, _>::from_literal(Base2K(BASE2K as u32), case.lit, &host_module).expect("EvalMod::from_literal");

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
                    .ckks_eval_mod(&mut ct_res, &ct_run, &params, &tsk_prepared, &mut sc)
                    .unwrap();
            }
            let lb_out = ct_res.log_budget();
            ((lb_in - lb_out) / LOG_DELTA, lb_in, lb_out)
        };
        eprintln!(
            "[eval_mod/{label} {case_label}] L={levels} depth_pred={depth} ({lb_in}→{lb_out} budget bits)",
            case_label = case.label,
            depth = params.eval_depth(),
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
                        .ckks_eval_mod(black_box(&mut ct_res), &ct_run, black_box(&params), &tsk_prepared, &mut sc)
                        .unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_ckks_eval_mod(c: &mut Criterion) {
    bench_ntt120_ref(c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::ckks_criterion_config();
    targets = bench_ckks_eval_mod
}
criterion_main!(benches);
