//! Times the ciphertext → P(ciphertext) BSGS evaluation pipeline across a
//! degree sweep, comparing the two split strategies (`MinDepth`, `MinMult`).
//! Polynomial: dense random monomial coefficients (parity = Full). Per
//! (degree, strategy), prints the baby-step size and the level consumption
//! (log_budget delta) measured from the actual run.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSMeta,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    leveled::api::{CKKSAllOpsTmpBytes, CKKSCopyOps, PolynomialEvaluation},
    polynomial::{Basis, ComplexPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
    power_basis::{PowerBasis, PowerBasisGen},
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
const CT_K: usize = 400;
const LOG_DELTA: usize = 30;
const DSIZE: usize = 1;
const DEGREES: &[usize] = &[7, 15, 31, 63, 127];
const STRATEGIES: &[(SplitStrategy, &str)] = &[(SplitStrategy::MinDepth, "min-depth"), (SplitStrategy::MinMult, "min-mult")];

const COEFF_META: CKKSLayout = CKKSLayout {
    glwe_layout: GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision((LOG_DELTA + 1) as u32),
        rank: Rank(1),
    },
    meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: LOG_DELTA,
    },
};

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

fn random_coeffs(degree: usize) -> Vec<f64> {
    let mut state = (0x9e37_79b9_7f4a_7c15_u64).wrapping_add(degree as u64);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as i64) as f64 / (1u64 << 31) as f64 * 0.5
    };
    (0..=degree).map(|_| next()).collect()
}

macro_rules! poly_eval_bench {
    ($fn:ident, $be:ty, $label:expr) => {
fn $fn(c: &mut Criterion) {
    type BE = $be;
    let label = $label;
    let module = Module::<BE>::new(N as u64);
    let host_module = Module::<HostBytesBackend>::new(N as u64);
    let glwe_layout = glwe_layout();
    let tsk_layout = tsk_layout();
    let input_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta: LOG_DELTA,
    };

    let ct_template = module.ckks_ciphertext_alloc_from_infos(&glwe_layout);
    // The all-ops aggregate already includes the giant-step engine's scratch.
    let scratch_bytes = module
        .ckks_all_ops_tmp_bytes(&ct_template, &tsk_layout, &COEFF_META)
        .max(module.ckks_copy_tmp_bytes());
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);

    let mut ct_x = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
    ct_x.set_meta_checked(input_meta).unwrap();

    let mut group = c.benchmark_group(format!("ckks_poly_eval::{label}"));
    for &degree in DEGREES {
        let coeffs = random_coeffs(degree);
        let poly = Polynomial::new(Basis::Monomial, coeffs);
        let parity = poly.parity;

        for &(strategy, strategy_label) in STRATEGIES {
            let bsgs = poly
                .encode_bsgs_with(&host_module, Base2K(BASE2K as u32), COEFF_META, strategy)
                .expect("encode_bsgs_with");
            let log_split = bsgs.base().trailing_zeros() as usize;

            // Untimed dry run for level / log_budget reporting.
            let (levels, log_budget_in, log_budget_out) = {
                let mut ct_x_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module.ckks_copy(&mut ct_x_run, &ct_x, &mut sc).unwrap();
                }
                let lb_in = ct_x_run.log_budget();
                let mut pb = PowerBasis::new(Basis::Monomial, ct_x_run);
                {
                    let mut sc = scratch.borrow();
                    pb.populate(degree, log_split, parity, &module, &tsk_prepared, &mut sc)
                        .unwrap();
                }
                let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module
                        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
                            &mut ct_res,
                            &bsgs,
                            &pb,
                            &tsk_prepared,
                            &mut sc,
                        )
                        .unwrap();
                }
                let lb_out = ct_res.log_budget();
                ((lb_in - lb_out) / LOG_DELTA, lb_in, lb_out)
            };
            eprintln!(
                "[poly_eval/{label} d={degree:3} {strategy_label:>9}] k={k:2} baby_steps={n_baby:2} L={levels} ({lb_in}→{lb_out} budget bits)",
                k = bsgs.base(),
                n_baby = bsgs.baby_steps().len(),
                lb_in = log_budget_in,
                lb_out = log_budget_out,
            );

            group.bench_function(format!("{strategy_label}/d{degree}"), |b| {
                b.iter(|| {
                    let mut ct_x_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                    {
                        let mut sc = scratch.borrow();
                        module.ckks_copy(&mut ct_x_run, &ct_x, &mut sc).unwrap();
                    }
                    let mut pb = PowerBasis::new(Basis::Monomial, ct_x_run);
                    {
                        let mut sc = scratch.borrow();
                        pb.populate(degree, log_split, parity, &module, &tsk_prepared, &mut sc)
                            .unwrap();
                    }
                    let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                    {
                        let mut sc = scratch.borrow();
                        module
                            .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
                                black_box(&mut ct_res),
                                black_box(&bsgs),
                                black_box(&pb),
                                &tsk_prepared,
                                &mut sc,
                            )
                            .unwrap();
                    }
                });
            });
        }

        // Complex-coefficient evaluation (dense random re+im, Full parity), MinDepth.
        let im_coeffs: Vec<f64> = random_coeffs(degree).iter().rev().map(|c| c * 0.5 - 0.05).collect();
        let complex_poly = ComplexPolynomial::new(Basis::Monomial, random_coeffs(degree), im_coeffs);
        let complex_bsgs = complex_poly
            .encode_bsgs_with(&host_module, Base2K(BASE2K as u32), COEFF_META, SplitStrategy::MinDepth)
            .expect("complex encode_bsgs_with");
        let complex_log_split = complex_bsgs.re.base().trailing_zeros() as usize;

        let (clevels, clb_in, clb_out) = {
            let mut ct_x_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
            {
                let mut sc = scratch.borrow();
                module.ckks_copy(&mut ct_x_run, &ct_x, &mut sc).unwrap();
            }
            let lb_in = ct_x_run.log_budget();
            let mut pb = PowerBasis::new(Basis::Monomial, ct_x_run);
            {
                let mut sc = scratch.borrow();
                pb.populate(degree, complex_log_split, parity, &module, &tsk_prepared, &mut sc)
                    .unwrap();
            }
            let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
            {
                let mut sc = scratch.borrow();
                module
                    .ckks_eval_poly_complex_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
                        &mut ct_res,
                        &complex_bsgs,
                        &pb,
                        &tsk_prepared,
                        &mut sc,
                    )
                    .unwrap();
            }
            let lb_out = ct_res.log_budget();
            ((lb_in - lb_out) / LOG_DELTA, lb_in, lb_out)
        };
        eprintln!(
            "[poly_eval/{label} d={degree:3}   complex] k={k:2} baby_steps={n_baby:2} L={clevels} ({clb_in}→{clb_out} budget bits)",
            k = complex_bsgs.re.base(),
            n_baby = complex_bsgs.re.baby_steps().len(),
        );

        group.bench_function(format!("complex/d{degree}"), |b| {
            b.iter(|| {
                let mut ct_x_run = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module.ckks_copy(&mut ct_x_run, &ct_x, &mut sc).unwrap();
                }
                let mut pb = PowerBasis::new(Basis::Monomial, ct_x_run);
                {
                    let mut sc = scratch.borrow();
                    pb.populate(degree, complex_log_split, parity, &module, &tsk_prepared, &mut sc)
                        .unwrap();
                }
                let mut ct_res = module.ckks_ciphertext_alloc(Base2K(BASE2K as u32), TorusPrecision(CT_K as u32));
                {
                    let mut sc = scratch.borrow();
                    module
                        .ckks_eval_poly_complex_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
                            black_box(&mut ct_res),
                            black_box(&complex_bsgs),
                            &pb,
                            &tsk_prepared,
                            &mut sc,
                        )
                        .unwrap();
                }
            });
        });
    }
    group.finish();
}
    };
}

poly_eval_bench!(bench_ntt4x30_ref, poulpy_cpu_ref::NTT4x30Ref, "ntt4x30-ref");
#[cfg(feature = "enable-avx")]
poly_eval_bench!(bench_ntt4x30_avx, poulpy_cpu_avx::NTT4x30Avx, "ntt4x30-avx");
#[cfg(feature = "enable-ifma")]
poly_eval_bench!(bench_ntt_ifma, poulpy_cpu_avx512::NTT3x42Ifma, "ntt-ifma");

fn bench_ckks_poly_eval(c: &mut Criterion) {
    bench_ntt4x30_ref(c);
    #[cfg(feature = "enable-avx")]
    bench_ntt4x30_avx(c);
    #[cfg(feature = "enable-ifma")]
    bench_ntt_ifma(c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::ckks_criterion_config();
    targets = bench_ckks_poly_eval
}
criterion_main!(benches);
