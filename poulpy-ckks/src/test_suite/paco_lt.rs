//! Homomorphic evaluation tests for the PaCo factor chains.
//!
//! For both PaCo chains — the partial CoeffToSlot (the blockwise `2C`-point
//! Encode factorization, seqPaCo line 7) and the decomposed SlotToCoeff (the
//! `C/2`-point Decode factorization, line 16) — this encrypts a random
//! **dense** slot vector, applies the grouped factor chain homomorphically
//! through the standard [`CKKSLinearTransformationOps`] BSGS engine (factors
//! sparse-encoded as gap-mapped sub-ring plaintexts), and checks:
//!
//! 1. **Correctness** against the cleartext tiled-matrix model
//!    (`mul_vec_tiled`) via the GLWE noise of the result against an
//!    encoding of the expected slots (the same structural noise bound as the
//!    DFT suite).
//! 2. **Budget accounting**: the chain consumes exactly
//!    `num_factors × log_delta` bits.
//!
//! The input is deliberately *not* periodic: a sparse-encoded `m × m` factor
//! acts on the full `N/2`-slot vector as the tiled matrix, so a dense input
//! exercises the index arithmetic that a period-`m` input would mask.
//!
//! PaCo dimensions are derived from the suite's ring degree (`N = 256`):
//! `h = 4`, `C = 8` → `B = 16`, `k = 2`, `n = 2hC = 64`, with `C/2 = 4` for
//! the SlotToCoeff chain. Grouping radices `g ∈ {1, 2}` cover both the
//! one-layer-per-factor and merged schedules.

use std::collections::HashMap;

use anyhow::ensure;

use crate::SlotsKind;
use crate::{
    CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    default::paco::lt::{mul_vec_tiled, paco_c2s_factors, paco_stc_factors},
    encoding::paco::cpx::Cpx,
    layouts::ComplexDiagonals,
    layouts::PaCoPlan,
    layouts::{
        CKKSEncodingBuffer, CKKSModuleAlloc, ScratchArenaTakeCKKS, copy_encoding_buffer_into_reim_host,
        copy_host_into_encoding_buffer,
    },
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_scratch, ckks_encrypt, gen_atk,
            gen_sk_with_raw, test_vector_1,
        },
    },
};
use poulpy_core::{
    GLWENoise,
    layouts::{Base2K, LWEInfos, LinearTransformationStrategy, TorusPrecision},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module},
};

/// PaCo test dimensions on the suite's `N = 256` ring: `B = 16`, chunk count
/// `k = 2`, `n = 2hC = 64`, StC dimension `C/2 = 4`.
const PACO_H: usize = 4;
const PACO_C: usize = 8;

/// Structural noise bound, as in the DFT suite: a basis/permutation/scale error
/// puts the noise at signal level (`log2 ≈ 0`); a correct chain stays far below.
fn noise_bound(log_delta: usize) -> f64 {
    -(log_delta as f64) + 16.0
}

/// Test params re-sized for `num_factors` chained plaintext multiplies
/// (input scale + headroom), keeping the suite's ring degree and `base2k`.
fn chain_params(base: &CKKSTestParams, num_factors: usize) -> CKKSTestParams {
    let log_delta = base.prec().log_delta();
    CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k: (log_delta * (num_factors + 3)).next_multiple_of(base.base2k),
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 10,
        hw: base.hw.min(base.n / 2),
        dsize: base.dsize,
        rank: 1,
    }
}

/// Encrypts a dense random vector, applies the grouped factor chain
/// homomorphically, and checks correctness (vs `mul_vec_tiled`) and
/// budget consumption.
fn run_chain<BE, F, E>(base: &CKKSTestParams, factors: &[ComplexDiagonals<F>], label: &str)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    assert!(!factors.is_empty(), "{label}: empty chain");
    let params = chain_params(base, factors.len());
    let m_full = params.n / 2;
    let log_delta = params.prec().log_delta();

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<F>(m_full).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    // Factor plaintext layout: per-factor scale log_delta; sub-`N/2` factors
    // are gap-mapped sparse encodings (tiled slot values), tracked via
    // log_sparsity. Chains may mix dimensions (the fused StC first factor is
    // a 2C tile, the remaining Decode factors C/2), so the encoder and layout
    // are built per factor.
    let lts: Vec<_> = factors
        .iter()
        .map(|cd| {
            let m_dim = cd.slots();
            let factor_layout = CoeffsMeta {
                k: TorusPrecision((log_delta + 10) as u32),
                meta: CKKSMeta {
                    log_sparsity: (m_full / m_dim).trailing_zeros() as usize,
                    log_delta,
                    slots: SlotsKind::Complex,
                },
            };
            crate::default::ckks_encode_linear_transformation_from_diagonals(
                &module,
                Base2K(params.base2k as u32),
                factor_layout,
                cd,
                LinearTransformationStrategy::Bsgs { giant_step: 2 },
                false,
                &mut scratch.borrow(),
            )
            .unwrap()
        })
        .collect();

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for lt in &lts {
        for p_el in lt.galois_elements(order) {
            atks.entry(p_el)
                .or_insert_with(|| gen_atk(&params, &module, p_el, &sk_raw, &mut scratch.borrow()));
        }
    }

    // Dense (non-periodic) input.
    let (a_re, a_im) = test_vector_1::<F>(m_full);
    let mut ct = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder_full,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let budget_before = ct.log_budget();

    for lt in &lts {
        module
            .ckks_eval_linear_transformation_self_assign(&mut ct, lt, &atks, &mut scratch.borrow())
            .unwrap();
    }

    assert_eq!(
        budget_before - ct.log_budget(),
        factors.len() * log_delta,
        "{label}: chain must consume num_factors × log_delta budget bits"
    );

    // Cleartext oracle: the tiled-matrix chain on the full slot vector, at
    // the working precision F end to end.
    let mut w: Vec<Cpx<F>> = a_re.iter().zip(&a_im).map(|(&r, &i)| Cpx::new(r, i)).collect();
    for f in factors {
        w = mul_vec_tiled(f, &w);
    }
    let want_re: Vec<F> = w.iter().map(|x| x.re).collect();
    let want_im: Vec<F> = w.iter().map(|x| x.im).collect();

    let mut pt_want = module.ckks_pt_vec_alloc(ct.base2k(), ct.k());
    pt_want.set_meta(CKKSMeta {
        log_sparsity: ct.log_sparsity(),
        log_delta: ct.log_delta(),
        slots: SlotsKind::Complex,
    });
    encoder_full.encode_reim(&mut pt_want, &want_re, &want_im).unwrap();
    let noise = module.glwe_noise(&ct, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(noise < bound, "{label}: noise log2={noise:.1} (bound {bound:.1})");
}

/// Partial CoeffToSlot chain (`log C + 1` blockwise Encode layers on
/// `n = 2hC` slots), grouped at radices 1 and 2.
pub fn test_paco_partial_c2s<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let base = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, 29).unwrap();
    // Butterfly schedules over the log 2C = 4 layers, in both slot-order
    // conventions (BitRevLow exercises the conjugated diagonal offsets and
    // their Galois elements through the same self-consistent gate).
    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        let p = base.clone().with_slot_order(slot_order);
        for schedule in [vec![1usize, 1, 1, 1], vec![2, 2]] {
            run_chain::<BE, F, E>(
                &params,
                &paco_c2s_factors::<F>(&p, &schedule),
                &format!("paco_partial_c2s({schedule:?},{slot_order:?})"),
            );
        }
    }
}

/// Decomposed SlotToCoeff chain (`log C − 1` Decode layers on
/// `C/2` slots), grouped at radices 1 and 2.
pub fn test_paco_stc<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let base = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, 29).unwrap();
    // Schedules over the stc unit list [pack, L1, L2]: pack composed with the
    // first layer, standalone, and fully merged — in both slot-order
    // conventions (BitRevLow uses the raw, unfolded Decode factors).
    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        for schedule in [vec![2usize, 1], vec![1, 1, 1], vec![3]] {
            let p = base
                .clone()
                .with_slot_order(slot_order)
                .with_evaluation(
                    0,
                    0,
                    base.c2s().clone(),
                    crate::layouts::PaCoDFTPlan::new(schedule.clone(), vec![2; schedule.len()], 0, 0, 1.0).unwrap(),
                )
                .unwrap();
            run_chain::<BE, F, E>(
                &params,
                &paco_stc_factors::<F>(&p),
                &format!("paco_stc({schedule:?},{slot_order:?})"),
            );
        }
    }
}

/// The packing basis and its homomorphic inverse, gated on the backend's
/// FFT: (1) `pack_chunk` (the encoder's FFT + bit-reversal gather) matches
/// the naive Vandermonde formula `packed[v·2C+k] = Σ_j c_j·x_k^j` with
/// `x_k = ζ^{5^{bitrev(k)}}` (every point a root of `Z^{2C} = i`); (2) the
/// generator-built `paco_c2s_factors` chain inverts the packing per block in
/// natural order, at every grouping radix, applied tiled — the cross-artifact
/// pin between the encoder (packing) and the DFT generator (homomorphic
/// inverse).
pub fn test_paco_packing<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    use crate::default::paco::ops::ext_bitrev_low;
    use crate::layouts::paco::secret::pack_chunk;

    let base = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, 29).unwrap();
    let (two_c, n) = (2 * base.c(), base.slots());
    let log_two_c = two_c.trailing_zeros();
    let log_p = base.log_c() - 1;
    let mut scratch = alloc_scratch(&params, module);

    let dense: Vec<Cpx> = (0..n)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(6364136223846793005)) as f64 / u64::MAX as f64;
            let y = ((i as u64 + 31).wrapping_mul(1442695040888963407)) as f64 / u64::MAX as f64;
            Cpx::new(2.0 * x - 1.0, 2.0 * y - 1.0)
        })
        .collect();
    let dense_f: Vec<Cpx<F>> = dense
        .iter()
        .map(|z| Cpx::new(F::from_f64(z.re).unwrap(), F::from_f64(z.im).unwrap()))
        .collect();

    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        let p = base.clone().with_slot_order(slot_order);
        // Slot k reads the evaluation point br(k) under Natural, br(P(k))
        // under BitRevLow (the gather composes the tiled low-bit reversal).
        let point_of = |k: usize| -> usize {
            let k = match slot_order {
                crate::layouts::PaCoSlotOrder::Natural => k,
                crate::layouts::PaCoSlotOrder::BitRevLow => ext_bitrev_low(k, log_p),
            };
            (k as u32).reverse_bits() as usize >> (u32::BITS - log_two_c)
        };
        let packed: Vec<Cpx> = pack_chunk(
            &p,
            &mut |coeffs, re, im| {
                let mut scratch = scratch.borrow();
                let required = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(coeffs.len());
                ensure!(scratch.available() >= required);
                scratch.scope(|arena| {
                    let (mut values, _) = arena.take_ckks_encoding_buffer_scratch::<F>(coeffs.len());
                    copy_host_into_encoding_buffer::<BE, F, _>(&mut values, coeffs)?;
                    module.ckks_coeffs_to_slots_assign(&mut values)?;
                    copy_encoding_buffer_into_reim_host::<BE, F, _>(&values, re, im)
                })
            },
            &dense_f,
        )
        .unwrap()
        .into_iter()
        .map(|z| z.to_f64().unwrap())
        .collect();

        // (1) Vandermonde pin.
        let zeta = |e: usize| {
            let theta = 2.0 * std::f64::consts::PI * (e as f64) / ((4 * two_c) as f64);
            Cpx::new(theta.cos(), theta.sin())
        };
        assert!((zeta(two_c) - Cpx::new(0.0, 1.0)).abs() < 1e-12, "α = i");
        for k in 0..two_c {
            let mut pow5 = 1usize;
            for _ in 0..point_of(k) {
                pow5 = (pow5 * 5) % (4 * two_c);
            }
            for v in 0..p.h() {
                let mut want = Cpx::ZERO;
                for j in 0..two_c {
                    want = want + dense[v * two_c + j] * zeta((pow5 * j) % (4 * two_c));
                }
                let got = packed[v * two_c + k];
                assert!((got - want).abs() < 1e-9, "vandermonde({slot_order:?}): block {v} slot {k}");
            }
        }

        // (2) The generator's c2s chain inverts the encoder's packing — onto
        // natural order under Natural, onto the `P`-relabeled coefficients
        // under BitRevLow (the telescoped chain's output side).
        for schedule in [vec![1usize, 1, 1, 1], vec![2, 2], vec![4]] {
            let mut w = packed.clone();
            for f in crate::default::paco::lt::paco_c2s_factors::<f64>(&p, &schedule) {
                w = mul_vec_tiled(&f, &w);
            }
            for j in 0..n {
                let want = match slot_order {
                    crate::layouts::PaCoSlotOrder::Natural => dense[j],
                    crate::layouts::PaCoSlotOrder::BitRevLow => dense[ext_bitrev_low(j, log_p)],
                };
                assert!((w[j] - want).abs() < 1e-9, "inversion({slot_order:?}): {schedule:?} slot {j}");
            }
        }
    }
}
