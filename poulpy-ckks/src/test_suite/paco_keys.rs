//! Tests for structured PaCo key material and the blind-rotation/partial-C2S
//! pipeline prefix.
//!
//! This is the first full integration point composed against **real**
//! ciphertexts under the **structured** secret key.
//!
//! 1. **Structured secret**: [`PaCoSecretSpec::fill_glwe_secret`] loads the
//!    Algorithm-1 secret into a host `GLWESecret`, which is uploaded/prepared
//!    like any other key.
//! 2. **Residue extraction** (`coeff_encodings_from_ciphertext`): validated
//!    *independently* of the pipeline by checking the cleartext decryption
//!    identity `ct0 + s·ct1 ≈ m (mod q)` on the extracted residues — this
//!    pins the body/mask column convention, the `b + a·s` sign, and the
//!    base2k limb weights all at once (and is deliberately not circular,
//!    unlike comparing the pipeline against an oracle fed the same residues).
//! 3. **bsk gate**: each encrypted bootstrapping key `bsk_t` must decrypt to
//!    the σ_t packing of Algorithm 2.
//! 4. **Lines 2–6 gate**: getCoeffEnc (from the real ciphertext) → four
//!    plaintext×ciphertext blind rotations → `Tr_{N/2→n}` → grouped partial
//!    CoeffToSlot must land, slot for slot, on the cleartext model's Eq. 11
//!    packing relation: `b'_{λ_v}` coefficients in natural order.

use crate::api::CKKSEncodingOps;
use std::collections::HashMap;

use poulpy_core::{
    ModuleTransfer,
    layouts::{Base2K, GLWESecretPreparedFactory, LinearTransformationStrategy, ModuleCoreAlloc},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, galois_elements_from_rotations},
    source::Source,
};

use crate::SlotsKind;
use crate::{
    CKKSInfos, CKKSMeta,
    api::{CKKSAddOps, CKKSLinearTransformationOps, CKKSMulOps},
    default::paco::{
        lt::paco_c2s_factors,
        ops::{PaCoSlotOps, fold_rotations},
    },
    encoding::paco::{coeff_enc::glwe_column_residues, cpx::Cpx},
    layouts::{PaCoPlan, PaCoSecretSpec},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_ct, alloc_scratch, ckks_encrypt,
            ckks_encrypt_coeffs, ckks_spec, gen_atk,
        },
        paco_ops::assert_slots,
        paco_reference_model::{centered, monomial_mul, seq_paco_reference},
    },
};

/// PaCo dimensions on the suite ring (`N = 256`): `B = 16`, chunk count
/// `k = 2`, `n = 2hC = 64`.
const PACO_H: usize = 4;
const PACO_C: usize = 8;

/// Lines 2–6 of seqPaCo on real ciphertexts under the structured key, gated
/// against the cleartext model.
pub fn test_paco_partial_pipeline<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + PaCoSlotOps<BE> + CnvPVecAlloc<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        test_paco_partial_pipeline_with::<BE, F, E>(&params, slot_order);
    }
}

/// One slot-order convention's run of the lines 2–6 gate. The oracle's `z_7`
/// is natural-order by definition; under `BitRevLow` the production pipeline
/// lands on its `P`-relabel (`slot j = z_7[P(j)]`).
fn test_paco_partial_pipeline_with<BE, F, E>(params: &CKKSTestParams, slot_order: crate::layouts::PaCoSlotOrder)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + PaCoSlotOps<BE> + CnvPVecAlloc<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let params = *params;
    // Exhausted-input modulus: a multiple of base2k around 38 bits, so the
    // residues are exact in f64 and the limb weights are integral.
    let k_in = params.base2k * 38usize.div_ceil(params.base2k);
    let p = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, k_in as u32)
        .unwrap()
        .with_slot_order(slot_order);
    let factors = paco_c2s_factors::<F>(&p, &[2, 2]);

    // Budget: one blind-rotation pt×ct multiply + the C2S chain, plus headroom.
    let log_delta = params.prec().log_delta();
    let params = CKKSTestParams {
        k: (log_delta * (1 + factors.len() + 3)).next_multiple_of(params.base2k),
        ..params
    };
    let (m_full, n_paco) = (params.n / 2, p.slots());

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<F>(m_full).unwrap();
    let encoder_block = ReferenceEncoder::<E>::new::<F>(2 * p.c()).unwrap();
    let mut scratch = alloc_scratch(&params, &module);

    // ── Structured secret (Algorithm 1) into a real GLWE secret ────────────
    let mut source = Source::new([7u8; 32]);
    let spec = PaCoSecretSpec::sample(&p, &mut source).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&p, &mut sk_host).unwrap();
    let sk_raw = module.upload_glwe_secret(&sk_host);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    // ── bsk_t = Enc(σ_t) (Algorithm 2) + decrypt gate ───────────────────────
    let sigma: Vec<Vec<Cpx<F>>> = (0..4)
        .map(|t| spec.sigma_slots_with::<F, _>(&p, t, &mut |coeffs, re, im| encoder_block.unpack_reim_coeffs(coeffs, re, im)))
        .collect::<anyhow::Result<_>>()
        .unwrap();
    let bsk: Vec<_> = sigma
        .iter()
        .enumerate()
        .map(|(t, s)| {
            let re: Vec<F> = s.iter().map(|x| x.re).collect();
            let im: Vec<F> = s.iter().map(|x| x.im).collect();
            let ct = ckks_encrypt(
                &params,
                &module,
                &host_module,
                &encoder_full,
                &sk,
                params.k,
                &re,
                &im,
                &mut scratch.borrow(),
            );
            let s64: Vec<Cpx> = s.iter().map(|x| x.to_f64().unwrap()).collect();
            assert_slots::<BE, F, E>(
                &format!("paco_bsk_{t}"),
                &module,
                &host_module,
                &encoder_full,
                &ct,
                &sk,
                &s64,
                -(params.prec().log_delta() as f64) + 16.0,
                &mut scratch.borrow(),
            );
            ct
        })
        .collect();

    // ── Real exhausted input ciphertext at k_in under the same secret ──────
    let log_delta_in = k_in - 10;
    let coeffs: Vec<F> = (0..params.n)
        .map(|i| F::from_f64(0.4 * (((i.wrapping_mul(2654435761) % 1024) as f64) / 512.0 - 1.0)).unwrap())
        .collect();
    let ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        k_in,
        &coeffs,
        ckks_spec(params.n, params.base2k, log_delta_in, 10),
        &mut scratch.borrow(),
    );

    // ── Residue extraction, with an independent decryption-identity check ──
    let host_in = ct_in.to_host_owned::<BE>();
    let ct0 = glwe_column_residues(host_in.data(), 0, k_in, params.base2k).unwrap();
    let ct1 = glwe_column_residues(host_in.data(), 1, k_in, params.base2k).unwrap();
    {
        // ct0 + s·ct1 must equal the encoded message up to encryption noise.
        let mut s_ct1 = vec![0i64; params.n];
        let secret_coeffs = spec.sk_coeffs(&p).unwrap();
        for (pos, _) in secret_coeffs.iter().enumerate().filter(|&(_, &x)| x != 0) {
            let shifted = monomial_mul(&ct1, pos as i64, k_in as u32);
            s_ct1.iter_mut().zip(&shifted).for_each(|(a, &b)| *a = a.wrapping_add(b));
        }
        for i in 0..params.n {
            let got = centered(ct0[i].wrapping_add(s_ct1[i]), k_in as u32);
            let want = (coeffs[i].to_f64().unwrap() * (1u64 << log_delta_in) as f64).round() as i64;
            assert!(
                (got - want).abs() < 1 << 20,
                "decryption identity at coefficient {i}: got {got}, want {want}"
            );
        }
    }

    // ── getCoeffEnc (Algorithm 3) from the real ciphertext and cleartext model ──
    let beta = crate::encoding::paco::coeff_enc::coeff_encodings_with::<F, _>(&ct0, &ct1, &p, &mut |coeffs, re, im| {
        encoder_block.unpack_reim_coeffs(coeffs, re, im)
    })
    .unwrap();
    let oracle = seq_paco_reference(&ct0, &ct1, &spec, &p, &encoder_block, log_delta_in);

    // β_t as CKKS plaintexts at the factor scale.
    let beta_pt: Vec<_> = beta
        .iter()
        .map(|b| {
            let re: Vec<F> = b.iter().map(|x| x.re).collect();
            let im: Vec<F> = b.iter().map(|x| x.im).collect();
            crate::test_suite::helpers::encode_and_upload_pt::<BE, F, E>(
                &host_module,
                &module,
                &encoder_full,
                Base2K(params.base2k as u32),
                ckks_spec(params.n, params.base2k, log_delta, 10),
                &re,
                &im,
            )
        })
        .collect();

    // Partial-C2S factors and the key material for the whole pipeline prefix.
    let lts: Vec<_> = factors
        .iter()
        .map(|cd| {
            crate::default::ckks_encode_linear_transformation_from_diagonals(
                &module,
                Base2K(params.base2k as u32),
                crate::CoeffsMeta {
                    k: (log_delta + 10).into(),
                    meta: CKKSMeta {
                        log_sparsity: (m_full / n_paco).trailing_zeros() as usize,
                        log_delta,
                        slots: SlotsKind::Complex,
                    },
                },
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
    for p_el in galois_elements_from_rotations(fold_rotations(m_full, n_paco), order)
        .into_iter()
        .chain(lts.iter().flat_map(|lt| lt.galois_elements(order)))
    {
        atks.entry(p_el)
            .or_insert_with(|| gen_atk(&params, &module, p_el, &sk_raw, &mut scratch.borrow()));
    }

    // ── Lines 4–7: blind rotation, trace, partial CoeffToSlot ──────────────
    let mut acc = alloc_ct(&params, &module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut acc, &bsk[0], &beta_pt[0], &mut scratch.borrow())
        .unwrap();
    let mut tmp = alloc_ct(&params, &module, params.k);
    for t in 1..4 {
        module
            .ckks_mul_pt_vec_into(&mut tmp, &bsk[t], &beta_pt[t], &mut scratch.borrow())
            .unwrap();
        module.ckks_add_assign(&mut acc, &tmp, &mut scratch.borrow()).unwrap();
    }
    module
        .ckks_slot_trace_assign(&mut acc, m_full, n_paco, &atks, &mut scratch.borrow())
        .unwrap();
    for lt in &lts {
        module
            .ckks_eval_linear_transformation_self_assign(&mut acc, lt, &atks, &mut scratch.borrow())
            .unwrap();
    }

    // ── Gate: slots must equal the oracle's z_7 (Eq. 11 packing relation) ──
    // — in natural order under `Natural`, `P`-relabeled under `BitRevLow`.
    let z7: Vec<Cpx> = match slot_order {
        crate::layouts::PaCoSlotOrder::Natural => oracle.partial_c2s.clone(),
        crate::layouts::PaCoSlotOrder::BitRevLow => {
            let log_p = p.log_c() - 1;
            (0..oracle.partial_c2s.len())
                .map(|j| oracle.partial_c2s[crate::default::paco::ops::ext_bitrev_low(j, log_p)])
                .collect()
        }
    };
    assert_slots::<BE, F, E>(
        &format!("paco_partial_pipeline_z7({slot_order:?})"),
        &module,
        &host_module,
        &encoder_full,
        &acc,
        &sk,
        &z7,
        -(log_delta as f64) + 16.0,
        &mut scratch.borrow(),
    );
}
