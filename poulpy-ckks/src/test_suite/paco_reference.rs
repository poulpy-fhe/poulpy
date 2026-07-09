//! PaCo cleartext-model tests that need a concrete FFT.
//!
//! The cleartext packing (`pack_chunk`) runs
//! on the [`ReferenceEncoder`]'s own FFT, so the reference-model gates that exercise
//! it — the Eq. 11 packing relation, end-to-end sparse-coefficient recovery,
//! and the Algorithm 5 coverage check — live here, parameterized by the
//! backend's FFT implementation, rather than in the reference model's unit
//! tests (poulpy-ckks itself carries no FFT implementation). Everything is
//! cleartext: no ciphertexts, no keys, no backend modules are touched.

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew},
    layouts::{HostBytesBackend, Module},
    source::Source,
};

use crate::{
    encoding::paco::{
        coeff_enc::{a_tilde, psi},
        cpx::Cpx,
    },
    layouts::{PaCoPlan, PaCoSecretSpec},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextHostModule, TestContextModule, TestScalar},
        paco_reference_model::{fabricate_ciphertext, monomial_mul, seq_paco_reference},
    },
};

fn source(seed: u8) -> Source {
    Source::new([seed; 32])
}

/// Random message with |m_i| < 2^log_msg.
fn random_message(n: usize, log_msg: u32, src: &mut Source) -> Vec<i64> {
    let bound = 1i64 << log_msg;
    let range = 2 * bound as u64;
    (0..n).map(|_| src.next_u64n(range, range - 1) as i64 - bound).collect()
}

/// For each selected v, the chunk index r and blind-rotation shift.
fn selected_shift(key: &PaCoSecretSpec, p: &PaCoPlan, lambda: usize) -> (usize, usize) {
    let k = p.k();
    let r = (k - key.u()[lambda] % k) % k;
    (r, (key.u()[lambda] + r) / k)
}

/// The cleartext reference model, gated on the backend's FFT:
///
/// 1. **Eq. (11)**: after blind rotation + trace + partial CoeffToSlot, slot
///    `v·2C + i` holds `b'_{λ_v, i}` in natural order, where
///    `b'_λ = Z^{⌈u_λ/k⌉} · b̃_λ^{([−u_λ]_k)}` (degree < 2C, no reduction).
/// 2. **End-to-end**: seqPaCo recovers `m_{i·N/C}` for all `i` within the
///    small-angle error bound, across several parameter sets.
/// 3. **Algorithm 5**: κ monomial-shifted sequential runs recover disjoint
///    coefficient classes covering all multiples of `N/D`, `D = κ·C`.
pub fn test_paco_cleartext_reference<BE, F, E>(
    _params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // ── Eq. (11) packing relation ───────────────────────────────────────────
    for (log_n, h, c) in [(9usize, 8usize, 4usize), (9, 4, 8), (10, 8, 8)] {
        let p = PaCoPlan::new(log_n, h, c, 29).expect("validated reference parameters");
        let encoder = ReferenceEncoder::<E>::new::<F>(2 * p.c()).expect("validated reference encoder dimension");
        let mut src = source(2);
        let key = PaCoSecretSpec::sample(&p, &mut src).expect("validated reference secret parameters");
        let m = random_message(p.n(), 10, &mut src);
        let (ct0, ct1) = fabricate_ciphertext(&m, &key, &p, &mut src);

        let trace = seq_paco_reference(&ct0, &ct1, &key, &p, &encoder, 0);
        let k = p.k();

        for v in 0..h {
            let lambda = (0..4)
                .map(|t| t * h + v)
                .find(|&x| key.d()[x])
                .expect("validated secret selects one group per residue class");
            let (r, shift) = selected_shift(&key, &p, lambda);
            // b'_λ directly: ψ(ã) coefficients shifted by Z^shift.
            let mut b_prime = vec![Cpx::ZERO; 2 * c];
            for jj in 0..c {
                let coefficient = a_tilde(&ct0, &ct1, &p, lambda, jj * k + r).expect("validated reference coefficient index");
                b_prime[shift + jj] = psi(coefficient, p.log_q()).expect("validated reference circle embedding");
            }
            for (i, &want) in b_prime.iter().enumerate() {
                let got = trace.partial_c2s[v * 2 * c + i];
                assert!(
                    (got - want).abs() < 1e-9,
                    "eq11 ({log_n},{h},{c}) v={v} i={i}: got ({},{}) want ({},{})",
                    got.re,
                    got.im,
                    want.re,
                    want.im
                );
            }
        }
    }

    // ── End-to-end sparse-coefficient recovery ──────────────────────────────
    for (log_n, h, c, log_q, log_msg) in [
        (9usize, 8usize, 4usize, 29u32, 10u32),
        (9, 8, 16, 29, 10),
        (9, 4, 8, 20, 6),
        (10, 8, 8, 29, 10),
        (10, 16, 2, 29, 10),
        (11, 16, 8, 29, 10),
    ] {
        let p = PaCoPlan::new(log_n, h, c, log_q).expect("validated reference parameters");
        let encoder = ReferenceEncoder::<E>::new::<F>(2 * p.c()).expect("validated reference encoder dimension");
        let mut src = source(3);
        let key = PaCoSecretSpec::sample(&p, &mut src).expect("validated reference secret parameters");
        let m = random_message(p.n(), log_msg, &mut src);
        let (ct0, ct1) = fabricate_ciphertext(&m, &key, &p, &mut src);

        let trace = seq_paco_reference(&ct0, &ct1, &key, &p, &encoder, 0);

        let mut max_err = 0.0f64;
        for i in 0..c {
            let want = m[i * p.n() / c] as f64;
            let got = trace.m_recovered[i];
            max_err = max_err.max((got - want).abs());
        }
        // Small-angle bound: |err| ≲ |m|·(2π·|m|/q)²/6 plus f64 noise (the
        // 1.5e-7 floor is the empirical f64 noise of the packing + factor
        // chains at these dimensions).
        let bound = {
            let mm = (1u64 << log_msg) as f64;
            let q = p.q() as f64;
            let angle = 2.0 * std::f64::consts::PI * mm / q;
            (mm * angle * angle / 6.0).max(1.5e-7) * 10.0
        };
        assert!(
            max_err < bound,
            "e2e ({log_n},{h},{c},q=2^{log_q}): max err {max_err:.3e} exceeds bound {bound:.3e}"
        );
    }

    // ── Algorithm 5 coverage ────────────────────────────────────────────────
    {
        let (log_n, h, c, kappa) = (9usize, 8usize, 4usize, 4usize);
        let p = PaCoPlan::new(log_n, h, c, 29).expect("validated reference parameters");
        let encoder = ReferenceEncoder::<E>::new::<F>(2 * p.c()).expect("validated reference encoder dimension");
        let n = p.n();
        let d = kappa * c;
        let mut src = source(4);
        let key = PaCoSecretSpec::sample(&p, &mut src).expect("validated reference secret parameters");
        let m = random_message(n, 10, &mut src);
        let (ct0, ct1) = fabricate_ciphertext(&m, &key, &p, &mut src);

        let mut covered = vec![false; d];
        for r in 0..kappa {
            let e = -((r * n / d) as i64);
            let ct0_r = monomial_mul(&ct0, e, p.log_q());
            let ct1_r = monomial_mul(&ct1, e, p.log_q());
            let trace = seq_paco_reference(&ct0_r, &ct1_r, &key, &p, &encoder, 0);

            // Expected: coefficients of X^{-r·N/D}·m at indices i·N/C.
            for i in 0..c {
                let raw = i * n / c + r * n / d;
                let (idx, neg) = (raw % n, (raw / n) % 2 == 1);
                let want = if neg { -m[idx] } else { m[idx] } as f64;
                let got = trace.m_recovered[i];
                assert!((got - want).abs() < 1e-4, "branch r={r} slot i={i}: got {got} want {want}");
                assert_eq!(idx % (n / d), 0);
                let class = idx / (n / d);
                assert!(!covered[class], "coefficient class {class} recovered twice");
                covered[class] = true;
            }
        }
        assert!(covered.iter().all(|&x| x), "all N/D-multiples covered");
    }
}
