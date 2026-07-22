//! Cleartext slot-level simulator of seqPaCo (paper Algorithm 4).
//!
//! Every homomorphic operation of the pipeline is executed on plain complex
//! slot vectors of length `N/2`:
//!
//! | Homomorphic op | Cleartext counterpart |
//! |---|---|
//! | plaintext×ciphertext multiplication | [`hadamard`] |
//! | rotation by `k` (Galois `5^k`) | [`rotate_left`] |
//! | conjugation (Galois `−1`) | [`conjugate`] |
//! | `Tr_{a→b}` / `Pr_{a→b}` | [`trace_slots`] / [`product_slots`] |
//! | per-block coefficient/slot conversion | [`ReferenceEncoder`] pack/unpack |
//!
//! [`seq_paco_reference`] returns the independent intermediate and final
//! values used as test gates. The oracle deliberately does not call PaCo's
//! production coefficient encodings or linear-transformation generators: it
//! spells out the paper formulas and only uses the generic [`ReferenceEncoder`] FFT
//! convention at the packing boundary.
//!
//! The module also provides integer negacyclic helpers modulo `q = 2^log_q`
//! ([`monomial_mul`], [`fabricate_ciphertext`]) so tests can build valid
//! `(ct0, ct1)` pairs satisfying `ct0 + s·ct1 = m mod q` without any
//! encryption machinery.

use std::fmt::Debug;

use num_traits::{FromPrimitive, ToPrimitive};
use poulpy_hal::{api::NegacyclicFFT, source::Source};

use crate::{default::dft::DftScalar, test_suite::reference_encoder::ReferenceEncoder};
use crate::{
    encoding::paco::cpx::Cpx,
    layouts::{PaCoPlan, PaCoSecretSpec},
};

/// Paper Eq. (8): coefficient `4h·j` of `a_v`, where `a_0 = ct0 + ct1`
/// and `a_v = X^v·ct1` for `v > 0` in the negacyclic ring.
fn reference_a_tilde(ct0: &[i64], ct1: &[i64], p: &PaCoPlan, v: usize, j: usize) -> i64 {
    let stride = 4 * p.h();
    assert!(v < stride);
    assert!(j < p.b());
    let target = stride * j;
    if v == 0 {
        ct0[target].wrapping_add(ct1[target])
    } else if target >= v {
        ct1[target - v]
    } else {
        ct1[p.n() + target - v].wrapping_neg()
    }
}

/// Paper's circle embedding `exp(2πi·a/q)`, evaluated at the oracle's
/// working precision and independently of the production embedding helper.
fn reference_circle<F: DftScalar + FromPrimitive>(a: i64, log_q: u32) -> Cpx<F> {
    let modulus = 1u64 << log_q;
    let a = <F as FromPrimitive>::from_i64(a).expect("reference coefficient is representable by the working scalar");
    let modulus = <F as FromPrimitive>::from_u64(modulus).expect("reference modulus is representable by the working scalar");
    let angle = (F::one() + F::one()) * F::PI() * a / modulus;
    Cpx::new(angle.cos(), angle.sin())
}

/// Independent implementation of the paper's block packing. Each `2C`-wide
/// block is interpreted as complex polynomial coefficients, converted to the
/// encoder's canonical slot order, then permuted to PaCo's bit-reversed point
/// order.
fn reference_pack_chunk<F, E>(p: &PaCoPlan, encoder: &ReferenceEncoder<E>, chunk: &[Cpx<F>]) -> Vec<Cpx<F>>
where
    F: DftScalar + Debug,
    E: NegacyclicFFT<F>,
{
    assert_eq!(chunk.len(), p.slots());
    let width = 2 * p.c();
    let log_width = width.trailing_zeros();
    let bit_reverse = |x: usize| x.reverse_bits() >> (usize::BITS - log_width);
    let mut out = vec![Cpx::zero(); chunk.len()];
    let mut coeffs = vec![F::zero(); 2 * width];
    let mut re = vec![F::zero(); width];
    let mut im = vec![F::zero(); width];

    for block in 0..p.h() {
        let start = block * width;
        for (j, value) in chunk[start..start + width].iter().enumerate() {
            coeffs[j] = value.re;
            coeffs[j + width] = value.im;
        }
        encoder
            .unpack_reim_coeffs(&coeffs, &mut re, &mut im)
            .expect("reference block packing uses a matching 2C-slot encoder");
        for k in 0..width {
            let point = bit_reverse(k);
            out[start + k] = Cpx::new(re[point], im[point]);
        }
    }
    out
}

/// Inverse of [`reference_pack_chunk`], tiled over the complete `N/2` slot
/// vector after the trace. This is the paper's partial CoeffToSlot result,
/// computed without the production DFT matrices.
fn reference_unpack_blocks<F, E>(p: &PaCoPlan, encoder: &ReferenceEncoder<E>, packed: &[Cpx<F>]) -> Vec<Cpx<F>>
where
    F: DftScalar + Debug,
    E: NegacyclicFFT<F>,
{
    assert_eq!(packed.len(), p.half_n());
    let width = 2 * p.c();
    let log_width = width.trailing_zeros();
    let bit_reverse = |x: usize| x.reverse_bits() >> (usize::BITS - log_width);
    let mut out = vec![Cpx::zero(); packed.len()];
    let mut re = vec![F::zero(); width];
    let mut im = vec![F::zero(); width];
    let mut coeffs = vec![F::zero(); 2 * width];

    for start in (0..packed.len()).step_by(width) {
        for k in 0..width {
            let point = bit_reverse(k);
            re[point] = packed[start + k].re;
            im[point] = packed[start + k].im;
        }
        encoder
            .pack_reim_coeffs(&mut coeffs, &re, &im)
            .expect("reference block unpacking uses a matching 2C-slot encoder");
        for j in 0..width {
            out[start + j] = Cpx::new(coeffs[j], coeffs[j + width]);
        }
    }
    out
}

/// Algorithm 3, written directly from the paper's `b̃_v^(r)` definition.
fn reference_beta<F, E>(ct0: &[i64], ct1: &[i64], p: &PaCoPlan, encoder: &ReferenceEncoder<E>) -> [Vec<Cpx<F>>; 4]
where
    F: DftScalar + Debug + FromPrimitive,
    E: NegacyclicFFT<F>,
{
    assert_eq!(ct0.len(), p.n());
    assert_eq!(ct1.len(), p.n());
    std::array::from_fn(|t| {
        let mut packed = Vec::with_capacity(p.half_n());
        for r in 0..p.k() {
            let mut raw = vec![Cpx::zero(); p.slots()];
            for v in 0..p.h() {
                let class = t * p.h() + v;
                for i in 0..p.c() {
                    let coefficient = reference_a_tilde(ct0, ct1, p, class, i * p.k() + r);
                    raw[v * 2 * p.c() + i] = reference_circle(coefficient, p.log_q());
                }
            }
            packed.extend(reference_pack_chunk(p, encoder, &raw));
        }
        assert_eq!(packed.len(), p.half_n());
        packed
    })
}

/// Algorithm 2, written directly from the structured-key selector rule.
fn reference_sigma<F, E>(key: &PaCoSecretSpec, p: &PaCoPlan, t: usize, encoder: &ReferenceEncoder<E>) -> Vec<Cpx<F>>
where
    F: DftScalar + Debug,
    E: NegacyclicFFT<F>,
{
    assert!(t < 4);
    let mut packed = Vec::with_capacity(p.half_n());
    for r in 0..p.k() {
        let mut raw = vec![Cpx::zero(); p.slots()];
        for v in 0..p.h() {
            let lambda = t * p.h() + v;
            let shifted = key.u()[lambda] + r;
            if key.d()[lambda] && shifted.is_multiple_of(p.k()) {
                raw[v * 2 * p.c() + shifted / p.k()] = Cpx::one();
            }
        }
        packed.extend(reference_pack_chunk(p, encoder, &raw));
    }
    assert_eq!(packed.len(), p.half_n());
    packed
}

/// `out[i] = v[(i+k) mod len]` — the slot action of the Galois rotation `5^k`.
pub fn rotate_left(v: &[Cpx], k: usize) -> Vec<Cpx> {
    let len = v.len();
    (0..len).map(|i| v[(i + k) % len]).collect()
}

/// Slot-wise complex conjugation (Galois element `−1`).
pub fn conjugate(v: &[Cpx]) -> Vec<Cpx> {
    v.iter().map(|x| x.conj()).collect()
}

/// Slot-wise product.
pub fn hadamard(a: &[Cpx], b: &[Cpx]) -> Vec<Cpx> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// `Tr_{a→b}`: folds a period-`a` vector to period `b` by summing slots
/// `i, i+b, i+2b, …` (`log(a/b)` rotate-and-add steps; no-op when `a == b`).
pub fn trace_slots(v: &[Cpx], a: usize, b: usize) -> Vec<Cpx> {
    debug_assert!(a.is_power_of_two() && a.is_multiple_of(b));
    let mut v = v.to_vec();
    let steps = (a / b).trailing_zeros();
    for l in 0..steps {
        let rot = rotate_left(&v, a >> (l + 1));
        v.iter_mut().zip(&rot).for_each(|(x, &y)| *x = *x + y);
    }
    v
}

/// `Pr_{a→b}`: folds a period-`a` vector to period `b` by multiplying slots
/// `i, i+b, i+2b, …` (`log(a/b)` rotate-and-multiply steps — the modular
/// additions of the decryption sum, simulated on the unit circle).
pub fn product_slots(v: &[Cpx], a: usize, b: usize) -> Vec<Cpx> {
    debug_assert!(a.is_power_of_two() && a.is_multiple_of(b));
    let mut v = v.to_vec();
    let steps = (a / b).trailing_zeros();
    for l in 0..steps {
        let rot = rotate_left(&v, a >> (l + 1));
        v.iter_mut().zip(&rot).for_each(|(x, &y)| *x = *x * y);
    }
    v
}

/// Centered representative of `x mod 2^log_q` in `[−q/2, q/2)`.
pub fn centered(x: i64, log_q: u32) -> i64 {
    let q = 1i64 << log_q;
    let x = x & (q - 1);
    if x >= q / 2 { x - q } else { x }
}

/// Negacyclic `X^e · poly` modulo `q = 2^log_q` (`e` may be negative);
/// outputs are reduced to `[0, q)`.
pub fn monomial_mul(coeffs: &[i64], e: i64, log_q: u32) -> Vec<i64> {
    let n = coeffs.len();
    let mask = (1i64 << log_q) - 1;
    let e = e.rem_euclid(2 * n as i64) as usize; // X^{2N} = 1
    let mut out = vec![0i64; n];
    for (j, &c) in coeffs.iter().enumerate() {
        let raw = j + e;
        let (idx, neg) = (raw % n, (raw / n) % 2 == 1);
        out[idx] = if neg { c.wrapping_neg() & mask } else { c & mask };
    }
    out
}

/// Builds a valid exhausted ciphertext for message `m` (signed coefficients)
/// under the structured key: `ct1` uniform mod `q`, `ct0 = m − s·ct1 mod q`.
pub fn fabricate_ciphertext(m: &[i64], key: &PaCoSecretSpec, p: &PaCoPlan, source: &mut Source) -> (Vec<i64>, Vec<i64>) {
    let n = p.n();
    debug_assert_eq!(m.len(), n);
    let mask = p.q_mask() as i64;
    let ct1: Vec<i64> = (0..n).map(|_| source.next_u64n(p.q(), p.q_mask()) as i64).collect();

    // s·ct1 negacyclically, exploiting that s is sparse with 0/1 coefficients.
    let mut s_ct1 = vec![0i64; n];
    let secret_coeffs = key.sk_coeffs(p).expect("validated reference secret");
    for (pos, &coefficient) in secret_coeffs.iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        let shifted = monomial_mul(&ct1, pos as i64, p.log_q());
        s_ct1.iter_mut().zip(&shifted).for_each(|(a, &b)| *a = a.wrapping_add(b));
    }

    let ct0: Vec<i64> = (0..n).map(|j| m[j].wrapping_sub(s_ct1[j]) & mask).collect();
    (ct0, ct1)
}

/// Independent intermediate and final values used to validate seqPaCo.
///
/// The slot vectors have `N/2` entries. Only values consumed by the test
/// gates are retained; transient pipeline state stays local to the oracle.
#[derive(Clone, Debug)]
pub struct PaCoTrace {
    /// Line 7 (`z_7`): after partial CoeffToSlot — slot `v·2C + i` holds the
    /// block coefficient `b'_{λ_v,i}`, in natural order. **Cleartext-only**
    /// intermediate (pins Eq. 11): homomorphically, lines 7–10 are one fused
    /// stage. The oracle is convention-agnostic (it models the paper math in
    /// natural order and never calls the production packing); gates comparing
    /// production mid-pipeline data under
    /// [`PaCoSlotOrder::BitRevLow`](crate::layouts::PaCoSlotOrder) relabel
    /// this field by the low-bit reversal `P` themselves.
    pub partial_c2s: Vec<Cpx>,
    /// Lines 12/14/15/16, fused: the `msg`-valued CKKS slot evaluations of
    /// the recovered coefficient pairs, constructed directly from the
    /// defining Vandermonde rather than the production SlotToCoeff′ chain.
    pub final_slots: Vec<Cpx>,
    /// The recovered coefficients `m̃_i ≈ m_{i·N/C}`, read off the packed
    /// pairs `w_j = m̃_j + i·m̃_{j+C/2}`.
    pub m_recovered: Vec<f64>,
}

/// Runs the cleartext seqPaCo pipeline (Algorithm 4) on the exhausted
/// ciphertext `(ct0, ct1)` (residues mod `q`) under `key`. `encoder` is the
/// `2C`-slot encoder used at the per-block packing boundary.
/// `log_delta_in` is the exhausted ciphertext's own scale: the homomorphic
/// pipeline re-anchors its output onto it (the output decodes to the same
/// values the input decoded to), so [`PaCoTrace::final_slots`] carries
/// `msg = m̃/2^{log_delta_in}`-valued slots; pass `0` to read raw `m̃` values.
pub fn seq_paco_reference<F, E>(
    ct0: &[i64],
    ct1: &[i64],
    key: &PaCoSecretSpec,
    p: &PaCoPlan,
    encoder: &ReferenceEncoder<E>,
    log_delta_in: usize,
) -> PaCoTrace
where
    F: DftScalar + Debug + FromPrimitive + ToPrimitive,
    E: NegacyclicFFT<F>,
{
    let (n, c) = (p.slots(), p.c());
    let half_n = p.half_n();
    let q = p.q() as f64;

    // Lines 2–5: cleartext encodings, blind rotation (pt×ct), sum over t.
    // The packings are computed at the working precision F and truncated to
    // the oracle's f64 here (the oracle's gates carry f64-level bounds).
    let beta = reference_beta::<F, E>(ct0, ct1, p, encoder);
    let mut blind_rotated = vec![Cpx::ZERO; half_n];
    for (t, b) in beta.iter().enumerate() {
        let sigma: Vec<Cpx> = reference_sigma::<F, E>(key, p, t, encoder)
            .into_iter()
            .map(|x| x.to_f64().expect("reference sigma value is representable as f64"))
            .collect();
        let b64: Vec<Cpx> = b
            .iter()
            .map(|x| x.to_f64().expect("reference coefficient value is representable as f64"))
            .collect();
        let prod = hadamard(&b64, &sigma);
        blind_rotated.iter_mut().zip(&prod).for_each(|(x, &y)| *x = *x + y);
    }

    // Line 6: Tr_{N/2 → n} selects, per v, the single contributing chunk r.
    let traced = trace_slots(&blind_rotated, half_n, n);

    // Line 7: undo the per-block packing through the generic encoder itself,
    // independently of the production CoeffToSlot factor generator.
    let traced_f: Vec<Cpx<F>> = traced
        .iter()
        .map(|value| {
            Cpx::new(
                <F as FromPrimitive>::from_f64(value.re).expect("reference real slot is representable by the working scalar"),
                <F as FromPrimitive>::from_f64(value.im)
                    .expect("reference imaginary slot is representable by the working scalar"),
            )
        })
        .collect();
    let partial_c2s: Vec<Cpx> = reference_unpack_blocks(p, encoder, &traced_f)
        .into_iter()
        .map(|value| {
            Cpx::new(
                value.re.to_f64().expect("reference real coefficient is representable as f64"),
                value
                    .im
                    .to_f64()
                    .expect("reference imaginary coefficient is representable as f64"),
            )
        })
        .collect();

    // Lines 8–10: ψ(a'_{v,i}) = b'_{v,i} + conj(b'_{v,i+C}), then the mask μ
    // keeping the lower half of every 2C block (homomorphically fused into
    // the last C2S factor as A·w + B·conj(w)).
    let conj_rot = conjugate(&rotate_left(&partial_c2s, c));
    let psi_masked: Vec<Cpx> = partial_c2s
        .iter()
        .zip(&conj_rot)
        .enumerate()
        .map(|(i, (&x, &y))| if i % (2 * c) < c { x + y } else { Cpx::ZERO })
        .collect();

    // Line 11: Pr_{n → 2C} — the product over the h blocks.
    let product = product_slots(&psi_masked, n, 2 * c);

    // Line 13: 2i·Im(·), on the 2C-periodic layout (the upper halves are
    // exact zeros; the paper's line-12 fold lives inside the StC chain).
    let conj = conjugate(&product);
    let imag_extracted: Vec<Cpx> = product.iter().zip(&conj).map(|(&x, &y)| x - y).collect();

    // Read m̃ off the packed pairs w_j = (q/4π)·(−i·v_j + v_{j+C/2}) =
    // m̃_j + i·m̃_{j+C/2} (natural order), computed directly from the
    // extraction output v.
    let s = q / (4.0 * std::f64::consts::PI);
    let mut m_recovered = vec![0.0f64; c];
    for j in 0..c / 2 {
        let z = (Cpx::new(0.0, -1.0) * imag_extracted[j] + imag_extracted[j + c / 2]) * Cpx::new(s, 0.0);
        m_recovered[j] = z.re;
        m_recovered[j + c / 2] = z.im;
    }

    // Lines 14–16 target the ordinary CKKS encoding of the recovered
    // coefficient pairs. Construct those slot evaluations directly instead
    // of applying the production SlotToCoeff factor chain. For the C/2-slot
    // sub-ring, slot k evaluates the polynomial with complex coefficients
    // m̃_j + i·m̃_{j+C/2} at xi^(5^k), xi = exp(2πi/(2C)); the result
    // repeats across the full N/2 slots and is re-anchored to the exhausted
    // input's scale.
    let input_scale_inverse = 2.0f64.powf(-(log_delta_in as f64));
    let half_c = c / 2;
    let cyclotomic_order = 4 * half_c;
    let mut encoded_period = Vec::with_capacity(half_c);
    let mut exponent = 1usize;
    for _ in 0..half_c {
        let mut slot = Cpx::ZERO;
        for j in 0..half_c {
            let angle = 2.0 * std::f64::consts::PI * ((exponent * j) % cyclotomic_order) as f64 / cyclotomic_order as f64;
            let root = Cpx::new(angle.cos(), angle.sin());
            slot = slot + Cpx::new(m_recovered[j], m_recovered[j + half_c]) * root;
        }
        encoded_period.push(slot * Cpx::new(input_scale_inverse, 0.0));
        exponent = (exponent * 5) % cyclotomic_order;
    }
    let final_slots = (0..half_n).map(|slot| encoded_period[slot % half_c]).collect();

    PaCoTrace {
        partial_c2s,
        final_slots,
        m_recovered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Eq. (9): the decryption reformulation is exact over the integers:
    /// m_{i·N/C} = Σ_{v selected} ±ã_{λ_v, ·} mod q, with the Y^C = −1 wrap.
    #[test]
    fn eq9_decryption_reformulation() {
        for (log_n, h, c) in [(9usize, 8usize, 4usize), (9, 4, 8), (10, 8, 8), (9, 8, 16)] {
            let p = PaCoPlan::new(log_n, h, c, 29).expect("validated reference parameters");
            let mut src = source(1);
            let key = PaCoSecretSpec::sample(&p, &mut src).expect("validated reference secret parameters");
            let m = random_message(p.n(), 10, &mut src);
            let (ct0, ct1) = fabricate_ciphertext(&m, &key, &p, &mut src);

            let k = p.k();
            for i in 0..c {
                let mut acc = 0i64;
                for lambda in (0..4 * h).filter(|&v| key.d()[v]) {
                    let (r, shift) = selected_shift(&key, &p, lambda);
                    if i >= shift {
                        acc = acc.wrapping_add(reference_a_tilde(&ct0, &ct1, &p, lambda, (i - shift) * k + r));
                    } else {
                        // Y^C = −1 wrap: coefficient i picks up −ã at index i − shift + C.
                        acc = acc.wrapping_sub(reference_a_tilde(&ct0, &ct1, &p, lambda, (i + c - shift) * k + r));
                    }
                }
                assert_eq!(
                    centered(acc, p.log_q()),
                    m[i * p.n() / c],
                    "params ({log_n},{h},{c}) coefficient {i}"
                );
            }
        }
    }
}
