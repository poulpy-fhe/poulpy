//! Plaintext generation of the factorized homomorphic (I)DFT matrices.
//!
//! Builds the radix-2 FFT/IFFT butterfly layers, merges them into the requested
//! number of factor matrices, applies the DFT normalization + caller scaling, and
//! returns one [`ComplexDiagonals<F>`] per factor (in evaluation order). No FHE
//! types are touched here — the output is validated in the clear and only later
//! encoded into CKKS plaintexts by the evaluation layer.
//!
//! Generic over the real scalar `F` ([`DftScalar`]) so the matrices can be
//! generated at `f64` today and at higher precision later.
//!
//! Covers all formats: `Standard`, `SplitRealAndImag`, and `RepackImagAsReal`
//! (under full packing the latter two share the same dense matrices; under sparse
//! packing `RepackImagAsReal` uses `dslots = 2·slots` butterflies and the special
//! repack matrix). Sparse coefficient placement at the codec boundary is provided
//! by [`CKKSPlaintextVecHostCodec::encode_host_floats_sparse`](crate::layouts::CKKSPlaintextVecHostCodec);
//! the [`CKKSMeta::log_sparsity`](crate::CKKSMeta) field carries the packing factor.
//!
//! Conventions match the rest of the crate: a diagonal at index `i` is the vector
//! `diag_i[j] = M[j][(j+i) mod slots]` (see [`poulpy_core::layouts::Diagonals`]),
//! and the canonical embedding uses the Galois generator 5 (`pow5`), identical to
//! the reim [`Encoder`](crate::encoding::reim).

use poulpy_core::layouts::Diagonals;
use rand_distr::num_traits::{Float, FloatConst};

use crate::layouts::{
    ComplexDiagonals,
    dft::{DFTOutputFormat, DFTPlan, DFTType},
};

/// Real scalar used for plaintext DFT-matrix generation: any float carrying the
/// constants needed for the roots of unity. Implemented for `f64` today; a
/// higher-precision float can be plugged in for more accurate matrices.
pub trait DftScalar: Float + FloatConst {}
impl<F: Float + FloatConst> DftScalar for F {}

/// A minimal complex *scalar* for the butterfly math (roots, the per-layer
/// coefficient vectors `a`/`b`/`c`, and `rotate_and_mul`). The diagonal *maps*
/// themselves are the crate's [`ComplexDiagonals`] (re/im split); `Cpx` only makes
/// the contiguous-vector FFT arithmetic readable.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cpx<F> {
    re: F,
    im: F,
}

impl<F: Float> Cpx<F> {
    #[inline]
    fn zero() -> Self {
        Cpx {
            re: F::zero(),
            im: F::zero(),
        }
    }

    #[inline]
    fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    #[inline]
    fn mul(self, o: Cpx<F>) -> Cpx<F> {
        Cpx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }

    #[inline]
    fn neg(self) -> Cpx<F> {
        Cpx::new(-self.re, -self.im)
    }
}

/// `roots[k] = exp(2πi·k / n)` for `k in 0..n`. `n = 4·slots`.
fn roots_of_unity<F: DftScalar>(n: usize) -> Vec<Cpx<F>> {
    let two = F::from(2.0).unwrap();
    let nf = F::from(n).unwrap();
    (0..n)
        .map(|k| {
            let theta = two * F::PI() * F::from(k).unwrap() / nf;
            Cpx::new(theta.cos(), theta.sin())
        })
        .collect()
}

/// `pow5[i] = 5^i mod 4·slots`, for `i in 0..=2·slots`.
fn pow5_table(slots: usize) -> Vec<usize> {
    let modulus = slots << 2;
    let mut pow5 = vec![0usize; (slots << 1) + 1];
    pow5[0] = 1;
    for i in 1..pow5.len() {
        pow5[i] = (pow5[i - 1] * 5) & (modulus - 1);
    }
    pow5
}

/// Bit-reverses the first `n` elements of `v` in place (`n` a power of two).
fn bit_reverse_in_place<F>(v: &mut [Cpx<F>], n: usize) {
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (u32::BITS - log_n)) as usize;
        if i < j {
            v.swap(i, j);
        }
    }
}

/// Butterfly coefficient triple `(a, b, c)` for one (I)FFT layer; each part is
/// `dslots` wide.
type ButterflyLayer<F> = (Vec<Cpx<F>>, Vec<Cpx<F>>, Vec<Cpx<F>>);

/// Coefficients `(a, b, c)` of one (I)FFT butterfly layer at `level`
/// (`0`-indexed in evaluation order: `level == 0` is the first layer applied).
///
/// `kind` selects the direction; the two variants differ only in three things:
/// - butterfly width `m`: `2 << level` for Decode (FFT), `slots >> level` for
///   Encode (IFFT) — both walk `2 → slots` in opposite orders;
/// - twiddle index: the IFFT conjugates the FFT one (`(4m − raw)` instead of
///   `raw`, equivalent to `roots[-raw·gap]`);
/// - twiddle slot: FFT writes it at `b[idx1]`, IFFT at `c[idx2]`; the other
///   slot gets the identity `roots[0]`.
///
/// Always `slots = 2^log_slots` wide: in the sparse-repack case the layer
/// values are identical in the two halves of the `dslots = 2·slots` working
/// vector (the butterfly acts the same on `[Re | Im]`), so the merge can simply
/// replicate the layer via modular indexing — see [`rotate_and_mul`]. Produced
/// on demand so [`gen_dft_matrices`] can merge layer-by-layer without holding
/// all `log_slots` layers in memory at once.
fn plain_layer<F: DftScalar>(
    kind: DFTType,
    log_slots: usize,
    roots: &[Cpx<F>],
    pow5: &[usize],
    level: usize,
) -> ButterflyLayer<F> {
    let slots = 1usize << log_slots;
    let m = match kind {
        DFTType::Decode => 2usize << level,
        DFTType::Encode => slots >> level,
    };
    debug_assert!(
        (2..=slots).contains(&m),
        "layer level {level} out of range for log_slots {log_slots}"
    );

    let mut a_m = vec![Cpx::zero(); slots];
    let mut b_m = vec![Cpx::zero(); slots];
    let mut c_m = vec![Cpx::zero(); slots];
    let tt = m >> 1;
    let gap = slots / m;
    let mask = (m << 2) - 1;
    let four_m = m << 2;
    let decode = kind == DFTType::Decode;

    let mut i = 0;
    while i < slots {
        for (j, &p5) in pow5.iter().enumerate().take(m >> 1) {
            let raw = p5 & mask;
            let k = if decode { raw } else { four_m - raw } * gap;
            let idx1 = i + j;
            let idx2 = i + j + tt;
            a_m[idx1] = roots[0];
            a_m[idx2] = roots[k].neg();
            // FFT puts the twiddle at b[idx1]; IFFT puts it at c[idx2].
            if decode {
                b_m[idx1] = roots[k];
                c_m[idx2] = roots[0];
            } else {
                b_m[idx1] = roots[0];
                c_m[idx2] = roots[k];
            }
        }
        i += m;
    }

    (a_m, b_m, c_m)
}

/// `true` when the layer's rotation uses `1 << (level-1)` (vs `1 << (logL-level)`).
/// Encode forward and Decode bit-reversed share one branch.
fn rot_uses_level(kind: DFTType, bit_reversed: bool) -> bool {
    (kind == DFTType::Encode && !bit_reversed) || (kind == DFTType::Decode && bit_reversed)
}

/// Reads diagonal `index` of a [`ComplexDiagonals`] as a contiguous `Cpx` vector
/// (zeros where a side is absent).
fn cd_get<F: DftScalar>(cd: &ComplexDiagonals<F>, index: i64, dslots: usize) -> Vec<Cpx<F>> {
    let re = cd.re.get(index);
    let im = cd.im.get(index);
    (0..dslots)
        .map(|j| Cpx::new(re.map_or(F::zero(), |v| v[j]), im.map_or(F::zero(), |v| v[j])))
        .collect()
}

/// Accumulating insert into a [`ComplexDiagonals`]: `cd[index] += vec` (set if
/// absent). The re/im parts are stored in the two underlying [`Diagonals`].
fn cd_accumulate<F: DftScalar>(cd: &mut ComplexDiagonals<F>, index: i64, vec: &[Cpx<F>]) {
    let new_re = match cd.re.get(index) {
        Some(cur) => cur.iter().zip(vec).map(|(&a, c)| a + c.re).collect(),
        None => vec.iter().map(|c| c.re).collect(),
    };
    cd.re.set(index, new_re);
    let new_im = match cd.im.get(index) {
        Some(cur) => cur.iter().zip(vec).map(|(&a, c)| a + c.im).collect(),
        None => vec.iter().map(|c| c.im).collect(),
    };
    cd.im.set(index, new_im);
}

/// An empty `dslots`-wide complex diagonal map.
fn empty_cd<F: DftScalar>(dslots: usize) -> ComplexDiagonals<F> {
    ComplexDiagonals::new(Diagonals::<F>::new(dslots), Diagonals::<F>::new(dslots))
}

/// Element-wise `out[i] = multiplier[i & (multiplier.len() − 1)] · rotated[(i + k) & (rotated.len() − 1)]`,
/// `out.len() == rotated.len()`.
///
/// The multiplier mask is what makes the sparse-repack path work with a
/// `slots`-wide butterfly layer (`multiplier`) against a `dslots = 2·slots`-wide
/// factor diagonal (`rotated`): indexing `multiplier` modulo its own length
/// replicates the layer across both halves of the working vector, exactly
/// matching the previous expanded representation.
fn rotate_and_mul<F: DftScalar>(rotated: &[Cpx<F>], k: i64, multiplier: &[Cpx<F>]) -> Vec<Cpx<F>> {
    let rot_mask = (rotated.len() - 1) as i64;
    let mul_mask = (multiplier.len() - 1) as i64;
    (0..rotated.len())
        .map(|i| {
            let m = multiplier[(i as i64 & mul_mask) as usize];
            let r = rotated[((i as i64 + k) & rot_mask) as usize];
            m.mul(r)
        })
        .collect()
}

/// Bit-reverses a `slots`-wide butterfly coefficient layer (no-op when
/// `bit_reversed == false`).
fn maybe_bit_reverse<F: DftScalar>(v: &[Cpx<F>], log_l: usize, bit_reversed: bool) -> Vec<Cpx<F>> {
    if !bit_reversed {
        return v.to_vec();
    }
    let slots = 1usize << log_l;
    let mut out = v.to_vec();
    bit_reverse_in_place(&mut out, slots);
    out
}

/// The `slots × slots` identity diagonal matrix as a [`ComplexDiagonals`]:
/// one diagonal at index `0` with all-ones real part (imaginary part empty).
///
/// Used as the merge accumulator's initial state, so every butterfly layer —
/// including the first of each factor — flows through [`merge_next_layer`]
/// uniformly.
fn identity_diag<F: DftScalar>(dslots: usize) -> ComplexDiagonals<F> {
    let mut diag = empty_cd(dslots);
    diag.re.set(0, vec![F::one(); dslots]);
    diag
}

/// Merges the next butterfly layer into an existing factor matrix.
#[allow(clippy::too_many_arguments)]
fn merge_next_layer<F: DftScalar>(
    vec: &ComplexDiagonals<F>,
    log_l: usize,
    n: usize,
    next_level: usize,
    a: &[Cpx<F>],
    b: &[Cpx<F>],
    c: &[Cpx<F>],
    kind: DFTType,
    bit_reversed: bool,
    dslots: usize,
) -> ComplexDiagonals<F> {
    let mask = (n - 1) as i64;
    let rot = if rot_uses_level(kind, bit_reversed) {
        (1i64 << (next_level - 1)) & mask
    } else {
        (1i64 << (log_l - next_level)) & mask
    };
    let a = maybe_bit_reverse(a, log_l, bit_reversed);
    let b = maybe_bit_reverse(b, log_l, bit_reversed);
    let c = maybe_bit_reverse(c, log_l, bit_reversed);

    let mut new_vec = empty_cd(dslots);
    for i in vec.indexes() {
        let vi = cd_get(vec, i, dslots);
        cd_accumulate(&mut new_vec, i, &rotate_and_mul(&vi, 0, &a));
        cd_accumulate(&mut new_vec, (i + rot) & mask, &rotate_and_mul(&vi, rot, &b));
        cd_accumulate(&mut new_vec, (i - rot) & mask, &rotate_and_mul(&vi, -rot, &c));
    }
    new_vec
}

/// The special initial matrix for sparse Decode repack.
///
/// Two diagonals over `dslots = 2·slots`: index `0` with value `(1 | i)` (real 1
/// in the left half, imag unit in the right half) and index `slots` with value
/// `(i | 1)`. Prepended (merged) before the first DFT layer when decoding a
/// sparsely-packed `RepackImagAsReal` vector; it recombines the `[Re | Im]` real
/// packing back into the complex form.
fn gen_repack_matrix<F: DftScalar>(log_l: usize, dslots: usize) -> ComplexDiagonals<F> {
    let slots = 1usize << log_l;
    debug_assert_eq!(dslots, 2 * slots);
    let (zero, one) = (F::zero(), F::one());
    let mut a = vec![Cpx::zero(); dslots];
    let mut b = vec![Cpx::zero(); dslots];
    for i in 0..slots {
        a[i] = Cpx::new(one, zero);
        a[i + slots] = Cpx::new(zero, one);
        b[i] = Cpx::new(zero, one);
        b[i + slots] = Cpx::new(one, zero);
    }
    let mut diag = empty_cd(dslots);
    cd_accumulate(&mut diag, 0, &a);
    cd_accumulate(&mut diag, slots as i64, &b);
    diag
}

/// Generates the ordered factor matrices of the homomorphic (I)DFT described by
/// `literal`, in evaluation order.
///
/// `log_n` is the ring degree exponent; `log_max_slots = log_n − 1`. When
/// `log_slots < log_max_slots` and the format is `RepackImagAsReal`, the **sparse
/// repack** path is taken: `dslots = 2·slots` butterflies, the repack matrix
/// prepended to the first Decode matrix, and the right half of the last Encode
/// matrix zeroed. Otherwise the dense path is used (and full-packing
/// `RepackImagAsReal` ≡ `SplitRealAndImag`). Panics on an invalid literal
/// (`log_slots < depth`).
pub fn gen_dft_matrices<F: DftScalar>(literal: &DFTPlan, log_n: usize) -> Vec<ComplexDiagonals<F>> {
    literal.check().expect("invalid DFTMatrixLiteral");

    let log_slots = literal.log_slots();
    let slots = 1usize << log_slots;
    let max_depth = literal.num_factors();
    let kind = literal.kind;
    let bit_reversed = literal.bit_reversed;

    let log_max_slots = log_n.saturating_sub(1);
    let imag_repack = literal.format == DFTOutputFormat::RepackImagAsReal;
    let sparse = log_slots < log_max_slots;
    // dslots == 2·slots only for the sparse repack path; otherwise dense.
    let dslots = if sparse && imag_repack { slots << 1 } else { slots };

    let roots = roots_of_unity::<F>(slots << 2);
    let pow5 = pow5_table(slots);

    // Fetch one butterfly layer on demand (peak memory drops by ~log_slots
    // vs. materializing every layer up front).
    let layer = |level: usize| -> ButterflyLayer<F> { plain_layer(kind, log_slots, &roots, &pow5, level) };

    // `factorization_depth` is consumed in evaluation order (factor 0 applied
    // first; see the convention on [`DFTPlan::factorization_depth`]). No implicit
    // reordering by `kind`: a Decode that inverts an Encode is the same schedule
    // reversed, which is the caller's responsibility.
    let mut plain_vector: Vec<ComplexDiagonals<F>> = Vec::with_capacity(max_depth);
    let mut fft_level = log_slots;
    for (i, &m) in literal.factorization_depth.iter().enumerate() {
        let repack_first = sparse && imag_repack && kind == DFTType::Decode && i == 0;
        // Sparse-repack merges wrap rotation indices mod `2·slots`; otherwise mod `slots`.
        let merge_n = if repack_first { slots << 1 } else { slots };

        // Start the factor from the repack matrix (sparse Decode, first factor)
        // or the slot-identity; every butterfly layer of the factor is then
        // merged in uniformly via `merge_next_layer`.
        let mut factor = if repack_first {
            gen_repack_matrix(log_slots, dslots)
        } else {
            identity_diag(dslots)
        };
        let mut next_level = fft_level as i64;

        // Merge the factor's `m` butterfly layers one at a time; each layer's
        // coefficient buffers are dropped at the end of its iteration.
        for _ in 0..m {
            let nl = next_level as usize;
            let (a_l, b_l, c_l) = layer(log_slots - nl);
            factor = merge_next_layer(&factor, log_slots, merge_n, nl, &a_l, &b_l, &c_l, kind, bit_reversed, dslots);
            next_level -= 1;
        }

        plain_vector.push(factor);
        fft_level -= m;
    }

    // Sparse-repack Encode: zero the right half of the last matrix's diagonals.
    if sparse && imag_repack && kind == DFTType::Encode {
        let last = plain_vector.last_mut().expect("dft has at least one factor");
        for idx in last.indexes() {
            let mut re = last.re.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); dslots]);
            let mut im = last.im.get(idx).cloned().unwrap_or_else(|| vec![F::zero(); dslots]);
            for x in 0..slots {
                re[x + slots] = F::zero();
                im[x + slots] = F::zero();
            }
            last.re.set(idx, re);
            last.im.set(idx, im);
        }
    }

    apply_scaling(&mut plain_vector, literal);
    plain_vector
}

/// Applies the DFT `1/N` normalization (Encode only) and the caller `scaling`,
/// spread as the `depth`-th root across all factor matrices.
fn apply_scaling<F: DftScalar>(factors: &mut [ComplexDiagonals<F>], literal: &DFTPlan) {
    let slots = 1usize << literal.log_slots();
    let depth = literal.num_factors();

    let mut scaling = literal.scaling.unwrap_or(1.0);
    if literal.kind == DFTType::Encode {
        // Real/imag extraction carries an extra 1/2 factor.
        let denom = match literal.format {
            DFTOutputFormat::Standard => slots as f64,
            DFTOutputFormat::SplitRealAndImag | DFTOutputFormat::RepackImagAsReal => 2.0 * slots as f64,
        };
        scaling /= denom;
    }
    // Spread across the matrices so the product accumulates to `scaling`.
    let per_factor = F::from(scaling.powf(1.0 / depth as f64)).unwrap();

    for factor in factors.iter_mut() {
        scale_complex_diagonals(factor, per_factor);
    }
}

/// Scales every stored diagonal value (re and im) of a [`ComplexDiagonals`] by `s`.
fn scale_complex_diagonals<F: DftScalar>(cd: &mut ComplexDiagonals<F>, s: F) {
    for idx in cd.re.indexes() {
        if let Some(v) = cd.re.get(idx) {
            let scaled: Vec<F> = v.iter().map(|&x| x * s).collect();
            cd.re.set(idx, scaled);
        }
    }
    for idx in cd.im.indexes() {
        if let Some(v) = cd.im.get(idx) {
            let scaled: Vec<F> = v.iter().map(|&x| x * s).collect();
            cd.im.set(idx, scaled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{Evaluate, LinearTransformationStrategy};

    fn literal(kind: DFTType, factorization_depth: Vec<usize>, bit_reversed: bool) -> DFTPlan {
        DFTPlan {
            kind,
            factorization_depth,
            format: DFTOutputFormat::Standard,
            scaling: Some(1.0),
            bit_reversed,
            factor_log_delta: 0,
        }
    }

    /// Applies a chain of complex factor matrices to `(re, im)` in the clear.
    fn eval_chain(factors: &[ComplexDiagonals<f64>], mut re: Vec<f64>, mut im: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        for f in factors {
            let (r, i) = f.evaluate((re.as_slice(), im.as_slice()), LinearTransformationStrategy::Direct);
            re = r;
            im = i;
        }
        (re, im)
    }

    /// Under full packing, `RepackImagAsReal` must generate exactly the same
    /// factor matrices as `SplitRealAndImag` (the sparse-only steps are skipped
    /// when `logSlots == logMaxSlots`; the two share the `1/(2·slots)` scaling).
    #[test]
    fn repack_equals_split_when_dense() {
        for kind in [DFTType::Encode, DFTType::Decode] {
            let mut split = literal(kind, vec![1, 1, 1, 1], false);
            split.format = DFTOutputFormat::SplitRealAndImag;
            split.scaling = None;
            let mut repack = split.clone();
            repack.format = DFTOutputFormat::RepackImagAsReal;

            let fs = gen_dft_matrices::<f64>(&split, 5);
            let fr = gen_dft_matrices::<f64>(&repack, 5);
            assert_eq!(fs.len(), fr.len());
            for (a, b) in fs.iter().zip(&fr) {
                assert_eq!(a.indexes(), b.indexes());
                for idx in a.indexes() {
                    assert_eq!(a.re.get(idx), b.re.get(idx), "re diag {idx}");
                    assert_eq!(a.im.get(idx), b.im.get(idx), "im diag {idx}");
                }
            }
        }
    }

    /// Sparse `RepackImagAsReal` generation produces `dslots = 2·slots`-wide
    /// diagonals, a Decode repack diagonal at index `slots`, and a zeroed right
    /// half on the last Encode matrix.
    #[test]
    fn sparse_repack_generation_structure() {
        // N = 64 (log_n = 6, log_max_slots = 5); log_slots = 2 < 5 → sparse.
        let (log_n, slots, dslots) = (6usize, 4usize, 8usize);
        let mk = |kind| DFTPlan {
            kind,
            factorization_depth: vec![1, 1], // sum = log_slots = 2
            format: DFTOutputFormat::RepackImagAsReal,
            scaling: None,
            bit_reversed: false,
            factor_log_delta: 0,
        };

        let dec = gen_dft_matrices::<f64>(&mk(DFTType::Decode), log_n);
        assert_eq!(dec[0].slots(), dslots, "decode value width");
        assert!(dec[0].indexes().contains(&(slots as i64)), "repack diagonal at index slots");

        let enc = gen_dft_matrices::<f64>(&mk(DFTType::Encode), log_n);
        assert_eq!(enc.last().unwrap().slots(), dslots, "encode value width");
        let last = enc.last().unwrap();
        for idx in last.re.indexes() {
            let re = last.re.get(idx).unwrap();
            let im = last.im.get(idx).unwrap();
            for x in slots..dslots {
                assert_eq!(re[x], 0.0, "encode right-half re zeroed at idx {idx} pos {x}");
                assert_eq!(im[x], 0.0, "encode right-half im zeroed at idx {idx} pos {x}");
            }
        }
    }

    /// A dense `RepackImagAsReal` (full packing: `log_slots == log_max_slots`)
    /// must not trigger the sparse branch.
    #[test]
    fn dense_repack_not_sparse() {
        // log_n = 5 → log_max_slots = 4; log_slots = 4 → dense.
        let f = gen_dft_matrices::<f64>(
            &DFTPlan {
                kind: DFTType::Encode,
                factorization_depth: vec![1, 1, 1, 1], // sum = log_slots = 4
                format: DFTOutputFormat::RepackImagAsReal,
                scaling: None,
                bit_reversed: false,
                factor_log_delta: 0,
            },
            5,
        );
        assert_eq!(f[0].slots(), 16, "dense → dslots == slots (no doubling)");
    }

    /// Encode (IDFT) followed by Decode (DFT) must recover the input vector, for
    /// every factorization schedule and both bit-reversal settings. Basis-independent:
    /// it exercises the butterflies, the merge, the rotation formulas, and the scaling
    /// without assuming a slot ordering.
    #[test]
    fn encode_then_decode_is_identity() {
        for log_slots in 1..=6usize {
            let slots = 1usize << log_slots;
            for bit_reversed in [false, true] {
                for levels in schedules(log_slots) {
                    let enc = gen_dft_matrices::<f64>(&literal(DFTType::Encode, levels.clone(), bit_reversed), log_slots + 1);
                    // Decode inverts Encode, so it uses the reversed schedule
                    // (evaluation-order convention; see `DFTPlan::factorization_depth`).
                    let dec_levels: Vec<usize> = levels.iter().rev().copied().collect();
                    let dec = gen_dft_matrices::<f64>(&literal(DFTType::Decode, dec_levels, bit_reversed), log_slots + 1);

                    let re: Vec<f64> = (0..slots).map(|j| (0.3 * (j as f64 + 1.0)).sin()).collect();
                    let im: Vec<f64> = (0..slots).map(|j| (0.7 * (j as f64 + 2.0)).cos()).collect();

                    let (er, ei) = eval_chain(&enc, re.clone(), im.clone());
                    let (rr, ri) = eval_chain(&dec, er, ei);

                    for j in 0..slots {
                        assert!(
                            (rr[j] - re[j]).abs() < 1e-9 && (ri[j] - im[j]).abs() < 1e-9,
                            "ls={log_slots} br={bit_reversed} levels={levels:?} slot {j}: \
                             got ({:.3e},{:.3e}) want ({:.3e},{:.3e})",
                            rr[j],
                            ri[j],
                            re[j],
                            im[j]
                        );
                    }
                }
            }
        }
    }

    /// Each factor is a sparse matrix; merging `m` layers gives at most `3^m`
    /// diagonals. Sanity-checks the merge does not explode.
    #[test]
    fn factor_sparsity() {
        // 3 factors merging 2, 2, 1 layers (log_slots = 2+2+1 = 5).
        let enc = gen_dft_matrices::<f64>(&literal(DFTType::Encode, vec![2, 2, 1], false), 6);
        assert_eq!(enc.len(), 3, "one factor per schedule entry");
        // First factor merges 2 layers -> at most 3^2 = 9 diagonals.
        assert!(enc[0].indexes().len() <= 9);
    }

    /// Candidate `levels` schedules whose sum == log_slots.
    fn schedules(log_slots: usize) -> Vec<Vec<usize>> {
        let mut out = vec![vec![1usize; log_slots]]; // one layer per matrix (no merge)
        out.push(vec![log_slots]); // single fully-merged matrix
        if log_slots >= 2 {
            let half = log_slots / 2;
            out.push(vec![half, log_slots - half]);
        }
        if log_slots >= 3 {
            out.push(vec![1, log_slots - 1]);
        }
        out
    }
}
