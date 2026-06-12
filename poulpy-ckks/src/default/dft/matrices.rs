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
    dft::{DFTFormat, DFTMatrixLiteral, DFTType},
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
        Cpx { re: F::zero(), im: F::zero() }
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

/// FFT (decoding) butterfly coefficient layers `a,b,c`, each `[log_slots][dslots]`.
fn fft_plain_vec<F: DftScalar>(
    log_slots: usize,
    dslots: usize,
    roots: &[Cpx<F>],
    pow5: &[usize],
) -> (Vec<Vec<Cpx<F>>>, Vec<Vec<Cpx<F>>>, Vec<Vec<Cpx<F>>>) {
    let big_n = 1usize << log_slots;
    let size = if 2 * big_n == dslots { 2 } else { 1 };

    let mut a = Vec::with_capacity(log_slots);
    let mut b = Vec::with_capacity(log_slots);
    let mut c = Vec::with_capacity(log_slots);

    let mut m = 2;
    while m <= big_n {
        let mut a_m = vec![Cpx::zero(); dslots];
        let mut b_m = vec![Cpx::zero(); dslots];
        let mut c_m = vec![Cpx::zero(); dslots];
        let tt = m >> 1;

        let mut i = 0;
        while i < big_n {
            let gap = big_n / m;
            let mask = (m << 2) - 1;
            for j in 0..(m >> 1) {
                let k = (pow5[j] & mask) * gap;
                let idx1 = i + j;
                let idx2 = i + j + tt;
                for u in 0..size {
                    a_m[idx1 + u * big_n] = roots[0];
                    a_m[idx2 + u * big_n] = roots[k].neg();
                    b_m[idx1 + u * big_n] = roots[k];
                    c_m[idx2 + u * big_n] = roots[0];
                }
            }
            i += m;
        }

        a.push(a_m);
        b.push(b_m);
        c.push(c_m);
        m <<= 1;
    }

    (a, b, c)
}

/// IFFT (encoding) butterfly coefficient layers.
fn ifft_plain_vec<F: DftScalar>(
    log_slots: usize,
    dslots: usize,
    roots: &[Cpx<F>],
    pow5: &[usize],
) -> (Vec<Vec<Cpx<F>>>, Vec<Vec<Cpx<F>>>, Vec<Vec<Cpx<F>>>) {
    let big_n = 1usize << log_slots;
    let size = if 2 * big_n == dslots { 2 } else { 1 };

    let mut a = Vec::with_capacity(log_slots);
    let mut b = Vec::with_capacity(log_slots);
    let mut c = Vec::with_capacity(log_slots);

    let mut m = big_n;
    while m >= 2 {
        let mut a_m = vec![Cpx::zero(); dslots];
        let mut b_m = vec![Cpx::zero(); dslots];
        let mut c_m = vec![Cpx::zero(); dslots];
        let tt = m >> 1;

        let mut i = 0;
        while i < big_n {
            let gap = big_n / m;
            let mask = (m << 2) - 1;
            for j in 0..(m >> 1) {
                let k = ((m << 2) - (pow5[j] & mask)) * gap;
                let idx1 = i + j;
                let idx2 = i + j + tt;
                for u in 0..size {
                    a_m[idx1 + u * big_n] = roots[0];
                    a_m[idx2 + u * big_n] = roots[k].neg();
                    b_m[idx1 + u * big_n] = roots[0];
                    c_m[idx2 + u * big_n] = roots[k];
                }
            }
            i += m;
        }

        a.push(a_m);
        b.push(b_m);
        c.push(c_m);
        m >>= 1;
    }

    (a, b, c)
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

/// `c[i] = b[i] · a[(i+k) mod len]`.
fn rotate_and_mul<F: DftScalar>(a: &[Cpx<F>], k: i64, b: &[Cpx<F>]) -> Vec<Cpx<F>> {
    let len = a.len();
    let mask = (len - 1) as i64;
    (0..len).map(|i| b[i].mul(a[((i as i64 + k) & mask) as usize])).collect()
}

/// Maybe-bit-reverse a butterfly coefficient layer (both halves if `dslots > slots`).
fn maybe_bit_reverse<F: DftScalar>(v: &[Cpx<F>], log_l: usize, bit_reversed: bool) -> Vec<Cpx<F>> {
    if !bit_reversed {
        return v.to_vec();
    }
    let slots = 1usize << log_l;
    let mut out = v.to_vec();
    bit_reverse_in_place(&mut out, slots);
    if out.len() > slots {
        let (_, hi) = out.split_at_mut(slots);
        bit_reverse_in_place(hi, slots);
    }
    out
}

/// First layer of a factor matrix.
#[allow(clippy::too_many_arguments)]
fn gen_fft_diag_matrix<F: DftScalar>(
    log_l: usize,
    fft_level: usize,
    a: &[Cpx<F>],
    b: &[Cpx<F>],
    c: &[Cpx<F>],
    kind: DFTType,
    bit_reversed: bool,
    dslots: usize,
) -> ComplexDiagonals<F> {
    let rot = if rot_uses_level(kind, bit_reversed) {
        1i64 << (fft_level - 1)
    } else {
        1i64 << (log_l - fft_level)
    };
    let a = maybe_bit_reverse(a, log_l, bit_reversed);
    let b = maybe_bit_reverse(b, log_l, bit_reversed);
    let c = maybe_bit_reverse(c, log_l, bit_reversed);

    let mut diag = empty_cd(dslots);
    cd_accumulate(&mut diag, 0, &a);
    cd_accumulate(&mut diag, rot, &b);
    cd_accumulate(&mut diag, (1i64 << log_l) - rot, &c);
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

/// Distributes `log_slots` butterfly layers across `max_depth` factor matrices.
/// The order is reversed for Decode (this collapses the layers so the transform
/// needs fewer distinct rotations / keys).
fn merge_schedule(log_slots: usize, max_depth: usize, kind: DFTType) -> Vec<usize> {
    let mut merge = vec![0usize; max_depth];
    let mut level = log_slots;
    for i in 0..max_depth {
        let depth = (level as f64 / (max_depth - i) as f64).ceil() as usize;
        if kind == DFTType::Encode {
            merge[i] = depth;
        } else {
            merge[max_depth - i - 1] = depth;
        }
        level -= depth;
    }
    merge
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
pub fn gen_dft_matrices<F: DftScalar>(literal: &DFTMatrixLiteral, log_n: usize) -> Vec<ComplexDiagonals<F>> {
    literal.check().expect("invalid DFTMatrixLiteral");

    let log_slots = literal.log_slots;
    let slots = 1usize << log_slots;
    let max_depth = literal.depth(false);
    let kind = literal.kind;
    let bit_reversed = literal.bit_reversed;

    let log_max_slots = log_n.saturating_sub(1);
    let imag_repack = literal.format == DFTFormat::RepackImagAsReal;
    let sparse = log_slots < log_max_slots;
    // dslots == 2·slots only for the sparse repack path; otherwise dense.
    let dslots = if sparse && imag_repack { slots << 1 } else { slots };

    let roots = roots_of_unity::<F>(slots << 2);
    let pow5 = pow5_table(slots);

    let (a, b, c) = match kind {
        DFTType::Encode => ifft_plain_vec(log_slots, dslots, &roots, &pow5),
        DFTType::Decode => fft_plain_vec(log_slots, dslots, &roots, &pow5),
    };

    let merge = merge_schedule(log_slots, max_depth, kind);

    let mut plain_vector: Vec<ComplexDiagonals<F>> = Vec::with_capacity(max_depth);
    let mut fft_level = log_slots;
    for (i, &m) in merge.iter().enumerate() {
        let repack_first = sparse && imag_repack && kind == DFTType::Decode && i == 0;
        // Sparse-repack merges wrap rotation indices mod `2·slots`; otherwise mod `slots`.
        let merge_n = if repack_first { slots << 1 } else { slots };

        let (mut factor, mut next) = if repack_first {
            // Special repack matrix, then merge the first DFT layer into it.
            let repack = gen_repack_matrix(log_slots, dslots);
            let merged = merge_next_layer(
                &repack,
                log_slots,
                merge_n,
                fft_level,
                &a[log_slots - fft_level],
                &b[log_slots - fft_level],
                &c[log_slots - fft_level],
                kind,
                bit_reversed,
                dslots,
            );
            (merged, fft_level as i64 - 1)
        } else {
            let f = gen_fft_diag_matrix(
                log_slots,
                fft_level,
                &a[log_slots - fft_level],
                &b[log_slots - fft_level],
                &c[log_slots - fft_level],
                kind,
                bit_reversed,
                dslots,
            );
            (f, fft_level as i64 - 1)
        };

        // Merge the remaining `m - 1` layers of this factor.
        for _ in 0..(m.saturating_sub(1)) {
            let nl = next as usize;
            factor = merge_next_layer(
                &factor,
                log_slots,
                merge_n,
                nl,
                &a[log_slots - nl],
                &b[log_slots - nl],
                &c[log_slots - nl],
                kind,
                bit_reversed,
                dslots,
            );
            next -= 1;
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
fn apply_scaling<F: DftScalar>(factors: &mut [ComplexDiagonals<F>], literal: &DFTMatrixLiteral) {
    let slots = 1usize << literal.log_slots;
    let depth = literal.depth(false);

    let mut scaling = literal.scaling.unwrap_or(1.0);
    if literal.kind == DFTType::Encode {
        // Real/imag extraction carries an extra 1/2 factor.
        let denom = match literal.format {
            DFTFormat::Standard => slots as f64,
            DFTFormat::SplitRealAndImag | DFTFormat::RepackImagAsReal => 2.0 * slots as f64,
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

    fn literal(kind: DFTType, log_slots: usize, levels: Vec<usize>, bit_reversed: bool) -> DFTMatrixLiteral {
        DFTMatrixLiteral {
            kind,
            log_slots,
            levels,
            format: DFTFormat::Standard,
            scaling: Some(1.0),
            bit_reversed,
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
            let mut split = literal(kind, 4, vec![1, 1, 1, 1], false);
            split.format = DFTFormat::SplitRealAndImag;
            split.scaling = None;
            let mut repack = split.clone();
            repack.format = DFTFormat::RepackImagAsReal;

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
        let (log_n, log_slots, slots, dslots) = (6usize, 2usize, 4usize, 8usize);
        let mk = |kind| DFTMatrixLiteral {
            kind,
            log_slots,
            levels: vec![1, 1],
            format: DFTFormat::RepackImagAsReal,
            scaling: None,
            bit_reversed: false,
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
            &DFTMatrixLiteral {
                kind: DFTType::Encode,
                log_slots: 4,
                levels: vec![1, 1, 1, 1],
                format: DFTFormat::RepackImagAsReal,
                scaling: None,
                bit_reversed: false,
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
                    let enc = gen_dft_matrices::<f64>(&literal(DFTType::Encode, log_slots, levels.clone(), bit_reversed), log_slots + 1);
                    let dec = gen_dft_matrices::<f64>(&literal(DFTType::Decode, log_slots, levels.clone(), bit_reversed), log_slots + 1);

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
        let log_slots = 5;
        let enc = gen_dft_matrices::<f64>(&literal(DFTType::Encode, log_slots, vec![2, 2, 1], false), log_slots + 1);
        assert_eq!(enc.len(), 5, "sum(levels) factors");
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
