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
//! by [`CKKSPlaintextVecHostCodec::encode_host_floats`](crate::layouts::CKKSPlaintextVecHostCodec);
//! the [`CKKSMeta::log_sparsity`](crate::CKKSMeta) field carries the packing factor.
//!
//! Conventions match the rest of the crate: a diagonal at index `i` is the vector
//! `diag_i[j] = M[j][(j+i) mod slots]` (see [`poulpy_core::layouts::Diagonals`]),
//! and the canonical embedding uses the Galois generator 5 (`pow5`), identical to
//! the backend CKKS encoding plans.

use num_traits::{Float, FloatConst};
use poulpy_core::layouts::Diagonals;

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
    debug_assert!(n >= 4 && n.is_power_of_two());
    let two = F::from(2.0).unwrap();
    let nf = F::from(n).unwrap();
    let step = two * F::PI() / nf;

    // `n == 4` has no complete octant; all of its roots are cardinal points.
    if n == 4 {
        return vec![
            Cpx::new(F::one(), F::zero()),
            Cpx::new(F::zero(), F::one()),
            Cpx::new(-F::one(), F::zero()),
            Cpx::new(F::zero(), -F::one()),
        ];
    }

    // Evaluate only [0, π/4], then derive the other seven octants with
    // exact sign changes and coordinate swaps. Unlike a recurrence, this adds
    // no rounding drift and preserves unit modulus to the sin_cos accuracy.
    let width = n >> 3;
    let octant: Vec<Cpx<F>> = (0..=width)
        .map(|k| {
            let angle = step * F::from(k).unwrap();
            let (sin, cos) = angle.sin_cos();
            Cpx::new(cos, sin)
        })
        .collect();
    let mut roots = Vec::with_capacity(n);
    for k in 0..n {
        let sector = k / width;
        let offset = k % width;
        let direct = octant[offset];
        let reflected = octant[width - offset];
        roots.push(match sector {
            0 => direct,
            1 => Cpx::new(reflected.im, reflected.re),
            2 => Cpx::new(-direct.im, direct.re),
            3 => Cpx::new(-reflected.re, reflected.im),
            4 => Cpx::new(-direct.re, -direct.im),
            5 => Cpx::new(-reflected.im, -reflected.re),
            6 => Cpx::new(direct.im, -direct.re),
            7 => Cpx::new(reflected.re, -reflected.im),
            _ => unreachable!(),
        });
    }

    // Pin the axes to their exact representations, including positive zero.
    roots[0] = Cpx::new(F::one(), F::zero());
    roots[n >> 2] = Cpx::new(F::zero(), F::one());
    roots[n >> 1] = Cpx::new(-F::one(), F::zero());
    roots[3 * (n >> 2)] = Cpx::new(F::zero(), -F::one());
    roots
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
    // Rotation geometry is single-sourced with the plan's index replay
    // (`DFTPlan::diagonal_indexes`), so key provisioning cannot drift from the
    // generated diagonals.
    let rot = crate::layouts::dft::dft_layer_rotation(kind, bit_reversed, next_level, log_l, mask);
    let a = maybe_bit_reverse(a, log_l, bit_reversed);
    let b = maybe_bit_reverse(b, log_l, bit_reversed);
    let c = maybe_bit_reverse(c, log_l, bit_reversed);

    let mut new_vec = empty_cd(dslots);
    for i in vec.indexes() {
        let vi = cd_get(vec, i, dslots);
        let [d_same, d_plus, d_minus] = crate::layouts::dft::dft_layer_spread(i, rot, mask);
        cd_accumulate(&mut new_vec, d_same, &rotate_and_mul(&vi, 0, &a));
        cd_accumulate(&mut new_vec, d_plus, &rotate_and_mul(&vi, rot, &b));
        cd_accumulate(&mut new_vec, d_minus, &rotate_and_mul(&vi, -rot, &c));
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

/// Merges the next `m` butterfly layers (from `fft_level`, descending) into a
/// starting `factor` — the shared inner loop of [`gen_dft_matrices`] (where
/// `merge_n`/`out_width` are the possibly-doubled sparse-repack slot counts)
/// and [`gen_dft_matrices_blockwise`] (where both are the tile width). Keeping
/// one copy guarantees the dense and blockwise generators compose their factor
/// matrices from bit-identical layer merges.
#[allow(clippy::too_many_arguments)]
fn merge_factor_layers<F: DftScalar>(
    mut factor: ComplexDiagonals<F>,
    m: usize,
    fft_level: usize,
    kind: DFTType,
    log_slots: usize,
    merge_n: usize,
    out_width: usize,
    bit_reversed: bool,
    roots: &[Cpx<F>],
    pow5: &[usize],
) -> ComplexDiagonals<F> {
    let mut next_level = fft_level;
    // Merge the factor's `m` butterfly layers one at a time; each layer's
    // coefficient buffers are dropped at the end of its iteration.
    for _ in 0..m {
        let (a_l, b_l, c_l) = plain_layer(kind, log_slots, roots, pow5, log_slots - next_level);
        factor = merge_next_layer(
            &factor,
            log_slots,
            merge_n,
            next_level,
            &a_l,
            &b_l,
            &c_l,
            kind,
            bit_reversed,
            out_width,
        );
        next_level -= 1;
    }
    factor
}

/// Generates the ordered factor matrices of the homomorphic (I)DFT described by
/// `literal`, in evaluation order.
///
/// `log_n` is the ring degree exponent; `log_max_slots = log_n − 1`. When
/// `log_slots < log_max_slots` and the format is `RepackImagAsReal`, the **sparse
/// repack** path is taken: `dslots = 2·slots` butterflies, the repack matrix
/// prepended to the first Decode matrix, and the right half of the last Encode
/// matrix zeroed. Otherwise the dense path is used (and full-packing
/// `RepackImagAsReal` ≡ `SplitRealAndImag`). Shape validity is guaranteed by
/// [`DFTPlan::new`].
pub fn gen_dft_matrices<F: DftScalar>(literal: &DFTPlan, log_n: usize) -> Vec<ComplexDiagonals<F>> {
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

    // The schedule is consumed in evaluation order (factor 0 applied first; see
    // the convention on [`DFTPlan`]). No implicit reordering by `kind`: a Decode
    // that inverts an Encode is the same schedule reversed, which is the
    // caller's responsibility.
    let mut plain_vector: Vec<ComplexDiagonals<F>> = Vec::with_capacity(max_depth);
    let mut fft_level = log_slots;
    for (i, m) in literal.schedule.steps.iter().map(|s| s.depth).enumerate() {
        let repack_first = sparse && imag_repack && kind == DFTType::Decode && i == 0;
        // Sparse-repack merges wrap rotation indices mod `2·slots`; otherwise mod `slots`.
        let merge_n = if repack_first { slots << 1 } else { slots };

        // Start the factor from the repack matrix (sparse Decode, first factor)
        // or the slot-identity; every butterfly layer of the factor is then
        // merged in uniformly via `merge_factor_layers`.
        let start = if repack_first {
            gen_repack_matrix(log_slots, dslots)
        } else {
            identity_diag(dslots)
        };
        let factor = merge_factor_layers(
            start,
            m,
            fft_level,
            kind,
            log_slots,
            merge_n,
            dslots,
            bit_reversed,
            &roots,
            &pow5,
        );

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

/// Generates the ordered factor matrices of a `2^log_slots`-point (I)DFT whose
/// butterflies act **block-locally** on each `2^log_slots`-wide block of a
/// larger `2^log_tile`-slot vector (block-diagonal replication across
/// `2^(log_tile − log_slots)` identical small transforms).
///
/// Unlike [`gen_dft_matrices`], whose factor diagonals are canonicalized modulo
/// `slots` (exact only against a `slots`-periodic working vector), the
/// diagonals here are canonicalized modulo the tile, so `+rot` and `−rot`
/// stay distinct and the factors are exact when applied (tiled) to a
/// tile-periodic slot vector holding **different** data in each block. The
/// replication is sound because a single butterfly layer never reads across
/// its own `m`-aligned group (the `b`/`c` coefficient supports pair each
/// position with its in-group partner at distance `m/2`), so the composed
/// factor never crosses a block boundary. With `log_tile == log_slots` this
/// degenerates to the dense [`gen_dft_matrices`] (`Standard` format, natural
/// order).
///
/// `factorization_depth` is consumed in evaluation order (factor 0 applied
/// first) and must sum to `log_slots`; `scaling` accumulates across the
/// factors, with the `1/slots` (I)DFT normalization folded in for `Encode`.
/// `bit_reversed` selects which side of the small transform is bit-reversed
/// (same convention as [`DFTPlan::bit_reversed`]): with `false` the
/// coefficient side is bit-reversed and the point side natural (the encoder
/// convention); with `true` the coefficient side is natural and the points
/// come in bit-reversed order — for a Decode/Encode pair generated with the
/// same flag the round-trip is the identity on positions either way.
pub fn gen_dft_matrices_blockwise<F: DftScalar>(
    kind: DFTType,
    log_slots: usize,
    log_tile: usize,
    factorization_depth: &[usize],
    scaling: f64,
    bit_reversed: bool,
) -> Vec<ComplexDiagonals<F>> {
    assert!(
        log_tile >= log_slots,
        "log_tile {log_tile} must be at least log_slots {log_slots}"
    );
    assert_eq!(
        factorization_depth.iter().sum::<usize>(),
        log_slots,
        "factorization_depth must sum to log_slots"
    );
    let slots = 1usize << log_slots;
    let tile = 1usize << log_tile;

    let roots = roots_of_unity::<F>(slots << 2);
    let pow5 = pow5_table(slots);

    let mut factors: Vec<ComplexDiagonals<F>> = Vec::with_capacity(factorization_depth.len());
    let mut fft_level = log_slots;
    for &m in factorization_depth {
        let factor = merge_factor_layers(
            identity_diag(tile),
            m,
            fft_level,
            kind,
            log_slots,
            tile,
            tile,
            bit_reversed,
            &roots,
            &pow5,
        );
        factors.push(factor);
        fft_level -= m;
    }

    // The caller scaling is spread uniformly (root computed natively in `F`,
    // see `nth_root_scalar`); the Encode 1/slots normalization is distributed
    // dyadically (2^−m per factor of m merged layers), keeping every
    // per-factor norm division exact — the factor values stay as accurate as
    // the roots themselves.
    let per_factor_caller = nth_root_scalar(F::from(scaling).unwrap(), factorization_depth.len());
    for (factor, &m) in factors.iter_mut().zip(factorization_depth) {
        let norm = if kind == DFTType::Encode { (1u64 << m) as f64 } else { 1.0 };
        scale_complex_diagonals(factor, per_factor_caller / F::from(norm).unwrap());
    }
    factors
}

/// `s^(1/depth)` computed natively in `F` (for positive finite `s`): an `f64`
/// seed refined by two Newton–Raphson steps in `F`
/// (`x ← x − (x·x^{d−1} − s)/(d·x^{d−1})`), converging quadratically
/// (53 → 106 → ≥113 correct bits, capped at `F`'s precision) using only `F`
/// multiply/divide — no reliance on an `F`-native `powf`, so the root is
/// mantissa-accurate for `F = Quad` on every configuration.
pub(crate) fn nth_root_scalar<F: DftScalar>(s: F, depth: usize) -> F {
    debug_assert!(depth >= 1, "nth_root_scalar: depth must be >= 1");
    if depth == 1 {
        return s;
    }
    let seed = s.to_f64().expect("finite scaling").powf(1.0 / depth as f64);
    let d = F::from(depth).unwrap();
    let mut x = F::from(seed).unwrap();
    for _ in 0..2 {
        let x_dm1 = x.powi(depth as i32 - 1);
        x = x - (x * x_dm1 - s) / (d * x_dm1);
    }
    x
}

/// Applies the DFT `1/N` normalization (Encode only) and the caller `scaling`,
/// spread as the `depth`-th root across all factor matrices. The ratio and its
/// root are computed natively in `F` ([`nth_root_scalar`]), so the per-factor
/// scale carries `F`'s full precision (the plan `scaling` itself is an `f64`
/// input; typical values — `1/K`, `2^log_msg_ratio` — are exactly
/// representable).
fn apply_scaling<F: DftScalar>(factors: &mut [ComplexDiagonals<F>], literal: &DFTPlan) {
    let slots = 1usize << literal.log_slots();
    let depth = literal.num_factors();

    let mut scaling = F::from(literal.scaling.unwrap_or(1.0)).unwrap();
    if literal.kind == DFTType::Encode {
        // Real/imag extraction carries an extra 1/2 factor. The denominator is
        // a power of two, so the division is exact in `F`.
        let denom = match literal.format {
            DFTOutputFormat::Standard => slots as f64,
            DFTOutputFormat::SplitRealAndImag | DFTOutputFormat::RepackImagAsReal => 2.0 * slots as f64,
        };
        scaling = scaling / F::from(denom).unwrap();
    }
    // Spread across the matrices so the product accumulates to `scaling`.
    let per_factor = nth_root_scalar(scaling, depth);

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

    /// Coefficient meta is irrelevant for clear-text factor generation.
    fn zero_meta() -> crate::CoeffsMeta {
        crate::CoeffsMeta::from_delta_budget(0, 0)
    }

    /// The per-factor scale root must be `F`-native: at `F = Quad` the residual
    /// `root^depth − s` must be quad-small (~2^−110 relative), far below what an
    /// f64-computed root (~2^−53) could achieve; at `F = f64` it stays
    /// f64-exact.
    #[test]
    fn nth_root_scalar_is_scalar_native() {
        use crate::Quad;
        use num_traits::{FromPrimitive, ToPrimitive};

        for &(s, depth) in &[(0.0625f64, 4usize), (1.0 / 3.0, 5), (2048.0, 3), (1e-9, 7), (1.0, 1)] {
            // f64: reconstruction error within a few ulps.
            let r64 = nth_root_scalar(s, depth);
            let rel64 = ((r64.powi(depth as i32) - s) / s).abs();
            assert!(rel64 < 1e-14, "f64 root off: s={s} depth={depth} rel={rel64:e}");

            // Quad: reconstruction error far below f64 precision.
            let sq = Quad::from_f64(s).unwrap();
            let rq = nth_root_scalar(sq, depth);
            let relq = ((rq.powi(depth as i32) - sq) / sq).abs().to_f64().unwrap();
            assert!(relq < 1e-30, "Quad root not quad-precise: s={s} depth={depth} rel={relq:e}");

            // And it genuinely carries sub-f64 mantissa: rounding the Quad root
            // to f64 must reproduce the f64 root (same value class)…
            let rq_f64 = rq.to_f64().unwrap();
            assert!(
                ((rq_f64 - r64) / r64).abs() < 1e-15,
                "Quad and f64 roots diverge: s={s} depth={depth}"
            );
            // …while the exact-root cases aside (depth 1, exact powers), the
            // Quad root minus its f64 rounding is a genuine low-order residue.
            if depth > 1 && s != 0.0625 {
                let residue = (rq - Quad::from_f64(rq_f64).unwrap()).abs().to_f64().unwrap();
                assert!(residue > 0.0, "Quad root carries no sub-f64 mantissa: s={s} depth={depth}");
            }
        }
    }

    #[test]
    fn symmetric_roots_add_no_recurrence_drift() {
        use crate::Quad;
        use num_traits::{Float, FromPrimitive, ToPrimitive};

        let n = 131072usize;
        let roots = roots_of_unity::<f64>(n);
        assert_eq!(roots[0], Cpx::new(1.0, 0.0));
        assert_eq!(roots[n / 4], Cpx::new(0.0, 1.0));
        assert_eq!(roots[n / 2], Cpx::new(-1.0, 0.0));
        assert_eq!(roots[3 * n / 4], Cpx::new(0.0, -1.0));

        let mut max_modulus_error = 0.0f64;
        for root in &roots {
            max_modulus_error = max_modulus_error.max((root.re.hypot(root.im) - 1.0).abs());
        }
        assert!(max_modulus_error <= f64::EPSILON, "modulus error: {max_modulus_error:e}");

        // Every derived octant is a bit-exact sign change / coordinate swap of
        // the first. This is platform independent and admits no drift budget.
        let width = n / 8;
        let assert_f64_bits = |got: Cpx<f64>, expected: Cpx<f64>| {
            assert_eq!(got.re.to_bits(), expected.re.to_bits());
            assert_eq!(got.im.to_bits(), expected.im.to_bits());
        };
        for offset in 1..width {
            let base = roots[offset];
            assert_f64_bits(roots[2 * width - offset], Cpx::new(base.im, base.re));
            assert_f64_bits(roots[2 * width + offset], Cpx::new(-base.im, base.re));
            assert_f64_bits(roots[4 * width - offset], Cpx::new(-base.re, base.im));
            assert_f64_bits(roots[4 * width + offset], Cpx::new(-base.re, -base.im));
            assert_f64_bits(roots[6 * width - offset], Cpx::new(-base.im, -base.re));
            assert_f64_bits(roots[6 * width + offset], Cpx::new(base.im, -base.re));
            assert_f64_bits(roots[8 * width - offset], Cpx::new(base.re, -base.im));
        }

        // Quad has the same exact symmetry and retains binary128 unit modulus.
        let roots = roots_of_unity::<Quad>(n);
        assert_eq!(roots[0].re.to_bits(), Quad::from_f64(1.0).unwrap().to_bits());
        assert_eq!(roots[0].im.to_bits(), Quad::from_f64(0.0).unwrap().to_bits());
        assert_eq!(roots[n / 4].re.to_bits(), Quad::from_f64(0.0).unwrap().to_bits());
        assert_eq!(roots[n / 4].im.to_bits(), Quad::from_f64(1.0).unwrap().to_bits());
        for offset in (1..width).step_by(257) {
            let base = roots[offset];
            assert_eq!(roots[2 * width - offset].re.to_bits(), base.im.to_bits());
            assert_eq!(roots[2 * width - offset].im.to_bits(), base.re.to_bits());
            assert_eq!(roots[6 * width + offset].re.to_bits(), base.im.to_bits());
            assert_eq!(roots[6 * width + offset].im.to_bits(), (-base.re).to_bits());
        }
        for root in roots.iter().step_by(2039) {
            let modulus_error = (root.re * root.re + root.im * root.im - Quad::from_f64(1.0).unwrap()).abs();
            assert!(modulus_error.to_f64().unwrap() <= 5e-34);
        }
    }

    fn literal(kind: DFTType, factorization_depth: Vec<usize>, bit_reversed: bool) -> DFTPlan {
        let schedule: Vec<(usize, usize)> = factorization_depth.into_iter().map(|d| (d, 1)).collect();
        DFTPlan::new(kind, schedule, DFTOutputFormat::Standard, zero_meta())
            .unwrap()
            .with_scaling(1.0)
            .unwrap()
            .with_bit_reversed(bit_reversed)
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

    /// The plan's value-free index replay ([`DFTPlan::diagonal_indexes`]) must
    /// name exactly the diagonals the generator produces, for every kind,
    /// schedule, bit-reversal, and packing — this is what key provisioning
    /// ([`DFTPlan::galois_elements`]) is derived from, so a mismatch silently
    /// under-provisions rotation keys. The rotation geometry itself is
    /// single-sourced (`dft_layer_rotation`/`dft_layer_spread`); this pins the
    /// remaining composition (sparse-repack seeding, repack widening, zero
    /// dropping) end to end.
    #[test]
    fn plan_diagonal_indexes_match_generated_factors() {
        let schedules: &[Vec<usize>] = &[vec![1, 1, 1, 1], vec![2, 2], vec![4], vec![1, 3], vec![3, 1], vec![2, 1, 1]];
        for kind in [DFTType::Encode, DFTType::Decode] {
            for bit_reversed in [false, true] {
                for schedule in schedules {
                    for (log_n, format) in [
                        // Dense: log_slots = 4 = log_n − 1.
                        (5usize, DFTOutputFormat::SplitRealAndImag),
                        // Sparse repack: log_slots = 4 < log_n − 1.
                        (7usize, DFTOutputFormat::RepackImagAsReal),
                    ] {
                        let mut plan = literal(kind, schedule.clone(), bit_reversed);
                        plan.format = format;
                        plan.scaling = None;
                        let factors = gen_dft_matrices::<f64>(&plan, log_n);
                        let replayed = plan.diagonal_indexes(log_n);
                        let generated: Vec<Vec<i64>> = factors.iter().map(|f| f.indexes()).collect();
                        assert_eq!(
                            replayed, generated,
                            "index replay diverges from generator: kind={kind:?} bit_reversed={bit_reversed} schedule={schedule:?} log_n={log_n} format={format:?}"
                        );
                    }
                }
            }
        }
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
        let mk = |kind| {
            // vec![(1, 1); 2]: sum of depths = log_slots = 2.
            DFTPlan::new(kind, vec![(1, 1); 2], DFTOutputFormat::RepackImagAsReal, zero_meta()).unwrap()
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
            // vec![(1, 1); 4]: sum of depths = log_slots = 4.
            &DFTPlan::new(
                DFTType::Encode,
                vec![(1, 1); 4],
                DFTOutputFormat::RepackImagAsReal,
                zero_meta(),
            )
            .unwrap(),
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
                    // (evaluation-order convention; see the `DFTPlan` docs).
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

    /// `log_tile == log_slots` degenerates the blockwise generator to the dense
    /// one (Standard format): same factor count and diagonal indexes, identical
    /// chain product. (Per-factor values may differ: the blockwise generator
    /// distributes the Encode 1/slots normalization dyadically instead of as a
    /// uniform `depth`-th root.)
    #[test]
    fn blockwise_tile_equals_dense() {
        let log_slots = 3usize;
        let slots = 1usize << log_slots;
        let re: Vec<f64> = (0..slots).map(|j| (0.29 * (j as f64 + 1.0)).sin()).collect();
        let im: Vec<f64> = (0..slots).map(|j| (0.41 * (j as f64 + 2.0)).cos()).collect();
        for kind in [DFTType::Encode, DFTType::Decode] {
            for bit_reversed in [false, true] {
                for depth in schedules(log_slots) {
                    let dense = gen_dft_matrices::<f64>(&literal(kind, depth.clone(), bit_reversed), log_slots + 1);
                    let block = gen_dft_matrices_blockwise::<f64>(kind, log_slots, log_slots, &depth, 1.0, bit_reversed);
                    assert_eq!(dense.len(), block.len(), "{kind:?} {depth:?}");
                    for (f, (a, b)) in dense.iter().zip(&block).enumerate() {
                        assert_eq!(a.indexes(), b.indexes(), "{kind:?} br={bit_reversed} {depth:?} factor {f}");
                    }
                    let (dr, di) = eval_chain(&dense, re.clone(), im.clone());
                    let (br, bi) = eval_chain(&block, re.clone(), im.clone());
                    for j in 0..slots {
                        assert!(
                            (dr[j] - br[j]).abs() < 1e-12 && (di[j] - bi[j]).abs() < 1e-12,
                            "{kind:?} br={bit_reversed} {depth:?} slot {j}"
                        );
                    }
                }
            }
        }
    }

    /// Blockwise factors act block-locally on a tile holding *different* data
    /// per block: dense per-block Decode (coefficients → evaluations) followed
    /// by the blockwise Encode chain over the whole tile returns every block's
    /// coefficients in natural order.
    #[test]
    fn blockwise_inverts_per_block() {
        let (log_slots, log_tile) = (3usize, 5usize);
        let (slots, tile) = (1usize << log_slots, 1usize << log_tile);
        let re: Vec<f64> = (0..tile).map(|j| (0.37 * (j as f64 + 1.0)).sin()).collect();
        let im: Vec<f64> = (0..tile).map(|j| (0.53 * (j as f64 + 2.0)).cos()).collect();

        // Per-block dense Decode, one fully-merged factor (bit_reversed = true:
        // natural coefficient side, the convention PaCo uses).
        let dec = gen_dft_matrices::<f64>(&literal(DFTType::Decode, vec![log_slots], true), log_slots + 1);
        let mut ev_re = vec![0.0; tile];
        let mut ev_im = vec![0.0; tile];
        for b in 0..tile / slots {
            let range = b * slots..(b + 1) * slots;
            let (r, i) = eval_chain(&dec, re[range.clone()].to_vec(), im[range.clone()].to_vec());
            ev_re[range.clone()].copy_from_slice(&r);
            ev_im[range].copy_from_slice(&i);
        }

        // Blockwise Encode chain applied across the full tile at once.
        let enc = gen_dft_matrices_blockwise::<f64>(DFTType::Encode, log_slots, log_tile, &[1, 2], 1.0, true);
        let (rr, ri) = eval_chain(&enc, ev_re, ev_im);
        for j in 0..tile {
            assert!(
                (rr[j] - re[j]).abs() < 1e-9 && (ri[j] - im[j]).abs() < 1e-9,
                "slot {j}: got ({:.3e},{:.3e}) want ({:.3e},{:.3e})",
                rr[j],
                ri[j],
                re[j],
                im[j]
            );
        }
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
