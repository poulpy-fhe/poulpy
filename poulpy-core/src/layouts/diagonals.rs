//! Generic diagonal representation of a square slot-matrix linear map.
//!
//! A homomorphic linear transformation evaluates `w = M·v` where `M` is an
//! `s×s` matrix over the `s = cols` slots and `v` is the slot vector. `M` is
//! stored in *diagonal form*: only the non-zero generalized diagonals
//!
//! ```text
//! diag_i[j] = M[j][(j + i) mod s]
//! ```
//!
//! are kept, in a map `i -> diag_i`. This module provides:
//!
//! - [`Diagonals<T>`] — the diagonal map over a *real* scalar `T`
//!   (`f64`, `i64`, ...).
//! - [`Evaluate`] — a trait abstracting "a diagonal map that can be evaluated
//!   on a plaintext-domain input". Implemented here for [`Diagonals<T>`];
//!   schemes whose slot domain is complex (CKKS) implement it for their own
//!   complex wrapper (e.g. `poulpy_ckks::layouts::ComplexDiagonals`).
//! - [`DiagonalArithmetic`] — the per-scalar arithmetic used by `evaluate`.
//!
//! [`Diagonals<T>`] is strictly **one-dimensional** — it represents the matrix
//! of a single slot vector of length `slots`. Multi-dimensional packings (e.g.
//! SIMD-batched rows × cols, where the same transform is applied independently
//! to several `slots`-length blocks of one larger value vector) are
//! scheme-specific and belong in a thin wrapper above this type: such a
//! wrapper iterates over each block and calls [`Diagonals::evaluate`] on the
//! corresponding sub-slice. Keeping the core map 1D keeps the algebra clean
//! (`(B·a)[j] = Σ_k B[j][k]·a[k]`) and pushes packing concerns to the scheme
//! layer.
//!
//! A diagonal is stored as **one period** of its values: [`Diagonals::set`]
//! takes a vector whose length divides `slots`, and the diagonal reads
//! `diag_i[j] = v[j mod v.len()]`. A full-length vector is a dense diagonal;
//! a shorter one is a periodic diagonal stored once, which a scheme encodes on
//! that many slots, at a degree smaller by the factor `slots / v.len()`. Every
//! reader indexes modulo the stored length.

use std::collections::BTreeMap;

use super::{LinearTransformationLayout, LinearTransformationStrategy};

/// Element-wise arithmetic on the real scalar carried by [`Diagonals`].
///
/// Blanket-implemented for every numeric scalar that is `Clone` and supports
/// `Zero` / `-` / `*` (which covers `f64`, `i64`, extended-precision floats such
/// as `f128`, etc.), so the diagonal helpers are generic over the scheme's slot
/// scalar without any per-type wiring.
pub trait DiagonalArithmetic: Clone {
    /// The additive identity.
    fn zero() -> Self;
    /// `self += rhs`.
    fn add_assign(&mut self, rhs: &Self);
    /// `self -= rhs`.
    fn sub_assign(&mut self, rhs: &Self);
    /// Returns `self * rhs`.
    fn mul(&self, rhs: &Self) -> Self;
}

impl<T> DiagonalArithmetic for T
where
    T: Clone + num_traits::Zero + core::ops::Sub<Output = T> + core::ops::Mul<Output = T>,
{
    fn zero() -> Self {
        <T as num_traits::Zero>::zero()
    }
    fn add_assign(&mut self, rhs: &Self) {
        // `num_traits::Zero` implies `Add<Output = Self>`.
        *self = self.clone() + rhs.clone();
    }
    fn sub_assign(&mut self, rhs: &Self) {
        *self = self.clone() - rhs.clone();
    }
    fn mul(&self, rhs: &Self) -> Self {
        self.clone() * rhs.clone()
    }
}

/// Plaintext-domain evaluator for "a diagonal map that represents a linear
/// transformation on slot vectors".
///
/// Abstracted as a trait so different flavors of diagonal map (real, complex,
/// integer, scheme-specific permutations) share the same call shape:
///
/// ```ignore
/// let out = d.evaluate(input, strategy);
/// ```
///
/// `In` / `Out` differ by flavor: real diagonals take `&[T]` → `Vec<T>`; the
/// CKKS complex wrapper takes `(&[T], &[T])` → `(Vec<T>, Vec<T>)` so the same
/// trait covers both.
pub trait Evaluate<In, Out> {
    /// Evaluates the linear transformation on `input`, following the BSGS
    /// `strategy` so the result matches the homomorphic engine bit-by-bit (up to
    /// scheme precision).
    fn evaluate(&self, input: In, strategy: LinearTransformationStrategy) -> Out;
}

/// The period of a slot-wise combination (sum or product) of two periodic
/// vectors: the least common multiple of their periods. It divides `slots`
/// whenever both periods do, and is their maximum when both are powers of
/// two.
pub fn period_lcm(a: usize, b: usize) -> usize {
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    a / gcd(a, b) * b
}

/// Left-rotates `src` by `k` slots into `out`: `out[j] = src[(j + k) mod src.len()]`.
///
/// Public utility matching the scheme slot-rotation convention
/// `rot(v, k)[j] = v[(j+k) mod n]`; downstream diagonal-map flavors (e.g. the
/// CKKS `ComplexDiagonals`) use it to apply baby / giant pre-rotations under
/// the same convention as the homomorphic engine. `src` is read as a periodic
/// vector, so `out` may be any multiple of its length: one stored period
/// rotates into any replication of itself.
pub fn rotate_slots_into<T: Clone>(src: &[T], k: i64, out: &mut [T]) {
    let n = src.len() as i64;
    debug_assert!(out.len().is_multiple_of(src.len()));
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = src[(j as i64 + k).rem_euclid(n) as usize].clone();
    }
}

/// The non-zero generalized diagonals of a square slot-matrix linear map.
///
/// One-dimensional: the matrix has `slots × slots` shape and every diagonal is a
/// length-`slots` vector, stored as one period of its values (see the module
/// docs). Keys are diagonal indexes normalized to `[0, slots)`.
/// See the module docs for the diagonal convention and for how schemes layer
/// multi-dimensional packings on top.
#[derive(Clone, Debug)]
pub struct Diagonals<T> {
    slots: usize,
    map: BTreeMap<i64, Vec<T>>,
}

impl<T> Diagonals<T> {
    /// Creates an empty map for a `slots × slots` matrix.
    pub fn new(slots: usize) -> Self {
        assert!(slots > 0, "diagonals require a non-zero slot count");
        Self {
            slots,
            map: BTreeMap::new(),
        }
    }

    /// Slot vector length (= the matrix dimension this map represents).
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Sets the diagonal at `index` (normalized modulo `slots`) from one period
    /// of its values: `values.len()` divides `slots` and the diagonal reads
    /// `diag[j] = values[j % values.len()]`. A full-length vector is a dense
    /// diagonal; a shorter one stores a periodic diagonal once.
    pub fn set(&mut self, index: i64, values: Vec<T>) {
        assert!(
            !values.is_empty() && self.slots.is_multiple_of(values.len()),
            "diagonal length ({}) must divide slots ({})",
            values.len(),
            self.slots,
        );
        self.map.insert(index.rem_euclid(self.slots as i64), values);
    }

    /// Returns one period of the diagonal at `index` (normalized modulo
    /// `slots`), if present: `diag[j] = v[j % v.len()]`.
    pub fn get(&self, index: i64) -> Option<&Vec<T>> {
        self.map.get(&index.rem_euclid(self.slots as i64))
    }

    /// The slot count after which the diagonal at `index` repeats, the length
    /// of its stored period; `None` when the diagonal is absent.
    pub fn period(&self, index: i64) -> Option<usize> {
        self.get(index).map(Vec::len)
    }

    /// The (normalized, sorted) indexes of the stored non-zero diagonals.
    pub fn indexes(&self) -> Vec<i64> {
        self.map.keys().copied().collect()
    }

    /// True when no diagonal is stored.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<T: DiagonalArithmetic> Diagonals<T> {
    /// The diagonals of the map whose output slot `i + q period` sums `w x[(j + q period) mod slots]` over the
    /// `terms` `(i, j, w)`, for every `q < slots / period`: a map that repeats under a common shift of its input and
    /// output by `period`, given by the terms of its first `period` outputs. Term `(i, j, w)` adds `w` at position `i`
    /// of diagonal `(j - i) mod slots`, so terms on one entry add up. Every diagonal a term reaches is stored over
    /// `period` slots, whatever its values.
    ///
    /// # Panics
    ///
    /// When `slots` is zero, `period` is zero or does not divide `slots`, or a term has `i >= period` or
    /// `j >= slots`.
    pub fn from_slot_map(slots: usize, period: usize, terms: impl IntoIterator<Item = (usize, usize, T)>) -> Self {
        let mut diagonals = Self::new(slots);
        assert!(
            period > 0 && slots.is_multiple_of(period),
            "slot map period must divide the slots"
        );
        for (i, j, w) in terms {
            assert!(i < period && j < slots, "slot map term out of range");
            let offset = (j as i64 - i as i64).rem_euclid(slots as i64);
            diagonals
                .map
                .entry(offset)
                .or_insert_with(|| vec![<T as DiagonalArithmetic>::zero(); period])[i]
                .add_assign(&w);
        }
        diagonals
    }
}

impl<T: Clone + PartialEq> Diagonals<T> {
    /// Stores every diagonal at the smallest period they all repeat with: the least common multiple of their smallest
    /// periods, the slot count an encoder encodes the map on.
    pub fn shrink_period(&mut self) {
        let period = self.map.values().map(|values| min_period(values)).fold(1, period_lcm);
        for values in self.map.values_mut() {
            let len = values.len();
            if len > period {
                values.truncate(period);
            } else if len < period {
                *values = (0..period).map(|t| values[t % len].clone()).collect();
            }
        }
    }
}

/// The smallest `p` dividing `v.len()` with `v[t] == v[t mod p]` for every `t`.
fn min_period<T: PartialEq>(v: &[T]) -> usize {
    let repeats = |p: usize| v.iter().enumerate().all(|(t, x)| *x == v[t % p]);
    let mut period = v.len();
    let (mut rest, mut f) = (v.len(), 2);
    // Periods dividing the length are closed under gcd, so dividing out each prime while the quotient still repeats
    // reaches the smallest.
    while rest > 1 {
        if f * f > rest {
            f = rest;
        }
        if rest.is_multiple_of(f) {
            while rest.is_multiple_of(f) {
                rest /= f;
            }
            while period.is_multiple_of(f) && repeats(period / f) {
                period /= f;
            }
        }
        f += 1;
    }
    period
}

impl<T> Diagonals<T> {
    /// Transposes the underlying matrix in place.
    ///
    /// Uses the identity `diag_i(Mᵀ) = rot(diag_(-i)(M), i)`: every
    /// `(j, diag_j(M))` entry is rebucketed to `((-j) mod slots,
    /// rot(diag_j(M), -j))`. The diagonal vectors are rotated in place via
    /// [`slice::rotate_left`] and the map's value storage is reused — no
    /// per-element clones, no extra `Vec<T>` allocations.
    ///
    /// After [`Self::transpose`], [`Self::evaluate`] on `self` computes
    /// `Mᵀ·v = v·M` (where `M` was the matrix before the call).
    ///
    /// A stored period rotates by the same amount modulo its length.
    pub fn transpose(&mut self) {
        let slots = self.slots as i64;
        let old = std::mem::take(&mut self.map);
        for (j, mut vec) in old {
            // In-place cyclic left rotation by (-j) mod slots, matching the
            // `rot(v, k)[i] = v[(i + k) mod n]` convention.
            let period = vec.len();
            vec.rotate_left((-j).rem_euclid(slots) as usize % period);
            self.map.insert((-j).rem_euclid(slots), vec);
        }
    }
}

impl<'a, T: DiagonalArithmetic> Evaluate<&'a [T], Vec<T>> for Diagonals<T> {
    /// Evaluates `M·input` in the clear (the raw `slots × slots` matrix-vector
    /// product), mirroring the homomorphic BSGS schedule (`strategy`) so the
    /// result matches a homomorphic evaluation of the same transform. `input`
    /// has length [`Self::slots`].
    fn evaluate(&self, input: &'a [T], strategy: LinearTransformationStrategy) -> Vec<T> {
        let slots = self.slots;
        assert_eq!(input.len(), slots, "input length must equal slots");
        let index = LinearTransformationLayout {
            indexes: self.indexes(),
            slots,
            strategy,
        }
        .index();

        let mut out = vec![T::zero(); slots];
        let mut buff = vec![T::zero(); slots];
        let mut rot_in = vec![T::zero(); slots];
        let mut rot_diag = vec![T::zero(); slots];

        for (g, &giant_rot) in index.giant_steps.iter().enumerate() {
            buff.iter_mut().for_each(|x| *x = T::zero());
            for &baby in &index.index[g] {
                let diag = self.get(giant_rot + baby).expect("schedule references a missing diagonal");
                rotate_slots_into(input, baby, &mut rot_in);
                rotate_slots_into(diag, -giant_rot, &mut rot_diag);
                for j in 0..slots {
                    let mut p = rot_in[j].clone();
                    p = p.mul(&rot_diag[j]);
                    buff[j].add_assign(&p);
                }
            }
            // Giant rotation, folded straight into the output.
            for j in 0..slots {
                out[j].add_assign(&buff[(j as i64 + giant_rot).rem_euclid(slots as i64) as usize]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (M·v)[j] = Σ_k M[j][k] v[k]
    fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }

    fn diagonals_of(m: &[Vec<f64>]) -> Diagonals<f64> {
        let s = m.len();
        let mut d = Diagonals::new(s);
        for i in 0..s {
            let col: Vec<f64> = (0..s).map(|j| m[j][(j + i) % s]).collect();
            if col.iter().any(|&x| x != 0.0) {
                d.set(i as i64, col);
            }
        }
        d
    }

    #[test]
    fn real_evaluate_matches_matvec() {
        let m = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let v = vec![1.0, -2.0, 0.5, 3.0];
        let want = matvec(&m, &v);
        let d = diagonals_of(&m);
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
        ] {
            let got = d.evaluate(&v, strategy);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-9, "{strategy:?}: {got:?} != {want:?}");
            }
        }
    }

    fn matvec_i64(m: &[Vec<i64>], v: &[i64]) -> Vec<i64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }

    /// The `slots x slots` matrix of [`Diagonals::from_slot_map`]'s terms, repeated under shifts by `period`.
    fn dense_of(slots: usize, period: usize, terms: &[(usize, usize, i64)]) -> Vec<Vec<i64>> {
        let mut m = vec![vec![0; slots]; slots];
        for q in 0..slots / period {
            for &(i, j, w) in terms {
                m[i + q * period][(j + q * period) % slots] += w;
            }
        }
        m
    }

    fn assert_evaluates(d: &Diagonals<i64>, m: &[Vec<i64>]) {
        let v: Vec<i64> = (0..m.len() as i64).map(|x| 3 * x - 7).collect();
        let want = matvec_i64(m, &v);
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
        ] {
            assert_eq!(d.evaluate(&v, strategy), want, "{strategy:?}");
        }
    }

    #[test]
    fn from_slot_map_matches_matvec() {
        // Two terms on entry (1, 5) add up, and the zero term stores diagonal 1 all the same.
        let terms = [(0, 3, 2), (1, 5, 4), (1, 5, -1), (2, 2, 7), (5, 0, 3), (7, 1, -2), (6, 7, 0)];
        let d = Diagonals::from_slot_map(8, 8, terms);
        assert_eq!(d.indexes(), vec![0, 1, 2, 3, 4]);
        assert!(d.indexes().iter().all(|&i| d.period(i) == Some(8)));
        assert_evaluates(&d, &dense_of(8, 8, &terms));
    }

    #[test]
    fn from_slot_map_repeats_over_the_period() {
        let terms = [(0, 1, 5), (1, 11, -3), (2, 2, 1), (3, 7, 2)];
        let d = Diagonals::from_slot_map(12, 4, terms);
        assert!(d.indexes().iter().all(|&i| d.period(i) == Some(4)));
        assert_evaluates(&d, &dense_of(12, 4, &terms));
    }

    #[test]
    fn shrink_period_stores_the_common_smallest_period() {
        let mut d = Diagonals::new(12);
        d.set(0, vec![1, 2, 3, 1, 2, 3]);
        d.set(1, vec![4, 5, 4, 5]);
        d.set(5, vec![6; 12]);
        let m: Vec<Vec<i64>> = (0..12)
            .map(|r| {
                (0..12)
                    .map(|c| d.get(c as i64 - r as i64).map_or(0, |v| v[r % v.len()]))
                    .collect()
            })
            .collect();
        d.shrink_period();
        assert_eq!(d.get(0), Some(&vec![1, 2, 3, 1, 2, 3]));
        assert_eq!(d.get(1), Some(&vec![4, 5, 4, 5, 4, 5]));
        assert_eq!(d.get(5), Some(&vec![6; 6]));
        assert_evaluates(&d, &m);

        let mut powers = Diagonals::from_slot_map(16, 16, (0..16).map(|i| (i, (i + 3) % 16, (i % 4) as i64)));
        powers.shrink_period();
        assert_eq!(powers.get(3), Some(&vec![0, 1, 2, 3]));

        let mut empty = Diagonals::<i64>::new(8);
        empty.shrink_period();
        assert!(empty.is_empty());
    }

    #[test]
    #[should_panic(expected = "slot map period must divide the slots")]
    fn from_slot_map_rejects_a_period_not_dividing_the_slots() {
        Diagonals::from_slot_map(12, 5, [(0, 0, 1i64)]);
    }

    #[test]
    #[should_panic(expected = "slot map period must divide the slots")]
    fn from_slot_map_rejects_a_zero_period() {
        Diagonals::from_slot_map(12, 0, [(0, 0, 1i64)]);
    }

    #[test]
    #[should_panic(expected = "slot map term out of range")]
    fn from_slot_map_rejects_an_output_past_the_period() {
        Diagonals::from_slot_map(12, 4, [(4, 0, 1i64)]);
    }

    #[test]
    #[should_panic(expected = "slot map term out of range")]
    fn from_slot_map_rejects_an_input_past_the_slots() {
        Diagonals::from_slot_map(12, 4, [(0, 12, 1i64)]);
    }

    #[test]
    fn transpose_evaluate_matches_matvec_transpose() {
        // After d.transpose(), d.evaluate(v) should equal Mᵀ·v == v·M.
        let m = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let v = vec![1.0, -2.0, 0.5, 3.0];
        let mt: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| m[j][i]).collect()).collect();
        let want = matvec(&mt, &v);
        let mut dt = diagonals_of(&m);
        dt.transpose();
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
        ] {
            let got = dt.evaluate(&v, strategy);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-9, "{strategy:?}: {got:?} != {want:?}");
            }
        }
    }
}
