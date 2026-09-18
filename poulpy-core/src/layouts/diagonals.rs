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

/// Left-rotates `src` by `k` slots into `out`: `out[j] = src[(j + k) mod len]`.
///
/// Public utility matching the scheme slot-rotation convention
/// `rot(v, k)[j] = v[(j+k) mod n]`; downstream diagonal-map flavors (e.g. the
/// CKKS `ComplexDiagonals`) use it to apply baby / giant pre-rotations under
/// the same convention as the homomorphic engine.
pub fn rotate_slots_into<T: Clone>(src: &[T], k: i64, out: &mut [T]) {
    let n = src.len() as i64;
    debug_assert_eq!(src.len(), out.len());
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = src[(j as i64 + k).rem_euclid(n) as usize].clone();
    }
}

/// The non-zero generalized diagonals of a square slot-matrix linear map.
///
/// One-dimensional: the matrix has `slots × slots` shape and every diagonal is a
/// length-`slots` vector. Keys are diagonal indexes normalized to `[0, slots)`.
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

    /// Sets the diagonal at `index` (normalized modulo `slots`). `values` must
    /// have length `slots`.
    pub fn set(&mut self, index: i64, values: Vec<T>) {
        assert_eq!(
            values.len(),
            self.slots,
            "diagonal length ({}) must equal slots ({})",
            values.len(),
            self.slots,
        );
        self.map.insert(index.rem_euclid(self.slots as i64), values);
    }

    /// Returns the diagonal at `index` (normalized modulo `slots`), if present.
    pub fn get(&self, index: i64) -> Option<&Vec<T>> {
        self.map.get(&index.rem_euclid(self.slots as i64))
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
    pub fn transpose(&mut self) {
        let slots = self.slots as i64;
        let old = std::mem::take(&mut self.map);
        for (j, mut vec) in old {
            // In-place cyclic left rotation by (-j) mod slots, matching the
            // `rot(v, k)[i] = v[(i + k) mod n]` convention.
            vec.rotate_left((-j).rem_euclid(slots) as usize);
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
