//! Complex slot-matrix linear map for CKKS, stored as the real and imaginary
//! parts of its diagonals.
//!
//! CKKS slots are complex; the canonical-embedding encoder takes the slot
//! vector as a pair `(re, im)` of real slices. To represent a complex matrix
//! `M = M_re + i·M_im` we keep two scheme-agnostic real diagonal maps
//! [`Diagonals<T>`](poulpy_core::layouts::Diagonals): one for the real parts
//! of the diagonals and one for the imaginary parts. A diagonal index is
//! "present" if either side carries it; the missing side is treated as zero.
//!
//! [`ComplexDiagonals`] implements [`Evaluate`] for the complex slot domain
//! `(re_in, im_in) -> (re_out, im_out)` using the standard
//! `(a + ib)(c + id) = (ac − bd) + i(ad + bc)` expansion (the user's
//! "split re/im + 4-term product" convention), so the plaintext reference
//! matches the homomorphic evaluation slot-by-slot. The companion
//! [`ComplexDiagonals::build_transform`] performs the BSGS pre-rotation +
//! giant-step bucketing in scheme-agnostic code; only the per-diagonal encode
//! into a CKKS plaintext is supplied by the scheme layer.

use std::collections::BTreeSet;

use poulpy_core::layouts::{
    DiagonalArithmetic, Diagonals, Evaluate, GLWELinearTransform, GLWELinearTransformDiagonal, GLWELinearTransformGiantStep,
    LinearTransformationStrategy, linear_transform_index, rotate_slots_into,
};

/// A complex slot-matrix linear map for CKKS.
///
/// Both real and imaginary diagonal maps share the same `rows`/`cols` packing.
#[derive(Clone, Debug)]
pub struct ComplexDiagonals<T> {
    /// Real parts of the diagonals.
    pub re: Diagonals<T>,
    /// Imaginary parts of the diagonals.
    pub im: Diagonals<T>,
}

impl<T> ComplexDiagonals<T> {
    /// Wraps a real/imaginary pair, asserting matching slot count.
    pub fn new(re: Diagonals<T>, im: Diagonals<T>) -> Self {
        assert_eq!(re.slots(), im.slots(), "complex diagonals slot count mismatch");
        Self { re, im }
    }

    /// Slot vector length shared by the real and imaginary diagonal maps.
    pub fn slots(&self) -> usize {
        self.re.slots()
    }

    /// Union of the real and imaginary non-zero diagonal indexes (sorted,
    /// normalized to `[0, slots)`).
    pub fn indexes(&self) -> Vec<i64> {
        let mut set: BTreeSet<i64> = self.re.indexes().into_iter().collect();
        set.extend(self.im.indexes());
        set.into_iter().collect()
    }

    /// Transposes the underlying complex matrix in place.
    ///
    /// Since `(Bre + i·Bim)ᵀ = (Bre)ᵀ + i·(Bim)ᵀ` (transpose is entry-wise on
    /// the complex matrix), this delegates to [`Diagonals::transpose`] on each
    /// part. After the call, [`Self::evaluate`] computes `Bᵀ·v = v·B` instead
    /// of `B·v`, with no allocation of new diagonal vectors.
    pub fn transpose(&mut self) {
        self.re.transpose();
        self.im.transpose();
    }
}

impl<'a, T: DiagonalArithmetic> Evaluate<(&'a [T], &'a [T]), (Vec<T>, Vec<T>)> for ComplexDiagonals<T> {
    /// Evaluates the complex `M·(u_re + i·u_im)` in the clear, returning
    /// `(re_out, im_out)`, by decomposing into four real matrix-vector products
    /// on the underlying [`Diagonals`]:
    ///
    /// ```text
    /// (Mre + i·Mim)·(u_re + i·u_im)
    ///     = (Mre·u_re − Mim·u_im) + i·(Mre·u_im + Mim·u_re)
    /// ```
    ///
    /// Each of the four terms is one `Diagonals::evaluate(.., strategy)` call,
    /// keeping the complex math at the scheme layer and the raw 1D matrix-vector
    /// product in the generic core.
    fn evaluate(&self, (re_in, im_in): (&'a [T], &'a [T]), strategy: LinearTransformationStrategy) -> (Vec<T>, Vec<T>) {
        let slots = self.slots();
        assert_eq!(re_in.len(), slots, "re input length must equal slots");
        assert_eq!(im_in.len(), slots, "im input length must equal slots");

        let mut out_re = self.re.evaluate(re_in, strategy); // Mre · u_re
        let mim_uim = self.im.evaluate(im_in, strategy); //    Mim · u_im
        for (acc, term) in out_re.iter_mut().zip(&mim_uim) {
            acc.sub_assign(term);
        }
        let mut out_im = self.re.evaluate(im_in, strategy); // Mre · u_im
        let mim_ure = self.im.evaluate(re_in, strategy); //    Mim · u_re
        for (acc, term) in out_im.iter_mut().zip(&mim_ure) {
            acc.add_assign(term);
        }
        (out_re, out_im)
    }
}

impl<T: DiagonalArithmetic> ComplexDiagonals<T> {
    /// Builds the unprepared [`GLWELinearTransform`] for this complex map.
    ///
    /// Performs the BSGS pre-rotation `ũ_{j,k} = rot(diag_{n1·j+k}, −n1·j)` and
    /// the giant-step bucketing; `encode` only turns a pre-rotated `(re, im)`
    /// diagonal (each `slots` long) into the scheme's encoded plaintext `P`
    /// (e.g. via `encode_reim`).
    pub fn build_transform<P>(
        &self,
        strategy: LinearTransformationStrategy,
        mut encode: impl FnMut(&[T], &[T]) -> P,
    ) -> GLWELinearTransform<P> {
        let slots = self.slots();
        let index = linear_transform_index(self.indexes(), slots, strategy);

        let mut pre_re = vec![T::zero(); slots];
        let mut pre_im = vec![T::zero(); slots];
        let zero_block = vec![T::zero(); slots];

        let mut giant_steps = Vec::with_capacity(index.giant_steps.len());
        for (g, &giant_rot) in index.giant_steps.iter().enumerate() {
            let mut diagonals = Vec::with_capacity(index.index[g].len());
            for &baby in &index.index[g] {
                let d = giant_rot + baby;
                let dre = self.re.get(d).map_or(&zero_block[..], |v| v.as_slice());
                let dim = self.im.get(d).map_or(&zero_block[..], |v| v.as_slice());
                rotate_slots_into(dre, -giant_rot, &mut pre_re);
                rotate_slots_into(dim, -giant_rot, &mut pre_im);
                let plaintext = encode(&pre_re, &pre_im);
                diagonals.push(GLWELinearTransformDiagonal { baby, plaintext });
            }
            giant_steps.push(GLWELinearTransformGiantStep {
                rot: giant_rot,
                diagonals,
            });
        }

        GLWELinearTransform {
            baby_steps: index.baby_steps,
            giant_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn complex_evaluate_matches_matvec() {
        let mre = vec![
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0, 1.0],
            vec![2.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 4.0],
        ];
        let mim = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 2.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let vre = vec![1.0, -1.0, 2.0, 0.5];
        let vim = vec![0.5, 2.0, -1.0, 1.0];

        let s = 4;
        let mut want_re = vec![0.0; s];
        let mut want_im = vec![0.0; s];
        for j in 0..s {
            for k in 0..s {
                want_re[j] += mre[j][k] * vre[k] - mim[j][k] * vim[k];
                want_im[j] += mre[j][k] * vim[k] + mim[j][k] * vre[k];
            }
        }

        let cd = ComplexDiagonals::new(diagonals_of(&mre), diagonals_of(&mim));
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
            LinearTransformationStrategy::Auto,
        ] {
            let (got_re, got_im) = cd.evaluate((vre.as_slice(), vim.as_slice()), strategy);
            for j in 0..s {
                assert!((got_re[j] - want_re[j]).abs() < 1e-9, "{strategy:?} re");
                assert!((got_im[j] - want_im[j]).abs() < 1e-9, "{strategy:?} im");
            }
        }
    }

    #[test]
    fn transpose_evaluate_matches_matvec_transpose() {
        // Same complex matrix as above; verify cd.transpose() then evaluate(v) == (Mᵀ·v) == (v·M).
        let mre = vec![
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0, 1.0],
            vec![2.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 4.0],
        ];
        let mim = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 2.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let vre = vec![1.0, -1.0, 2.0, 0.5];
        let vim = vec![0.5, 2.0, -1.0, 1.0];

        let s = 4;
        // (Mᵀ·v)[j] = Σ_k M[k][j] · v[k]  (i.e. v·M with v as a row vector).
        let mut want_re = vec![0.0; s];
        let mut want_im = vec![0.0; s];
        for j in 0..s {
            for k in 0..s {
                want_re[j] += mre[k][j] * vre[k] - mim[k][j] * vim[k];
                want_im[j] += mre[k][j] * vim[k] + mim[k][j] * vre[k];
            }
        }

        let cd = ComplexDiagonals::new(diagonals_of(&mre), diagonals_of(&mim));
        let mut cdt = cd;
        cdt.transpose();
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
            LinearTransformationStrategy::Auto,
        ] {
            let (got_re, got_im) = cdt.evaluate((vre.as_slice(), vim.as_slice()), strategy);
            for j in 0..s {
                assert!((got_re[j] - want_re[j]).abs() < 1e-9, "{strategy:?} re");
                assert!((got_im[j] - want_im[j]).abs() < 1e-9, "{strategy:?} im");
            }
        }
    }
}
