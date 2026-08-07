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
    DiagonalArithmetic, Diagonals, Evaluate, LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep,
    LinearTransformationLayout, LinearTransformationStrategy, rotate_slots_into,
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

impl<T: DiagonalArithmetic> Evaluate<(&[T], &[T]), (Vec<T>, Vec<T>)> for ComplexDiagonals<T> {
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
    fn evaluate(&self, (re_in, im_in): (&[T], &[T]), strategy: LinearTransformationStrategy) -> (Vec<T>, Vec<T>) {
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
    /// Entry-wise complex conjugation of the underlying matrix (negates the
    /// imaginary diagonals in place). Satisfies
    /// `conj(M·v) = conj(M)·conj(v)`.
    pub fn conjugate(&mut self) {
        for i in self.im.indexes() {
            let mut v = self.im.get(i).cloned().expect("indexed diagonal present");
            for x in v.iter_mut() {
                let mut neg = T::zero();
                neg.sub_assign(x);
                *x = neg;
            }
            self.im.set(i, v);
        }
    }

    /// Composition `self ∘ rhs` (apply `rhs` first): the diagonal form of the
    /// matrix product `Self · Rhs`.
    ///
    /// Both maps are read with the tiled convention (`out[j] = Σ_i
    /// diag_i[j mod slots] · v[j+i]`), so the two slot counts may differ as
    /// long as one divides the other; the result's slot count is the larger
    /// tile. Diagonal indexes add (`diag_{a+b}[j] += A_a[j] · B_b[j+a]`, all
    /// complex), so the output carries at most `|A|·|B|` diagonals, fewer when
    /// index sums collide.
    pub fn compose(&self, rhs: &Self) -> Self {
        let (ta, tb) = (self.slots() as i64, rhs.slots() as i64);
        assert!(
            ta % tb == 0 || tb % ta == 0,
            "compose requires nested tiles, got {ta} and {tb}"
        );
        let t = ta.max(tb);
        let mut out = ComplexDiagonals::new(Diagonals::new(t as usize), Diagonals::new(t as usize));
        let zeros = vec![T::zero(); t as usize];
        let read = |d: &Diagonals<T>, idx: i64, j: i64, tile: i64| -> T {
            d.get(idx).map_or_else(T::zero, |v| v[(j % tile) as usize].clone())
        };
        for ia in self.indexes() {
            for ib in rhs.indexes() {
                let idx = (ia + ib).rem_euclid(t);
                let mut re = out.re.get(idx).cloned().unwrap_or_else(|| zeros.clone());
                let mut im = out.im.get(idx).cloned().unwrap_or_else(|| zeros.clone());
                for j in 0..t {
                    let ar = read(&self.re, ia, j, ta);
                    let ai = read(&self.im, ia, j, ta);
                    let br = read(&rhs.re, ib, j + ia, tb);
                    let bi = read(&rhs.im, ib, j + ia, tb);
                    // (ar + i·ai)(br + i·bi)
                    let mut re_term = ar.mul(&br);
                    re_term.sub_assign(&ai.mul(&bi));
                    re[j as usize].add_assign(&re_term);
                    let mut im_term = ar.mul(&bi);
                    im_term.add_assign(&ai.mul(&br));
                    im[j as usize].add_assign(&im_term);
                }
                out.re.set(idx, re);
                out.im.set(idx, im);
            }
        }
        out
    }

    /// Builds the unprepared [`LinearTransformation`] for this complex map.
    ///
    /// Performs the BSGS pre-rotation `ũ_{j,k} = rot(diag_{n1·j+k}, −n1·j)` and
    /// the giant-step bucketing; `encode` only turns a pre-rotated `(re, im)`
    /// diagonal (each `slots` long) into the scheme's encoded plaintext `P`
    /// (e.g. via `encode_reim`).
    pub fn build_transform<P>(
        &self,
        strategy: LinearTransformationStrategy,
        mut encode: impl FnMut(&[T], &[T]) -> P,
    ) -> LinearTransformation<P> {
        let slots = self.slots();
        let index = LinearTransformationLayout {
            indexes: self.indexes(),
            slots,
            strategy,
        }
        .index();

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
                diagonals.push(LinearTransformationDiagonal { baby, plaintext });
            }
            giant_steps.push(LinearTransformationGiantStep {
                rot: giant_rot,
                diagonals,
            });
        }

        LinearTransformation {
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
        ] {
            let (got_re, got_im) = cd.evaluate((vre.as_slice(), vim.as_slice()), strategy);
            for j in 0..s {
                assert!((got_re[j] - want_re[j]).abs() < 1e-9, "{strategy:?} re");
                assert!((got_im[j] - want_im[j]).abs() < 1e-9, "{strategy:?} im");
            }
        }
    }

    #[test]
    fn compose_matches_sequential_evaluation() {
        let mre_a = vec![
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0, 1.0],
            vec![2.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 4.0],
        ];
        let mim_a = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 2.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let mre_b = vec![
            vec![0.5, 1.0, 0.0, 0.0],
            vec![0.0, 0.5, 1.0, 0.0],
            vec![0.0, 0.0, 0.5, 1.0],
            vec![1.0, 0.0, 0.0, 0.5],
        ];
        let mim_b = vec![
            vec![0.0, 0.0, 1.5, 0.0],
            vec![0.0, 0.0, 0.0, 1.5],
            vec![1.5, 0.0, 0.0, 0.0],
            vec![0.0, 1.5, 0.0, 0.0],
        ];
        let a = ComplexDiagonals::new(diagonals_of(&mre_a), diagonals_of(&mim_a));
        let b = ComplexDiagonals::new(diagonals_of(&mre_b), diagonals_of(&mim_b));
        let vre = vec![1.0, -1.0, 2.0, 0.5];
        let vim = vec![0.5, 2.0, -1.0, 1.0];

        let strategy = LinearTransformationStrategy::Direct;
        let (wre, wim) = b.evaluate((vre.as_slice(), vim.as_slice()), strategy);
        let (want_re, want_im) = a.evaluate((wre.as_slice(), wim.as_slice()), strategy);
        let (got_re, got_im) = a.compose(&b).evaluate((vre.as_slice(), vim.as_slice()), strategy);
        for j in 0..4 {
            assert!((got_re[j] - want_re[j]).abs() < 1e-9, "re slot {j}");
            assert!((got_im[j] - want_im[j]).abs() < 1e-9, "im slot {j}");
        }
    }

    #[test]
    fn compose_lifts_nested_tiles() {
        // a: tile 2 (values alternate per parity), b: tile 4; compose must
        // produce the tile-4 map matching the manual tiled evaluation
        // out[j] = Σ_i d_i[j mod tile] · v[(j + i) mod 4].
        let mut a = ComplexDiagonals::new(Diagonals::new(2), Diagonals::new(2));
        a.re.set(0, vec![2.0, 3.0]);
        a.im.set(1, vec![1.0, -1.0]);
        let mut b = ComplexDiagonals::new(Diagonals::new(4), Diagonals::new(4));
        b.re.set(0, vec![1.0, 0.0, -1.0, 0.5]);
        b.re.set(2, vec![0.0, 1.0, 2.0, 0.0]);

        let v: Vec<(f64, f64)> = vec![(1.0, 0.5), (-1.0, 2.0), (2.0, -1.0), (0.5, 1.0)];
        let tiled = |cd: &ComplexDiagonals<f64>, v: &[(f64, f64)]| -> Vec<(f64, f64)> {
            let t = cd.slots();
            (0..v.len())
                .map(|j| {
                    let mut acc = (0.0, 0.0);
                    for i in cd.indexes() {
                        let dr = cd.re.get(i).map_or(0.0, |d| d[j % t]);
                        let di = cd.im.get(i).map_or(0.0, |d| d[j % t]);
                        let (xr, xi) = v[(j + i as usize) % v.len()];
                        acc.0 += dr * xr - di * xi;
                        acc.1 += dr * xi + di * xr;
                    }
                    acc
                })
                .collect()
        };
        let want = tiled(&a, &tiled(&b, &v));
        let got = tiled(&a.compose(&b), &v);
        for j in 0..4 {
            assert!((got[j].0 - want[j].0).abs() < 1e-9, "re slot {j}");
            assert!((got[j].1 - want[j].1).abs() < 1e-9, "im slot {j}");
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
        ] {
            let (got_re, got_im) = cdt.evaluate((vre.as_slice(), vim.as_slice()), strategy);
            for j in 0..s {
                assert!((got_re[j] - want_re[j]).abs() < 1e-9, "{strategy:?} re");
                assert!((got_im[j] - want_im[j]).abs() < 1e-9, "{strategy:?} im");
            }
        }
    }
}
