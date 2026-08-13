//! Noise-variance estimation for parameter selection.
//!
//! This module provides closed-form noise formulas for the core
//! homomorphic operations (external product, key-switching, etc.).
//! These functions are intended for parameter-set design and
//! noise-budget analysis, not for runtime use.
//!
//! Most helper functions are `pub(crate)`.
//!
//! Notation, shared by every formula below:
//!
//! ```text
//! n      ring degree
//! B      gadget digit width in bits, dsize*base2k
//! d      digits consumed, min(dnum, ceil(k_in/B))
//! k      key precision, dnum*B + k_aux
//! k_in   operand precision
//! s      secret, Var(s_j) = var_xs per coefficient
//! ```
//!
//! Variances are torus variances. Errors of a ciphertext of precision `k` are
//! stated at coefficient scale, i.e. carry a factor `2^{-2k}`. Two facts are
//! used throughout: a product of independent degree-`n` polynomials satisfies
//! `Var((xy)_i) = n*Var(x)*Var(y)`, and independent contributions add.

pub(crate) mod gglwe;
pub(crate) mod ggsw;
pub(crate) mod glwe;

pub use crate::api::GGLWENoise;

use crate::layouts::{GGLWEInfos, GGLWELayout, GGSWInfos, GLWEInfos, LWEInfos};

/// Noise model of a gadget key, blanket-implemented for every [`GGLWEInfos`].
///
/// `k_aux` makes the key's torus precision a property of its layout
/// (`k() = dnum * dsize * base2k + k_aux`, see [`crate::layouts::key_k`]), so
/// the operations below take only the operands they are applied to and the
/// error variances, which are no property of a layout; everything else is read
/// off the key.
///
/// `var_xs_in` is the variance of the secret the operand is encrypted under,
/// `var_xs_out` the one of the secret the key is encrypted under.
pub(crate) trait GGLWENoiseModel: GGLWEInfos {
    /// `B = dsize * base2k`.
    fn digit_bits(&self) -> usize {
        self.dsize().as_usize() * self.base2k().as_usize()
    }

    /// `d = min(dnum, ceil(k_in / B))`.
    ///
    /// The decomposition consumes whole digits until the operand's precision is
    /// exhausted, and stops at the key's last row.
    fn digits<A: LWEInfos + ?Sized>(&self, input: &A) -> f64 {
        input.k().as_usize().div_ceil(self.digit_bits()).min(self.dnum().as_usize()) as f64
    }

    /// `Var(a_i) = 2^{2B} / 12`.
    ///
    /// A digit is signed with `|a_i| <= 2^{B-1}`, uniform over the `P = 2^B`
    /// two's-complement values of `[-2^{B-1}, 2^{B-1})`, and a discrete uniform
    /// over `P` values has variance `(P^2-1)/12`, taken here at `P^2/12`.
    fn var_digit(&self) -> f64 {
        let base: f64 = (self.digit_bits() as f64).exp2();
        base * base / 12f64
    }

    /// `Var(r) = 2^{-2dB} / 12` if `dB < k_in`, else `0`.
    ///
    /// Writing `a = sum_{i<d} a_i * 2^{-(i+1)B} + r`, the digits reproduce `a`
    /// down to `2^{-dB}` and leave `r` uniform of width `2^{-dB}`, hence the
    /// same `width^2/12`. Once `dB >= k_in` the operand holds no bits below the
    /// cover, so `r = 0`. What a dropped component then costs depends on what it
    /// would have been multiplied by, so each operation weights `Var(r)` itself.
    fn var_residue<A: LWEInfos + ?Sized>(&self, input: &A) -> f64 {
        let cover: usize = self.digits(input) as usize * self.digit_bits();
        if cover >= input.k().as_usize() {
            0f64
        } else {
            (-2.0 * cover as f64).exp2() / 12f64
        }
    }

    /// `Var = rank_in * n * d * Var(a_i) * (Var(e_body) + var_xs_out * Var(e_mask)) * 2^{-2k}`,
    /// the key's own error as one gadget product injects it.
    ///
    /// Each of the `rank_in` decomposed components is spent on `d` independent
    /// digits, each meeting one key row through a ring product (hence `n`). A row
    /// carries `e_body + s_out * e_mask` as decryption sees it, so its mask-side
    /// error is weighted by the secret the **key** is encrypted under.
    fn var_key_error<A: LWEInfos + ?Sized>(
        &self,
        input: &A,
        var_xs_out: f64,
        var_key_err_body: f64,
        var_key_err_mask: f64,
    ) -> f64 {
        (self.rank_in().as_usize() as f64)
            * (self.n().as_usize() as f64)
            * self.digits(input)
            * self.var_digit()
            * (var_key_err_body + var_xs_out * var_key_err_mask)
            * var_scale(self)
    }

    /// `Var = Var(e_in) * 2^{-2k_in} + Var_key + rank_in * n * var_xs_in * Var(r)`.
    ///
    /// A key-switch rewrites `sum_j a_j * s_in,j` and leaves the body untouched,
    /// so the operand's phase error carries over unchanged. Past the key error
    /// of [`Self::var_key_error`], every mask the gadget truncates loses
    /// `r_j * s_in,j` from the phase, a ring product against the **source**
    /// secret.
    #[allow(clippy::too_many_arguments)]
    fn var_noise_keyswitch<A: GLWEInfos>(
        &self,
        input: &A,
        var_xs_in: f64,
        var_xs_out: f64,
        var_in_err: f64,
        var_key_err_body: f64,
        var_key_err_mask: f64,
    ) -> f64 {
        var_in_err * var_scale(input)
            + self.var_key_error(input, var_xs_out, var_key_err_body, var_key_err_mask)
            + (self.rank_in().as_usize() as f64) * (self.n().as_usize() as f64) * var_xs_in * self.var_residue(input)
    }

    /// `log2 sqrt(Var)` of [`Self::var_noise_keyswitch`].
    #[allow(clippy::too_many_arguments)]
    fn log2_std_noise_keyswitch<A: GLWEInfos>(
        &self,
        input: &A,
        var_xs_in: f64,
        var_xs_out: f64,
        var_in_err: f64,
        var_key_err_body: f64,
        var_key_err_mask: f64,
    ) -> f64 {
        log2_std(self.var_noise_keyswitch(input, var_xs_in, var_xs_out, var_in_err, var_key_err_body, var_key_err_mask))
    }

    /// `log2 sqrt(Var)` of the `col`-th column of a GGSW key-switched (or
    /// automorphed) from `input` into `res` with `self`, then expanded with
    /// `tsk`:
    ///
    /// ```text
    /// col = 0: Var_ks
    /// col > 0: n * var_xs_out * Var_ks + Var_key(tsk) + rank * n^2 * var_xs_out^2 * Var(r_tsk)
    /// ```
    ///
    /// Column `0` is the key-switch of the row. Column `col` is rebuilt as
    /// `sum_j a_j * tsk_j` with the row's body re-inserted, undecomposed, in
    /// slot `col`, so decryption multiplies the whole row phase by `s_out,col`:
    /// one ring product over `Var_ks`, and the tensor key's own error enters
    /// unscaled next to it. Only the `rank` masks meet the gadget, and a residue
    /// there is lost against `s_j * s_col`, of variance `n * var_xs_out^2`. The
    /// operand of that second product is the key-switched row, hence `res`.
    /// This branch runs up to ~0.2 bits under the measurement, the square
    /// `s_col^2` carrying a mean a variance model ignores.
    #[allow(clippy::too_many_arguments)]
    fn log2_std_noise_ggsw_keyswitch<T: GGLWEInfos, A: GLWEInfos, B: GLWEInfos>(
        &self,
        tsk: &T,
        col: usize,
        input: &A,
        res: &B,
        var_xs_in: f64,
        var_xs_out: f64,
        var_in_err: f64,
        var_key_err_body: f64,
        var_key_err_mask: f64,
    ) -> f64 {
        let noise: f64 = self.var_noise_keyswitch(input, var_xs_in, var_xs_out, var_in_err, var_key_err_body, var_key_err_mask);

        if col == 0 {
            return log2_std(noise);
        }

        let n: f64 = tsk.n().as_usize() as f64;
        let rank: f64 = tsk.rank_in().as_usize() as f64;
        // sum_j Var(s_j * s_col) over the rank masks: (rank-1) off-diagonal at
        // n*var_xs^2, and the j = col square at twice that.
        let var_s_x_s_col: f64 = (rank + 1.0) * n * var_xs_out * var_xs_out;

        log2_std(
            n * var_xs_out * noise
                + tsk.var_key_error(res, var_xs_out, var_key_err_body, var_key_err_mask)
                + n * var_s_x_s_col * tsk.var_residue(res),
        )
    }
}

impl<T: GGLWEInfos + ?Sized> GGLWENoiseModel for T {}

/// Noise model of a GGSW, blanket-implemented for every [`GGSWInfos`].
pub(crate) trait GGSWNoiseModel: GGSWInfos {
    /// The GGSW seen as a gadget key: an external product decomposes all
    /// `rank + 1` components of the GLWE operand, so the equivalent key has
    /// `rank_in = rank + 1`.
    fn as_gadget_key(&self) -> GGLWELayout {
        GGLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            dnum: self.dnum(),
            k_aux: self.k_aux(),
            dsize: self.dsize(),
            rank_in: self.rank() + 1,
            rank_out: self.rank(),
        }
    }

    /// `log2 sqrt(Var)` of a GLWE (x) GGSW external product, the GGSW encrypting
    /// a message of variance `var_msg`:
    ///
    /// ```text
    /// fold = 1 + rank * n * var_xs
    /// Var  = Var_key + n * var_msg * fold * Var(r)
    ///                + n * var_msg * (Var(e_body) + rank * n * var_xs * Var(e_mask)) * 2^{-2k_in}
    /// ```
    ///
    /// The product is `(ct - r) * m` for `r` the residues the gadget drops, so
    /// the phase misses `m * (r_body - sum_j r_j * s_j)`: a dropped component
    /// rides the message through a ring product, the masks folding once more
    /// against the secret, which is the `fold` factor. A zero message therefore
    /// costs nothing. The operand's own error travels the same route, hence the
    /// identical weighting.
    #[allow(clippy::too_many_arguments)]
    fn log2_std_noise_external_product<A: GLWEInfos>(
        &self,
        input: &A,
        var_xs: f64,
        var_msg: f64,
        var_in_err_body: f64,
        var_in_err_mask: f64,
        var_key_err_body: f64,
        var_key_err_mask: f64,
    ) -> f64 {
        let key: GGLWELayout = self.as_gadget_key();
        let n: f64 = self.n().as_usize() as f64;
        let rank: f64 = self.rank().as_usize() as f64;
        let fold: f64 = 1.0 + rank * n * var_xs;

        let mut noise: f64 = key.var_key_error(input, var_xs, var_key_err_body, var_key_err_mask);
        noise += n * var_msg * fold * key.var_residue(input);
        noise += n * var_msg * (var_in_err_body + rank * n * var_xs * var_in_err_mask) * var_scale(input);
        log2_std(noise)
    }
}

impl<T: GGSWInfos + ?Sized> GGSWNoiseModel for T {}

/// `2^{-2k}`, the torus scale the operand's error sits at.
fn var_scale<A: LWEInfos + ?Sized>(operand: &A) -> f64 {
    (-2.0 * operand.k().as_usize() as f64).exp2()
}

/// `log2 sqrt(Var)`, clamped to the largest representable noise
/// (`[-2^{-1}, 2^{-1}]`).
fn log2_std(var: f64) -> f64 {
    var.sqrt().log2().min(-1.0)
}

/// `Var(<s (x) s, e>) / Var(e) = 1 + rank * n * var_xs + (rank(rank+3)/2) * n * (n * var_xs^2)`.
///
/// The tensor key of a rank-`r` secret holds one constant component, `r` linear
/// ones `s_i`, and the quadratic products (see
/// [`Distribution`](crate::dist::Distribution)): `r` diagonal `s_i^2`, whose
/// coefficients pair `s_a s_b` twice and so reach `2 * n * var_xs^2`, and
/// `r(r-1)/2` off-diagonal `s_i s_j` at `n * var_xs^2`. Each component meets an
/// independent error through a ring product, contributing `n * Var(component)`.
/// Collecting the quadratic ones gives `2r + r(r-1)/2 = r(r+3)/2` times
/// `n * var_xs^2`.
pub(crate) fn var_tensor_key(n: f64, rank: f64, var_xs: f64) -> f64 {
    let var_si_x_sj: f64 = n * var_xs * var_xs;
    1.0 + rank * n * var_xs + 0.5 * rank * (rank + 3.0) * n * var_si_x_sj
}

/// `Var = (Var(e_a) + Var(e_b)) * var_tensor_key * 2^{2*cnv_offset}`, the
/// decryption error of a GLWE tensor product **before** relinearization, at the
/// output scale.
///
/// Each tensor component carries both operands' errors, and decryption folds
/// the components against `s (x) s`, which amplifies by [`var_tensor_key`]. The
/// product is rescaled by `cnv_offset` dropped low bits, lifting everything
/// below it by `2^{cnv_offset}`.
///
/// Calibrated against the reference backend over rank, ring degree, torus
/// precision, secret variance, operand noise and convolution offset; it is an
/// upper estimate, running ~0.3 bits above the measured standard deviation.
/// Relinearization noise is additive and much smaller than this multiplicative
/// term, so the same bound covers the relinearized result.
pub(crate) fn var_noise_glwe_tensor(n: f64, rank: f64, var_xs: f64, var_e_a: f64, var_e_b: f64, cnv_offset: usize) -> f64 {
    let scale: f64 = ((2 * cnv_offset) as f64).exp2();
    (var_e_a + var_e_b) * var_tensor_key(n, rank, var_xs) * scale
}

/// `log2 sqrt(Var)` of [`var_noise_glwe_tensor`].
///
/// `sigma_a` / `sigma_b` are the operands' error standard deviations relative
/// to their torus precision `k_a` / `k_b`: an error placed at `k` with standard
/// deviation `sigma` has torus variance `(sigma * 2^{-k})^2`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn log2_std_noise_glwe_tensor(
    n: f64,
    rank: f64,
    var_xs: f64,
    sigma_a: f64,
    k_a: usize,
    sigma_b: f64,
    k_b: usize,
    cnv_offset: usize,
) -> f64 {
    let var_e = |sigma: f64, k: usize| {
        let s: f64 = sigma * (-(k as f64)).exp2();
        s * s
    };
    var_noise_glwe_tensor(n, rank, var_xs, var_e(sigma_a, k_a), var_e(sigma_b, k_b), cnv_offset)
        .sqrt()
        .log2()
        .min(-1.0)
}
