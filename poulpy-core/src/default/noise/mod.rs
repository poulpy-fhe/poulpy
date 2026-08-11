//! Noise-variance estimation for parameter selection.
//!
//! This module provides closed-form noise formulas for the core
//! homomorphic operations (external product, key-switching, etc.).
//! These functions are intended for parameter-set design and
//! noise-budget analysis, not for runtime use.
//!
//! Most helper functions are `pub(crate)`.

pub(crate) mod gglwe;
pub(crate) mod ggsw;
pub(crate) mod glwe;

pub use crate::api::GGLWENoise;

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn var_noise_gglwe_product(
    n: f64,
    base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank_in: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let a_logq: usize = a_logq.min(b_logq);
    let a_cols: usize = a_logq.div_ceil(base2k);

    let b_scale: f64 = (b_logq as f64).exp2();
    let a_scale: f64 = ((b_logq - a_logq) as f64).exp2();

    let base: f64 = (base2k as f64).exp2();
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * a_scale * a_scale * n;
    noise *= rank_in;
    noise /= b_scale * b_scale;
    noise
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn var_noise_gglwe_product_v2(
    n: f64,
    k_ksk: usize,
    dnum: usize,
    dsize: usize,
    base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank_in: f64,
) -> f64 {
    let base: f64 = ((dsize * base2k) as f64).exp2();
    let var_base: f64 = base * base / 12f64;
    let scale: f64 = (k_ksk as f64).exp2();

    let mut noise: f64 = (dnum as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a_err * var_base * n;
    noise *= rank_in;
    noise /= scale * scale;
    noise
}

/// Variance of `s (x) s`'s contribution to a tensor-product decryption.
///
/// The tensor key of a rank-`r` secret holds one constant component, `r`
/// linear components `s_i` (per-coefficient variance `var_xs`) and the
/// quadratic products (see [`Distribution`](crate::dist::Distribution)):
///
/// - `r` diagonal ones `s_i^2`, whose coefficients pair `s_a s_b` twice and
///   so have per-coefficient variance `2 * n * var_xs^2`;
/// - `r(r-1)/2` off-diagonal ones `s_i * s_j`, at `n * var_xs^2`.
///
/// Each component is hit by an independent error and enters through a ring
/// product, adding a factor `n` to every non-constant term. Collecting the
/// quadratic ones gives the `r(r+3)/2` multiplier below.
pub(crate) fn var_tensor_key(n: f64, rank: f64, var_xs: f64) -> f64 {
    let var_si_x_sj: f64 = n * var_xs * var_xs;
    1.0 + rank * n * var_xs + 0.5 * rank * (rank + 3.0) * n * var_si_x_sj
}

/// Variance of the decryption error of a GLWE tensor product, **before**
/// relinearization, at the output scale.
///
/// The operands' own errors are the dominant source: each tensor component
/// carries them, and decryption folds the components against `s (x) s`, which
/// amplifies by [`var_tensor_key`]. The convolution offset `cnv_offset`
/// rescales the product (it drops `cnv_offset` low bits), so everything below
/// it is amplified by `2^cnv_offset`.
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

/// `log2` of the standard deviation of [`var_noise_glwe_tensor`].
///
/// `sigma_a` / `sigma_b` are the operands' error standard deviations relative
/// to their torus precision `k_a` / `k_b` (an error placed at `k` with
/// standard deviation `sigma` has torus variance `(sigma * 2^-k)^2`).
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

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn log2_std_noise_gglwe_product(
    n: f64,
    base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank_in: f64,
    a_logq: usize,
    b_logq: usize,
) -> f64 {
    let mut noise: f64 = var_noise_gglwe_product(
        n,
        base2k,
        var_xs,
        var_msg,
        var_a_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank_in,
        a_logq,
        b_logq,
    );
    noise = noise.sqrt();
    noise.log2().min(-1.0).max(-(a_logq as f64)) // max noise is [-2^{-1}, 2^{-1}]
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn noise_ggsw_product(
    n: f64,
    base2k: usize,
    var_xs: f64,
    var_msg: f64,
    var_a0_err: f64,
    var_a1_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank: f64,
    k_in: usize,
    k_ggsw: usize,
) -> f64 {
    let a_logq: usize = k_in.min(k_ggsw);
    let a_cols: usize = a_logq.div_ceil(base2k);

    let b_scale: f64 = (k_ggsw as f64).exp2();
    let a_scale: f64 = ((k_ggsw - a_logq) as f64).exp2();

    let base: f64 = (base2k as f64).exp2();
    let var_base: f64 = base * base / 12f64;

    // lhs = a_cols * n * (var_base * var_gct_err_lhs + var_e_a * var_msg * p^2)
    // rhs = a_cols * n * var_base * var_gct_err_rhs * var_xs
    let mut noise: f64 = (rank + 1.0) * (a_cols as f64) * n * var_base * (var_gct_err_lhs + var_xs * var_gct_err_rhs);
    noise += var_msg * var_a0_err * a_scale * a_scale * n;
    noise += var_msg * var_a1_err * a_scale * a_scale * n * var_xs * rank;
    noise = noise.sqrt();
    noise /= b_scale;
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn noise_ggsw_keyswitch(
    n: f64,
    base2k: usize,
    col: usize,
    var_xs: f64,
    var_a_err: f64,
    var_gct_err_lhs: f64,
    var_gct_err_rhs: f64,
    rank: f64,
    k_ct: usize,
    k_ksk: usize,
    k_tsk: usize,
) -> f64 {
    let var_si_x_sj: f64 = n * var_xs * var_xs;

    // Initial KS for col = 0
    let mut noise: f64 = var_noise_gglwe_product(
        n,
        base2k,
        var_xs,
        var_xs,
        var_a_err,
        var_gct_err_lhs,
        var_gct_err_rhs,
        rank,
        k_ct,
        k_ksk,
    );

    // Other GGSW reconstruction for col > 0
    if col > 0 {
        noise += var_noise_gglwe_product(
            n,
            base2k,
            var_xs,
            var_si_x_sj,
            var_a_err + 1f64 / 12.0,
            var_gct_err_lhs,
            var_gct_err_rhs,
            rank,
            k_ct,
            k_tsk,
        );
        noise += n * noise * var_xs * 0.5;
    }

    noise = noise.sqrt();
    noise.log2().min(-1.0) // max noise is [-2^{-1}, 2^{-1}]
}
