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
