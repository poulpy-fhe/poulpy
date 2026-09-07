mod add;
mod arithmetic_ref;
mod automorphism;
mod automorphism_rotate;
mod copy;
mod mul;
mod neg;
mod normalization;
mod rotate;
mod sampling;
mod sub;
mod switch_ring;
mod zero;

pub use add::*;
pub use arithmetic_ref::*;
pub use automorphism::*;
pub use automorphism_rotate::*;
pub use copy::*;
pub use mul::*;
pub use neg::*;
pub use normalization::*;
pub use rotate::*;
pub use sub::*;
pub use switch_ring::*;
pub use zero::*;

pub use sampling::*;

pub trait ZnxAdd {
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxAddAssign {
    fn znx_add_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSub {
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxSubAssign {
    fn znx_sub_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSubNegateAssign {
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxAutomorphism {
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxAutomorphismRotate {
    /// Computes `res = X^k * auto(p, a)` (fused automorphism + rotation).
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxCopy {
    fn znx_copy(res: &mut [i64], a: &[i64]);
}

pub trait ZnxNegate {
    fn znx_negate(res: &mut [i64], src: &[i64]);
}

pub trait ZnxNegateAssign {
    fn znx_negate_assign(res: &mut [i64]);
}

pub trait ZnxRotate {
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]);
}

pub trait ZnxZero {
    fn znx_zero(res: &mut [i64]);
}

pub trait ZnxMulPowerOfTwo {
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulAddPowerOfTwo {
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulPowerOfTwoAssign {
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]);
}

pub trait ZnxSwitchRing {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]);
}

/// Starts centered normalization of i64 coefficients in `[-2^62, 2^62]`.
/// Requires `1 <= base2k <= 62`, `lsh < base2k`, and representable destination
/// sums when `OVERWRITE` is false.
pub trait ZnxNormalizeFirstStep {
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

/// Uses the input and radix bounds of [`ZnxNormalizeFirstStep`].
/// Incoming carries must lie in `[-2^62, 2^62]`; the output carry stays there.
/// These bounds keep the shifted digit sum and centered subtraction in i64.
pub trait ZnxNormalizeMiddleStep {
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

/// Uses the bounds of [`ZnxNormalizeMiddleStep`] and discards the final carry.
pub trait ZnxNormalizeFinalStep {
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

/// Uses the input and radix bounds of [`ZnxNormalizeFirstStep`].
pub trait ZnxNormalizeFirstStepCarryOnly {
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
}

/// In-place form with the input and radix bounds of [`ZnxNormalizeFirstStep`].
pub trait ZnxNormalizeFirstStepAssign {
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

/// Uses the carry bound of [`ZnxNormalizeMiddleStep`].
pub trait ZnxNormalizeMiddleStepCarryOnly {
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]);
}

/// Uses the carry bound of [`ZnxNormalizeMiddleStep`].
pub trait ZnxNormalizeMiddleStepAssign {
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

/// Uses the carry bound of [`ZnxNormalizeMiddleStep`].
pub trait ZnxNormalizeMiddleStepSub {
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

/// Uses the bounds of [`ZnxNormalizeMiddleStep`] and requires representable destination differences.
pub trait ZnxNormalizeFinalStepSub {
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]);
}

/// In-place form with the bounds of [`ZnxNormalizeMiddleStep`].
pub trait ZnxNormalizeFinalStepAssign {
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]);
}

/// Extracts a centered digit and adds it shifted by `lsh` to the destination.
/// Requires `1 <= base2k`, `base2k + lsh <= 62`, a representable destination
/// sum, and `src - get_digit_i64(base2k, src)` representable in i64.
pub trait ZnxExtractDigitAddMul {
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]);

    /// Extracts a shifted digit into `res`, replacing its previous contents.
    fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]);

    /// Extracts and adds a digit, then centers `res + carry` in the output base.
    /// Requires `1 <= res_base2k <= 62`, both accumulated sums to fit in i64,
    /// and their centered subtraction to fit in i64.
    fn znx_extract_digit_addmul_normalize<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i64],
        carry: &mut [i64],
    );
}

/// Centers the destination and adds its quotient to `src`.
/// Requires `1 <= base2k <= 62`, the centered subtraction from `res` to fit
/// in i64, and the updated `src` coefficient to fit in i64.
pub trait ZnxNormalizeDigit {
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]);
}

/// Extract a wide source digit and normalize the completed destination limb.
/// Source coefficients and incoming carries must lie in `[-2^126, 2^126]`.
/// Requires positive radix widths, `base2k + lsh <= 63`, `res_base2k <= 63`,
/// and a representable i64 destination sum.
pub trait ZnxExtractDigitAddMulI128 {
    const FUSE_NORMALIZE: bool = true;

    fn znx_extract_digit_mul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
        znx_extract_digit_mul_i128_ref(base2k, lsh, res, src);
    }

    fn znx_extract_digit_addmul_normalize_i128<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i128],
        carry: &mut [i128],
    ) {
        znx_extract_digit_addmul_normalize_i128_ref::<OVERWRITE>(base2k, lsh, res_base2k, res, src, carry);
    }
}
