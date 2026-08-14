//! Bivariate convolution over the operand domains.
//!
//! A method name lists the domain of each operand, then the output domain:
//!
//! ```text
//! cnv_apply_<left-domain>_<right-domain>_to_<output-domain>
//! ```
//!
//! Both operands are tiered, so the order is positional (left, then right):
//!
//! - `small`: coefficient domain, a [`VecZnx`](crate::layouts::VecZnx).
//! - `tvec`: transformed hot-prep, a [`CnvTVecL`](crate::layouts::CnvTVecL) /
//!   [`CnvTVecR`](crate::layouts::CnvTVecR).
//! - `pvec`: packed cold-prep, a [`CnvPVecL`](crate::layouts::CnvPVecL) /
//!   [`CnvPVecR`](crate::layouts::CnvPVecR).
//!
//! The output: `dft`, `dft_accumulate` (adding into `res` instead of
//! overwriting), `big` (IDFT of the `dft` result) or `small` (normalization of
//! the `big` result).
//!
//! Each `small` operand is followed by its `mask`, the same
//! `msb_mask_bottom_limb(base2k, k)` value the prepare methods take. A `VecZnx`
//! does not carry `base2k`/`k`, so the apply cannot derive it, and omitting it
//! would make a `small` apply disagree with prepare-then-apply whenever `k` is
//! not limb-aligned.
//!
//! The `small` operand variants prepare on every call, so they are one-shot
//! paths: prepare into a `CnvTVec*` or `CnvPVec*` when the same operand is
//! applied more than once.

use crate::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecRBackendMut, CnvPVecRBackendRef, CnvPVecROwned,
    CnvTVecLBackendMut, CnvTVecLBackendRef, CnvTVecLOwned, CnvTVecRBackendMut, CnvTVecRBackendRef, CnvTVecROwned, ScratchArena,
    VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
};
use crate::layouts::{CnvDftAccTermPvec, CnvDftAccTermTvec};

/// Allocates packed cold-prep convolution operands.
pub trait CnvPVecAlloc<BE: Backend> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecLOwned<BE>;
    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecROwned<BE>;
}

/// Returns the byte sizes for packed cold-prep convolution operands.
pub trait CnvPVecBytesOf {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize;
}

/// Allocates transformed hot-prep convolution operands.
pub trait CnvTVecAlloc<BE: Backend> {
    fn cnv_tvec_left_alloc(&self, cols: usize, size: usize) -> CnvTVecLOwned<BE>;
    fn cnv_tvec_right_alloc(&self, cols: usize, size: usize) -> CnvTVecROwned<BE>;
}

/// Returns the byte sizes for transformed hot-prep convolution operands.
pub trait CnvTVecBytesOf {
    fn bytes_of_cnv_tvec_left(&self, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_tvec_right(&self, cols: usize, size: usize) -> usize;
}

/// Emits the six apply forms plus their `_tmp_bytes` companions for one operand
/// pair of the CNV matrix.
///
/// `$pair` is the snake-case pair token (`small_pvec`), `$Pair` its camel form
/// (used for the accumulation term alias), and `$L`/`$R` the operand view types.
/// `$lm`/`$rm` name that side's mask parameter: `(a_mask)` for a `small`
/// operand, `()` for an already-prepared one.
macro_rules! cnv_tier_methods {
    ($tier:ident, $Tier:ident, $L:ty, $R:ty) => {
        paste::paste! {
            #[doc = concat!("Returns scratch bytes required for [`cnv_apply_", stringify!($tier), "_to_dft`](Self::cnv_apply_", stringify!($tier), "_to_dft).")]
            fn [<cnv_apply_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;

            #[doc = concat!("`res[res_col] = a[a_col] (x) b[b_col]`, scaled by `2^{cnv_offset * Base2K}`.")]
            ///
            /// See the module documentation for the bivariate convolution the
            /// whole family evaluates; the variants differ only in the domain of
            /// each operand and of the result.
            #[allow(clippy::too_many_arguments)]
            fn [<cnv_apply_ $tier _to_dft>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                a_col: usize,
                b: &$R,
                b_col: usize,
                scratch: &mut ScratchArena<'_, BE>,
            );

            #[doc = concat!("Returns scratch bytes required for [`cnv_apply_", stringify!($tier), "_to_dft_accumulate`](Self::cnv_apply_", stringify!($tier), "_to_dft_accumulate).")]
            fn [<cnv_apply_ $tier _to_dft_accumulate_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;

            #[doc = concat!("`res[res_col] += a[a_col] (x) b[b_col]`. Limbs beyond `min(res.size(), a.size() + b.size())` are left untouched.")]
            #[allow(clippy::too_many_arguments)]
            fn [<cnv_apply_ $tier _to_dft_accumulate>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                a_col: usize,
                b: &$R,
                b_col: usize,
                scratch: &mut ScratchArena<'_, BE>,
            );

            #[doc = concat!("Returns scratch bytes required for [`cnv_accumulate_", stringify!($tier), "_to_dft`](Self::cnv_accumulate_", stringify!($tier), "_to_dft).")]
            ///
            /// `a_size` and `b_size` are upper bounds over the term operand sizes.
            fn [<cnv_accumulate_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;

            #[doc = concat!("`res[res_col] = sum_t a_t (x) b_t` over `terms`, overwriting `res[res_col]`.")]
            ///
            /// With an empty `terms` slice the output column is zeroed. Backends
            /// may fuse the accumulation, so the result is congruent to but not
            /// necessarily bit-identical with a sequence of
            #[doc = concat!("[`cnv_apply_", stringify!($tier), "_to_dft_accumulate`](Self::cnv_apply_", stringify!($tier), "_to_dft_accumulate) calls.")]
            fn [<cnv_accumulate_ $tier _to_dft>]<'a>(
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                terms: &[[<CnvDftAccTerm $Tier>]<'a, BE>],
                scratch: &mut ScratchArena<'_, BE>,
            ) where
                BE: 'a;

            #[doc = concat!("Returns scratch bytes required for [`cnv_pairwise_apply_", stringify!($tier), "_to_dft`](Self::cnv_pairwise_apply_", stringify!($tier), "_to_dft).")]
            fn [<cnv_pairwise_apply_ $tier _to_dft_tmp_bytes>](
                &self,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;

            #[doc = concat!("`res = (a[i] + a[j]) (x) (b[i] + b[j])`; for `i == j` this is [`cnv_apply_", stringify!($tier), "_to_dft`](Self::cnv_apply_", stringify!($tier), "_to_dft) on column `i`.")]
            #[allow(clippy::too_many_arguments)]
            fn [<cnv_pairwise_apply_ $tier _to_dft>](
                &self,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$L,
                b: &$R,
                i: usize,
                j: usize,
                scratch: &mut ScratchArena<'_, BE>,
            );
        }
    };
}

/// Bivariate convolution over `Z[X, Y] mod (X^N + 1)` where `Y = 2^{-K}`.
///
/// # Example
///
/// ```text
///       1    X   X^2  X^3
/// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
///     Y [a01, a11, a21, a31]
///
/// b = 1 [b00, b10, b20, b30] = (b00 + b01 * 2^-K) + (b10 + b11 * 2^-K) * X ...
///     Y [b01, b11, b21, b31]
///
/// If cnv_offset = 0:
///
///            1    X   X^2  X^3
/// res = 1  [r00, r10, r20, r30] = (r00 + r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K) + ... * X + ...
///       Y  [r01, r11, r21, r31]
///       Y^2[r02, r12, r22, r32]
///       Y^3[r03, r13, r23, r33]
///
/// If cnv_offset = 1:
///
///            1    X   X^2  X^3
/// res = 1  [r01, r11, r21, r31]  = (r01 + r02 * 2^-K + r03 * 2^-2K) + ... * X + ...
///       Y  [r02, r12, r22, r32]
///       Y^2[r03, r13, r23, r33]
///       Y^3[  0,   0,   0 ,  0]
/// ```
///
/// If `res.size() < a.size() + b.size() + k`, the result is truncated
/// accordingly in the `Y` dimension.
pub trait Convolution<BE: Backend> {
    /// Returns scratch bytes required for [`cnv_prepare_left_pvec`](Convolution::cnv_prepare_left_pvec).
    fn cnv_prepare_left_pvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) into the
    /// packed cold-prep left operand.
    fn cnv_prepare_left_pvec(
        &self,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_right_pvec`](Convolution::cnv_prepare_right_pvec).
    fn cnv_prepare_right_pvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) into the
    /// packed cold-prep right operand.
    fn cnv_prepare_right_pvec(
        &self,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_self_pvec`](Convolution::cnv_prepare_self_pvec).
    fn cnv_prepare_self_pvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares both packed cold-prep operands from the same input polynomial,
    /// sharing the transform. An optimization for self-convolution (squaring).
    fn cnv_prepare_self_pvec(
        &self,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_left_tvec`](Convolution::cnv_prepare_left_tvec).
    fn cnv_prepare_left_tvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) into the
    /// transformed hot-prep left operand.
    fn cnv_prepare_left_tvec(
        &self,
        res: &mut CnvTVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_right_tvec`](Convolution::cnv_prepare_right_tvec).
    fn cnv_prepare_right_tvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) into the
    /// transformed hot-prep right operand.
    fn cnv_prepare_right_tvec(
        &self,
        res: &mut CnvTVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_self_tvec`](Convolution::cnv_prepare_self_tvec).
    fn cnv_prepare_self_tvec_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares both transformed hot-prep operands from the same input
    /// polynomial, sharing the transform.
    fn cnv_prepare_self_tvec(
        &self,
        left: &mut CnvTVecLBackendMut<'_, BE>,
        right: &mut CnvTVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_by_const_apply`](Convolution::cnv_by_const_apply).
    fn cnv_by_const_apply_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Bivariate convolution with `b` treated as a constant polynomial in `X`.
    ///
    /// ```text
    ///       1    X   X^2  X^3
    /// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     Y [a01, a11, a21, a31]
    ///
    /// b = 1 [b0] = (b00 + b01 * 2^-K)
    ///     Y [b0]
    /// ```
    ///
    /// Intended for multiplications by constants greater than the base2k.
    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    cnv_tier_methods!(pvec, Pvec, CnvPVecLBackendRef<'_, BE>, CnvPVecRBackendRef<'_, BE>);
    cnv_tier_methods!(tvec, Tvec, CnvTVecLBackendRef<'_, BE>, CnvTVecRBackendRef<'_, BE>);
}
