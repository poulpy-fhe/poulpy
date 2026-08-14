//! Vector-matrix product over the operand domains.
//!
//! A method name lists the domain of each operand, then the output domain:
//!
//! ```text
//! vmp_apply_<matrix-domain>_<vector-domain>_to_<output-domain>
//! ```
//!
//! Each domain token is the layout actually passed. The matrix operand:
//!
//! - `small`: coefficient domain, a [`MatZnx`](crate::layouts::MatZnx).
//! - `tmat`: transformed hot-prep, a [`VmpTMat`](crate::layouts::VmpTMat).
//! - `pmat`: packed cold-prep, a [`VmpPMat`](crate::layouts::VmpPMat).
//!
//! The vector operand: `small` for [`VecZnx`](crate::layouts::VecZnx), `dft`
//! for [`VecZnxDft`](crate::layouts::VecZnxDft). The output: `dft`,
//! `dft_accumulate` (adding into `res` instead of overwriting), `big` (IDFT of
//! the `dft` result) or `small` (normalization of the `big` result).
//!
//! The matrix leads in both the name and the signature. `limb_offset` is
//! carried only by the variants whose vector operand is already in DFT domain.
//!
//! The `small` matrix variants prepare the whole matrix on every call, so they
//! are one-shot paths: prepare into a `VmpTMat` or `VmpPMat` when the same
//! matrix is applied more than once.

use crate::layouts::{
    Backend, MatZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    VecZnxDftBackendRef,
};

/// Declares one `vmp_apply_<matrix>_<vector>_to_<out>` trait plus its
/// `*TmpBytes` companion, which is the same for every variant.
///
/// `$a` is the matrix-operand view type and `$b` the vector-operand view type.
/// A trailing `, limb_offset` in the operand list adds that parameter, which
/// only the DFT-domain vector variants carry. The trailing token selects the
/// output domain; `dft` also covers the `_accumulate` variants, whose signature
/// is identical.
macro_rules! vmp_apply_trait {
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($a:ty, $b:ty $(, $off:ident)?) -> dft
    ) => {
        vmp_apply_tmp_bytes_trait!($trait, $method);
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxDftBackendMut<'_, B>,
                a: &$a,
                b: &$b,
                $($off: usize,)?
                scratch: &mut ScratchArena<'_, B>,
            );
        }
    };
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($a:ty, $b:ty $(, $off:ident)?) -> big
    ) => {
        vmp_apply_tmp_bytes_trait!($trait, $method);
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxBigBackendMut<'_, B>,
                a: &$a,
                b: &$b,
                $($off: usize,)?
                scratch: &mut ScratchArena<'_, B>,
            );
        }
    };
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($a:ty, $b:ty $(, $off:ident)?) -> small
    ) => {
        vmp_apply_tmp_bytes_trait!($trait, $method);
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxBackendMut<'_, B>,
                res_base2k: usize,
                res_offset: i64,
                a: &$a,
                b: &$b,
                b_base2k: usize,
                $($off: usize,)?
                scratch: &mut ScratchArena<'_, B>,
            );
        }
    };
}

/// Declares the `*TmpBytes` companion of one apply trait. Split out only so the
/// three [`vmp_apply_trait!`] arms share it; the shape does not vary.
macro_rules! vmp_apply_tmp_bytes_trait {
    ($trait:ident, $method:ident) => {
        paste::paste! {
            #[doc = concat!("Returns scratch bytes required for [`", stringify!($trait), "`].")]
            pub trait [<$trait TmpBytes>] {
                #[allow(clippy::too_many_arguments)]
                fn [<$method _tmp_bytes>](
                    &self,
                    res_size: usize,
                    a_rows: usize,
                    a_cols_in: usize,
                    a_cols_out: usize,
                    a_size: usize,
                    b_size: usize,
                ) -> usize;
            }
        }
    };
}

vmp_apply_trait!(
    /// `res = a * b`, with an unprepared matrix and `b` in coefficient domain.
    VmpApplySmallSmallToDft::vmp_apply_small_small_to_dft(MatZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, with an unprepared matrix and `b` in DFT domain.
    VmpApplySmallDftToDft::vmp_apply_small_dft_to_dft(MatZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with an unprepared matrix and `b` in coefficient domain.
    VmpApplySmallSmallToDftAccumulate::vmp_apply_small_small_to_dft_accumulate(
        MatZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>
    ) -> dft
);
vmp_apply_trait!(
    /// `res += a * b`, with an unprepared matrix and `b` in DFT domain.
    VmpApplySmallDftToDftAccumulate::vmp_apply_small_dft_to_dft_accumulate(
        MatZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> dft
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with an unprepared matrix and `b` in coefficient domain.
    VmpApplySmallSmallToBig::vmp_apply_small_small_to_big(MatZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT applied, with an unprepared matrix and `b` in DFT domain.
    VmpApplySmallDftToBig::vmp_apply_small_dft_to_big(MatZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset) -> big
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with an unprepared matrix and `b` in coefficient domain.
    VmpApplySmallSmallToSmall::vmp_apply_small_small_to_small(MatZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
vmp_apply_trait!(
    /// `res = a * b`, IDFT and normalization applied, with an unprepared matrix and `b` in DFT domain.
    VmpApplySmallDftToSmall::vmp_apply_small_dft_to_small(
        MatZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>, limb_offset
    ) -> small
);

mod pmat;
mod tmat;

pub use pmat::*;
pub use tmat::*;
