//! Scalar-vector product, `res = scalar * vector`.
//!
//! A method name lists the domain of each operand in signature order, then the
//! output domain:
//!
//! ```text
//! svp_apply_<scalar-domain>_<vector-domain>_to_<output-domain>
//! ```
//!
//! Each domain token is the layout actually passed. The scalar operand:
//!
//! - `small`: coefficient domain, a [`ScalarZnx`](crate::layouts::ScalarZnx).
//! - `tpol`: transformed hot-prep, an [`SvpTPol`](crate::layouts::SvpTPol).
//! - `ppol`: packed cold-prep, an [`SvpPPol`](crate::layouts::SvpPPol).
//!
//! The vector operand: `small` for [`VecZnx`](crate::layouts::VecZnx), `dft`
//! for [`VecZnxDft`](crate::layouts::VecZnxDft). The output: `dft`, `big`
//! (IDFT of the `dft` result) or `small` (normalization of the `big` result).
//!
//! The `small` scalar variants prepare on every call, so they are one-shot
//! paths: prepare into an `SvpTPol` or `SvpPPol` when the same scalar is
//! applied more than once.

use crate::layouts::{
    Backend, ScalarZnxBackendRef, ScratchArena, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolOwned, SvpTPolBackendMut,
    SvpTPolBackendRef, SvpTPolOwned, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
    VecZnxDftBackendRef,
};

/// Allocates an [crate::layouts::SvpPPol].
pub trait SvpPPolAlloc<B: Backend> {
    fn svp_ppol_alloc(&self, cols: usize) -> SvpPPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpPPol].
pub trait SvpPPolBytesOf {
    fn bytes_of_svp_ppol(&self, cols: usize) -> usize;
}

/// Allocates an [crate::layouts::SvpTPol].
pub trait SvpTPolAlloc<B: Backend> {
    fn svp_tpol_alloc(&self, cols: usize) -> SvpTPolOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::SvpTPol].
pub trait SvpTPolBytesOf {
    fn bytes_of_svp_tpol(&self, cols: usize) -> usize;
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpPPol].
pub trait SvpPreparePPol<B: Backend> {
    fn svp_prepare_ppol(&self, res: &mut SvpPPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Prepare a [crate::layouts::ScalarZnx] into an [crate::layouts::SvpTPol].
pub trait SvpPrepareTPol<B: Backend> {
    fn svp_prepare_tpol(&self, res: &mut SvpTPolBackendMut<'_, B>, res_col: usize, a: &ScalarZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy one packed prepared scalar polynomial column into another.
pub trait SvpPPolCopyBackend<B: Backend> {
    fn svp_ppol_copy_backend(
        &self,
        res: &mut SvpPPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Copy one transformed prepared scalar polynomial column into another.
pub trait SvpTPolCopyBackend<B: Backend> {
    fn svp_tpol_copy_backend(
        &self,
        res: &mut SvpTPolBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Scratch bytes required by the `svp_apply_*_to_big` family, for a result of
/// `res_size` limbs.
pub trait SvpApplyToBigTmpBytes {
    fn svp_apply_to_big_tmp_bytes(&self, res_size: usize) -> usize;
}

/// Scratch bytes required by the `svp_apply_*_to_small` family, for an input
/// `b` of `b_size` limbs. The intermediate product is carved at the input limb
/// count, not the result's, so the normalization sees the full width.
pub trait SvpApplyToSmallTmpBytes {
    fn svp_apply_to_small_tmp_bytes(&self, b_size: usize) -> usize;
}

/// Declares one `svp_apply_<scalar>_<vector>_to_<out>` trait.
///
/// `$rhs` is the scalar-operand view type, `$b` the vector-operand view type,
/// and the trailing token selects the output-specific leading and trailing
/// parameters.
macro_rules! svp_apply_trait {
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($rhs:ty, $b:ty) -> dft
    ) => {
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxDftBackendMut<'_, B>,
                res_col: usize,
                a: &$rhs,
                a_col: usize,
                b: &$b,
                b_col: usize,
            );
        }
    };
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($rhs:ty, $b:ty) -> big
    ) => {
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxBigBackendMut<'_, B>,
                res_col: usize,
                a: &$rhs,
                a_col: usize,
                b: &$b,
                b_col: usize,
                scratch: &mut ScratchArena<'_, B>,
            );
        }
    };
    (
        $(#[$meta:meta])*
        $trait:ident :: $method:ident ($rhs:ty, $b:ty) -> small
    ) => {
        $(#[$meta])*
        pub trait $trait<B: Backend> {
            #[allow(clippy::too_many_arguments)]
            fn $method(
                &self,
                res: &mut VecZnxBackendMut<'_, B>,
                res_base2k: usize,
                res_offset: i64,
                res_col: usize,
                a: &$rhs,
                a_col: usize,
                b: &$b,
                b_base2k: usize,
                b_col: usize,
                scratch: &mut ScratchArena<'_, B>,
            );
        }
    };
}

svp_apply_trait!(
    /// `res = a * b`, with `a` an unprepared scalar and `b` in coefficient domain.
    SvpApplySmallSmallToDft::svp_apply_small_small_to_dft(ScalarZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
svp_apply_trait!(
    /// `res = a * b`, with `a` an unprepared scalar and `b` in DFT domain.
    SvpApplySmallDftToDft::svp_apply_small_dft_to_dft(ScalarZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> dft
);
svp_apply_trait!(
    /// `res = a * b`, with `a` hot-prepared and `b` in coefficient domain.
    SvpApplyTPolSmallToDft::svp_apply_tpol_small_to_dft(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
svp_apply_trait!(
    /// `res = a * b`, with `a` hot-prepared and `b` in DFT domain.
    SvpApplyTPolDftToDft::svp_apply_tpol_dft_to_dft(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> dft
);
svp_apply_trait!(
    /// `res = a * b`, with `a` cold-prepared and `b` in coefficient domain.
    SvpApplyPPolSmallToDft::svp_apply_ppol_small_to_dft(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> dft
);
svp_apply_trait!(
    /// `res = a * b`, with `a` cold-prepared and `b` in DFT domain.
    SvpApplyPPolDftToDft::svp_apply_ppol_dft_to_dft(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> dft
);

svp_apply_trait!(
    /// [`SvpApplySmallSmallToDft`] followed by an inverse DFT.
    SvpApplySmallSmallToBig::svp_apply_small_small_to_big(ScalarZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
svp_apply_trait!(
    /// [`SvpApplySmallDftToDft`] followed by an inverse DFT.
    SvpApplySmallDftToBig::svp_apply_small_dft_to_big(ScalarZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> big
);
svp_apply_trait!(
    /// [`SvpApplyTPolSmallToDft`] followed by an inverse DFT.
    SvpApplyTPolSmallToBig::svp_apply_tpol_small_to_big(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
svp_apply_trait!(
    /// [`SvpApplyTPolDftToDft`] followed by an inverse DFT.
    SvpApplyTPolDftToBig::svp_apply_tpol_dft_to_big(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> big
);
svp_apply_trait!(
    /// [`SvpApplyPPolSmallToDft`] followed by an inverse DFT.
    SvpApplyPPolSmallToBig::svp_apply_ppol_small_to_big(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> big
);
svp_apply_trait!(
    /// [`SvpApplyPPolDftToDft`] followed by an inverse DFT.
    SvpApplyPPolDftToBig::svp_apply_ppol_dft_to_big(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> big
);

svp_apply_trait!(
    /// [`SvpApplySmallSmallToBig`] followed by a normalization.
    SvpApplySmallSmallToSmall::svp_apply_small_small_to_small(ScalarZnxBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
svp_apply_trait!(
    /// [`SvpApplySmallDftToBig`] followed by a normalization.
    SvpApplySmallDftToSmall::svp_apply_small_dft_to_small(ScalarZnxBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> small
);
svp_apply_trait!(
    /// [`SvpApplyTPolSmallToBig`] followed by a normalization.
    SvpApplyTPolSmallToSmall::svp_apply_tpol_small_to_small(SvpTPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
svp_apply_trait!(
    /// [`SvpApplyTPolDftToBig`] followed by a normalization.
    SvpApplyTPolDftToSmall::svp_apply_tpol_dft_to_small(SvpTPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> small
);
svp_apply_trait!(
    /// [`SvpApplyPPolSmallToBig`] followed by a normalization.
    SvpApplyPPolSmallToSmall::svp_apply_ppol_small_to_small(SvpPPolBackendRef<'_, B>, VecZnxBackendRef<'_, B>) -> small
);
svp_apply_trait!(
    /// [`SvpApplyPPolDftToBig`] followed by a normalization.
    SvpApplyPPolDftToSmall::svp_apply_ppol_dft_to_small(SvpPPolBackendRef<'_, B>, VecZnxDftBackendRef<'_, B>) -> small
);

/// `res = a * res` with `a` an unprepared scalar.
pub trait SvpApplySmallDftToDftAssign<B: Backend> {
    fn svp_apply_small_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// `res = a * res` with `a` hot-prepared.
pub trait SvpApplyTPolDftToDftAssign<B: Backend> {
    fn svp_apply_tpol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, B>,
        a_col: usize,
    );
}

/// `res = a * res` with `a` cold-prepared.
pub trait SvpApplyPPolDftToDftAssign<B: Backend> {
    fn svp_apply_ppol_dft_to_dft_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, B>,
        a_col: usize,
    );
}
