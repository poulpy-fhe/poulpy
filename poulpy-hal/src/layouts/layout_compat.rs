//! Cross-backend layout-compatibility markers.
//!
//! Containers of different backends are distinct types even when their words
//! match; the only zero-copy way to move a buffer between backends is the
//! `into_backend` re-tag on each container, guarded by the markers below.
//! A backend pair declares a marker **per container family** — sharing a word
//! (sizing/element view) does not imply sharing a prepared layout, as the
//! `VmpPMat` split between the reference and accelerated NTT4x30 backends
//! shows (block-interleaved q120c vs prime-major planar).
//!
//! # Safety
//!
//! Implementing a marker asserts that the two backends produce **byte-identical
//! buffer layouts for that container family, for every shape** — same
//! convention, not merely the same size. Declared impls are validated by the
//! word-compatibility test suite
//! ([`test_suite::word_compat`](crate::test_suite::word_compat)); declare the
//! marker and instantiate the corresponding tests together.
//!
//! Every backend is compatible with itself (reflexive blanket impls).

use crate::layouts::Backend;

/// `B: VecZnxDftLayoutCompatible<B2>` asserts `VecZnxDft` buffers of `B` are
/// byte-layout-identical to those of `B2`.
///
/// # Safety
///
/// Implementors assert byte-identical buffer layouts between `Self` and `B2`
/// for this container family, for every shape — same convention, not merely
/// the same size. Validate with the word-compat test suite.
pub unsafe trait VecZnxDftLayoutCompatible<B2: Backend>: Backend<DftWord = <B2 as Backend>::DftWord> {}
unsafe impl<B: Backend> VecZnxDftLayoutCompatible<B> for B {}

/// `B: SvpPPolLayoutCompatible<B2>` asserts `SvpPPol` buffers of `B` are
/// byte-layout-identical to those of `B2`.
///
/// # Safety
///
/// Implementors assert byte-identical buffer layouts between `Self` and `B2`
/// for this container family, for every shape — same convention, not merely
/// the same size. Validate with the word-compat test suite.
pub unsafe trait SvpPPolLayoutCompatible<B2: Backend>: Backend<DftWord = <B2 as Backend>::DftWord> {}
unsafe impl<B: Backend> SvpPPolLayoutCompatible<B> for B {}

/// `B: VmpPMatLayoutCompatible<B2>` asserts `VmpPMat` buffers of `B` are
/// byte-layout-identical to those of `B2`.
///
/// Deliberately NOT declared between the reference and accelerated NTT4x30
/// backends: their prepared-matrix layouts differ under the shared word.
///
/// # Safety
///
/// Implementors assert byte-identical buffer layouts between `Self` and `B2`
/// for this container family, for every shape — same convention, not merely
/// the same size. Validate with the word-compat test suite.
pub unsafe trait VmpPMatLayoutCompatible<B2: Backend>: Backend<DftWord = <B2 as Backend>::DftWord> {}
unsafe impl<B: Backend> VmpPMatLayoutCompatible<B> for B {}

/// `B: VecZnxBigLayoutCompatible<B2>` asserts `VecZnxBig` buffers of `B` are
/// byte-layout-identical to those of `B2`.
///
/// # Safety
///
/// Implementors assert byte-identical buffer layouts between `Self` and `B2`
/// for this container family, for every shape — same convention, not merely
/// the same size. Validate with the word-compat test suite.
pub unsafe trait VecZnxBigLayoutCompatible<B2: Backend>: Backend<BigWord = <B2 as Backend>::BigWord> {}
unsafe impl<B: Backend> VecZnxBigLayoutCompatible<B> for B {}

/// `B: CnvPVecLayoutCompatible<B2>` asserts `CnvPVecL` **and** `CnvPVecR`
/// buffers of `B` are byte-layout-identical to those of `B2`.
///
/// # Safety
///
/// Implementors assert byte-identical buffer layouts between `Self` and `B2`
/// for this container family, for every shape — same convention, not merely
/// the same size. Validate with the word-compat test suite.
pub unsafe trait CnvPVecLayoutCompatible<B2: Backend>: Backend<DftWord = <B2 as Backend>::DftWord> {}
unsafe impl<B: Backend> CnvPVecLayoutCompatible<B> for B {}
