//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape. Validated by the word-compat
//! suite instantiated in the corresponding `tests.rs`. `VmpPMat` and
//! `CnvPVec*` markers are intentionally absent: the accelerated NTT4x30
//! `VmpPMat` uses a prime-major planar layout (unlike Ref's block-interleaved
//! q120c), and the `CnvPVec` layouts are unverified.

use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Neon;
use crate::NTT4x30Neon;

unsafe impl VecZnxDftLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64Neon {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64Neon {}
unsafe impl SvpPPolLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64Neon {}

unsafe impl VecZnxDftLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl VecZnxDftLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}
