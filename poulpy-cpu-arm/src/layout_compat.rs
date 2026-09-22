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
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};

unsafe impl VecZnxDftLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64Neon {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64Neon {}
unsafe impl SvpPPolLayoutCompatible<FFT64Neon> for FFT64Ref {}
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64Neon {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64NeonRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64NeonRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64NeonRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64NeonRayon {}

unsafe impl VecZnxDftLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl VecZnxDftLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Neon> for NTT4x30Ref {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Ref> for NTT4x30Neon {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<NTT4x30Ref> for NTT4x30NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<NTT4x30Ref> for NTT4x30NeonRayon {}

mod ci {
    use poulpy_cpu_ref::{FFT64CIRef, NTT4x30CIRef};
    use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

    use crate::FFT64CINeon;
    use crate::NTT4x30CINeon;
    #[cfg(feature = "enable-rayon")]
    use crate::{FFT64CINeonRayon, NTT4x30CINeonRayon};

    unsafe impl VecZnxDftLayoutCompatible<FFT64CINeon> for FFT64CIRef {}
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIRef> for FFT64CINeon {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CINeon> for FFT64CIRef {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIRef> for FFT64CINeon {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CINeon> for FFT64CIRef {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIRef> for FFT64CINeon {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CINeonRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIRef> for FFT64CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CINeonRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIRef> for FFT64CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CINeonRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIRef> for FFT64CINeonRayon {}

    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CINeon> for NTT4x30CIRef {}
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeon {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CINeon> for NTT4x30CIRef {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeon {}
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CINeon> for NTT4x30CIRef {}
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeon {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CIRef> for NTT4x30CINeonRayon {}
}
