//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the portable backend
//! for that container family, for every shape. Validated by the word-compat
//! suite instantiated in the corresponding `tests.rs`. `VmpPMat` and
//! `CnvPVec*` markers are intentionally absent: the accelerated NTT4x30
//! `VmpPMat` uses a prime-major planar layout (unlike Ref's block-interleaved
//! q120c), and the `CnvPVec` layouts are unverified.

use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Neon;
use crate::NTT4x30Neon;
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};

unsafe impl VecZnxDftLayoutCompatible<FFT64Neon> for FFT64Portable {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64Neon {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Neon> for FFT64Portable {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64Neon {}
unsafe impl SvpPPolLayoutCompatible<FFT64Neon> for FFT64Portable {}
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64Neon {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64NeonRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64NeonRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64NeonRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64NeonRayon {}

unsafe impl VecZnxDftLayoutCompatible<NTT4x30Neon> for NTT4x30Portable {}
unsafe impl VecZnxDftLayoutCompatible<NTT4x30Portable> for NTT4x30Neon {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Neon> for NTT4x30Portable {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30Neon {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Neon> for NTT4x30Portable {}
unsafe impl SvpPPolLayoutCompatible<NTT4x30Portable> for NTT4x30Neon {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<NTT4x30Portable> for NTT4x30NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30NeonRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<NTT4x30NeonRayon> for NTT4x30Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<NTT4x30Portable> for NTT4x30NeonRayon {}

mod ci {
    use poulpy_cpu_portable::{FFT64CIPortable, NTT4x30CIPortable};
    use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

    use crate::FFT64CINeon;
    use crate::NTT4x30CINeon;
    #[cfg(feature = "enable-rayon")]
    use crate::{FFT64CINeonRayon, NTT4x30CINeonRayon};

    unsafe impl VecZnxDftLayoutCompatible<FFT64CINeon> for FFT64CIPortable {}
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIPortable> for FFT64CINeon {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CINeon> for FFT64CIPortable {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIPortable> for FFT64CINeon {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CINeon> for FFT64CIPortable {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIPortable> for FFT64CINeon {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CINeonRayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIPortable> for FFT64CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CINeonRayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIPortable> for FFT64CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CINeonRayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIPortable> for FFT64CINeonRayon {}

    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CINeon> for NTT4x30CIPortable {}
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeon {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CINeon> for NTT4x30CIPortable {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeon {}
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CINeon> for NTT4x30CIPortable {}
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeon {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeonRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CINeonRayon> for NTT4x30CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<NTT4x30CIPortable> for NTT4x30CINeonRayon {}
}
