//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the portable backend
//! for that container family, for every shape.

use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Avx512;
use crate::NTT4x30Avx512;
#[cfg(feature = "enable-rayon")]
use crate::{FFT64Avx512Rayon, NTT4x30Avx512Rayon};

unsafe impl VecZnxDftLayoutCompatible<FFT64Avx512> for FFT64Portable {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64Avx512 {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Avx512> for FFT64Portable {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64Avx512 {}
unsafe impl SvpPPolLayoutCompatible<FFT64Avx512> for FFT64Portable {}
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64Avx512 {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Avx512Rayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64Avx512Rayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Avx512Rayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64Avx512Rayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Avx512Rayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64Avx512Rayon {}

unsafe impl VecZnxBigLayoutCompatible<NTT4x30Avx512> for NTT4x30Portable {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30Avx512 {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Avx512Rayon> for NTT4x30Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30Avx512Rayon {}

mod ci {
    use poulpy_cpu_portable::{FFT64CIPortable, NTT4x30CIPortable};
    use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

    use crate::FFT64CIAvx512;
    #[cfg(feature = "enable-rayon")]
    use crate::FFT64CIAvx512Rayon;
    use crate::NTT4x30CIAvx512;
    #[cfg(feature = "enable-rayon")]
    use crate::NTT4x30CIAvx512Rayon;

    unsafe impl VecZnxDftLayoutCompatible<FFT64CIAvx512> for FFT64CIPortable {}
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512 {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIAvx512> for FFT64CIPortable {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512 {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIAvx512> for FFT64CIPortable {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512 {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIAvx512Rayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512Rayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIAvx512Rayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512Rayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIAvx512Rayon> for FFT64CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIPortable> for FFT64CIAvx512Rayon {}

    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIAvx512> for NTT4x30CIPortable {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIPortable> for NTT4x30CIAvx512 {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIAvx512Rayon> for NTT4x30CIPortable {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIPortable> for NTT4x30CIAvx512Rayon {}
}
