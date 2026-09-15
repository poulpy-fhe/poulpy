//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
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
