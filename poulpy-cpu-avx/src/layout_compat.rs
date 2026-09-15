//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape.

use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Avx;
#[cfg(feature = "enable-rayon")]
use crate::FFT64AvxRayon;
use crate::NTT4x30Avx;
#[cfg(feature = "enable-rayon")]
use crate::NTT4x30AvxRayon;

unsafe impl VecZnxDftLayoutCompatible<FFT64Avx> for FFT64Portable {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64Avx {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Avx> for FFT64Portable {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64Avx {}
unsafe impl SvpPPolLayoutCompatible<FFT64Avx> for FFT64Portable {}
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64Avx {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64AvxRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Portable> for FFT64AvxRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64AvxRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Portable> for FFT64AvxRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64AvxRayon> for FFT64Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Portable> for FFT64AvxRayon {}

unsafe impl VecZnxBigLayoutCompatible<NTT4x30Avx> for NTT4x30Portable {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30Avx {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30AvxRayon> for NTT4x30Portable {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Portable> for NTT4x30AvxRayon {}
