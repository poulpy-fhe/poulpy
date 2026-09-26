//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape.

use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Avx;
#[cfg(feature = "enable-rayon")]
use crate::FFT64AvxRayon;
use crate::NTT4x30Avx;
#[cfg(feature = "enable-rayon")]
use crate::NTT4x30AvxRayon;

unsafe impl VecZnxDftLayoutCompatible<FFT64Avx> for FFT64Ref {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64Avx {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Avx> for FFT64Ref {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64Avx {}
unsafe impl SvpPPolLayoutCompatible<FFT64Avx> for FFT64Ref {}
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64Avx {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64AvxRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64AvxRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64AvxRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64AvxRayon {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64AvxRayon> for FFT64Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64AvxRayon {}

unsafe impl VecZnxBigLayoutCompatible<NTT4x30Avx> for NTT4x30Ref {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30Avx {}

#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30AvxRayon> for NTT4x30Ref {}
#[cfg(feature = "enable-rayon")]
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30AvxRayon {}

mod ci {
    use poulpy_cpu_ref::{FFT64CIRef, NTT4x30CIRef};
    use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

    use crate::FFT64CIAvx;
    #[cfg(feature = "enable-rayon")]
    use crate::FFT64CIAvxRayon;
    use crate::NTT4x30CIAvx;
    #[cfg(feature = "enable-rayon")]
    use crate::NTT4x30CIAvxRayon;

    unsafe impl VecZnxDftLayoutCompatible<FFT64CIAvx> for FFT64CIRef {}
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIRef> for FFT64CIAvx {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIAvx> for FFT64CIRef {}
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIRef> for FFT64CIAvx {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIAvx> for FFT64CIRef {}
    unsafe impl SvpPPolLayoutCompatible<FFT64CIRef> for FFT64CIAvx {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIAvxRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxDftLayoutCompatible<FFT64CIRef> for FFT64CIAvxRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIAvxRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<FFT64CIRef> for FFT64CIAvxRayon {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIAvxRayon> for FFT64CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl SvpPPolLayoutCompatible<FFT64CIRef> for FFT64CIAvxRayon {}

    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIAvx> for NTT4x30CIRef {}
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIRef> for NTT4x30CIAvx {}

    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIAvxRayon> for NTT4x30CIRef {}
    #[cfg(feature = "enable-rayon")]
    unsafe impl VecZnxBigLayoutCompatible<NTT4x30CIRef> for NTT4x30CIAvxRayon {}
}
