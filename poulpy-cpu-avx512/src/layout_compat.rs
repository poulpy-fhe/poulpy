//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape.

use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

use crate::FFT64Avx512;
use crate::NTT4x30Avx512;

unsafe impl VecZnxDftLayoutCompatible<FFT64Avx512> for FFT64Ref {}
unsafe impl VecZnxDftLayoutCompatible<FFT64Ref> for FFT64Avx512 {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Avx512> for FFT64Ref {}
unsafe impl VecZnxBigLayoutCompatible<FFT64Ref> for FFT64Avx512 {}
unsafe impl SvpPPolLayoutCompatible<FFT64Avx512> for FFT64Ref {}
unsafe impl SvpPPolLayoutCompatible<FFT64Ref> for FFT64Avx512 {}

unsafe impl VecZnxBigLayoutCompatible<NTT4x30Avx512> for NTT4x30Ref {}
unsafe impl VecZnxBigLayoutCompatible<NTT4x30Ref> for NTT4x30Avx512 {}
