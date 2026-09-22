//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape.

use crate::FFT64AvxBackend;
#[cfg(feature = "enable-rayon")]
use crate::FFT64AvxRayonBackend;
use crate::NTT4x30AvxBackend;
#[cfg(feature = "enable-rayon")]
use crate::NTT4x30AvxRayonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_cpu_ref::{FFT64RefBackend, NTT4x30RefBackend};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64AvxBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64AvxBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64AvxBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxBackend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64AvxRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64AvxRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64AvxRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64AvxRayonBackend<R> {}

unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30AvxBackend<R>> for NTT4x30RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30AvxBackend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30AvxRayonBackend<R>> for NTT4x30RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30AvxRayonBackend<R> {}
