//! Cross-backend layout-compatibility declarations.
//!
//! # Safety
//!
//! Each impl asserts byte-identical buffer layouts with the reference backend
//! for that container family, for every shape.

use crate::FFT64Avx512Backend;
use crate::NTT4x30Avx512Backend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_cpu_ref::{FFT64RefBackend, NTT4x30RefBackend};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

#[cfg(feature = "enable-rayon")]
use crate::{FFT64Avx512RayonBackend, NTT4x30Avx512RayonBackend};

unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64Avx512Backend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512Backend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64Avx512Backend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512Backend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64Avx512Backend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512Backend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64Avx512RayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512RayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64Avx512RayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512RayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64Avx512RayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64Avx512RayonBackend<R> {}

unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30Avx512Backend<R>> for NTT4x30RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30Avx512Backend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30Avx512RayonBackend<R>> for NTT4x30RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30Avx512RayonBackend<R> {}
