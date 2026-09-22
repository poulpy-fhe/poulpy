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

use crate::FFT64NeonBackend;
use crate::NTT4x30NeonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_cpu_ref::{FFT64RefBackend, NTT4x30RefBackend};
use poulpy_hal::layouts::{SvpPPolLayoutCompatible, VecZnxBigLayoutCompatible, VecZnxDftLayoutCompatible};

#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayonBackend, NTT4x30NeonRayonBackend};

unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64NeonBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64NeonBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64NeonBackend<R>> for FFT64RefBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonBackend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64NeonRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64NeonRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64NeonRayonBackend<R>> for FFT64RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<FFT64RefBackend<R>> for FFT64NeonRayonBackend<R> {}

unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<NTT4x30NeonBackend<R>> for NTT4x30RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30NeonBackend<R>> for NTT4x30RefBackend<R> {}
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<NTT4x30NeonBackend<R>> for NTT4x30RefBackend<R> {}
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonBackend<R> {}

#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<NTT4x30NeonRayonBackend<R>> for NTT4x30RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxDftLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30NeonRayonBackend<R>> for NTT4x30RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> VecZnxBigLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonRayonBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<NTT4x30NeonRayonBackend<R>> for NTT4x30RefBackend<R> {}
#[cfg(feature = "enable-rayon")]
unsafe impl<R: CpuRing> SvpPPolLayoutCompatible<NTT4x30RefBackend<R>> for NTT4x30NeonRayonBackend<R> {}
