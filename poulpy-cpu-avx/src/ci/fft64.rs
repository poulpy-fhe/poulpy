pub(crate) use crate::fft64::automorphism;
#[path = "../fft64/module.rs"]
pub(crate) mod module;
#[cfg(feature = "enable-rayon")]
#[path = "../fft64/rayon.rs"]
pub(crate) mod rayon;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CIAvx;
pub use FFT64CIAvx as FFT64Avx;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CIAvxRayon;
#[cfg(feature = "enable-rayon")]
pub use FFT64CIAvxRayon as FFT64AvxRayon;

pub(crate) use automorphism::fft64_vec_znx_dft_automorphism_avx;
