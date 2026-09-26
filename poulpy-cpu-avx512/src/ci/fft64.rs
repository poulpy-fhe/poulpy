pub(crate) use crate::fft64::automorphism;
#[path = "../fft64/module.rs"]
pub(crate) mod module;
#[cfg(feature = "enable-rayon")]
#[path = "../fft64/rayon.rs"]
pub(crate) mod rayon;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CIAvx512;
pub use FFT64CIAvx512 as FFT64Avx512;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CIAvx512Rayon;
#[cfg(feature = "enable-rayon")]
pub use FFT64CIAvx512Rayon as FFT64Avx512Rayon;

pub(crate) use automorphism::fft64_vec_znx_dft_automorphism_avx512;
