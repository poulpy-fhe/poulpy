#[path = "../fft64/module.rs"]
pub(crate) mod module;
#[cfg(feature = "enable-rayon")]
#[path = "../fft64/rayon.rs"]
pub(crate) mod rayon;
#[path = "../fft64/reim.rs"]
pub(crate) mod reim;
#[path = "../fft64/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CINeon;
pub use FFT64CINeon as FFT64Neon;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FFT64CINeonRayon;
#[cfg(feature = "enable-rayon")]
pub use FFT64CINeonRayon as FFT64NeonRayon;
