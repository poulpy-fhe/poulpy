pub(crate) use crate::ntt3x42_ifma::bbc_meta;
#[path = "../ntt3x42_ifma/convolution.rs"]
pub(crate) mod convolution;
pub(crate) use crate::ntt3x42_ifma::execution;
pub(crate) use crate::ntt3x42_ifma::kernels;
pub(crate) use crate::ntt3x42_ifma::mat_vec_ifma;
#[path = "../ntt3x42_ifma/module.rs"]
pub(crate) mod module;
#[path = "../ntt3x42_ifma/prim.rs"]
pub(crate) mod prim;
pub(crate) use crate::ntt3x42_ifma::primes;
#[cfg(feature = "enable-rayon")]
#[path = "../ntt3x42_ifma/rayon.rs"]
pub(crate) mod rayon;

#[path = "../ntt3x42_ifma/svp.rs"]
pub(crate) mod svp;
pub(crate) use crate::ntt3x42_ifma::tables;
pub(crate) use crate::ntt3x42_ifma::traits;
pub(crate) use crate::ntt3x42_ifma::types;
#[path = "../ntt3x42_ifma/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
#[path = "../ntt3x42_ifma/vec_znx_dft.rs"]
pub(crate) mod vec_znx_dft;
#[path = "../ntt3x42_ifma/vmp.rs"]
pub(crate) mod vmp;
#[path = "../ntt3x42_ifma/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT3x42CIIfma;
pub use NTT3x42CIIfma as NTT3x42Ifma;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT3x42CIIfmaRayon;
#[cfg(feature = "enable-rayon")]
pub use NTT3x42CIIfmaRayon as NTT3x42IfmaRayon;
#[cfg(all(feature = "enable-rayon", feature = "enable-ckks"))]
pub(crate) use rayon::vmp_apply_digits_strided_known_zero_prefix;
#[cfg(feature = "enable-rayon")]
pub type NTT3x42IfmaRayonExecutor = poulpy_cpu_rayon::RayonTaskExecutor;
#[cfg(feature = "enable-rayon")]
poulpy_hal::impl_backend_from!(NTT3x42IfmaRayon, NTT3x42Ifma, NTT3x42IfmaRayonExecutor);
