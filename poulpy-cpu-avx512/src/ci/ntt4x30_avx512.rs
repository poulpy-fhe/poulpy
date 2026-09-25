pub(crate) use crate::ntt4x30_avx512::arithmetic_avx512;
#[path = "../ntt4x30_avx512/convolution.rs"]
pub(crate) mod convolution;
pub(crate) use crate::ntt4x30_avx512::mat_vec_avx512;
#[path = "../ntt4x30_avx512/module.rs"]
pub(crate) mod module;
pub(crate) use crate::ntt4x30_avx512::ntt;
#[path = "../ntt4x30_avx512/prim.rs"]
pub(crate) mod prim;
#[cfg(feature = "enable-rayon")]
#[path = "../ntt4x30_avx512/rayon.rs"]
pub(crate) mod rayon;
#[path = "../ntt4x30_avx512/svp.rs"]
pub(crate) mod svp;
#[path = "../ntt4x30_avx512/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
#[path = "../ntt4x30_avx512/vec_znx_dft.rs"]
pub(crate) mod vec_znx_dft;
#[path = "../ntt4x30_avx512/vmp.rs"]
pub(crate) mod vmp;
#[path = "../ntt4x30_avx512/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CIAvx512;
pub use NTT4x30CIAvx512 as NTT4x30Avx512;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NTT4x30CIAvx512Rayon;
#[cfg(feature = "enable-rayon")]
pub use NTT4x30CIAvx512Rayon as NTT4x30Avx512Rayon;
