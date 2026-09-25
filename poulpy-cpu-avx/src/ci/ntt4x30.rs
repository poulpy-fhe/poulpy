pub(crate) use crate::ntt4x30::arithmetic_avx;
pub(crate) use crate::ntt4x30::convolution;
pub(crate) use crate::ntt4x30::mat_vec_avx;
#[path = "../ntt4x30/module.rs"]
pub(crate) mod module;
pub(crate) use crate::ntt4x30::ntt;
#[path = "../ntt4x30/prim.rs"]
pub(crate) mod prim;
#[cfg(feature = "enable-rayon")]
#[path = "../ntt4x30/rayon.rs"]
pub(crate) mod rayon;
#[path = "../ntt4x30/svp.rs"]
pub(crate) mod svp;
#[path = "../ntt4x30/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
pub(crate) use crate::ntt4x30::vec_znx_big_avx;
#[path = "../ntt4x30/vec_znx_dft.rs"]
pub(crate) mod vec_znx_dft;
#[path = "../ntt4x30/vmp.rs"]
pub(crate) mod vmp;
#[path = "../ntt4x30/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CIAvx;
pub use NTT4x30CIAvx as NTT4x30Avx;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NTT4x30CIAvxRayon;
#[cfg(feature = "enable-rayon")]
pub use NTT4x30CIAvxRayon as NTT4x30AvxRayon;
