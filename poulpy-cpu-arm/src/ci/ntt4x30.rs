#[path = "../ntt4x30/module.rs"]
pub(crate) mod module;
#[path = "../ntt4x30/prim.rs"]
pub(crate) mod prim;
#[cfg(feature = "enable-rayon")]
#[path = "../ntt4x30/rayon.rs"]
pub(crate) mod rayon;
#[path = "../ntt4x30/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
#[cfg(target_arch = "aarch64")]
#[path = "../ntt4x30/vmp.rs"]
pub(crate) mod vmp;
#[path = "../ntt4x30/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CINeon;
pub use NTT4x30CINeon as NTT4x30Neon;
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CINeonRayon;
#[cfg(feature = "enable-rayon")]
pub use NTT4x30CINeonRayon as NTT4x30NeonRayon;
