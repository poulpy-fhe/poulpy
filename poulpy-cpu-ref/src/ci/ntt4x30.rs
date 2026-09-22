#[path = "../ntt4x30/module.rs"]
pub(crate) mod module;
#[path = "../ntt4x30/prim.rs"]
pub(crate) mod prim;
#[path = "../ntt4x30/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
#[path = "../ntt4x30/znx.rs"]
pub(crate) mod znx;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CIRef;
pub use NTT4x30CIRef as NTT4x30Ref;
