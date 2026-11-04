//! Public GBFV operations, implemented on `Module<BE>` for every backend that
//! implements the matching [`crate::oep`] family.

mod ntt;

pub use ntt::GBFVNttOps;
