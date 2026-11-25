//! Derived dot products.
use poulpy_core::layouts::GGLWEInfos;
use poulpy_hal::layouts::{Module, Ring};

use crate::{CKKSCtBounds, oep::CKKSDotProductImpl};

pub(crate) fn ckks_dot_product_prepared_tmp_bytes_derived<BE, R, A, B, T, Rg: Ring>(
    module: &Module<BE>,
    n: usize,
    res: &R,
    a: &A,
    b: &B,
    tsk: &T,
) -> usize
where
    BE: CKKSDotProductImpl<Rg>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    T: GGLWEInfos,
{
    BE::ckks_dot_product_tmp_bytes(module, n, res, a, b, tsk)
}
