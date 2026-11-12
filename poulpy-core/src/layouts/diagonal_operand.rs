use poulpy_hal::layouts::{Backend, CnvPVecROwned, Ring, Standard};

use crate::layouts::GLWEBackendRef;

/// A linear-transformation diagonal as the right operand of the products of its giant step: prepared, on
/// `ceil(k / base2k)` limbs, or a GLWE of rank 0 (a plaintext) or of the source's rank (a ciphertext) the product
/// prepares; both give the same bits.
#[expect(
    clippy::large_enum_variant,
    reason = "a GLWE view carries its plaintext metadata, a box would allocate per diagonal"
)]
pub enum DiagonalOperand<'a, BE: Backend + 'a, Rg: Ring = Standard, P = CnvPVecROwned<BE, Rg>> {
    /// Prepared as a right operand, on `ceil(k / base2k)` limbs.
    Prepared(&'a P),
    /// Not prepared yet: the product prepares it.
    Unprepared(GLWEBackendRef<'a, BE, Rg>),
}
