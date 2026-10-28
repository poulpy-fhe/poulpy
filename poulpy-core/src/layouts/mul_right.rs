use poulpy_hal::layouts::{Backend, CnvPVecRToBackendRef, Data};

use crate::layouts::{GLWE, GLWEInfos, GLWEToBackendRef};

/// A right operand of
/// [`GLWEPreparedTensoring::glwe_mul_prepared_relinearize_sum`](crate::GLWEPreparedTensoring::glwe_mul_prepared_relinearize_sum):
/// a rank-1 ciphertext, either prepared as a right operand by [`crate::glwe_prepare_right`] or read as a ciphertext and
/// prepared by the product. Its infos are those of the ciphertext, and both forms give the same bits.
pub trait GLWEMulRight<BE: Backend>: GLWEInfos {
    type Prepared: CnvPVecRToBackendRef<BE>;
    type Ciphertext: GLWEToBackendRef<BE> + GLWEInfos;

    /// The operand in its form: prepared, on `ceil(k / base2k)` limbs of its infos, or the ciphertext itself.
    fn mul_right(&self) -> MulRightOperand<'_, Self::Prepared, Self::Ciphertext>;
}

/// The form of a [`GLWEMulRight`].
pub enum MulRightOperand<'a, P, C> {
    /// Prepared as a right operand, on `ceil(k / base2k)` limbs.
    Prepared(&'a P),
    /// A ciphertext, prepared in the product.
    Ciphertext(&'a C),
}

impl<BE: Backend, D: Data> GLWEMulRight<BE> for GLWE<D, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    type Prepared = poulpy_hal::layouts::CnvPVecROwned<BE>;
    type Ciphertext = Self;

    fn mul_right(&self) -> MulRightOperand<'_, Self::Prepared, Self> {
        MulRightOperand::Ciphertext(self)
    }
}
