//! Prepared operands for hoisted CKKS `ct×ct` multiplication.

use crate::CKKSInfos;
use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta, IntPolyInfos, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecLToBackendRef, CnvPVecRBackendRef, CnvPVecROwned, CnvPVecRToBackendRef,
    Ring, Standard,
};

/// A CKKS ciphertext prepared as the left operand of a `ct×ct` product.
///
/// Allocated by
/// [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepared_alloc_from_infos`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepared_alloc_from_infos)
/// and filled by
/// [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare),
/// or produced by
/// [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc),
/// and consumed by
/// [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared).
/// It bundles the backend-resident convolution operand with the layout and
/// metadata of the ciphertext it was prepared from, which its infos report, so
/// a product reads the operand without its source. It carries no noise
/// estimate: only fresh ciphertexts do.
pub struct CKKSLeftOperandPrepared<BE: Backend, Rg: Ring = Standard> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecLOwned<BE, Rg>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// Ring degree, radix, width `k` and rank of the source ciphertext.
    pub(crate) layout: GLWELayout,
    /// Plaintext metadata of the source ciphertext, `None` when it has none.
    pub(crate) meta: Option<GLWEPlaintextMeta>,
}

/// A CKKS ciphertext prepared as the right operand of a `ct×ct` product.
///
/// Allocated by
/// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepared_alloc_from_infos`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepared_alloc_from_infos)
/// and filled by
/// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepare),
/// or produced by
/// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc),
/// and consumed by
/// [`CKKSMulOps::ckks_mul_prepared_assign`](crate::api::CKKSMulOps::ckks_mul_prepared_assign)
/// and [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared).
/// It bundles the backend-resident convolution operand with the layout and
/// metadata of the ciphertext it was prepared from, which its infos report, so
/// the multiply can derive the result precision without re-reading the source.
/// It carries no noise estimate: only fresh ciphertexts do.
/// Preparing once and multiplying many times (e.g. the same `X^{gsp}` across a
/// BSGS giant-step level) hoists the forward transform out of the per-multiply
/// path.
pub struct CKKSRightOperandPrepared<BE: Backend, Rg: Ring = Standard> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecROwned<BE, Rg>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// Ring degree, radix, width `k` and rank of the source ciphertext;
    /// prepared operands are long-lived cached objects, so the products
    /// validate it against the destination before use.
    pub(crate) layout: GLWELayout,
    /// Plaintext metadata of the source ciphertext, `None` when it has none.
    pub(crate) meta: Option<GLWEPlaintextMeta>,
}

/// The former name of [`CKKSRightOperandPrepared`].
#[deprecated(note = "renamed to `CKKSRightOperandPrepared`")]
pub type CKKSPreparedRight<BE, Rg = Standard> = CKKSRightOperandPrepared<BE, Rg>;

impl<BE: Backend, Rg: Ring> CKKSLeftOperandPrepared<BE, Rg> {
    /// The prepared convolution operand, on `ceil(k / base2k)` limbs and `rank + 1` columns of the source.
    pub fn cnv(&self) -> &CnvPVecLOwned<BE, Rg> {
        &self.prep
    }
}

impl<BE: Backend, Rg: Ring> CKKSRightOperandPrepared<BE, Rg> {
    /// The prepared convolution operand, on `ceil(k / base2k)` limbs and `rank + 1` columns of the source.
    pub fn cnv(&self) -> &CnvPVecROwned<BE, Rg> {
        &self.prep
    }
}

// The core products read a prepared operand where it is, through its convolution operand.
impl<BE: Backend, Rg: Ring> CnvPVecLToBackendRef<BE, Rg> for CKKSLeftOperandPrepared<BE, Rg> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE, Rg> {
        self.prep.to_backend_ref()
    }
}

impl<BE: Backend, Rg: Ring> CnvPVecRToBackendRef<BE, Rg> for CKKSRightOperandPrepared<BE, Rg> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE, Rg> {
        self.prep.to_backend_ref()
    }
}

// The infos of both prepared operands report the source ciphertext, so they are
// `CKKSCtBounds`; `max_size` is the prepared limb count.
macro_rules! impl_prepared_infos {
    ($name:ident) => {
        impl<BE: Backend, Rg: Ring> LWEInfos for $name<BE, Rg> {
            fn n(&self) -> Degree {
                self.layout.n
            }

            fn base2k(&self) -> Base2K {
                self.layout.base2k
            }

            fn max_size(&self) -> usize {
                self.size
            }

            fn k(&self) -> TorusPrecision {
                self.layout.k
            }
        }

        impl<BE: Backend, Rg: Ring> GLWEInfos for $name<BE, Rg> {
            fn rank(&self) -> Rank {
                self.layout.rank
            }
        }

        impl<BE: Backend, Rg: Ring> GLWEPlaintextInfos for $name<BE, Rg> {
            fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
                self.meta
            }
        }

        impl<BE: Backend, Rg: Ring> CKKSInfos for $name<BE, Rg> {}
    };
}

impl_prepared_infos!(CKKSLeftOperandPrepared);
impl_prepared_infos!(CKKSRightOperandPrepared);

impl<BE: Backend, Rg: Ring> IntPolyInfos for CKKSRightOperandPrepared<BE, Rg> {
    /// Its whole stored width, as a plaintext's: every prepared limb enters a product.
    fn encoded_k(&self) -> TorusPrecision {
        self.max_k()
    }
}
