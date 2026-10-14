//! Prepared operands for hoisted CKKS `ct×ct` multiplication.

use crate::CKKSInfos;
use poulpy_core::ComponentNoise;
use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, CnvPVecLOwned, CnvPVecROwned};

/// A CKKS ciphertext prepared as the left operand of a `ct×ct` product.
///
/// Produced by
/// [`CKKSMulOps::ckks_prepare_left`](crate::api::CKKSMulOps::ckks_prepare_left)
/// and consumed by
/// [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared).
/// It bundles the backend-resident convolution operand with the layout,
/// metadata and noise estimate of the ciphertext it was prepared from, which
/// its infos report, so a product reads the operand without its source.
pub struct CKKSPreparedLeft<BE: Backend> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecLOwned<BE>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// Ring degree, radix, width `k` and rank of the source ciphertext.
    pub(crate) layout: GLWELayout,
    /// Plaintext metadata of the source ciphertext.
    pub(crate) meta: GLWEPlaintextMeta,
    /// Noise estimate of the source ciphertext.
    pub(crate) noise: Option<ComponentNoise>,
}

/// A CKKS ciphertext prepared as the right operand of a `ct×ct` product.
///
/// Produced by
/// [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right)
/// and consumed by
/// [`CKKSMulOps::ckks_mul_prepared_assign`](crate::api::CKKSMulOps::ckks_mul_prepared_assign)
/// and [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared).
/// It bundles the backend-resident convolution operand with the layout,
/// metadata and noise estimate of the ciphertext it was prepared from, which
/// its infos report, so the multiply can derive the result precision without
/// re-reading the source.
/// Preparing once and multiplying many times (e.g. the same `X^{gsp}` across a
/// BSGS giant-step level) hoists the forward transform out of the per-multiply
/// path.
pub struct CKKSPreparedRight<BE: Backend> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecROwned<BE>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// Ring degree, radix, width `k` and rank of the source ciphertext;
    /// prepared operands are long-lived cached objects, so the products
    /// validate it against the destination before use.
    pub(crate) layout: GLWELayout,
    /// Plaintext metadata of the source ciphertext.
    pub(crate) meta: GLWEPlaintextMeta,
    /// Noise estimate of the source ciphertext.
    pub(crate) noise: Option<ComponentNoise>,
}

impl<BE: Backend> CKKSPreparedLeft<BE> {
    /// The prepared convolution operand, on `ceil(k / base2k)` limbs and `rank + 1` columns of the source.
    pub fn cnv(&self) -> &CnvPVecLOwned<BE> {
        &self.prep
    }
}

impl<BE: Backend> CKKSPreparedRight<BE> {
    /// The prepared convolution operand, on `ceil(k / base2k)` limbs and `rank + 1` columns of the source.
    pub fn cnv(&self) -> &CnvPVecROwned<BE> {
        &self.prep
    }
}

// The infos of both prepared operands report the source ciphertext, so they are
// `CKKSCtBounds`; `max_size` is the prepared limb count.
macro_rules! impl_prepared_infos {
    ($name:ident) => {
        impl<BE: Backend> LWEInfos for $name<BE> {
            fn noise(&self) -> Option<ComponentNoise> {
                self.noise.clone()
            }

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

        impl<BE: Backend> GLWEInfos for $name<BE> {
            fn rank(&self) -> Rank {
                self.layout.rank
            }
        }

        impl<BE: Backend> GLWEPlaintextInfos for $name<BE> {
            fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
                Some(self.meta)
            }
        }

        impl<BE: Backend> CKKSInfos for $name<BE> {}
    };
}

impl_prepared_infos!(CKKSPreparedLeft);
impl_prepared_infos!(CKKSPreparedRight);
