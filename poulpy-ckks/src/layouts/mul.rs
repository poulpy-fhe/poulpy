//! Prepared right operand for hoisted CKKS `ct×ct` multiplication.

use crate::SlotsKind;
use poulpy_core::layouts::{GLWEInfos, GLWELayout};
use poulpy_hal::layouts::Backend;
use poulpy_hal::layouts::CnvPVecROwned;

/// A CKKS ciphertext prepared as the right operand of a `ct×ct` multiply.
///
/// Produced by
/// [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right)
/// and consumed by
/// [`CKKSMulOps::ckks_mul_prepared_assign`](crate::api::CKKSMulOps::ckks_mul_prepared_assign).
/// It bundles the backend-resident convolution operand with the scale metadata of
/// the ciphertext it was prepared from, so the multiply can derive the result
/// precision without re-reading the source. Preparing once and multiplying many
/// times (e.g. the same `X^{gsp}` across a BSGS giant-step level) hoists the
/// forward transform out of the per-multiply path.
pub struct CKKSPreparedRight<BE: Backend> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecROwned<BE>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// `log_delta` of the source ciphertext.
    pub(crate) log_delta: usize,
    /// Torus width `k` of the source ciphertext.
    pub(crate) k: usize,
    /// `log_sparsity` of the source ciphertext.
    pub(crate) log_sparsity: usize,
    /// Slot kind of the source ciphertext.
    pub(crate) slots: SlotsKind,
    /// Ring/radix/rank identity captured at prepare time; prepared operands are
    /// long-lived cached objects, so `ckks_mul_prepared_assign` validates this
    /// against the destination before use.
    pub(crate) layout: GLWELayout,
}

/// What a prepared right operand contributes to a scratch query: its layout,
/// its effective width and the limb count it was prepared at.
///
/// Lets a query describe an operand it has not built, which is what the
/// lockstep EvalMod scratch replay needs.
pub trait CKKSPreparedRightInfos {
    fn prepared_layout(&self) -> GLWELayout;
    fn prepared_k(&self) -> usize;
    fn prepared_size(&self) -> usize;
}

impl<BE: Backend> CKKSPreparedRightInfos for CKKSPreparedRight<BE> {
    fn prepared_layout(&self) -> GLWELayout {
        self.layout
    }

    fn prepared_k(&self) -> usize {
        self.k
    }

    fn prepared_size(&self) -> usize {
        self.size
    }
}

/// Metadata-only stand-in for a [`CKKSPreparedRight`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CKKSPreparedRightLayout {
    pub layout: GLWELayout,
    /// Torus width `k` of the source ciphertext.
    pub k: usize,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub size: usize,
}

impl CKKSPreparedRightLayout {
    /// The layout [`CKKSMulOps::ckks_prepare_right`] would produce from `a`.
    ///
    /// [`CKKSMulOps::ckks_prepare_right`]: crate::api::CKKSMulOps::ckks_prepare_right
    pub fn of<A>(a: &A) -> Self
    where
        A: GLWEInfos,
    {
        let k: usize = a.k().into();
        Self {
            layout: GLWELayout {
                n: a.n(),
                base2k: a.base2k(),
                k: a.k(),
                rank: a.rank(),
            },
            k,
            size: k.div_ceil(a.base2k().as_usize()),
        }
    }
}

impl CKKSPreparedRightInfos for CKKSPreparedRightLayout {
    fn prepared_layout(&self) -> GLWELayout {
        self.layout
    }

    fn prepared_k(&self) -> usize {
        self.k
    }

    fn prepared_size(&self) -> usize {
        self.size
    }
}
