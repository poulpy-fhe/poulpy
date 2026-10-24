//! Reference bodies of the required SHIP hooks: the theta-column masking
//! accumulation and the hoisted base-B mux blind rotation.

use std::collections::HashMap;

use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{GGLWEInfos, LWEInfos},
    reference::keyswitching::glwe::GGLWEProductReference,
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, CnvPVecL, Module, ScratchArena},
};

use crate::layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, ShipPlan};

pub(crate) mod masking;
pub(crate) mod mux;

/// Reference bodies of the required [`CKKSShipImpl`](crate::oep::CKKSShipImpl) hooks.
pub trait CKKSShipReference<BE: Backend>: Sized {
    /// Scratch of [`Self::ckks_ship_masking_accumulate_reference`].
    fn ckks_ship_masking_accumulate_tmp_bytes_reference(&self, plan: &ShipPlan, base2k: usize) -> usize
    where
        Self: ModuleN
            + Convolution<BE>
            + CnvPVecBytesOf
            + VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes,
    {
        masking::ship_masking_tmp_bytes(self, plan, base2k)
    }

    /// `acc = Σ_i masks[i] · pis[i]`: each plaintext prepared once, the convolutions
    /// summed in the DFT domain, one inverse transform and normalization per column.
    fn ckks_ship_masking_accumulate_reference(
        &self,
        acc: &mut CKKSCiphertextOwned<BE>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
        pis: &[CKKSPlaintextOwned<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + Convolution<BE>
            + CnvPVecBytesOf
            + VecZnxDftBytesOf
            + VecZnxIdftApply<BE>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigNormalize<BE>,
    {
        masking::ship_masking_accumulate(self, acc, plan, masks, pis, scratch)
    }

    /// Scratch of [`Self::ckks_ship_mux_rotate_reference`] for `term_count` keys like `key`.
    fn ckks_ship_mux_rotate_tmp_bytes_reference<C, K>(&self, ct: &C, key: &K, term_count: usize) -> usize
    where
        Self: ModuleN
            + VecZnxDftBytesOf
            + VecZnxBigBytesOf
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes
            + VecZnxGadgetDecomposeTmpBytes
            + GGLWEProductReference<BE>,
        C: LWEInfos,
        K: GGLWEInfos,
    {
        mux::ship_mux_rotate_tmp_bytes(self, ct, key, term_count)
    }

    /// `ct ← Σ_d β_d · Rot_{rot_d}(ct)` over the keys of one digit position: the
    /// gadget input shared by every key, one product per key and its DFT-domain
    /// automorphism under `plans`, one inverse transform and normalization.
    fn ckks_ship_mux_rotate_reference(
        &self,
        ct: &mut CKKSCiphertextOwned<BE>,
        keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
        plans: &HashMap<i64, <Self as VecZnxDftAutomorphismPlan<BE>>::Plan>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + VecZnxDftApply<BE>
            + VecZnxDftZero<BE>
            + VecZnxDftAddAssign<BE>
            + VecZnxDftAutomorphism<BE>
            + VecZnxIdftApply<BE>
            + VecZnxIdftApplyTmpBytes
            + VecZnxBigNormalize<BE>
            + VecZnxDftBytesOf
            + VecZnxGadgetDecompose<BE>
            + GGLWEProductReference<BE>,
    {
        mux::ship_mux_rotate(self, ct, keys, plans, scratch)
    }
}

impl<BE: Backend> CKKSShipReference<BE> for Module<BE> {}
