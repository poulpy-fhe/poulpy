//! Reference bodies of the SHIP family: the half-bootstrapping drivers over the selected CKKS operations, poulpy-core's
//! key switch and zero and the masking and mux hooks, with their preflight and scratch sizing, and the hooks
//! themselves: the theta-column masking accumulation and the hoisted base-B mux blind rotation.

use std::{borrow::Borrow, collections::HashMap};

use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendRef, LWEInfos},
    oep::{GLWEKeyswitchImpl, GLWEZeroImpl},
    reference::keyswitching::glwe::GGLWEProductReference,
};
use poulpy_hal::{
    api::{
        CnvOps, CnvPVecBytesOf, ModuleN, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftZero, VecZnxGadgetDecompose,
        VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, CnvPVecL, Module, ScratchArena},
    oep::HalVecZnxDftImpl,
};

use crate::{
    CKKSCtBounds,
    api::ShipScalar,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, ShipKeysPrepared, ShipPlan},
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSEncodingImpl, CKKSImagImpl, CKKSMulImpl, CKKSShipCoeffEncodingImpl, CKKSShipImpl,
        CKKSSubImpl,
    },
};

mod bootstrap;
pub(crate) mod masking;
pub(crate) mod mux;
mod preflight;

/// The families the SHIP drivers and their query compose.
pub(super) trait ShipBackend:
    CKKSShipImpl
    + CKKSShipCoeffEncodingImpl
    + CKKSMulImpl
    + CKKSAddImpl
    + CKKSSubImpl
    + CKKSImagImpl
    + CKKSConjugateImpl
    + GLWEKeyswitchImpl
    + GLWEZeroImpl
    + HalVecZnxDftImpl
{
}

impl<BE> ShipBackend for BE where
    BE: CKKSShipImpl
        + CKKSShipCoeffEncodingImpl
        + CKKSMulImpl
        + CKKSAddImpl
        + CKKSSubImpl
        + CKKSImagImpl
        + CKKSConjugateImpl
        + GLWEKeyswitchImpl
        + GLWEZeroImpl
        + HalVecZnxDftImpl
{
}

/// Automorphism plans of the mux output twists, keyed by Galois element.
pub(crate) type ShipMuxPlans<BE> = HashMap<i64, <BE as HalVecZnxDftImpl>::AutomorphismPlan>;

/// Builds the automorphism plans of every distinct mux Galois element once;
/// the same rotation amounts recur across all support slots.
pub(crate) fn ship_mux_plans<'a, BE>(
    module: &Module<BE>,
    groups: impl Iterator<Item = &'a [HMuxRotKeyPrepared<BE::OwnedBuf, BE>]>,
) -> ShipMuxPlans<BE>
where
    BE: HalVecZnxDftImpl + 'a,
{
    let mut plans = ShipMuxPlans::<BE>::new();
    for group in groups {
        for key in group {
            if key.gal_el != 1 {
                plans
                    .entry(key.gal_el)
                    .or_insert_with(|| BE::vec_znx_dft_automorphism_plan(module, module.n(), key.gal_el));
            }
        }
    }
    plans
}

/// Reference bodies of [`CKKSShipImpl`], forwarded by `impl_ckks_ship_reference_full!`. The drivers reach the masking
/// and mux hooks and their queries through the selected [`CKKSShipImpl`] and the coefficient encoding through the
/// selected [`CKKSShipCoeffEncodingImpl`].
pub trait CKKSShipReference<BE: Backend>: Borrow<Module<BE>> + Sized {
    /// The runtime validation, refusing as the drivers do, then the largest of the encapsulation (the larger of
    /// poulpy-core's `glwe_keyswitch_tmp_bytes` and the coefficient-encoding query, plus the bottom ciphertext it
    /// carves), the product tree (the add-plaintext, the selected `ckks_ship_masking_accumulate_tmp_bytes`,
    /// `ckks_ship_mux_rotate_tmp_bytes` of each group at its key count, and `ckks_mul_tmp_bytes` queries) and the final
    /// combination (its temporaries plus the conjugate, add, sub and multiply-by-`i` queries), the complex combination
    /// counted whenever the keys carry its masks.
    fn ckks_ship_bootstrap_tmp_bytes_reference<F, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    ) -> Result<usize>
    where
        BE: CKKSShipImpl
            + CKKSShipCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSImagImpl
            + CKKSConjugateImpl
            + GLWEKeyswitchImpl
            + GLWEZeroImpl
            + HalVecZnxDftImpl,
        F: ShipScalar,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        preflight::ckks_ship_bootstrap_tmp_bytes::<BE, F, Src>(self.borrow(), output, input, keys)
    }

    /// The runtime validation of the output, input and key layouts, then an arena below the selected
    /// `ckks_ship_bootstrap_tmp_bytes`, refuse before any write. Then poulpy-core's `glwe_keyswitch` of `input` into a
    /// carved ciphertext under the dense-to-sparse key (carrying the input's metadata), the selected
    /// `ckks_ship_coeff_encodings`, a leaf 0 (poulpy-core's `glwe_zero` then `ckks_add_pt_vec_assign` of the constant
    /// encoding), one leaf per index key (the selected `ckks_ship_masking_accumulate` then `ckks_ship_mux_rotate` per
    /// mux group), and a binary product tree of `ckks_mul_into` under the tensor key, an odd leftover carrying to the
    /// next level, a tree that exhausts the budget being `MultiplicationPrecisionUnderflow`, and
    /// `output = root + conj(root)` with the selected `ckks_conjugate_into` and `ckks_add_into`, the slot kind set to
    /// real.
    fn ckks_ship_bootstrap_into_reference<F, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSShipImpl
            + CKKSShipCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSImagImpl
            + CKKSConjugateImpl
            + GLWEKeyswitchImpl
            + GLWEZeroImpl
            + HalVecZnxDftImpl,
        F: ShipScalar,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        bootstrap::ckks_ship_bootstrap_into::<BE, F, Src>(self.borrow(), output, input, keys, scratch)
    }

    /// The runtime validation of the output, input and key layouts, then an arena below the selected
    /// `ckks_ship_bootstrap_tmp_bytes`, refuse before any write. Then the encapsulation and coefficient encodings of
    /// [`Self::ckks_ship_bootstrap_into_reference`] once, then for each of the two coefficient halves its own leaves
    /// (the second half with the second mask set) and product tree as there, and with `v1` and `v2` the two roots,
    /// `ckks_mul_i_into` of `v2`, `ckks_add_into` and `ckks_sub_into` of it with `v1`, `ckks_conjugate_into` of the
    /// difference, and `output = (v1 + i·v2) + conj(v1 − i·v2)` by `ckks_add_into`.
    fn ckks_ship_bootstrap_complex_into_reference<F, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSShipImpl
            + CKKSShipCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSImagImpl
            + CKKSConjugateImpl
            + GLWEKeyswitchImpl
            + GLWEZeroImpl
            + HalVecZnxDftImpl,
        F: ShipScalar,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        bootstrap::ckks_ship_bootstrap_complex_into::<BE, F, Src>(self.borrow(), output, input, keys, scratch)
    }

    /// Scratch of [`Self::ckks_ship_masking_accumulate_reference`].
    fn ckks_ship_masking_accumulate_tmp_bytes_reference(&self, plan: &ShipPlan, base2k: usize) -> usize
    where
        Self: ModuleN
            + CnvOps<BE>
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
            + CnvOps<BE>
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
