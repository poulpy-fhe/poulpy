use anyhow::{Result, ensure};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, Compact, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, ScratchArena, TransferFrom};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps, DFTOps},
    layouts::{BootstrappingContext, BootstrappingKeys, CKKSCiphertext, CKKSModuleAlloc},
};

/// Default (backend-generic) implementation of the CKKS bootstrapping
/// primitives.
///
/// The only genuinely new primitive bootstrapping needs is the modulus raise
/// ([`Self::ckks_mod_up_into_default`]); the full refresh
/// ([`Self::ckks_bootstrap_default`]) composes it with CoeffsToSlots, EvalMod and
/// SlotsToCoeffs from the existing op traits. See
/// [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) for the documented
/// semantics.
#[doc(hidden)]
pub trait CKKSBootstrappingOpsDefault<BE: Backend> {
    fn ckks_mod_up_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_mod_up_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        // `dst` is the (freshly-allocated) widened target: ModUp raises the modulus
        // into its full allocated capacity, so the widening width is `dst.max_k()`.
        // `dst.k()` is meta-derived and is `0` on a fresh ciphertext, which would
        // spuriously fail the "must widen" check below.
        let k_large: usize = dst.max_k().as_usize();
        let k_small: usize = src.k().as_usize();

        ensure!(
            k_large >= k_small,
            "ckks_mod_up: dst.k ({k_large}) < src.k ({k_small}); ModUp must widen, not shrink, the modulus"
        );

        // MSB-align `src` into the wider `dst`: the value occupies the top
        // `k_small` bits and the freshly-introduced low-order limbs are zero.
        self.glwe_copy(dst, src);

        // Shift the digits down to their natural integer magnitude. This is the
        // modulus raise: the raised-from modulus `q = 2^k_small` becomes an
        // explicit, un-reduced multiple `I(X)·q` living in the `[0, 2^k_large)`
        // window, which EvalMod subsequently removes. (A zero shift — when the
        // input already fills the storage — is a no-op.)
        self.glwe_rsh(k_large - k_small, dst, scratch);

        // `log_delta` is unchanged (the integer `Δ·m` keeps its scale); the
        // remaining headroom now spans the full raised modulus, so the torus
        // width `k` becomes `k_large` and `log_budget = k_large - log_delta`.
        dst.set_meta(CKKSMeta {
            log_delta: src.log_delta(),
            log_sparsity: src.log_sparsity(),
        });
        dst.set_k(k_large.into());

        Ok(())
    }

    /// Backend-generic reference for
    /// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    /// Pipeline is selected from the context's
    /// [`coeffs_to_slots_bypass`](BootstrappingContext::coeffs_to_slots_bypass).
    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_default<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        Self: GLWECopy<BE>
            + GLWEShift<BE>
            + CKKSModuleAlloc<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + DFTOps<BE>
            + CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertext<BE::OwnedBuf>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + Compact + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let base2k = ct_in.base2k();
        let k_boot = ct_out.k();
        let log_modulus_in = ct_in.k();

        // (encapsulate) denseToSparse → ModUp → sparseToDense. With encapsulation the
        // integer wrap-around `I·q` exposed by ModUp is bounded by the *sparse* secret's
        // Hamming weight (https://eprint.iacr.org/2022/024).
        let mut ct = self.ckks_ciphertext_alloc(base2k, k_boot);
        match keys.encapsulation_keys() {
            Some((dense_to_sparse, sparse_to_dense)) => {
                let mut ct0 = self.ckks_ciphertext_alloc(base2k, log_modulus_in);
                self.ckks_copy(&mut ct0, ct_in, scratch)?;
                self.glwe_keyswitch_assign(&mut ct0, dense_to_sparse, dense_to_sparse.max_size(), scratch);
                self.ckks_mod_up_into_default(&mut ct, &ct0, scratch)?;
                self.glwe_keyswitch_assign(&mut ct, sparse_to_dense, sparse_to_dense.max_size(), scratch);
            }
            None => {
                self.ckks_mod_up_into_default(&mut ct, ct_in, scratch)?;
            }
        }

        // Relabel at the input-modulus scale (free /message-ratio): the integer
        // wrap-around becomes the integer part, the message the residue.
        ct.set_meta(CKKSMeta {
            log_sparsity: ct_in.log_sparsity(),
            log_delta: log_modulus_in.as_usize(),
        });

        // CoeffsToSlots (split): coefficients → (real, imag) slots. In the standard
        // pipeline this feeds EvalMod directly; in EvalRound+ it is the low-precision
        // transform feeding the round.
        let mut r0 = self.ckks_ciphertext_alloc(base2k, k_boot);
        let mut i0 = self.ckks_ciphertext_alloc(base2k, k_boot);
        self.ckks_coeffs_to_slots_split(
            &mut r0,
            &mut i0,
            &ct,
            &ctx.coeffs_to_slots,
            keys.rotation_keys(),
            keys.conjugation_key(),
            scratch,
        )?;

        // EvalMod each half (scale-preserving; removes the integer part / leaves `Δm + e`).
        let mut res_real = self.ckks_ciphertext_alloc(base2k, k_boot);
        let mut res_imag = self.ckks_ciphertext_alloc(base2k, k_boot);
        self.ckks_eval_mod(&mut res_real, &r0, &ctx.eval_mod, keys.tensor_key(), scratch)?;
        self.ckks_eval_mod(&mut res_imag, &i0, &ctx.eval_mod, keys.tensor_key(), scratch)?;

        match &ctx.coeffs_to_slots_bypass {
            // Standard: EvalMod's clean residue goes straight to SlotsToCoeffs.
            None => {
                self.ckks_slots_to_coeffs_split(
                    ct_out,
                    &res_real,
                    &res_imag,
                    &ctx.slots_to_coeffs,
                    keys.rotation_keys(),
                    scratch,
                )?;
            }
            // EvalRound+: r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m). The
            // integer part and the low-precision error `e` both cancel, leaving the
            // message at the high-precision (bypass) transform's precision.
            Some(bypass) => {
                let mut r0_hp = self.ckks_ciphertext_alloc(base2k, k_boot);
                let mut i0_hp = self.ckks_ciphertext_alloc(base2k, k_boot);
                self.ckks_coeffs_to_slots_split(
                    &mut r0_hp,
                    &mut i0_hp,
                    &ct,
                    bypass,
                    keys.rotation_keys(),
                    keys.conjugation_key(),
                    scratch,
                )?;

                let log2_k = ctx.eval_mod.plan.f_mod_interval.trailing_zeros() as usize;
                self.ckks_mul_pow2_assign(&mut r0, log2_k, scratch)?;
                self.ckks_mul_pow2_assign(&mut i0, log2_k, scratch)?;
                self.ckks_sub_assign(&mut r0_hp, &r0, scratch)?;
                self.ckks_sub_assign(&mut i0_hp, &i0, scratch)?;
                self.ckks_add_assign(&mut r0_hp, &res_real, scratch)?;
                self.ckks_add_assign(&mut i0_hp, &res_imag, scratch)?;

                self.ckks_slots_to_coeffs_split(ct_out, &r0_hp, &i0_hp, &ctx.slots_to_coeffs, keys.rotation_keys(), scratch)?;
            }
        }

        Ok(())
    }
}
