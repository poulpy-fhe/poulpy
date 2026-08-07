use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        Rank, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertext, CKKSModuleAlloc,
        ScratchArenaTakeCKKS,
    },
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

    /// Scratch upper bound for [`Self::ckks_bootstrap_default`]: the pipeline
    /// intermediates it carves from scratch (rank-1 bootstrap-width working
    /// ciphertexts — five on the standard path, seven with the EvalRound+
    /// bypass, plus the input-width copy when encapsulation is enabled) on top
    /// of the largest nested stage (the per-family all-ops bound, EvalMod, and
    /// the encapsulation key switches).
    fn ckks_bootstrap_tmp_bytes_default<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Self: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertext<BE::OwnedBuf>: CKKSCtBounds,
    {
        let base2k = ct_in.base2k();
        // `log_delta = 0` maximizes `log_budget`, which upper-bounds the EvalMod
        // working width.
        let boot_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: ct_out.n(),
                base2k,
                k: ct_out.k(),
                rank: Rank(1),
            },
            meta: CKKSMeta {
                log_delta: 0,
                log_sparsity: 0,
            },
        };
        let boot_ct_bytes = GLWE::<Vec<u8>>::bytes_of_from_infos(&boot_layout);
        // The coefficient-plaintext proxy for the plaintext ops: EvalMod's
        // encoded polynomial coefficients, the widest plaintext the pipeline
        // multiplies by.
        let coeffs_meta = ctx.eval_mod.plan.coeffs_meta;
        let coeffs_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: ct_out.n(),
                base2k,
                k: coeffs_meta.k,
                rank: Rank(1),
            },
            meta: coeffs_meta.meta,
        };

        // Worst-path carve: `ct`, `r0`, `i0`, `res_real`, `res_imag` are
        // co-resident through EvalMod; the bypass adds `r0_hp`/`i0_hp`.
        let mut carved = 5 * boot_ct_bytes;
        if ctx.coeffs_to_slots_bypass.is_some() {
            carved += 2 * boot_ct_bytes;
        }

        let mut nested = self
            .ckks_all_ops_with_atk_tmp_bytes(
                &boot_layout,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                &coeffs_layout,
            )
            .max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, &ctx.eval_mod, &keys_layout.tensor_key));

        if let Some(encaps) = &keys_layout.encapsulation {
            let in_layout = GLWELayout {
                n: ct_in.n(),
                base2k,
                k: ct_in.k(),
                rank: Rank(1),
            };
            carved += GLWE::<Vec<u8>>::bytes_of_from_infos(&in_layout);
            nested = nested
                .max(self.glwe_keyswitch_tmp_bytes(&in_layout, &in_layout, &encaps.dense_to_sparse))
                .max(self.glwe_keyswitch_tmp_bytes(&boot_layout, &boot_layout, &encaps.sparse_to_dense));
        }

        carved + nested
    }

    fn ckks_mod_up_into_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos,
    {
        // `dst` is the (freshly-allocated) widened target: ModUp raises the modulus
        // to the destination's requested modulus, so the widening width is `dst.k()`.
        // `dst.k()` is meta-derived and is `0` on a fresh ciphertext, which would
        // spuriously fail the "must widen" check below.
        let k_large: usize = dst.k().as_usize();
        let k_small: usize = src.k().as_usize();

        ckks_ensure!(
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
        Self: GLWECopy<BE>
            + GLWEShift<BE>
            + CKKSModuleAlloc<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertext<BE::OwnedBuf>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        // TODO(HalfBTS): remove this guard when S2C-first is wired.
        ckks_ensure!(
            ctx.pipeline == BootstrappingPipeline::C2SFirst,
            "S2C-first bootstrapping is not implemented"
        );

        // All pipeline intermediates are rank-1 working ciphertexts carved from
        // scratch (accounted for by `ckks_bootstrap_tmp_bytes`); reject
        // higher-rank inputs up front.
        ckks_ensure!(
            ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
            "ckks_bootstrap supports rank-1 ciphertexts only, got rank {} -> {}",
            ct_in.rank().as_usize(),
            ct_out.rank().as_usize()
        );

        let encapsulation_keys = keys.encapsulation_keys();
        ckks_ensure!(
            encapsulation_keys.is_some() == ctx.sparse_secret_hamming_weight().is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe (expected {}, got {})",
            ctx.sparse_secret_hamming_weight().is_some(),
            encapsulation_keys.is_some()
        );

        let base2k = ct_in.base2k();
        let k_boot = ct_out.k();
        let log_modulus_in = ct_in.k();
        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k,
            k: k_boot,
            rank: Rank(1),
        };

        scratch.scope(|scratch_local| {
            // (encapsulate) denseToSparse → ModUp → sparseToDense. With encapsulation the
            // integer wrap-around `I·q` exposed by ModUp is bounded by the *sparse* secret's
            // Hamming weight (https://eprint.iacr.org/2022/024).
            let (mut ct, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            match encapsulation_keys {
                Some((dense_to_sparse, sparse_to_dense)) => {
                    // The input-width copy is scoped so its scratch is released
                    // right after ModUp widens it into `ct`.
                    scratch_local.scope(|scratch_inner| {
                        let (mut ct0, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(
                            &GLWELayout {
                                n: ct_in.n(),
                                base2k,
                                k: log_modulus_in,
                                rank: Rank(1),
                            },
                            ct_in.meta(),
                        );
                        self.ckks_copy(&mut ct0, ct_in, &mut scratch_inner)?;
                        self.glwe_keyswitch_assign(&mut ct0, dense_to_sparse, &mut scratch_inner);
                        self.ckks_mod_up_into_default(&mut ct, &ct0, &mut scratch_inner)
                    })?;
                    self.glwe_keyswitch_assign(&mut ct, sparse_to_dense, &mut scratch_local);
                }
                None => {
                    self.ckks_mod_up_into_default(&mut ct, ct_in, &mut scratch_local)?;
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
            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            self.ckks_coeffs_to_slots_split(
                &mut r0,
                &mut i0,
                &ct,
                &ctx.coeffs_to_slots,
                keys.rotation_keys(),
                keys.conjugation_key(),
                &mut scratch_local,
            )?;

            // EvalMod each half (scale-preserving; removes the integer part / leaves `Δm + e`).
            let (mut res_real, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            let (mut res_imag, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            self.ckks_eval_mod(&mut res_real, &r0, &ctx.eval_mod, keys.tensor_key(), &mut scratch_local)?;
            self.ckks_eval_mod(&mut res_imag, &i0, &ctx.eval_mod, keys.tensor_key(), &mut scratch_local)?;

            match &ctx.coeffs_to_slots_bypass {
                // Standard: EvalMod's clean residue goes straight to SlotsToCoeffs.
                None => {
                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &res_real,
                        &res_imag,
                        &ctx.slots_to_coeffs,
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;
                }
                // EvalRound+: r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m). The
                // integer part and the low-precision error `e` both cancel, leaving the
                // message at the high-precision (bypass) transform's precision.
                Some(bypass) => {
                    let (mut r0_hp, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
                    let (mut i0_hp, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
                    self.ckks_coeffs_to_slots_split(
                        &mut r0_hp,
                        &mut i0_hp,
                        &ct,
                        bypass,
                        keys.rotation_keys(),
                        keys.conjugation_key(),
                        &mut scratch_local,
                    )?;

                    // `BootstrappingContext::compile` rejects this, but re-check
                    // before the power-of-two shift.
                    ckks_ensure!(
                        ctx.eval_mod.plan.f_mod_interval.is_power_of_two(),
                        "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                        ctx.eval_mod.plan.f_mod_interval
                    );
                    let log2_k = ctx.eval_mod.plan.f_mod_interval.trailing_zeros() as usize;
                    self.ckks_mul_pow2_assign(&mut r0, log2_k, &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut i0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut r0_hp, &r0, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut i0_hp, &i0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut r0_hp, &res_real, &mut scratch_local)?;
                    self.ckks_add_assign(&mut i0_hp, &res_imag, &mut scratch_local)?;

                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &r0_hp,
                        &i0_hp,
                        &ctx.slots_to_coeffs,
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;
                }
            }

            Result::Ok(())
        })
    }
}
