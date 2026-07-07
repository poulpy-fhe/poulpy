use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        Rank, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSImagOps,
        CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_multi},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertext, CKKSModuleAlloc,
        CKKSPlaintext, EncodedLut, ScratchArenaTakeCKKS,
    },
    polynomial::ComplexBSGSPolynomial,
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

    /// Scratch upper bound for [`Self::ckks_bootstrap_default`].
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

        let in_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: ct_in.n(),
                base2k,
                k: ct_in.k(),
                rank: Rank(1),
            },
            meta: CKKSMeta {
                log_delta: 0,
                log_sparsity: 0,
            },
        };
        let in_ct_bytes = GLWE::<Vec<u8>>::bytes_of_from_infos(&in_layout);

        let mut carved = match ctx.pipeline {
            BootstrappingPipeline::C2SFirst => 5 * boot_ct_bytes,
            BootstrappingPipeline::S2CFirst => (3 * boot_ct_bytes).max(boot_ct_bytes + 4 * in_ct_bytes),
        };
        if ctx.pipeline == BootstrappingPipeline::C2SFirst && ctx.coeffs_to_slots_bypass.is_some() {
            carved += 2 * boot_ct_bytes;
        }

        let mut nested = self
            .ckks_all_ops_with_atk_tmp_bytes(
                &boot_layout,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                &coeffs_layout,
            )
            .max(self.ckks_all_ops_with_atk_tmp_bytes(
                &in_layout,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                &coeffs_layout,
            ))
            .max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, &ctx.eval_mod, &keys_layout.tensor_key));

        if let Some(encaps) = &keys_layout.encapsulation {
            if ctx.pipeline == BootstrappingPipeline::C2SFirst {
                carved += in_ct_bytes;
            }
            nested = nested
                .max(self.glwe_keyswitch_tmp_bytes(&in_layout.glwe_layout, &in_layout.glwe_layout, &encaps.dense_to_sparse))
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

    fn ckks_bootstrap_mod_up_from_mut<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE>,
        K: BootstrappingKeys<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        match keys.encapsulation_keys() {
            Some((dense_to_sparse, sparse_to_dense)) => {
                self.glwe_keyswitch_assign(src, dense_to_sparse, scratch);
                self.ckks_mod_up_into_default(dst, src, scratch)?;
                self.glwe_keyswitch_assign(dst, sparse_to_dense, scratch);
            }
            None => self.ckks_mod_up_into_default(dst, src, scratch)?,
        }
        Ok(())
    }

    fn recombine_halves<R1, R2>(&self, res_re: &mut R1, res_im: &mut R2, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSImagOps<BE> + CKKSAddOps<BE>,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        self.ckks_mul_i_assign(res_im, scratch)?;
        self.ckks_add_assign(res_re, &*res_im, scratch)?;
        Ok(())
    }

    fn ckks_bootstrap_coeffs_to_slots<F, K, C, R>(
        &self,
        ct: &C,
        r0: &mut R,
        i0: &mut R,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDFTOps<BE>,
        K: BootstrappingKeys<BE>,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        self.ckks_coeffs_to_slots_split(
            r0,
            i0,
            ct,
            &ctx.coeffs_to_slots,
            keys.rotation_keys(),
            keys.conjugation_key(),
            scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_eval_mod_halves<F, K, C, R1, R2>(
        &self,
        r0: &C,
        i0: &C,
        res_real: &mut R1,
        res_imag: &mut R2,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    {
        self.ckks_eval_mod(res_real, r0, &ctx.eval_mod, keys.tensor_key(), scratch)?;
        self.ckks_eval_mod(res_imag, i0, &ctx.eval_mod, keys.tensor_key(), scratch)?;
        Ok(())
    }

    fn ckks_bootstrap_s2c_first<F, K>(
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
            + GLWEKeyswitch<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertext<BE::OwnedBuf>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k: ct_in.base2k(),
            k: ct_out.k(),
            rank: Rank(1),
        };

        scratch.scope(|scratch_local| {
            let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            self.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;

            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            self.ckks_bootstrap_coeffs_to_slots(&ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
            self.ckks_bootstrap_eval_mod_halves(&r0, &i0, ct_out, &mut ct_raised, ctx, keys, &mut scratch_local)?;
            self.recombine_halves(ct_out, &mut ct_raised, &mut scratch_local)?;
            ct_out.set_meta(CKKSMeta {
                log_sparsity: ct_in.log_sparsity(),
                log_delta: ct_in.log_delta(),
            });
            Result::Ok(())
        })
    }

    fn ckks_bootstrap_s2c_mod_up<F, K, R>(
        &self,
        ct_raised: &mut R,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>,
        K: BootstrappingKeys<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        CKKSCiphertext<BE::OwnedBuf>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    {
        let input_layout = GLWELayout {
            n: ct_in.n(),
            base2k: ct_in.base2k(),
            k: ct_in.k(),
            rank: Rank(1),
        };

        scratch.scope(|scratch_inner| {
            let (mut ct_coeffs, scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
            let (mut conj, scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
            let (mut re_half, scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
            let (mut im_half, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());

            self.ckks_conjugate_into(&mut conj, ct_in, keys.conjugation_key(), &mut scratch_inner)?;
            self.ckks_add_into(&mut re_half, ct_in, &conj, &mut scratch_inner)?;
            self.ckks_sub_into(&mut im_half, ct_in, &conj, &mut scratch_inner)?;
            self.ckks_div_i_assign(&mut im_half, &mut scratch_inner)?;
            self.ckks_slots_to_coeffs_split(
                &mut ct_coeffs,
                &re_half,
                &im_half,
                &ctx.slots_to_coeffs,
                keys.rotation_keys(),
                &mut scratch_inner,
            )?;

            let log_modulus_in = ct_coeffs.k().as_usize();
            self.ckks_bootstrap_mod_up_from_mut(ct_raised, &mut ct_coeffs, keys, &mut scratch_inner)?;
            ct_raised.set_meta(CKKSMeta {
                log_sparsity: ct_in.log_sparsity(),
                log_delta: log_modulus_in,
            });
            Result::Ok(())
        })
    }

    /// Backend-generic reference for
    /// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    /// Pipeline is selected from the context.
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
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertext<BE::OwnedBuf>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
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

        if ctx.pipeline == BootstrappingPipeline::S2CFirst {
            return self.ckks_bootstrap_s2c_first(ct_out, ct_in, ctx, keys, scratch);
        }

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
                Some(_) => {
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
                        self.ckks_bootstrap_mod_up_from_mut(&mut ct, &mut ct0, keys, &mut scratch_inner)
                    })?;
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
            self.ckks_bootstrap_coeffs_to_slots(&ct, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;

            // EvalMod each half (scale-preserving; removes the integer part / leaves `Δm + e`).
            let (mut res_real, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            let (mut res_imag, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            self.ckks_bootstrap_eval_mod_halves(&r0, &i0, &mut res_real, &mut res_imag, ctx, keys, &mut scratch_local)?;

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

#[allow(clippy::too_many_arguments)]
pub fn ckks_functional_bootstrap_default<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
    ct_in: &CKKSCiphertext<BE::OwnedBuf>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSBootstrappingOpsDefault<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + CKKSDFTOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertext<BE::OwnedBuf>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ckks_ensure!(
        ctx.pipeline == BootstrappingPipeline::S2CFirst,
        "functional bootstrapping requires an S2C-first context"
    );
    ckks_ensure!(
        ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
        "functional bootstrapping supports rank-1 ciphertexts only"
    );
    ckks_ensure!(
        keys.encapsulation_keys().is_some() == ctx.sparse_secret_hamming_weight().is_some(),
        "bootstrapping key encapsulation does not match the compiled recipe"
    );

    let boot_layout = GLWELayout {
        n: ct_out.n(),
        base2k: ct_in.base2k(),
        k: ct_out.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch_local| {
        let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        module.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;
        let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        module.ckks_bootstrap_coeffs_to_slots(&ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;

        let mut res_im = module.ckks_ciphertext_alloc(ct_out.base2k(), ct_out.max_k());
        match lut {
            EncodedLut::General(series) => {
                ckks_eval_lut(
                    module,
                    ct_out,
                    &r0,
                    &ctx.eval_mod,
                    series,
                    keys.conjugation_key(),
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
                ckks_eval_lut(
                    module,
                    &mut res_im,
                    &i0,
                    &ctx.eval_mod,
                    series,
                    keys.conjugation_key(),
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
            }
            EncodedLut::Binary {
                cos,
                affine,
                log_interval_reduction,
            } => {
                ckks_eval_lut_binary(
                    module,
                    ct_out,
                    &r0,
                    cos,
                    *log_interval_reduction,
                    affine,
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
                ckks_eval_lut_binary(
                    module,
                    &mut res_im,
                    &i0,
                    cos,
                    *log_interval_reduction,
                    affine,
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
            }
        }
        module.recombine_halves(ct_out, &mut res_im, &mut scratch_local)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn ckks_functional_bootstrap_multi_default<BE, F, K>(
    module: &Module<BE>,
    ct_outs: &mut [CKKSCiphertext<BE::OwnedBuf>],
    ct_in: &CKKSCiphertext<BE::OwnedBuf>,
    ctx: &BootstrappingContext<BE, F>,
    luts: &[ComplexBSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>],
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSBootstrappingOpsDefault<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + CKKSDFTOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertext<BE::OwnedBuf>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ckks_ensure!(
        !luts.is_empty(),
        "ckks_functional_bootstrap_multi: at least one LUT is required"
    );
    ckks_ensure!(
        ct_outs.len() == luts.len(),
        "ckks_functional_bootstrap_multi: ct_outs/luts length mismatch ({} vs {})",
        ct_outs.len(),
        luts.len()
    );

    ckks_ensure!(
        ctx.pipeline == BootstrappingPipeline::S2CFirst,
        "functional bootstrapping requires an S2C-first context"
    );
    let head = &ct_outs[0];
    let (out_n, out_base2k, out_k, out_max_k) = (head.n(), head.base2k(), head.k(), head.max_k());
    ckks_ensure!(
        ct_in.rank().as_usize() == 1
            && ct_outs.iter().all(|ct| {
                ct.rank().as_usize() == 1 && ct.n() == head.n() && ct.base2k() == head.base2k() && ct.max_k() == head.max_k()
            }),
        "functional bootstrapping outputs must share one rank-1 layout"
    );
    ckks_ensure!(
        keys.encapsulation_keys().is_some() == ctx.sparse_secret_hamming_weight().is_some(),
        "bootstrapping key encapsulation does not match the compiled recipe"
    );

    let boot_layout = GLWELayout {
        n: out_n,
        base2k: ct_in.base2k(),
        k: out_k,
        rank: Rank(1),
    };
    scratch.scope(|scratch_local| {
        let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        module.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;
        let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        module.ckks_bootstrap_coeffs_to_slots(&ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;

        let mut im_outs: Vec<_> = (0..luts.len())
            .map(|_| module.ckks_ciphertext_alloc(out_base2k, out_max_k))
            .collect();
        ckks_eval_lut_multi(
            module,
            ct_outs,
            &r0,
            &ctx.eval_mod,
            luts,
            keys.conjugation_key(),
            keys.tensor_key(),
            &mut scratch_local,
        )?;
        ckks_eval_lut_multi(
            module,
            &mut im_outs,
            &i0,
            &ctx.eval_mod,
            luts,
            keys.conjugation_key(),
            keys.tensor_key(),
            &mut scratch_local,
        )?;

        for (ct_out, im_i) in ct_outs.iter_mut().zip(im_outs.iter_mut()) {
            module.recombine_halves(ct_out, im_i, &mut scratch_local)?;
        }
        Result::Ok(())
    })
}
