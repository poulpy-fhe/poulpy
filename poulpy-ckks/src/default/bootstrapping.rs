use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
        SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
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
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSModuleAlloc, CKKSPlaintextOwned, EncodedLut, EncodedLutKind, EvalModType, ScratchArenaTakeCKKS, SlotsKind,
    },
};
use poulpy_core::GLWEBytesOf;

fn bootstrap_layouts<C1, C2>(ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
where
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let base2k = ct_in.base2k();
    let boot_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k,
            k: ct_out.k(),
            rank: Rank(1),
        },
        // `log_delta = 0` maximizes `log_budget`, which upper-bounds the
        // EvalMod working width.
        meta: CKKSMeta::default(),
    };
    let in_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_in.n(),
            base2k,
            k: ct_in.k(),
            rank: Rank(1),
        },
        meta: CKKSMeta::default(),
    };
    (boot_layout, in_layout)
}

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
        Self: GLWEBytesOf<BE>,
        Self: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
        let base2k = ct_in.base2k();
        let boot_ct_bytes = self.glwe_bytes_of_from_infos(&boot_layout);
        // The coefficient-plaintext proxy for the plaintext ops: EvalMod's
        // encoded polynomial coefficients, the widest plaintext the pipeline
        // multiplies by.
        let coeffs_meta = ctx.eval_mod().plan.coeffs_meta;
        let coeffs_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: ct_out.n(),
                base2k,
                k: coeffs_meta.k,
                rank: Rank(1),
            },
            meta: coeffs_meta.meta,
        };

        let in_ct_bytes = self.glwe_bytes_of_from_infos(&in_layout);

        let post_mod_up = if ctx.coeffs_to_slots_bypass().is_some() {
            5 * boot_ct_bytes
        } else {
            3 * boot_ct_bytes
        };
        let mut carved = match ctx.pipeline() {
            BootstrappingPipeline::C2SFirst => post_mod_up,
            BootstrappingPipeline::S2CFirst => post_mod_up.max(boot_ct_bytes + in_ct_bytes),
        };

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
            .max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), &keys_layout.tensor_key));

        if let Some(encaps) = &keys_layout.encapsulation {
            if ctx.pipeline() == BootstrappingPipeline::C2SFirst {
                carved += in_ct_bytes;
            }
            nested = nested
                .max(self.glwe_keyswitch_tmp_bytes(&in_layout.glwe_layout, &in_layout.glwe_layout, &encaps.dense_to_sparse))
                .max(self.glwe_keyswitch_tmp_bytes(&boot_layout, &boot_layout, &encaps.sparse_to_dense));
        }

        carved + nested
    }

    fn ckks_functional_bootstrap_tmp_bytes_default<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Self: GLWEBytesOf<BE>,
        Self: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let base = self.ckks_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, keys_layout);
        let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
        let boot_bytes = self.glwe_bytes_of_from_infos(&boot_layout);
        let in_bytes = self.glwe_bytes_of_from_infos(&in_layout);
        // The complex path holds `ct_raised`, `r0`, and `i0` while carving one
        // LUT-evaluation temporary. Direct S2C instead holds one boot-width
        // ciphertext beside its input-width buffer.
        let carved = (4 * boot_bytes).max(boot_bytes + in_bytes);
        let mut nested = self
            .ckks_all_ops_with_atk_tmp_bytes(
                &boot_layout,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                lut.coeffs(),
            )
            .max(self.ckks_all_ops_with_atk_tmp_bytes(
                &in_layout,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                lut.coeffs(),
            ));
        if lut.requires_eval_mod() {
            nested =
                nested.max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), &keys_layout.tensor_key));
        }
        base.max(carved + nested)
    }

    fn ckks_functional_bootstrap_multi_tmp_bytes_default<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Self: GLWEBytesOf<BE>,
        Self: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let single = luts
            .iter()
            .map(|lut| self.ckks_functional_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, lut, keys_layout))
            .max()
            .unwrap_or(0);
        let (boot_layout, _) = bootstrap_layouts(ct_out, ct_in);
        single.saturating_add(luts.len().saturating_mul(self.glwe_bytes_of_from_infos(&boot_layout)))
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
            ctx.coeffs_to_slots(),
            keys.rotation_keys(),
            keys.conjugation_key(),
            scratch,
        )
    }

    fn ckks_bootstrap_coeffs_to_slots_real<F, K, R1, R2>(
        &self,
        ct: &mut R1,
        conjugate: &mut R2,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDFTOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
        K: BootstrappingKeys<BE>,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        self.ckks_dft_evaluate_assign(ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)?;
        self.ckks_conjugate_into(conjugate, &*ct, keys.conjugation_key(), scratch)?;
        self.ckks_add_assign(ct, &*conjugate, scratch)
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
        self.ckks_eval_mod(res_real, r0, ctx.eval_mod(), keys.tensor_key(), scratch)?;
        self.ckks_eval_mod(res_imag, i0, ctx.eval_mod(), keys.tensor_key(), scratch)?;
        Ok(())
    }

    fn ckks_bootstrap_s2c_first<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
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
            + CKKSEvalModOps<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertextOwned<BE>:
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
            match ctx.coeffs_to_slots_bypass() {
                None => {
                    self.ckks_bootstrap_eval_mod_halves(&r0, &i0, ct_out, &mut ct_raised, ctx, keys, &mut scratch_local)?;
                    self.recombine_halves(ct_out, &mut ct_raised, &mut scratch_local)?;
                }
                Some(bypass) => {
                    let (mut r0_hp, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                    let (mut i0_hp, mut scratch_local) =
                        scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                    self.ckks_coeffs_to_slots_split(
                        &mut r0_hp,
                        &mut i0_hp,
                        &ct_raised,
                        bypass,
                        keys.rotation_keys(),
                        keys.conjugation_key(),
                        &mut scratch_local,
                    )?;

                    self.ckks_bootstrap_eval_mod_halves(&r0, &i0, ct_out, &mut ct_raised, ctx, keys, &mut scratch_local)?;

                    ckks_ensure!(
                        ctx.eval_mod().plan.f_mod_interval.is_power_of_two(),
                        "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                        ctx.eval_mod().plan.f_mod_interval
                    );
                    let log2_k = ctx.eval_mod().plan.f_mod_interval.trailing_zeros() as usize;
                    self.ckks_mul_pow2_assign(&mut r0, log2_k, &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut i0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut r0_hp, &r0, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut i0_hp, &i0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut r0_hp, ct_out, &mut scratch_local)?;
                    self.ckks_add_assign(&mut i0_hp, &ct_raised, &mut scratch_local)?;
                    self.recombine_halves(&mut r0_hp, &mut i0_hp, &mut scratch_local)?;
                    self.ckks_copy(ct_out, &r0_hp, &mut scratch_local)?;
                }
            }
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
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSDFTOps<BE>,
        K: BootstrappingKeys<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    {
        let input_layout = GLWELayout {
            n: ct_in.n(),
            base2k: ct_in.base2k(),
            k: ct_in.k(),
            rank: Rank(1),
        };

        scratch.scope(|scratch_inner| {
            let (mut ct_coeffs, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
            self.ckks_copy(&mut ct_coeffs, ct_in, &mut scratch_inner)?;
            // A `Split` decode matrix is numerically identical to the standard
            // matrix after the two halves are recombined. Preserve the split
            // path's normalization, which reconstructed `2 * ct_in`, then
            // intentionally use the format-agnostic evaluator directly.
            self.ckks_mul_pow2_assign(&mut ct_coeffs, 1, &mut scratch_inner)?;
            self.ckks_dft_evaluate_assign(
                &mut ct_coeffs,
                ctx.slots_to_coeffs(),
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
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
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
        CKKSCiphertextOwned<BE>:
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

        if ctx.pipeline() == BootstrappingPipeline::S2CFirst {
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
            match ctx.coeffs_to_slots_bypass() {
                // Standard: EvalMod's clean residue goes straight to SlotsToCoeffs.
                None => {
                    self.ckks_eval_mod(&mut ct, &r0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    r0.set_k(k_boot);
                    self.ckks_eval_mod(&mut r0, &i0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &ct,
                        &r0,
                        ctx.slots_to_coeffs(),
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
                        ctx.eval_mod().plan.f_mod_interval.is_power_of_two(),
                        "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                        ctx.eval_mod().plan.f_mod_interval
                    );
                    let log2_k = ctx.eval_mod().plan.f_mod_interval.trailing_zeros() as usize;

                    self.ckks_eval_mod(&mut ct, &r0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut r0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut r0_hp, &r0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut r0_hp, &ct, &mut scratch_local)?;

                    r0.set_k(k_boot);
                    self.ckks_eval_mod(&mut r0, &i0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut i0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut i0_hp, &i0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut i0_hp, &r0, &mut scratch_local)?;

                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &r0_hp,
                        &i0_hp,
                        ctx.slots_to_coeffs(),
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;
                }
            }

            Result::Ok(())
        })
    }

    fn ckks_bootstrap_real_default<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
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
            + CKKSConjugateOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ckks_ensure!(
            ctx.pipeline() == BootstrappingPipeline::S2CFirst,
            "real-slot bootstrapping requires an S2C-first context"
        );
        ckks_ensure!(
            ctx.coeffs_to_slots_bypass().is_none(),
            "real-slot bootstrapping does not support EvalRound+"
        );
        ckks_ensure!(
            ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
            "ckks_bootstrap_real supports rank-1 ciphertexts only"
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
            let (mut ct_raised, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            self.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;
            self.ckks_bootstrap_coeffs_to_slots_real(&mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
            self.ckks_eval_mod(ct_out, &ct_raised, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
            ct_out.set_meta(CKKSMeta {
                log_sparsity: ct_in.log_sparsity(),
                log_delta: ct_in.log_delta(),
            });
            Result::Ok(())
        })
    }
}

fn ensure_unit_circle_exp_context<BE: Backend, F>(ctx: &BootstrappingContext<BE, F>) -> Result<()> {
    ckks_ensure!(
        ctx.eval_mod().plan.eval_mod_type == EvalModType::ExpCmplx,
        "general functional bootstrapping requires an ExpCmplx EvalMod context"
    );
    ckks_ensure!(
        ctx.eval_mod().plan.scaling == Some(std::f64::consts::TAU),
        "general functional bootstrapping requires a unit-circle ExpCmplx context (scaling = 2π)"
    );
    Ok(())
}

fn functional_output_contract<BE: Backend, F>(
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    bootstrap_k: usize,
) -> Result<(CKKSMeta, usize)> {
    let log_delta = ct_in
        .log_delta()
        .checked_add(lut.log_msg_ratio())
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap output scale overflow")))?;
    let eval_mod = if lut.requires_eval_mod() {
        ctx.eval_mod().plan.consumed_bits()
    } else {
        0
    };
    let consumed = ctx
        .coeffs_to_slots()
        .consumed_bits()
        .checked_add(eval_mod)
        .and_then(|bits| bits.checked_add(lut.consumed_bits(log_delta)))
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap budget overflow")))?;
    let output_k = bootstrap_k.checked_sub(consumed).ok_or_else(|| {
        crate::CKKSError::Internal(anyhow::anyhow!(
            "functional bootstrap output width {bootstrap_k} is smaller than its {consumed}-bit post-ModUp cost"
        ))
    })?;
    ckks_ensure!(
        output_k >= log_delta,
        "functional bootstrap leaves output k={output_k} below log_delta={log_delta}"
    );
    Ok((
        CKKSMeta {
            log_delta,
            log_sparsity: ct_in.log_sparsity(),
        },
        output_k,
    ))
}

fn ckks_eval_encoded_lut<BE, F, K, C, R>(
    module: &Module<BE>,
    ct_out: &mut R,
    ct_in: &C,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    match lut.kind() {
        EncodedLutKind::General(series) => ckks_eval_lut(
            module,
            ct_out,
            ct_in,
            ctx.eval_mod(),
            series,
            keys.conjugation_key(),
            keys.tensor_key(),
            scratch,
        )?,
        EncodedLutKind::Binary {
            cos,
            affine,
            log_interval_reduction,
        } => ckks_eval_lut_binary(
            module,
            ct_out,
            ct_in,
            cos,
            *log_interval_reduction,
            affine,
            keys.tensor_key(),
            scratch,
        )?,
    }
    Ok(())
}

pub fn ckks_functional_bootstrap_default<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
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
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ckks_functional_bootstrap_with_slots(module, ct_out, ct_in, ctx, lut, keys, SlotsKind::Complex, scratch)
}

pub fn ckks_functional_bootstrap_real_default<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
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
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ckks_functional_bootstrap_with_slots(module, ct_out, ct_in, ctx, lut, keys, SlotsKind::Real, scratch)
}

#[allow(clippy::too_many_arguments)]
fn ckks_functional_bootstrap_with_slots<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    keys: &K,
    slots_kind: SlotsKind,
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
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ckks_ensure!(
        ctx.pipeline() == BootstrappingPipeline::S2CFirst,
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
    if lut.requires_eval_mod() {
        ensure_unit_circle_exp_context(ctx)?;
    }
    ensure_functional_message_ratio(ct_in, ctx.slots_to_coeffs().consumed_bits(), lut.log_msg_ratio())?;
    let (output_meta, output_k) = functional_output_contract(ct_in, ctx, lut, ct_out.k().as_usize())?;

    let boot_layout = GLWELayout {
        n: ct_out.n(),
        base2k: ct_in.base2k(),
        k: ct_out.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch_local| {
        let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        module.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;
        if slots_kind == SlotsKind::Real {
            let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            module.ckks_bootstrap_coeffs_to_slots_real(&mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
            return ckks_eval_encoded_lut(module, ct_out, &ct_raised, ctx, lut, keys, &mut scratch_local);
        }

        let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        module.ckks_bootstrap_coeffs_to_slots(&ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
        ckks_eval_encoded_lut(module, ct_out, &r0, ctx, lut, keys, &mut scratch_local)?;
        ckks_eval_encoded_lut(module, &mut ct_raised, &i0, ctx, lut, keys, &mut scratch_local)?;
        module.recombine_halves(ct_out, &mut ct_raised, &mut scratch_local)
    })?;
    ct_out.set_meta(output_meta);
    ckks_ensure!(
        ct_out.k().as_usize() >= output_k,
        "functional bootstrap produced k={}, below required {output_k}",
        ct_out.k().as_usize()
    );
    ct_out.set_k(output_k.into());
    Ok(())
}

pub fn ckks_functional_bootstrap_multi_default<BE, F, K>(
    module: &Module<BE>,
    ct_outs: &mut [CKKSCiphertextOwned<BE>],
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
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
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
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
    let log_msg_ratio = luts[0].log_msg_ratio();
    ckks_ensure!(
        luts.iter().all(|lut| lut.log_msg_ratio() == log_msg_ratio),
        "functional bootstrap LUTs must have the same message ratio"
    );
    let use_shared_general_evaluation = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());

    ckks_ensure!(
        ctx.pipeline() == BootstrappingPipeline::S2CFirst,
        "functional bootstrapping requires an S2C-first context"
    );
    if luts.iter().any(EncodedLut::requires_eval_mod) {
        ensure_unit_circle_exp_context(ctx)?;
    }
    let head = &ct_outs[0];
    let (out_n, out_k) = (head.n(), head.k());
    ckks_ensure!(
        ct_in.rank().as_usize() == 1
            && ct_outs.iter().all(|ct| {
                ct.rank().as_usize() == 1 && ct.n() == head.n() && ct.base2k() == head.base2k() && ct.k() == head.k()
            }),
        "functional bootstrapping outputs must share one rank-1 layout"
    );
    ckks_ensure!(
        keys.encapsulation_keys().is_some() == ctx.sparse_secret_hamming_weight().is_some(),
        "bootstrapping key encapsulation does not match the compiled recipe"
    );
    ensure_functional_message_ratio(ct_in, ctx.slots_to_coeffs().consumed_bits(), log_msg_ratio)?;
    let output_contracts = ct_outs
        .iter()
        .zip(luts)
        .map(|(ct_out, lut)| functional_output_contract(ct_in, ctx, lut, ct_out.k().as_usize()))
        .collect::<Result<Vec<_>>>()?;

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

        let (mut im_outs, mut scratch_local) =
            scratch_local.take_ckks_ciphertext_slice_scratch(luts.len(), &boot_layout, CKKSMeta::default());
        if use_shared_general_evaluation {
            ckks_eval_lut_multi(
                module,
                ct_outs,
                &r0,
                ctx.eval_mod(),
                luts,
                keys.conjugation_key(),
                keys.tensor_key(),
                &mut scratch_local,
            )?;
            ckks_eval_lut_multi(
                module,
                &mut im_outs,
                &i0,
                ctx.eval_mod(),
                luts,
                keys.conjugation_key(),
                keys.tensor_key(),
                &mut scratch_local,
            )?;
        } else {
            for ((ct_out, im_out), lut) in ct_outs.iter_mut().zip(&mut im_outs).zip(luts) {
                ckks_eval_encoded_lut(module, ct_out, &r0, ctx, lut, keys, &mut scratch_local)?;
                ckks_eval_encoded_lut(module, im_out, &i0, ctx, lut, keys, &mut scratch_local)?;
            }
        }

        for (ct_out, im_i) in ct_outs.iter_mut().zip(im_outs.iter_mut()) {
            module.recombine_halves(ct_out, im_i, &mut scratch_local)?;
        }
        Result::Ok(())
    })?;
    for (ct_out, (meta, expected_k)) in ct_outs.iter_mut().zip(output_contracts) {
        ct_out.set_meta(meta);
        ckks_ensure!(
            ct_out.k().as_usize() >= expected_k,
            "functional bootstrap produced k={}, below required {expected_k}",
            ct_out.k().as_usize()
        );
        ct_out.set_k(expected_k.into());
    }
    Ok(())
}

fn ensure_functional_message_ratio<C>(ct_in: &C, s2c_consumed_bits: usize, expected: usize) -> Result<()>
where
    C: CKKSInfos,
{
    let log_budget = ct_in.log_budget();
    ckks_ensure!(
        log_budget >= s2c_consumed_bits,
        "functional bootstrap needs log_budget >= {s2c_consumed_bits}, got {log_budget}"
    );
    let actual = log_budget - s2c_consumed_bits;
    ckks_ensure!(
        actual == expected,
        "functional bootstrap LUT requires log_msg_ratio {expected}, got {actual}"
    );
    Ok(())
}
