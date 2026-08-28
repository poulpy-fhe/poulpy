use crate::{CKKSResult as Result, ckks_ensure, layouts::eval_mod::EvalModPlan};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
        SetBSGSMeta,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    execution::{TaskExecutor, worker_scratch_bytes},
    layouts::{Backend, Module, ScratchArena},
};
use std::ops::Deref;

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSImagOps,
        CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_from_basis, ckks_lut_power_basis},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSModuleAlloc, CKKSPlaintextOwned, EncodedLut, EncodedLutKind, EvalModType, ScratchArenaTakeCKKS,
    },
    oep::CKKSEncapsulatedModUpImpl,
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

/// Private backend-generic implementation of the CKKS bootstrapping
/// composition. Keeping this as a helper value, rather than another trait,
/// leaves the public operation trait as the single execution abstraction.
pub(crate) struct BootstrappingDefault<'a, BE: Backend>(&'a Module<BE>);

impl<'a, BE: Backend> BootstrappingDefault<'a, BE> {
    pub(crate) fn new(module: &'a Module<BE>) -> Self {
        Self(module)
    }
}

impl<BE: Backend> Deref for BootstrappingDefault<'_, BE> {
    type Target = Module<BE>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<BE: Backend + CKKSEncapsulatedModUpImpl<BE>> BootstrappingDefault<'_, BE> {
    pub(crate) fn ckks_mod_up_tmp_bytes_default(&self) -> usize
    where
        Module<BE>: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    /// Scratch upper bound for [`Self::ckks_bootstrap_default`].
    pub(crate) fn ckks_bootstrap_tmp_bytes_default<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
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

        let eval_mod_tmp = self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), &keys_layout.tensor_key);
        let eval_mod_tmp = if <BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
            2 * worker_scratch_bytes::<BE>(eval_mod_tmp)
        } else {
            eval_mod_tmp
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
            .max(eval_mod_tmp);

        if ctx.pipeline() == BootstrappingPipeline::C2SFirst {
            carved += in_ct_bytes;
        }

        if let Some(encaps) = &keys_layout.encapsulation {
            nested = nested.max(BE::ckks_encapsulated_mod_up_tmp_bytes(
                self.0,
                &boot_layout,
                &in_layout,
                &encaps.dense_to_sparse,
                &encaps.sparse_to_dense,
            ));
        }

        carved + nested
    }

    /// Scratch upper bound for [`ckks_functional_bootstrap_default`], for one
    /// LUT or a batch: the batch folds each imaginary half as it is produced,
    /// so the carved working set does not grow with `luts.len()`.
    pub(crate) fn ckks_functional_bootstrap_tmp_bytes_default<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let base = self.ckks_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, keys_layout);
        let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
        let boot_bytes = self.glwe_bytes_of_from_infos(&boot_layout);
        let in_bytes = self.glwe_bytes_of_from_infos(&in_layout);
        // The complex path holds `ct_raised`, `r0` and `i0` while carving one
        // LUT-evaluation temporary. Direct S2C instead holds one boot-width
        // ciphertext beside its input-width buffer.
        let carved = (4 * boot_bytes).max(boot_bytes + in_bytes);
        let mut nested = 0usize;
        for lut in luts {
            nested = nested
                .max(self.ckks_all_ops_with_atk_tmp_bytes(
                    &boot_layout,
                    &keys_layout.tensor_key,
                    &keys_layout.automorphism_key,
                    lut.coeffs(),
                ))
                .max(self.ckks_all_ops_with_atk_tmp_bytes(
                    &in_layout,
                    &keys_layout.tensor_key,
                    &keys_layout.automorphism_key,
                    lut.coeffs(),
                ));
        }
        if luts.iter().any(EncodedLut::requires_eval_mod) {
            nested =
                nested.max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), &keys_layout.tensor_key));
        }
        base.max(carved + nested)
    }

    /// `scale_up` lifts the message that many bits above its natural magnitude,
    /// fused into the widening shift rather than costing a second pass.
    pub(crate) fn ckks_mod_up_into_default<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        scale_up: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
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
            k_large >= k_small + scale_up,
            "ckks_mod_up: dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up}); ModUp must widen, not shrink, the modulus"
        );

        // MSB-align `src` into the wider `dst`: the value occupies the top
        // `k_small` bits and the freshly-introduced low-order limbs are zero.
        self.glwe_copy(dst, src);

        // Shift the digits down to their natural integer magnitude. This is the
        // modulus raise: the raised-from modulus `q = 2^k_small` becomes an
        // explicit, un-reduced multiple `I(X)·q` living in the `[0, 2^k_large)`
        // window, which EvalMod subsequently removes. (A zero shift, when the
        // input already fills the storage, is a no-op.) Stopping `scale_up` bits
        // short of the natural magnitude is the fused lift: the value lands
        // multiplied by `2^scale_up`, absorbed by the matching `log_delta`.
        self.glwe_rsh(k_large - k_small - scale_up, dst, scratch);

        // `log_delta` picks up the fused lift, so the encoded value is unchanged;
        // the remaining headroom now spans the full raised modulus, so the torus
        // width `k` becomes `k_large` and `log_budget = k_large - log_delta`.
        dst.set_meta(CKKSMeta {
            log_delta: src.log_delta() + scale_up,
            log_sparsity: src.log_sparsity(),
            slots: src.slots(),
        });
        dst.set_k(k_large.into());

        Ok(())
    }

    /// Whole raise step: lift to the plan's message ratio, (encapsulate) ModUp
    /// with the second lift fused into its shift, then relabel by the message
    /// ratio so `I(X)·q` is the integer part and the message the residue.
    pub(crate) fn ckks_bootstrap_mod_up_default<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE> + CKKSCopyOps<BE>,
        K: BootstrappingKeys<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        // The pipeline carves rank-1 intermediates, so reject higher-rank inputs
        // rather than silently copying into a rank-1 scratch ciphertext.
        ckks_ensure!(
            src.rank().as_usize() == 1 && dst.rank().as_usize() == 1,
            "ckks_bootstrap_mod_up supports rank-1 ciphertexts only, got rank {} -> {}",
            src.rank().as_usize(),
            dst.rank().as_usize()
        );

        // The input-width copy is scoped so its scratch is released right after
        // ModUp widens it into `dst`.
        scratch.scope(|scratch_inner| {
            let (mut ct0, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(src, src.meta());
            self.ckks_copy(&mut ct0, src, &mut scratch_inner)?;
            self.ckks_bootstrap_mod_up_from_mut(dst, &mut ct0, Some(eval_mod), keys, &mut scratch_inner)
        })?;

        dst.set_meta(CKKSMeta {
            log_sparsity: src.log_sparsity(),
            log_delta: dst.log_delta() + eval_mod.log_msg_ratio,
            slots: src.slots(),
        });
        Ok(())
    }

    fn ckks_bootstrap_mod_up_from_mut<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        lift: Option<&EvalModPlan>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        K: BootstrappingKeys<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        // Both lifts come from the plan; `None` skips them (for pipelines whose
        // input does not meet the plan's message-ratio contract). The first
        // normalizes the input to that ratio, the second takes the raised
        // ciphertext up to EvalMod's own scale and is fused into ModUp's shift.
        let mut scale_up = 0;
        if let Some(plan) = lift {
            let k: usize = src.k().as_usize();
            let log_delta: usize = src.log_delta();
            let scale_up_in = plan.input_scale_up(k, log_delta)?;
            if scale_up_in != 0 {
                self.ckks_mul_pow2_assign(src, scale_up_in, scratch)?;
                // Not `set_log_delta`: it preserves `log_budget` and grows `k`, which
                // cancels the shift. The lift spends budget at a fixed modulus.
                let mut meta = src.meta();
                meta.log_delta = log_delta + scale_up_in;
                src.set_meta(meta);
            }
            scale_up = plan.raised_scale_up(src.log_delta())?;
        }

        match keys.encapsulation_keys() {
            Some((dense_to_sparse, sparse_to_dense)) => {
                BE::ckks_encapsulated_mod_up(self.0, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)?;
            }
            None => self.ckks_mod_up_into_default(dst, src, scale_up, scratch)?,
        }
        Ok(())
    }

    fn recombine_halves<R1, R2>(&self, res_re: &mut R1, res_im: &mut R2, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Module<BE>: CKKSImagOps<BE> + CKKSAddOps<BE>,
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
        Module<BE>: CKKSDFTOps<BE>,
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
        Module<BE>: CKKSDFTOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
        K: BootstrappingKeys<BE>,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        self.ckks_dft_evaluate_assign(ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)?;
        self.ckks_conjugate_into(conjugate, &*ct, keys.conjugation_key(), scratch)?;
        self.ckks_add_assign(ct, &*conjugate, scratch)?;
        // `z + conj(z) = 2·Re(z)` holds the input polynomial's coefficients.
        ct.set_slots(SlotsKind::Real);
        Ok(())
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
        Module<BE>: CKKSEvalModOps<BE>,
        F: Sync,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
        C: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    {
        if !<BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
            self.ckks_eval_mod(res_real, r0, ctx.eval_mod(), keys.tensor_key(), scratch)?;
            self.ckks_eval_mod(res_imag, i0, ctx.eval_mod(), keys.tensor_key(), scratch)?;
            return Ok(());
        }

        let task_bytes = self
            .ckks_eval_mod_tmp_bytes(res_real, r0, ctx.eval_mod(), keys.tensor_key())
            .max(self.ckks_eval_mod_tmp_bytes(res_imag, i0, ctx.eval_mod(), keys.tensor_key()));
        let task_bytes = worker_scratch_bytes::<BE>(task_bytes);
        let (arenas, _) = scratch.borrow().split(2, task_bytes);
        let mut arenas = arenas.into_iter();
        let mut scratch_real = arenas.next().unwrap();
        let mut scratch_imag = arenas.next().unwrap();
        let (real, imag) = <BE::TaskExecutor as TaskExecutor>::join(
            || self.ckks_eval_mod(res_real, r0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_real),
            || self.ckks_eval_mod(res_imag, i0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_imag),
        );
        real?;
        imag?;
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
        F: Sync,
        Module<BE>: GLWECopy<BE>
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
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
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
                    {
                        let r0_ref = r0.to_backend_view_ref();
                        let i0_ref = i0.to_backend_view_ref();
                        self.ckks_bootstrap_eval_mod_halves(
                            &r0_ref,
                            &i0_ref,
                            ct_out,
                            &mut ct_raised,
                            ctx,
                            keys,
                            &mut scratch_local,
                        )?;
                    }
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

                    {
                        let r0_ref = r0.to_backend_view_ref();
                        let i0_ref = i0.to_backend_view_ref();
                        self.ckks_bootstrap_eval_mod_halves(
                            &r0_ref,
                            &i0_ref,
                            ct_out,
                            &mut ct_raised,
                            ctx,
                            keys,
                            &mut scratch_local,
                        )?;
                    }

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
                slots: ct_in.slots(),
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
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSDFTOps<BE>,
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
            self.ckks_bootstrap_mod_up_from_mut(ct_raised, &mut ct_coeffs, None, keys, &mut scratch_inner)?;
            ct_raised.set_meta(CKKSMeta {
                log_sparsity: ct_in.log_sparsity(),
                log_delta: log_modulus_in,
                slots: ct_in.slots(),
            });
            Result::Ok(())
        })
    }

    /// Backend-generic reference for
    /// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    /// Pipeline is selected from the context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ckks_bootstrap_default<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWECopy<BE>
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
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
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
            // Known-real slots skip the imaginary nonlinear branch: one EvalMod
            // instead of two. EvalRound+ has no real-slot form, so a bypass
            // recipe falls back to the general pipeline rather than erroring.
            if ct_in.slots().is_real() && ctx.coeffs_to_slots_bypass().is_none() {
                return self.bootstrap_real_inner(ct_out, ct_in, ctx, keys, scratch);
            }
            return self.ckks_bootstrap_s2c_first(ct_out, ct_in, ctx, keys, scratch);
        }

        let base2k = ct_in.base2k();
        let k_boot = ct_out.k();
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
            self.ckks_bootstrap_mod_up_default(&mut ct, ct_in, &ctx.eval_mod().plan, keys, &mut scratch_local)?;

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
            ct_out.set_slots(ct_in.slots());
            Result::Ok(())
        })
    }

    /// Real-slot S2C-first pipeline: one EvalMod instead of two. Selected by
    /// [`Self::ckks_bootstrap_default`] from `ct_in.slots()`; the recipe
    /// preconditions are part of that selection, not checked again here.
    fn bootstrap_real_inner<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
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
                slots: ct_in.slots(),
            });
            Result::Ok(())
        })
    }

    /// Backend-generic reference for
    /// [`CKKSBootstrappingOps::ckks_functional_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_functional_bootstrap).
    ///
    /// One LUT or many: the batch shares the S2C, ModUp and CoeffsToSlots stages,
    /// and equal-arity general LUTs additionally share the power basis of each
    /// transformed half. The slot kind of `ct_in` selects the pipeline: real slots
    /// skip the imaginary branch entirely.
    pub(crate) fn ckks_functional_bootstrap_default<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
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
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ckks_ensure!(!luts.is_empty(), "functional bootstrap requires at least one LUT");
        ckks_ensure!(
            ct_outs.len() == luts.len(),
            "functional bootstrap ct_outs/luts length mismatch ({} vs {})",
            ct_outs.len(),
            luts.len()
        );
        ckks_ensure!(
            ctx.pipeline() == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first context"
        );
        let log_msg_ratio = luts[0].log_msg_ratio();
        ckks_ensure!(
            luts.iter().all(|lut| lut.log_msg_ratio() == log_msg_ratio),
            "functional bootstrap LUTs must have the same message ratio"
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

        // A shared power basis pays for itself only across several general LUTs; a
        // single LUT, or any batch containing a binary one, evaluates per LUT.
        let shared = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());
        let boot_layout = GLWELayout {
            n: out_n,
            base2k: ct_in.base2k(),
            k: out_k,
            rank: Rank(1),
        };
        scratch.scope(|scratch_local| {
            let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            self.ckks_bootstrap_s2c_mod_up(&mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;

            if ct_in.slots().is_real() {
                let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                self.ckks_bootstrap_coeffs_to_slots_real(&mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
                return eval_lut_batch(self, ct_outs, &ct_raised, ctx, luts, keys, shared, &mut scratch_local);
            }

            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            self.ckks_bootstrap_coeffs_to_slots(&ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;

            eval_lut_batch(self, ct_outs, &r0, ctx, luts, keys, shared, &mut scratch_local)?;

            // `ct_raised` is dead after CoeffsToSlots, so it serves as the single
            // imaginary accumulator: each LUT's imaginary half is folded into its
            // output before the next one is evaluated, which keeps the carved
            // working set independent of the batch size. Multiplication produces
            // its result at the destination's requested `k`, so the accumulator is
            // restored to the full boot width before each reuse; otherwise the
            // first LUT's consumption would clamp every later one.
            let im_meta = ct_raised.meta();
            if shared {
                let basis = ckks_lut_power_basis(
                    self,
                    &i0,
                    &boot_layout,
                    ctx.eval_mod(),
                    luts,
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_lut_from_basis(
                        self,
                        &mut ct_raised,
                        lut,
                        &basis,
                        keys.conjugation_key(),
                        keys.tensor_key(),
                        &mut scratch_local,
                    )?;
                    self.recombine_halves(ct_out, &mut ct_raised, &mut scratch_local)?;
                }
            } else {
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_encoded_lut(self, &mut ct_raised, &i0, ctx, lut, keys, &mut scratch_local)?;
                    self.recombine_halves(ct_out, &mut ct_raised, &mut scratch_local)?;
                }
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
}

/// Reference SSE pipeline: switch to the sparse secret before ModUp to bound
/// wrap-around, then restore the dense secret afterward.
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_default<BE, Dst, Src, D2S, S2D>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl<BE>,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    D2S: poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    S2D: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    module.glwe_keyswitch_assign(src, dense_to_sparse, scratch);
    // The lift is fused into ModUp, so the message is already at its final scale
    // when sparse-to-dense adds its noise.
    BootstrappingDefault::new(module).ckks_mod_up_into_default(dst, src, scale_up, scratch)?;
    module.glwe_keyswitch_assign(dst, sparse_to_dense, scratch);
    Ok(())
}

/// Scratch bound for [`ckks_encapsulated_mod_up_default`].
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_tmp_bytes_default<BE, Dst, Src, D2S, S2D>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    dense_to_sparse_infos: &D2S,
    sparse_to_dense_infos: &S2D,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    S2D: GGLWEInfos,
{
    module
        .glwe_keyswitch_tmp_bytes(src_infos, src_infos, dense_to_sparse_infos)
        .max(module.glwe_shift_tmp_bytes())
        .max(module.glwe_keyswitch_tmp_bytes(dst_infos, dst_infos, sparse_to_dense_infos))
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
            slots: ct_in.slots(),
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

/// Evaluates every LUT of a batch against one transformed half, writing into
/// `outs`. Equal-arity general LUTs share one power basis; a batch containing a
/// binary LUT falls back to evaluating each LUT against the shared input.
#[allow(clippy::too_many_arguments)]
fn eval_lut_batch<BE, F, K, C>(
    module: &Module<BE>,
    outs: &mut [CKKSCiphertextOwned<BE>],
    ct: &C,
    ctx: &BootstrappingContext<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    shared: bool,
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
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    if shared {
        let layout = GLWELayout {
            n: outs[0].n(),
            base2k: outs[0].base2k(),
            k: outs[0].k(),
            rank: outs[0].rank(),
        };
        let basis = ckks_lut_power_basis(module, ct, &layout, ctx.eval_mod(), luts, keys.tensor_key(), scratch)?;
        for (out, lut) in outs.iter_mut().zip(luts) {
            ckks_eval_lut_from_basis(module, out, lut, &basis, keys.conjugation_key(), keys.tensor_key(), scratch)?;
        }
        return Ok(());
    }
    for (out, lut) in outs.iter_mut().zip(luts) {
        ckks_eval_encoded_lut(module, out, ct, ctx, lut, keys, scratch)?;
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
