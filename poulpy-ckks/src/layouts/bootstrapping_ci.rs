//! S2C-first bootstrapping with EvalMod on the conjugate-invariant (CI) ring.
//!
//! CoeffsToSlots leaves the `N` coefficients in `N/2` complex slots, and the real
//! polynomial of EvalMod runs on each half; the CI ring of degree `N` has `N` real
//! slots, so one EvalMod there reduces them all. The ciphertext goes there and
//! back through the standard ring of degree `2N` (embedding as `z(X²)`, products
//! by `u` and `v`, traces); `docs/bootstrapping.md` details the conversions.
//!
//! A plan selects it with [`BootstrappingPlan::with_eval_mod_on_ci_ring`]. The
//! context and the prepared keys then hold EvalMod and its tensor key on the CI
//! backend (`BootstrappingContext<BE, F, BCI>`), beside the conversion
//! plaintexts and keys of degree `2N`; there is no conjugation key.

use anyhow::{Result, ensure};
use poulpy_core::{
    DiagonalProd, GLWECIKeyEncryptSk, GLWETensorKeyEncryptSk, GetDistribution, GetDistributionMut,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWEToBackendRef, GLWECIEmbedKey, GLWECIKeyPreparedFactory, GLWECITraceKey, GLWEInfos,
        GLWESecret, GLWESecretLayout, GLWESecretSampling, GLWETensorKey, GLWETensorKeyPreparedFactory, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    layouts::{
        Backend, ConjugateInvariant, Data, HostDataMut, HostDataRef, Module, ScratchArena, Standard, ZnxView, ZnxViewMut, ZnxWord,
    },
    source::Source,
};

use crate::{
    Base2K, CKKSCtBounds, CKKSModuleInfos, GLWEPlaintextMeta, Scale, SlotsKind,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        BootstrappingContext, BootstrappingContextUnprepared, BootstrappingKeySet, BootstrappingKeysLayout,
        BootstrappingKeysPrepared, BootstrappingPlan, CKKSModuleAlloc, CKKSPlaintextOwned, EvalModCIPlan, EvalModRing,
        bootstrapping_keys::prepare_linear_keys, eval_mod::compile_eval_mod,
    },
};

impl<BE, F, BCI> BootstrappingContext<BE, F, BCI>
where
    BE: Backend<Ring = Standard>,
    BCI: Backend<Ring = ConjugateInvariant>,
    F: CKKSEncodingScalar,
{
    /// Compiles a plan that runs EvalMod on the conjugate-invariant ring: the
    /// transforms on `module` (degree `N`), EvalMod on `ci` (CI, degree `N`) and
    /// the conversion plaintexts on `wide` (degree `2N`).
    ///
    /// `scratch` serves the transforms and the plaintexts, `scratch_ci` EvalMod;
    /// as [`Self::compile`], `scratch` also serves the preparation.
    pub fn compile_ci(
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSModuleAlloc<BCI> + CKKSEncodingOps<BCI, F>,
        CKKSPlaintextOwned<BE>: poulpy_core::layouts::GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        Ok(Self::compile_unprepared_ci(module, wide, ci, base2k, plan, scratch, scratch_ci)?.prepare(module, scratch))
    }

    /// First stage of [`Self::compile_ci`], as [`Self::compile_unprepared`] is of
    /// [`Self::compile`].
    pub fn compile_unprepared_ci(
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<BootstrappingContextUnprepared<BE, F, BCI>>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSModuleAlloc<BCI> + CKKSEncodingOps<BCI, F>,
    {
        ensure!(
            wide.n() == 2 * module.n() && ci.n() == module.n(),
            "EvalMod on the CI ring takes a standard module of degree 2N and a CI module of degree N"
        );
        let EvalModRing::ConjugateInvariant(conversion) = plan.eval_mod_ring() else {
            anyhow::bail!("the plan runs EvalMod on the ring of the bootstrap: compile it with BootstrappingContext::compile");
        };
        let eval_mod = compile_eval_mod::<BCI, F>(base2k, plan.eval_mod, ci, scratch_ci)?;
        let conversions = conversion_plaintexts::<BE, F>(module, wide, base2k, conversion, scratch)?;
        BootstrappingContextUnprepared::with_eval_mod(module, base2k, plan, eval_mod, Some(conversions), scratch)
    }

    /// Generates the unprepared keys for the dense secret `sk_dense` of degree
    /// `N`: those of the linear stages, without conjugation, as
    /// [`BootstrappingContext::generate_keys`] does; a CI secret of the same
    /// Hamming weight and its tensor key; and the conversion keys of degree `2N`
    /// between it and `s(X²)`.
    ///
    /// `scratch` serves the keys of the standard rings, `scratch_ci` the CI tensor key.
    #[expect(
        clippy::too_many_arguments,
        reason = "key generation takes three modules, three sources and a scratch arena per ring"
    )]
    pub fn generate_keys_ci(
        &self,
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        sk_dense: &BackendGLWESecret<BE>,
        layout: &BootstrappingKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        BCI: Backend<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + poulpy_hal::layouts::CyclotomicOrder
            + poulpy_core::GLWEAutomorphismKeyEncryptSk<BE>
            + poulpy_core::GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + poulpy_hal::api::VecZnxAutomorphism<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BCI>: ModuleCoreAlloc<OwnedBuf = BCI::OwnedBuf, ZnxWord = BCI::ZnxWord>
            + GLWETensorKeyEncryptSk<BCI>
            + GLWESecretSampling<BCI>,
        BackendGLWESecret<BCI>: poulpy_core::layouts::GLWESecretToBackendRef<BE>,
    {
        ensure!(
            wide.n() == 2 * module.n() && ci.n() == module.n(),
            "EvalMod on the CI ring takes a standard module of degree 2N and a CI module of degree N"
        );
        let Some(conversion) = &layout.conversion else {
            anyhow::bail!("bootstrapping key layout has no conversion keys, which EvalMod on the CI ring uses");
        };
        let rotation_keys = self.generate_rotation_keys(module, sk_dense, layout, source_xe, source_xa, scratch)?;
        let (encapsulation_keys, first_factor_keys) =
            self.generate_raise_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch);

        // The CI secret takes the dense secret's Hamming weight. The secrets derived
        // here are wiped before returning, so nothing fails between their creation and
        // their wipe.
        let hamming_weight = match sk_dense.dist() {
            poulpy_core::Distribution::TernaryFixed(hw) => *hw,
            dist => anyhow::bail!("EvalMod on the CI ring takes a fixed-weight ternary dense secret, got {dist:?}"),
        };

        // `s(X²)`, the secret of the embedded ciphertext.
        let mut sk_wide = wide.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: wide.n().into(),
            rank: sk_dense.rank(),
        });
        for (i, &c) in sk_dense.data().at(0, 0).iter().enumerate() {
            sk_wide.data_mut().at_mut(0, 0)[2 * i] = c;
        }
        *sk_wide.dist_mut() = *sk_dense.dist();

        // The CI secret, of the dense secret's Hamming weight in the CI basis.
        let mut sk_ci = ci.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: ci.n().into(),
            rank: sk_dense.rank(),
        });
        ci.glwe_secret_fill_ternary_hw(&mut sk_ci, hamming_weight, source_xs);

        let mut tensor_key = ci.glwe_tensor_key_alloc_from_infos(&layout.tensor_key);
        ci.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_ci, source_xe, source_xa, scratch_ci);
        let mut trace_key = wide.glwe_ci_trace_key_alloc_from_infos(&conversion.trace_key);
        wide.glwe_ci_trace_key_encrypt_sk(&mut trace_key, &sk_ci, &sk_wide, source_xe, source_xa, scratch);
        let mut embed_key = wide.glwe_ci_embed_key_alloc_from_infos(&conversion.embed_key);
        wide.glwe_ci_embed_key_encrypt_sk(&mut embed_key, &sk_ci, &sk_wide, source_xe, source_xa, scratch);
        wipe_secret(&mut sk_wide);
        wipe_secret(&mut sk_ci);

        Ok(BootstrappingKeySet {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
            first_factor_keys,
            conversion_keys: Some((trace_key, embed_key)),
        })
    }
}

/// The plaintexts `u` and `v` of degree `2N` of the conversions, at the scales of
/// `conversion`: slot `j` of degree `N` is slot `j` and slot `j + N/2` of degree `2N`.
fn conversion_plaintexts<BE, F>(
    module: &Module<BE>,
    wide: &Module<BE>,
    base2k: Base2K,
    conversion: EvalModCIPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<(CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)>
where
    BE: Backend<Ring = Standard>,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
{
    let half = module.n() / 2;
    let pair = |first: (f64, f64), second: (f64, f64), log_delta: usize, scratch: &mut ScratchArena<'_, BE>| {
        let (mut re, mut im) = (vec![F::zero(); 2 * half], vec![F::zero(); 2 * half]);
        for j in 0..half {
            re[j] = F::from_f64(first.0).unwrap();
            im[j] = F::from_f64(first.1).unwrap();
            re[j + half] = F::from_f64(second.0).unwrap();
            im[j + half] = F::from_f64(second.1).unwrap();
        }
        // Values of modulus one: two bits above the scale hold them.
        let mut pt = wide.ckks_pt_vec_alloc(wide.ckks_max_slots(), base2k, (log_delta + 2).into());
        crate::SetCKKSInfos::set_meta(
            &mut pt,
            GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            },
        );
        wide.ckks_encode_reim_into(&mut pt, &re, &im, scratch)?;
        Result::<CKKSPlaintextOwned<BE>>::Ok(pt)
    };
    let pre = pair((1.0, 0.0), (0.0, -1.0), conversion.pre_log_delta, scratch)?;
    let post = pair((1.0, 0.0), (0.0, 1.0), conversion.post_log_delta, scratch)?;
    Ok((pre, post))
}

/// Zeroes a secret that key generation derived for itself, before it is
/// dropped: the writes are volatile, so they are kept although the buffer is
/// freed right after.
fn wipe_secret<D: HostDataMut, W: ZnxWord>(sk: &mut GLWESecret<D, W>) {
    for col in 0..sk.rank().as_usize() {
        for x in sk.data_mut().at_mut(col, 0) {
            // SAFETY: `x` is a valid, aligned, exclusive reference.
            unsafe { std::ptr::write_volatile(x, W::from_i64(0)) };
        }
    }
}

impl<D: Data, W: ZnxWord> BootstrappingKeySet<D, W> {
    /// Prepares a key set of a plan with EvalMod on the conjugate-invariant ring:
    /// the keys of the linear stages on `module`, the CI tensor key on `ci`, the
    /// conversion keys on `wide`.
    pub fn prepare_ci<BE, BCI>(
        &self,
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>>
    where
        BE: Backend<Ring = Standard, ZnxWord = W>,
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = W>,
        D: HostDataRef,
        poulpy_core::layouts::GLWEAutomorphismKey<D, W>:
            GGLWEToBackendRef<BE> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BCI> + GGLWEInfos,
        poulpy_core::layouts::GLWESwitchingKey<D, W>:
            GGLWEToBackendRef<BE> + poulpy_core::layouts::GLWESwitchingKeyDegrees + GGLWEInfos,
        GLWECITraceKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWECIEmbedKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + poulpy_core::layouts::GLWEAutomorphismKeyPreparedFactory<BE>
            + poulpy_core::layouts::GLWESwitchingKeyPreparedFactory<BE>
            + GLWECIKeyPreparedFactory<BE>,
        Module<BCI>: GLWETensorKeyPreparedFactory<BCI>,
    {
        let Some((trace, embed)) = &self.conversion_keys else {
            anyhow::bail!("the key set has no conversion keys, which EvalMod on the CI ring uses");
        };
        let (rotation_keys, encapsulation_keys, first_factor_keys) = prepare_linear_keys(
            module,
            &self.rotation_keys,
            self.encapsulation_keys.as_ref(),
            &self.first_factor_keys,
            scratch,
        );
        let mut tensor_key = ci.alloc_tensor_key_prepared_from_infos(&self.tensor_key);
        ci.prepare_tensor_key(&mut tensor_key, &self.tensor_key, scratch_ci);
        let mut trace_key = wide.glwe_ci_trace_key_prepared_alloc_from_infos(trace);
        wide.glwe_ci_key_prepare(&mut trace_key, trace, scratch);
        let mut embed_key = wide.glwe_ci_embed_key_prepared_alloc_from_infos(embed);
        wide.glwe_ci_key_prepare(&mut embed_key, embed, scratch);
        Ok(BootstrappingKeysPrepared {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
            first_factor_keys,
            conversion_keys: Some((trace_key, embed_key)),
        })
    }
}
