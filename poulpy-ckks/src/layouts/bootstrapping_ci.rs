//! S2C-first bootstrapping with EvalMod on the conjugate-invariant ring.
//!
//! On the standard ring of degree `N`, CoeffsToSlots leaves the `N` real
//! coefficients of the raised input in `N/2` complex slots, `z_j = c_j + i·c_{j+N/2}`,
//! and the polynomial of EvalMod, which is real, only reduces real inputs: the
//! standard pipeline splits `z` into its real and imaginary parts and evaluates
//! EvalMod twice. The conjugate-invariant (CI) ring of degree `N` has `N` real
//! slots, so a single EvalMod there reduces every coefficient.
//!
//! The pipeline ([`CKKSBootstrappingCIOps`](crate::api::CKKSBootstrappingCIOps))
//! moves `z` there and back through the standard ring of degree `2N`, which
//! contains both:
//!
//! 1. `z(X)` is embedded as `z(X²)`, where each slot of degree `N` holds two slots
//!    of degree `2N`, `j` and `j + N/2`. Its secret is `s(X²)`.
//! 2. A product with the plaintext `u` (`1` on the first copy, `−i` on the second)
//!    spends [`EvalModCIPlan::pre_log_delta`] bits, and the trace onto the CI ring
//!    ([`ckks_ci_trace`](crate::api::CKKSCIRingMapOps::ckks_ci_trace)) keeps the
//!    real part of each slot: `c_j` and `c_{j+N/2}`, each in its own slot, twice
//!    over, which is the input the split EvalMod also takes.
//! 3. EvalMod runs once on the CI ring.
//! 4. The embedding back ([`ckks_ci_embed`](crate::api::CKKSCIRingMapOps::ckks_ci_embed)),
//!    a product with `v` (`1` and `i`) at [`EvalModCIPlan::post_log_delta`] bits
//!    and the trace `x + x(−X)`, which needs no key since `s(X²)` is invariant
//!    under `X -> −X`, pair the reduced slots back as `r_j + i·r_{j+N/2}` on the
//!    even coefficients, the ciphertext of degree `N` under `s`.
//!
//! The keys of the conversions are of degree `2N`, from `s(X²)` to the CI secret
//! and back; EvalMod relinearizes with CI tensor keys of degree `N`.

use anyhow::{Result, ensure};
use poulpy_core::{
    DiagonalProd, GLWECIKeyEncryptSk, GLWETensorKeyEncryptSk, GetDistribution, GetDistributionMut,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWEToBackendRef, GLWECIEmbedKey, GLWECIEmbedKeyPrepared, GLWECIKeyPreparedFactory,
        GLWECITraceKey, GLWECITraceKeyPrepared, GLWEInfos, GLWESecretLayout, GLWESecretSampling, GLWESwitchingKeyLayout,
        GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, ModuleCoreAlloc,
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
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        BootstrappingContext, BootstrappingKeySet, BootstrappingKeysLayout, BootstrappingKeysPrepared, BootstrappingPipeline,
        BootstrappingPlan, CKKSModuleAlloc, CKKSPlaintextOwned,
        eval_mod::{EvalMod, EvalModType, compile_eval_mod},
    },
};

/// The two conversions of a [`BootstrappingCIPlan`]: the scales, in bits, of the
/// plaintext products before and after EvalMod. Each spends its scale from the
/// budget: the levels the conversions add to the bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalModCIPlan {
    /// Scale of `u`, which pairs each slot of degree `N` into two real CI slots.
    pub pre_log_delta: usize,
    /// Scale of `v`, which pairs the reduced CI slots back.
    pub post_log_delta: usize,
}

/// A [`BootstrappingPlan`] whose EvalMod runs on the conjugate-invariant ring.
///
/// The base plan is S2C-first, without functional bootstrapping or EvalRound+,
/// and its EvalMod is one of the real families. Its widths gain the two
/// conversion levels.
#[derive(Clone, Debug)]
pub struct BootstrappingCIPlan {
    base: BootstrappingPlan,
    conversion: EvalModCIPlan,
}

impl BootstrappingCIPlan {
    pub fn new(base: BootstrappingPlan, conversion: EvalModCIPlan) -> Result<Self> {
        ensure!(
            base.pipeline() == BootstrappingPipeline::S2CFirst,
            "EvalMod on the conjugate-invariant ring requires an S2C-first plan"
        );
        ensure!(
            base.coeffs_to_slots_bypass().is_none(),
            "EvalMod on the conjugate-invariant ring does not take the EvalRound+ bypass"
        );
        ensure!(
            base.functional_message_modulus.is_none(),
            "EvalMod on the conjugate-invariant ring does not run functional bootstrapping"
        );
        ensure!(
            base.eval_mod().eval_mod_type != EvalModType::ExpCmplx,
            "EvalMod on the conjugate-invariant ring needs a real EvalMod family"
        );
        ensure!(
            conversion.pre_log_delta > 0 && conversion.post_log_delta > 0,
            "the conversion scales must be positive"
        );
        Ok(Self { base, conversion })
    }

    /// The S2C-first plan around EvalMod.
    pub fn base(&self) -> &BootstrappingPlan {
        &self.base
    }

    /// The scales of the conversions.
    pub fn conversion(&self) -> EvalModCIPlan {
        self.conversion
    }

    /// Budget the two conversions consume.
    pub fn conversion_consumed_bits(&self) -> usize {
        self.conversion.pre_log_delta + self.conversion.post_log_delta
    }

    /// Input width for a given modulus at ModUp, as in the base plan.
    pub fn input_k(&self, log_modulus: usize) -> usize {
        self.base.input_k(log_modulus)
    }

    /// Bootstrap width for an output at the input scale `log_delta`: the base
    /// width and the two conversion levels.
    pub fn bootstrap_k(&self, output_k: usize, log_delta: usize) -> usize {
        self.base.bootstrap_k(output_k, log_delta) + self.conversion_consumed_bits()
    }

    /// Total budget the pipeline consumes.
    pub fn consumed_bits(&self) -> usize {
        self.base.consumed_bits() + self.conversion_consumed_bits()
    }
}

/// Compiled [`BootstrappingCIPlan`]: the base context, EvalMod encoded on the CI
/// module, and the conversion plaintexts `u` and `v` at degree `2N`.
pub struct BootstrappingCIContext<BE: Backend, BCI: Backend, F> {
    pub(crate) base: BootstrappingContext<BE, F>,
    pub(crate) eval_mod: EvalMod<F, CKKSPlaintextOwned<BCI>>,
    pub(crate) pre: CKKSPlaintextOwned<BE>,
    pub(crate) post: CKKSPlaintextOwned<BE>,
}

impl<BE: Backend, BCI: Backend, F> BootstrappingCIContext<BE, BCI, F> {
    /// The context of the standard stages.
    pub fn base(&self) -> &BootstrappingContext<BE, F> {
        &self.base
    }

    /// EvalMod encoded on the CI module.
    pub fn eval_mod(&self) -> &EvalMod<F, CKKSPlaintextOwned<BCI>> {
        &self.eval_mod
    }

    /// The plaintext `u` of degree `2N` before EvalMod.
    pub fn pre(&self) -> &CKKSPlaintextOwned<BE> {
        &self.pre
    }

    /// The plaintext `v` of degree `2N` after EvalMod.
    pub fn post(&self) -> &CKKSPlaintextOwned<BE> {
        &self.post
    }
}

impl<BE, BCI, F> BootstrappingCIContext<BE, BCI, F>
where
    BE: Backend<Ring = Standard>,
    BCI: Backend<Ring = ConjugateInvariant>,
    F: CKKSEncodingScalar,
{
    /// Compiles `plan`: the base context on `module` (degree `N`), EvalMod on `ci`
    /// (CI, degree `N`) and the conversion plaintexts on `wide` (degree `2N`).
    ///
    /// `scratch` serves the base context and the plaintexts, `scratch_ci` EvalMod.
    pub fn compile(
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        base2k: Base2K,
        plan: &BootstrappingCIPlan,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<Self>
    where
        Module<BE>:
            crate::api::CKKSDFTOps<BE> + crate::api::CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSModuleAlloc<BCI> + CKKSEncodingOps<BCI, F>,
        CKKSPlaintextOwned<BE>: poulpy_core::layouts::GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        ensure!(
            wide.n() == 2 * module.n() && ci.n() == module.n(),
            "EvalMod on the CI ring takes a standard module of degree 2N and a CI module of degree N"
        );
        let base = BootstrappingContext::compile(module, base2k, plan.base(), scratch)?;
        let eval_mod = compile_eval_mod::<BCI, F>(base2k, plan.base().eval_mod, ci, scratch_ci)?;
        let conversion = plan.conversion();
        // Slot `j` of degree `N` is slot `j` and slot `j + N/2` of degree `2N`.
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
        Ok(Self {
            base,
            eval_mod,
            pre,
            post,
        })
    }
}

/// Key layouts of a [`BootstrappingCIPlan`].
#[derive(Clone, Debug)]
pub struct BootstrappingCIKeysLayout {
    /// The keys of the standard stages, as for the base plan.
    pub base: BootstrappingKeysLayout,
    /// CI tensor key for EvalMod, covering its widest product.
    pub tensor_key: GLWETensorKeyLayout,
    /// Key of degree `2N` from `s(X²)` to the embedded CI secret, covering the
    /// CoeffsToSlots output.
    pub trace_key: GLWESwitchingKeyLayout,
    /// Key of degree `2N` from the embedded CI secret to `s(X²)`, covering the
    /// EvalMod output.
    pub embed_key: GLWESwitchingKeyLayout,
}

/// The unprepared keys of a [`BootstrappingCIPlan`].
pub struct BootstrappingCIKeySet<D: Data, W: ZnxWord> {
    /// The keys of the standard stages.
    pub base: BootstrappingKeySet<D, W>,
    /// CI tensor key for EvalMod.
    pub tensor_key: GLWETensorKey<D, W>,
    /// From `s(X²)` to the embedded CI secret.
    pub trace_key: GLWECITraceKey<D, W>,
    /// From the embedded CI secret to `s(X²)`.
    pub embed_key: GLWECIEmbedKey<D, W>,
}

/// The prepared keys of a [`BootstrappingCIPlan`].
pub struct BootstrappingCIKeysPrepared<D: Data, BE: Backend, BCI: Backend> {
    /// The keys of the standard stages.
    pub base: BootstrappingKeysPrepared<D, BE>,
    /// CI tensor key for EvalMod.
    pub tensor_key: GLWETensorKeyPrepared<D, BCI>,
    /// From `s(X²)` to the embedded CI secret.
    pub trace_key: GLWECITraceKeyPrepared<D, BE>,
    /// From the embedded CI secret to `s(X²)`.
    pub embed_key: GLWECIEmbedKeyPrepared<D, BE>,
}

impl<BE, BCI, F> BootstrappingCIContext<BE, BCI, F>
where
    BE: Backend<Ring = Standard>,
    BCI: Backend<Ring = ConjugateInvariant>,
{
    /// Generates the unprepared keys for the dense secret `sk_dense` of degree `N`:
    /// the base keys ([`BootstrappingContext::generate_keys`]), a CI secret of the
    /// same Hamming weight, its tensor keys, and the conversion keys of degree `2N`
    /// between it and `s(X²)`.
    ///
    /// `scratch` serves the base and conversion keys, `scratch_ci` the CI tensor keys.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_keys(
        &self,
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        sk_dense: &BackendGLWESecret<BE>,
        layout: &BootstrappingCIKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<BootstrappingCIKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        BCI: Backend<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + poulpy_hal::layouts::CyclotomicOrder
            + poulpy_core::GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
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
        let base = self
            .base
            .generate_keys(module, sk_dense, &layout.base, source_xs, source_xe, source_xa, scratch)?;

        // `s(X²)`, the secret of the embedded ciphertext.
        let mut sk_wide = wide.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: wide.n().into(),
            rank: sk_dense.rank(),
        });
        {
            let src = sk_dense.data().at(0, 0).to_vec();
            let dst = sk_wide.data_mut().at_mut(0, 0);
            dst.iter_mut().for_each(|x| *x = BE::ZnxWord::from_i64(0));
            for (i, c) in src.into_iter().enumerate() {
                dst[2 * i] = c;
            }
        }
        *sk_wide.dist_mut() = *sk_dense.dist();

        // The CI secret, of the dense secret's Hamming weight in the CI basis.
        let mut sk_ci = ci.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: ci.n().into(),
            rank: sk_dense.rank(),
        });
        let hamming_weight = match sk_dense.dist() {
            poulpy_core::Distribution::TernaryFixed(hw) => *hw,
            dist => anyhow::bail!("EvalMod on the CI ring takes a fixed-weight ternary dense secret, got {dist:?}"),
        };
        ci.glwe_secret_fill_ternary_hw(&mut sk_ci, hamming_weight, source_xs);

        let mut tensor_key = ci.glwe_tensor_key_alloc_from_infos(&layout.tensor_key);
        ci.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_ci, source_xe, source_xa, scratch_ci);

        let mut trace_key = wide.glwe_ci_trace_key_alloc_from_infos(&layout.trace_key);
        wide.glwe_ci_trace_key_encrypt_sk(&mut trace_key, &sk_ci, &sk_wide, source_xe, source_xa, scratch);
        let mut embed_key = wide.glwe_ci_embed_key_alloc_from_infos(&layout.embed_key);
        wide.glwe_ci_embed_key_encrypt_sk(&mut embed_key, &sk_ci, &sk_wide, source_xe, source_xa, scratch);

        Ok(BootstrappingCIKeySet {
            base,
            tensor_key,
            trace_key,
            embed_key,
        })
    }
}

impl<D: Data, W: ZnxWord> BootstrappingCIKeySet<D, W> {
    /// Prepares every key: the base keys on `module`, the CI tensor keys on `ci`,
    /// the conversion keys on `wide`.
    pub fn prepare<BE, BCI>(
        &self,
        module: &Module<BE>,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> BootstrappingCIKeysPrepared<BE::OwnedBuf, BE, BCI>
    where
        BE: Backend<Ring = Standard, ZnxWord = W>,
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = W>,
        D: HostDataRef,
        poulpy_core::layouts::GLWEAutomorphismKey<D, W>:
            GGLWEToBackendRef<BE> + poulpy_core::layouts::GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEToBackendRef<BCI> + GGLWEInfos,
        poulpy_core::layouts::GLWESwitchingKey<D, W>:
            GGLWEToBackendRef<BE> + poulpy_core::layouts::GLWESwitchingKeyDegrees + GGLWEInfos,
        GLWECITraceKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWECIEmbedKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + poulpy_core::layouts::GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + poulpy_core::layouts::GLWESwitchingKeyPreparedFactory<BE>
            + GLWECIKeyPreparedFactory<BE>,
        Module<BCI>: GLWETensorKeyPreparedFactory<BCI>,
    {
        let base = self.base.prepare(module, scratch);
        let mut tensor_key = ci.alloc_tensor_key_prepared_from_infos(&self.tensor_key);
        ci.prepare_tensor_key(&mut tensor_key, &self.tensor_key, scratch_ci);
        let mut trace_key = wide.glwe_ci_trace_key_prepared_alloc_from_infos(&self.trace_key);
        wide.glwe_ci_key_prepare(&mut trace_key, &self.trace_key, scratch);
        let mut embed_key = wide.glwe_ci_embed_key_prepared_alloc_from_infos(&self.embed_key);
        wide.glwe_ci_key_prepare(&mut embed_key, &self.embed_key, scratch);
        BootstrappingCIKeysPrepared {
            base,
            tensor_key,
            trace_key,
            embed_key,
        }
    }
}
