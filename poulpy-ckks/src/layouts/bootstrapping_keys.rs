//! Evaluation keys for CKKS bootstrapping.
//!
//! The bootstrapping pipelines consume four kinds of key material:
//!
//! - **rotation (automorphism) keys** for the two homomorphic DFTs — the union
//!   of the Galois elements of `CoeffsToSlots` (incl. the optional high-precision
//!   bypass) and `SlotsToCoeffs`;
//! - a **conjugation key** (Galois element `−1`) for the split `CoeffsToSlots`;
//! - a **tensor (relinearization) key** for EvalMod's `ct×ct` squaring;
//! - optionally, the **sparse-secret encapsulation** key-switching keys
//!   (`denseToSparse`, `sparseToDense`; <https://eprint.iacr.org/2022/024>),
//!   whose sparse ephemeral secret is sampled inside
//!   [`BootstrappingContext::generate_keys`] and never escapes it.
//!
//! ## Unprepared vs prepared
//!
//! [`BootstrappingContext::generate_keys`] returns an **unprepared**
//! [`BootstrappingKeySet`] — the encrypted, *not yet preprocessed* keys. Keys are
//! kept unprepared on purpose: the unprepared form is what one serializes to send
//! online, and on accelerators (GPU) it is what lives in device memory, prepared
//! on the fly right before use.
//!
//! [`BootstrappingKeySet::prepare`] produces a **prepared**
//! [`BootstrappingKeysPrepared`] bundle (everything preprocessed up front) for the
//! straightforward CPU path. [`BootstrappingKeysPrepared`] implements
//! [`BootstrappingKeys`], the pipeline-facing access trait; a custom key manager
//! (lazy / streaming / on-the-fly-prepared) can implement [`BootstrappingKeys`]
//! directly instead of materializing the whole bundle.

use std::collections::{BTreeSet, HashMap};

use anyhow::Result;
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWEAutomorphismKey,
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecretLayout,
        GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
        GetGaloisElement, GetTensorKey, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    layouts::{Backend, CyclotomicOrder, Data, HostDataMut, HostDataRef, Module, ScratchArena, ZnxWord},
    source::Source,
};

use crate::layouts::BootstrappingContext;
use poulpy_core::layouts::GLWESecretSampling;
use poulpy_core::{Distribution, GetDistributionMut};

/// Pipeline-facing access to the **prepared** evaluation keys a CKKS bootstrap
/// consumes.
///
/// The bootstrap stages take their keys through this trait, so any key store that
/// can answer the four queries below can drive a bootstrap. The key types are
/// associated (not fixed to a backend buffer), so an implementation is free to
/// back them with any data representation and to prepare them lazily / on the fly.
/// [`BootstrappingKeysPrepared`] is the eager in-memory implementation.
pub trait BootstrappingKeys<BE: Backend> {
    /// The rotation-key collection passed to the homomorphic DFT stages.
    type RotationKeys: poulpy_core::layouts::GetAutomorphismKey<BE>;

    /// The prepared tensor (relinearization) key type for EvalMod.
    type TensorKey: GetTensorKey<BE>;

    /// The prepared key-switching key type for sparse-secret encapsulation.
    type SwitchingKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Rotation (automorphism) keys, Galois element `−1` among them.
    fn rotation_keys(&self) -> &crate::layouts::CKKSKey<Self::RotationKeys>;

    /// Relinearization (tensor) key for EvalMod's `ct×ct` squaring.
    fn tensor_key(&self) -> &crate::layouts::CKKSKey<Self::TensorKey>;

    /// Sparse-secret encapsulation keys `(denseToSparse, sparseToDense)`, or
    /// `None` when the trick is disabled.
    #[allow(clippy::type_complexity)]
    fn encapsulation_keys(
        &self,
    ) -> Option<(
        &crate::layouts::CKKSKey<Self::SwitchingKey>,
        &crate::layouts::CKKSKey<Self::SwitchingKey>,
    )>;
}

/// The **unprepared** bootstrapping keys: encrypted but not yet preprocessed.
///
/// This is the form produced by [`BootstrappingContext::generate_keys`] and the
/// form to serialize / store (incl. GPU device memory). Generic over the key data
/// buffer `D` (backend agnostic). Call [`Self::prepare`] to obtain the prepared
/// [`BootstrappingKeysPrepared`] the pipeline consumes, or prepare individual keys
/// on the fly.
pub struct BootstrappingKeySet<D: Data, W: ZnxWord> {
    /// Rotation keys indexed by Galois element (the engine-wide convention),
    /// conjugation (`−1`) among them.
    pub rotation_keys: HashMap<i64, crate::layouts::CKKSKey<GLWEAutomorphismKey<D, W>>>,
    /// Relinearization (tensor) key for EvalMod.
    pub tensor_key: crate::layouts::CKKSKey<GLWETensorKey<D, W>>,
    /// `(denseToSparse, sparseToDense)` encapsulation keys, or `None`.
    #[allow(clippy::type_complexity)]
    pub encapsulation_keys: Option<(
        crate::layouts::CKKSKey<GLWESwitchingKey<D, W>>,
        crate::layouts::CKKSKey<GLWESwitchingKey<D, W>>,
    )>,
}

/// The **prepared** (preprocessed) bootstrapping keys, ready for the pipeline.
///
/// Generic over the key data buffer `D`; implements [`BootstrappingKeys`]. Built
/// eagerly by [`BootstrappingKeySet::prepare`].
pub struct BootstrappingKeysPrepared<D: Data, BE: Backend> {
    /// Prepared rotation keys indexed by Galois element, conjugation among them.
    pub rotation_keys: crate::layouts::CKKSKey<HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>>,
    /// Prepared relinearization (tensor) key for EvalMod.
    pub tensor_key: crate::layouts::CKKSKey<GLWETensorKeyPrepared<D, BE>>,
    /// Prepared `(denseToSparse, sparseToDense)` encapsulation keys, or `None`.
    #[allow(clippy::type_complexity)]
    pub encapsulation_keys: Option<(
        crate::layouts::CKKSKey<GLWESwitchingKeyPrepared<D, BE>>,
        crate::layouts::CKKSKey<GLWESwitchingKeyPrepared<D, BE>>,
    )>,
}

impl<D: Data, BE: Backend> BootstrappingKeys<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type RotationKeys = HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>;
    type TensorKey = GLWETensorKeyPrepared<D, BE>;
    type SwitchingKey = GLWESwitchingKeyPrepared<D, BE>;

    fn rotation_keys(&self) -> &crate::layouts::CKKSKey<Self::RotationKeys> {
        &self.rotation_keys
    }

    fn tensor_key(&self) -> &crate::layouts::CKKSKey<Self::TensorKey> {
        &self.tensor_key
    }

    fn encapsulation_keys(
        &self,
    ) -> Option<(
        &crate::layouts::CKKSKey<Self::SwitchingKey>,
        &crate::layouts::CKKSKey<Self::SwitchingKey>,
    )> {
        self.encapsulation_keys.as_ref().map(|(d2s, s2d)| (d2s, s2d))
    }
}

impl<D: Data, W: ZnxWord> BootstrappingKeySet<D, W> {
    /// Preprocesses every key into a [`BootstrappingKeysPrepared`] bundle.
    ///
    /// Convenience for the CPU path that prepares the whole set up front; streaming
    /// callers (e.g. GPU) can instead prepare individual keys on the fly from the
    /// public fields. `scratch` must hold the per-key prepare scratch.
    pub fn prepare<BE: Backend>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingKeysPrepared<BE::OwnedBuf, BE>>
    where
        D: HostDataRef,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>,
    {
        let ring = crate::api::CKKSModuleInfos::ckks_ring(module);
        for key in self.rotation_keys.values() {
            ring.check("bootstrap key preparation", key.key_ring())?;
        }
        ring.check("bootstrap key preparation", self.tensor_key.key_ring())?;
        if let Some((a, b)) = &self.encapsulation_keys {
            ring.check("bootstrap key preparation", a.key_ring())?;
            ring.check("bootstrap key preparation", b.key_ring())?;
        }
        let rotation_keys = self
            .rotation_keys
            .iter()
            .map(|(&p, key)| Ok((p, key.prepare_automorphism(module, scratch)?)))
            .collect::<Result<_>>()?;
        let rotation_keys = crate::layouts::CKKSKey::from_keys(rotation_keys, ring)?;
        let tensor_key = self.tensor_key.prepare_tensor(module, scratch)?;
        let encapsulation_keys = self
            .encapsulation_keys
            .as_ref()
            .map(|(a, b)| -> Result<_> { Ok((a.prepare_switching(module, scratch)?, b.prepare_switching(module, scratch)?)) })
            .transpose()?;
        Ok(BootstrappingKeysPrepared {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
        })
    }
}

/// Layout parameters for the evaluation keys produced by
/// [`BootstrappingContext::generate_keys`].
///
/// Each layout is wrapped with the default encryption noise
/// ([`EncryptionLayout::new_from_default_sigma`]) at generation time. The
/// Galois elements of the rotation keys are read from the compiled DFT matrices,
/// so they are not part of this layout — only the shared automorphism-key shape.
#[derive(Clone, Copy, Debug)]
pub struct BootstrappingKeysLayout {
    /// Shared layout of the rotation and conjugation automorphism keys.
    pub automorphism_key: GLWEAutomorphismKeyLayout,
    /// Layout of the EvalMod relinearization (tensor) key.
    pub tensor_key: GLWETensorKeyLayout,
    /// Physical layouts for the sparse-secret encapsulation keys required by
    /// the recipe, or `None` when the recipe disables the technique.
    pub encapsulation: Option<EncapsulationKeysLayout>,
}

/// Layout of the sparse-secret encapsulation key-switching keys
/// (<https://eprint.iacr.org/2022/024>).
///
/// The compiled recipe is the source of truth for whether encapsulation is
/// enabled and for the ephemeral secret's Hamming weight. This type describes
/// only the two physical key-switch layouts. Key generation rejects a layout
/// whose optional presence disagrees with the recipe.
#[derive(Clone, Copy, Debug)]
pub struct EncapsulationKeysLayout {
    /// `denseToSparse` key layout (sized at the input modulus).
    pub dense_to_sparse: GLWESwitchingKeyLayout,
    /// `sparseToDense` key layout (sized at the bootstrap modulus).
    pub sparse_to_dense: GLWESwitchingKeyLayout,
}

impl<BE: Backend, F> BootstrappingContext<BE, F> {
    /// Generates the **unprepared** [`BootstrappingKeySet`] for `sk_dense`.
    ///
    /// The keys are encrypted but **not preprocessed** (see the [module
    /// docs](self#unprepared-vs-prepared)): call [`BootstrappingKeySet::prepare`]
    /// (or prepare on the fly) before running the pipeline.
    ///
    /// The rotation keys cover the union of the Galois elements of the compiled
    /// `CoeffsToSlots` (and its high-precision bypass, if any) and `SlotsToCoeffs`
    /// matrices; the conjugation key is the Galois-element-`−1` automorphism; the
    /// tensor key relinearizes EvalMod; and, when the compiled recipe enables
    /// sparse-secret encapsulation, a fresh sparse ephemeral secret is sampled
    /// from `source_xs` at the recipe's Hamming weight and the two encapsulation
    /// key-switching keys are derived from `sk_dense`.
    ///
    /// The ephemeral secret never leaves this call and is tagged
    /// [`Distribution::ENCAPSULATED`], so it can neither back a public key nor
    /// be serialized.
    ///
    /// `scratch` must be large enough for the key encrypt operations.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_keys(
        &self,
        module: &Module<BE>,
        sk_dense: &crate::layouts::CKKSKey<BackendGLWESecret<BE>>,
        layout: &BootstrappingKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>,
    {
        let ring = crate::api::CKKSModuleInfos::ckks_ring(module);
        anyhow::ensure!(
            ring.kind == crate::layouts::CKKSRingKind::Standard,
            "bootstrapping keys require a standard ring"
        );
        ring.check("bootstrap key generation", sk_dense.key_ring())?;
        for factor in &self.coeffs_to_slots().inner().factors {
            ring.check("bootstrap context", factor.ring())?;
        }
        let sparse_secret_hamming_weight = self.sparse_secret_hamming_weight();
        anyhow::ensure!(
            sparse_secret_hamming_weight.is_some() == layout.encapsulation.is_some(),
            "bootstrapping key layout encapsulation does not match the compiled recipe (expected {}, got {})",
            sparse_secret_hamming_weight.is_some(),
            layout.encapsulation.is_some()
        );

        let order = module.cyclotomic_order();
        let atk_enc = EncryptionLayout::new_from_default_sigma(layout.automorphism_key)?;

        // Rotation keys: the union of both DFTs' (and the bypass') Galois elements.
        let mut gal_set: BTreeSet<i64> = BTreeSet::new();
        gal_set.extend(self.coeffs_to_slots().galois_elements(order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            gal_set.extend(bypass.galois_elements(order));
        }
        gal_set.extend(self.slots_to_coeffs().galois_elements(order));
        // Conjugation, for the split forward transform.
        gal_set.insert(-1);

        let mut rotation_keys = HashMap::with_capacity(gal_set.len());
        for p in gal_set {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_enc);
            module.glwe_automorphism_key_encrypt_sk(&mut atk, p, sk_dense.as_core(), &atk_enc, source_xe, source_xa, scratch);
            rotation_keys.insert(
                p,
                crate::layouts::CKKSKey::from_raw_parts(atk, crate::api::CKKSModuleInfos::ckks_ring(module))?,
            );
        }

        // Tensor (relinearization) key for EvalMod's ct×ct squaring.
        let tsk_enc = EncryptionLayout::new_from_default_sigma(layout.tensor_key)?;
        let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_enc);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_key, sk_dense.as_core(), &tsk_enc, source_xe, source_xa, scratch);

        // Sparse-secret encapsulation key-switching keys.
        let encapsulation_keys = match (sparse_secret_hamming_weight, &layout.encapsulation) {
            (Some(hamming_weight), Some(encaps)) => {
                let sk_layout = GLWESecretLayout {
                    n: sk_dense.n(),
                    rank: sk_dense.rank(),
                };
                let mut sk_sparse = module.glwe_secret_alloc_from_infos(&sk_layout);
                module.glwe_secret_fill_ternary_hw(&mut sk_sparse, hamming_weight, source_xs);
                *sk_sparse.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");

                let d2s_enc = EncryptionLayout::new_from_default_sigma(encaps.dense_to_sparse)?;
                let s2d_enc = EncryptionLayout::new_from_default_sigma(encaps.sparse_to_dense)?;

                let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&d2s_enc);
                module.glwe_switching_key_encrypt_sk(
                    &mut dense_to_sparse,
                    sk_dense.as_core(),
                    &sk_sparse,
                    &d2s_enc,
                    source_xe,
                    source_xa,
                    scratch,
                );
                let mut sparse_to_dense = module.glwe_switching_key_alloc_from_infos(&s2d_enc);
                module.glwe_switching_key_encrypt_sk(
                    &mut sparse_to_dense,
                    &sk_sparse,
                    sk_dense.as_core(),
                    &s2d_enc,
                    source_xe,
                    source_xa,
                    scratch,
                );
                Some((
                    crate::layouts::CKKSKey::from_raw_parts(dense_to_sparse, crate::api::CKKSModuleInfos::ckks_ring(module))?,
                    crate::layouts::CKKSKey::from_raw_parts(sparse_to_dense, crate::api::CKKSModuleInfos::ckks_ring(module))?,
                ))
            }
            (None, None) => None,
            _ => unreachable!("recipe/layout encapsulation mismatch validated above"),
        };

        Ok(BootstrappingKeySet {
            rotation_keys,
            tensor_key: crate::layouts::CKKSKey::from_raw_parts(tensor_key, crate::api::CKKSModuleInfos::ckks_ring(module))?,
            encapsulation_keys,
        })
    }
}

/// Standard bootstrap material and the two switches between the application
/// secret and the degree-doubled standard secret.
pub struct CIBootstrappingKeys<K, S> {
    pub bootstrap_keys: K,
    /// Switches from the unfolded CI secret `a_0 + Σ a_k(X^k - X^(2N-k))`
    /// to the independent standard secret, both at physical degree `2N`.
    pub ci_to_standard: crate::layouts::CKKSKey<S>,
    /// Reverse of `ci_to_standard`, from the standard secret to the unfolded CI secret.
    pub standard_to_ci: crate::layouts::CKKSKey<S>,
}

/// Unprepared keys for conjugate invariant bootstrapping.
pub type CIBootstrappingKeySet<D, W> = CIBootstrappingKeys<BootstrappingKeySet<D, W>, GLWESwitchingKey<D, W>>;

/// Prepared keys for conjugate invariant bootstrapping.
pub type CIBootstrappingKeysPrepared<D, BE> =
    CIBootstrappingKeys<BootstrappingKeysPrepared<D, BE>, GLWESwitchingKeyPrepared<D, BE>>;

/// Physical key layouts, all at the standard ring degree.
#[derive(Clone, Copy, Debug)]
pub struct CIBootstrappingKeysLayout {
    pub bootstrap_keys: BootstrappingKeysLayout,
    pub ci_to_standard: GLWESwitchingKeyLayout,
    pub standard_to_ci: GLWESwitchingKeyLayout,
}

impl<D: Data, W: ZnxWord> CIBootstrappingKeySet<D, W> {
    /// Validates the ring tags and prepares every key under the standard module.
    pub fn prepare<BE: Backend>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CIBootstrappingKeysPrepared<BE::OwnedBuf, BE>>
    where
        D: HostDataRef,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>,
    {
        use crate::CKKSModuleInfos;
        let ring = module.ckks_ring();
        anyhow::ensure!(
            ring.kind == crate::CKKSRingKind::Standard,
            "CI bootstrap keys require a standard preparation module"
        );
        ring.check("CI bootstrap preparation", self.ci_to_standard.key_ring())?;
        ring.check("CI bootstrap preparation", self.standard_to_ci.key_ring())?;
        Ok(CIBootstrappingKeys {
            bootstrap_keys: self.bootstrap_keys.prepare(module, scratch)?,
            ci_to_standard: self.ci_to_standard.prepare_switching(module, scratch)?,
            standard_to_ci: self.standard_to_ci.prepare_switching(module, scratch)?,
        })
    }
}

impl<BE: Backend<ZnxWord = i64>, F> crate::layouts::CIBootstrappingContext<BE, F> {
    /// Generates the standard bootstrap material and the two ring-switch keys.
    /// The CI secret has degree `N`; the independent standard secret and all
    /// evaluation keys have degree `2N`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_keys(
        &self,
        standard_module: &Module<BE>,
        ci_sk: &crate::layouts::CKKSKey<BackendGLWESecret<BE>>,
        standard_sk: &crate::layouts::CKKSKey<BackendGLWESecret<BE>>,
        layout: &CIBootstrappingKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CIBootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>,
    {
        use crate::{CKKSModuleInfos, CKKSRing, CKKSRingKind};
        use poulpy_core::{
            GetDistribution,
            layouts::{GLWESecretToBackendMut, GLWESecretToBackendRef},
        };
        use poulpy_hal::layouts::{ZnxView, ZnxViewMut};
        let ring = CKKSRing {
            kind: CKKSRingKind::Standard,
            n: standard_module.n().into(),
        };
        ring.check("CI bootstrap standard module", standard_module.ckks_ring())?;
        ring.check("CI bootstrap standard secret", standard_sk.key_ring())?;
        CKKSRing {
            kind: CKKSRingKind::ConjugateInvariant,
            n: (standard_module.n() / 2).into(),
        }
        .check("CI bootstrap application secret", ci_sk.key_ring())?;
        anyhow::ensure!(
            ci_sk.rank().as_usize() == 1 && standard_sk.rank().as_usize() == 1,
            "CI bootstrapping requires rank-1 secrets"
        );
        for key in [&layout.ci_to_standard, &layout.standard_to_ci] {
            anyhow::ensure!(
                key.n == ring.n && key.rank_in.as_usize() == 1 && key.rank_out.as_usize() == 1,
                "invalid CI switching-key layout"
            );
        }
        let bootstrap_keys = self.standard.generate_keys(
            standard_module,
            standard_sk,
            &layout.bootstrap_keys,
            source_xs,
            source_xe,
            source_xa,
            scratch,
        )?;
        let mut mapped_ci = standard_module.glwe_secret_alloc(ci_sk.rank());
        let n = ci_sk.n().as_usize();
        {
            let input = GLWESecretToBackendRef::<BE>::to_backend_ref(ci_sk.as_core());
            let mut output = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut mapped_ci);
            let src = input.data().at(0, 0);
            let dst = output.data_mut().at_mut(0, 0);
            dst.fill(0);
            dst[..n].copy_from_slice(src);
            for k in 1..n {
                dst[2 * n - k] = -src[k];
            }
        }
        *mapped_ci.dist_mut() = *ci_sk.dist();
        let enc = EncryptionLayout::new_from_default_sigma(layout.ci_to_standard)?;
        let mut ci_to_standard = standard_module.glwe_switching_key_alloc_from_infos(&enc);
        standard_module.glwe_switching_key_encrypt_sk(
            &mut ci_to_standard,
            &mapped_ci,
            standard_sk.as_core(),
            &enc,
            source_xe,
            source_xa,
            scratch,
        );
        let enc = EncryptionLayout::new_from_default_sigma(layout.standard_to_ci)?;
        let mut standard_to_ci = standard_module.glwe_switching_key_alloc_from_infos(&enc);
        standard_module.glwe_switching_key_encrypt_sk(
            &mut standard_to_ci,
            standard_sk.as_core(),
            &mapped_ci,
            &enc,
            source_xe,
            source_xa,
            scratch,
        );
        Ok(CIBootstrappingKeys {
            bootstrap_keys,
            ci_to_standard: crate::layouts::CKKSKey::from_raw_parts(ci_to_standard, ring)?,
            standard_to_ci: crate::layouts::CKKSKey::from_raw_parts(standard_to_ci, ring)?,
        })
    }
}
