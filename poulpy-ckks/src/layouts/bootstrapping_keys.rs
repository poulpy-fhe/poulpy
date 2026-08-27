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

use crate::CKKSAtkBounds;
use std::collections::{BTreeSet, HashMap};

use anyhow::Result;
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWEAutomorphismKey,
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyMap, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWEInfos, GLWESecretLayout, GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
        GetGaloisElement, LWEInfos, ModuleCoreAlloc,
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
    /// The prepared automorphism-key type returned for rotations and conjugation.
    type AutomorphismKey: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos;

    /// The rotation-key collection passed to the homomorphic DFT stages.
    type RotationKeys: GLWEAutomorphismKeyMap<Self::AutomorphismKey, BE>;

    /// The prepared tensor (relinearization) key type for EvalMod.
    type TensorKey: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// The prepared key-switching key type for sparse-secret encapsulation.
    type SwitchingKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Rotation (automorphism) keys for `CoeffsToSlots` / `SlotsToCoeffs`.
    fn rotation_keys(&self) -> &Self::RotationKeys;

    /// Conjugation key (Galois element `−1`) for the split `CoeffsToSlots`.
    fn conjugation_key(&self) -> &Self::AutomorphismKey;

    /// Relinearization (tensor) key for EvalMod's `ct×ct` squaring.
    fn tensor_key(&self) -> &Self::TensorKey;

    /// Sparse-secret encapsulation keys `(denseToSparse, sparseToDense)`, or
    /// `None` when the trick is disabled.
    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, &Self::SwitchingKey)>;
}

/// The **unprepared** bootstrapping keys: encrypted but not yet preprocessed.
///
/// This is the form produced by [`BootstrappingContext::generate_keys`] and the
/// form to serialize / store (incl. GPU device memory). Generic over the key data
/// buffer `D` (backend agnostic). Call [`Self::prepare`] to obtain the prepared
/// [`BootstrappingKeysPrepared`] the pipeline consumes, or prepare individual keys
/// on the fly.
pub struct BootstrappingKeySet<D: Data, W: ZnxWord> {
    /// Rotation keys indexed by Galois element (the engine-wide convention).
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
    /// Conjugation key (Galois element `−1`).
    pub conjugation_key: GLWEAutomorphismKey<D, W>,
    /// Relinearization (tensor) key for EvalMod.
    pub tensor_key: GLWETensorKey<D, W>,
    /// `(denseToSparse, sparseToDense)` encapsulation keys, or `None`.
    pub encapsulation_keys: Option<(GLWESwitchingKey<D, W>, GLWESwitchingKey<D, W>)>,
}

/// The **prepared** (preprocessed) bootstrapping keys, ready for the pipeline.
///
/// Generic over the key data buffer `D`; implements [`BootstrappingKeys`]. Built
/// eagerly by [`BootstrappingKeySet::prepare`].
pub struct BootstrappingKeysPrepared<D: Data, BE: Backend> {
    /// Prepared rotation keys indexed by Galois element.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    /// Prepared conjugation key (Galois element `−1`).
    pub conjugation_key: GLWEAutomorphismKeyPrepared<D, BE>,
    /// Prepared relinearization (tensor) key for EvalMod.
    pub tensor_key: GLWETensorKeyPrepared<D, BE>,
    /// Prepared `(denseToSparse, sparseToDense)` encapsulation keys, or `None`.
    pub encapsulation_keys: Option<(GLWESwitchingKeyPrepared<D, BE>, GLWESwitchingKeyPrepared<D, BE>)>,
}

impl<D: Data, BE: Backend> BootstrappingKeys<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: CKKSAtkBounds<BE>,
    GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type AutomorphismKey = GLWEAutomorphismKeyPrepared<D, BE>;
    type RotationKeys = HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>;
    type TensorKey = GLWETensorKeyPrepared<D, BE>;
    type SwitchingKey = GLWESwitchingKeyPrepared<D, BE>;

    fn rotation_keys(&self) -> &Self::RotationKeys {
        &self.rotation_keys
    }

    fn conjugation_key(&self) -> &Self::AutomorphismKey {
        &self.conjugation_key
    }

    fn tensor_key(&self) -> &Self::TensorKey {
        &self.tensor_key
    }

    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, &Self::SwitchingKey)> {
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
    ) -> BootstrappingKeysPrepared<BE::OwnedBuf, BE>
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
        let mut rotation_keys = HashMap::with_capacity(self.rotation_keys.len());
        for (&p, atk) in &self.rotation_keys {
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(atk);
            module.glwe_automorphism_key_prepare(&mut prepared, atk, scratch);
            rotation_keys.insert(p, prepared);
        }

        let conjugation_key = {
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&self.conjugation_key);
            module.glwe_automorphism_key_prepare(&mut prepared, &self.conjugation_key, scratch);
            prepared
        };

        let tensor_key = {
            let mut prepared = module.alloc_tensor_key_prepared_from_infos(&self.tensor_key);
            module.prepare_tensor_key(&mut prepared, &self.tensor_key, scratch);
            prepared
        };

        let encapsulation_keys = self.encapsulation_keys.as_ref().map(|(d2s, s2d)| {
            let mut d2s_prepared = module.glwe_switching_key_prepared_alloc_from_infos(d2s);
            module.glwe_switching_key_prepare(&mut d2s_prepared, d2s, scratch);
            let mut s2d_prepared = module.glwe_switching_key_prepared_alloc_from_infos(s2d);
            module.glwe_switching_key_prepare(&mut s2d_prepared, s2d, scratch);
            (d2s_prepared, s2d_prepared)
        });

        BootstrappingKeysPrepared {
            rotation_keys,
            conjugation_key,
            tensor_key,
            encapsulation_keys,
        }
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
        sk_dense: &BackendGLWESecret<BE>,
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

        let mut rotation_keys = HashMap::with_capacity(gal_set.len());
        for p in gal_set {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_enc);
            module.glwe_automorphism_key_encrypt_sk(&mut atk, p, sk_dense, &atk_enc, source_xe, source_xa, scratch);
            rotation_keys.insert(p, atk);
        }

        // Conjugation key for the split forward transform (Galois element −1).
        let mut conjugation_key = module.glwe_automorphism_key_alloc_from_infos(&atk_enc);
        module.glwe_automorphism_key_encrypt_sk(&mut conjugation_key, -1, sk_dense, &atk_enc, source_xe, source_xa, scratch);

        // Tensor (relinearization) key for EvalMod's ct×ct squaring.
        let tsk_enc = EncryptionLayout::new_from_default_sigma(layout.tensor_key)?;
        let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_enc);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_key, sk_dense, &tsk_enc, source_xe, source_xa, scratch);

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
                    sk_dense,
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
                    sk_dense,
                    &s2d_enc,
                    source_xe,
                    source_xa,
                    scratch,
                );
                Some((dense_to_sparse, sparse_to_dense))
            }
            (None, None) => None,
            _ => unreachable!("recipe/layout encapsulation mismatch validated above"),
        };

        Ok(BootstrappingKeySet {
            rotation_keys,
            conjugation_key,
            tensor_key,
            encapsulation_keys,
        })
    }
}
