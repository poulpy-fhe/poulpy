//! Evaluation keys for CKKS bootstrapping.
//!
//! The bootstrapping pipelines consume four kinds of key material:
//!
//! - **rotation (automorphism) keys** for the two homomorphic DFTs — the union
//!   of the Galois elements of `CoeffsToSlots` (incl. the optional high-precision
//!   bypass) and `SlotsToCoeffs`;
//! - a **conjugation key** (Galois element `−1`) for the split `CoeffsToSlots`;
//! - a **tensor (relinearization) key** for EvalMod's `ct×ct` squaring, on the ring
//!   the plan runs EvalMod on;
//! - optionally, one **first-factor key** per diagonal of the first
//!   `CoeffsToSlots` factor, when the plan leaves that factor to the modulus
//!   raise ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise));
//! - optionally, the **sparse-secret encapsulation** key-switching keys
//!   (`denseToSparse`, `sparseToDense`; <https://eprint.iacr.org/2022/024>),
//!   whose sparse ephemeral secret is sampled inside
//!   [`BootstrappingContextPrepared::generate_keys`] and never escapes it;
//! - with EvalMod on the conjugate-invariant ring, the **trace** and **embedding**
//!   keys there and back.
//!
//! ## Unprepared vs prepared
//!
//! [`BootstrappingContextPrepared::generate_keys`] returns an **unprepared**
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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use poulpy_core::{
    GGLWEBytesOf, GLWEAutomorphismKeyEncryptSk, GLWEBytesOf, GLWECIKeyEncryptSk, GLWESwitchingKeyEncryptSk,
    GLWETensorKeyEncryptSk, ScratchArenaTakeGGLWE, ScratchArenaTakeGLWESecret,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWECIKeyPreparedFactory, GLWESwitchingKeyPreparedFactory,
        GLWETensorKeyPreparedFactory,
    },
    layouts::{
        BackendGLWESecret, Degree, GGLWEBackendRef, GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendMut,
        GGLWEPreparedToBackendRef, GGLWEToBackendRef, GGLWEViewMut, GLWEAutomorphismKey, GLWEAutomorphismKeyAlloc,
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWECIEmbedKey, GLWECIEmbedKeyAlloc, GLWECIEmbedKeyPrepared,
        GLWECIKeyPrepared, GLWECITraceKey, GLWECITraceKeyAlloc, GLWECITraceKeyPrepared, GLWEInfos, GLWESecretBackendRef,
        GLWESecretToBackendRef, GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
        GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared, GLWETensorKey, GLWETensorKeyAlloc, GLWETensorKeyLayout,
        GLWETensorKeyPrepared, GetAutomorphismKey, GetGaloisElement, GetTensorKey, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef,
        },
    },
};
use poulpy_hal::{
    api::VecZnxAutomorphism,
    layouts::{
        Backend, ConjugateInvariant, CopyToHost, CyclotomicOrder, Data, HostDataMut, HostDataRef, Module, ScratchArena, Standard,
        ZnxWord, galois_element, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    CKKSCompositionError,
    layouts::{BootstrappingContextPrepared, BootstrappingPlan, EvalModRing},
};
use poulpy_core::api::{GLWESecretSampling, TransferInto};
use poulpy_core::{Distribution, GetDistribution, GetDistributionMut};

/// Pipeline-facing access to the **prepared** keys of the linear stages of a CKKS
/// bootstrap: the raise and the homomorphic transforms, which every bootstrap runs
/// beside EvalMod.
///
/// The key types are associated (not fixed to a backend buffer), so an implementation
/// is free to back them with any data representation and to prepare them lazily / on
/// the fly. [`BootstrappingKeysPrepared`] is the eager in-memory implementation.
///
/// As a [`GetAutomorphismKey`] it answers with the rotation keys of the homomorphic DFT
/// stages, Galois element `−1` among them, so a DFT operation takes these keys directly.
pub trait BootstrappingLinearKeys<BE: Backend>: GetAutomorphismKey<BE> {
    /// The prepared key-switching key type for sparse-secret encapsulation.
    type SwitchingKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Sparse-secret encapsulation keys `(denseToSparse, sparseToDense)`, or
    /// `None` when the trick is disabled. There is no `sparseToDense` key when
    /// the first `CoeffsToSlots` factor is applied on the raise, whose keys
    /// switch to the dense secret instead.
    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, Option<&Self::SwitchingKey>)>;

    /// Key of the diagonal of slot rotation `rotation` of a first
    /// `CoeffsToSlots` factor left to the modulus raise: a switching key with a
    /// row offset from the rotated raise secret to the dense one (see
    /// [`BootstrappingKeysLayout::first_factor_key`]). None by default.
    fn first_factor_key(&self, _rotation: i64) -> Option<&Self::SwitchingKey> {
        None
    }

    /// Layouts of these keys, against whose scratch query the bootstrap checks its
    /// arena at entry, or `None`, the default, to leave the arena unchecked.
    fn keys_layout(&self) -> Option<&BootstrappingKeysLayout> {
        None
    }
}

/// The prepared keys of a CKKS bootstrap: the [linear keys](BootstrappingLinearKeys) and
/// EvalMod's relinearization key, on the ring the plan runs EvalMod on, with the conversion
/// keys on the conjugate-invariant ring. A key of the other ring is `None`, and a bootstrap
/// refuses the absence of one it reads.
pub trait BootstrappingKeys<BE: Backend>: BootstrappingLinearKeys<BE> {
    /// The prepared tensor (relinearization) key type for EvalMod on the standard ring.
    type TensorKey: GetTensorKey<BE>;

    /// The prepared tensor (relinearization) key type for EvalMod on the conjugate-invariant ring.
    type CITensorKey: GetTensorKey<BE, ConjugateInvariant>;

    /// The prepared trace key type, from `s(X²)` to the embedded CI secret.
    type TraceKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// The prepared embedding key type, from the embedded CI secret to `s(X²)`.
    type EmbedKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Relinearization (tensor) key for EvalMod's `ct×ct` squaring on the standard ring,
    /// covering the widest ciphertext EvalMod relinearizes.
    fn tensor_key(&self) -> Option<&Self::TensorKey>;

    /// Relinearization (tensor) key for EvalMod on the conjugate-invariant ring, covering
    /// the widest ciphertext EvalMod relinearizes.
    fn ci_tensor_key(&self) -> Option<&Self::CITensorKey>;

    /// Trace and embedding keys of degree `2N`, with EvalMod on the conjugate-invariant ring.
    fn conversion_keys(&self) -> Option<(&Self::TraceKey, &Self::EmbedKey)>;
}

/// Sparse-secret encapsulation keys `(denseToSparse, sparseToDense)`, without
/// `sparseToDense` when the first `CoeffsToSlots` factor is applied on the raise.
pub type EncapsulationKeys<K> = (K, Option<K>);

/// The **unprepared** bootstrapping keys: encrypted but not yet preprocessed.
///
/// This is the form produced by [`BootstrappingContextPrepared::generate_keys`] and the
/// form to serialize / store (incl. GPU device memory). Generic over the key data
/// buffer `D` (backend agnostic). Call [`Self::prepare`] to obtain the prepared
/// [`BootstrappingKeysPrepared`] the pipeline consumes, or prepare individual keys
/// on the fly.
pub struct BootstrappingKeySet<D: Data, W: ZnxWord> {
    /// Rotation keys indexed by Galois element (the engine-wide convention),
    /// conjugation (`−1`) among them.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
    /// Relinearization (tensor) key for EvalMod on the standard ring, covering its widest ciphertext.
    pub tensor_key: Option<GLWETensorKey<D, W>>,
    /// Relinearization (tensor) key for EvalMod on the conjugate-invariant ring, covering its
    /// widest ciphertext.
    pub ci_tensor_key: Option<GLWETensorKey<D, W, ConjugateInvariant>>,
    /// `(denseToSparse, sparseToDense)` encapsulation keys, or `None`; no
    /// `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKey<D, W>>>,
    /// First-factor keys by slot rotation, possibly none: switching keys from
    /// the rotated raise secret (the sparse one under encapsulation) to the
    /// dense one, see [`BootstrappingKeysLayout::first_factor_key`].
    pub first_factor_keys: HashMap<i64, GLWESwitchingKey<D, W>>,
    /// Trace and embedding keys of degree `2N` with EvalMod on the conjugate-invariant ring.
    pub conversion_keys: Option<(GLWECITraceKey<D, W>, GLWECIEmbedKey<D, W>)>,
}

/// The **prepared** (preprocessed) bootstrapping keys, ready for the pipeline.
///
/// Generic over the key data buffer `D`; implements [`BootstrappingKeys`]. Built
/// eagerly by [`BootstrappingKeySet::prepare`]. The plan's ring says which tensor key a
/// run reads: a single generic impl cannot serve both rings otherwise.
pub struct BootstrappingKeysPrepared<D: Data, BE: Backend> {
    /// Prepared rotation keys indexed by Galois element, conjugation among them.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    /// Prepared relinearization (tensor) key for EvalMod on the standard ring, covering its
    /// widest ciphertext.
    pub tensor_key: Option<GLWETensorKeyPrepared<D, BE>>,
    /// Prepared relinearization (tensor) key for EvalMod on the conjugate-invariant ring,
    /// covering its widest ciphertext.
    pub ci_tensor_key: Option<GLWETensorKeyPrepared<D, BE, ConjugateInvariant>>,
    /// Prepared `(denseToSparse, sparseToDense)` encapsulation keys, or `None`;
    /// no `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<D, BE>>>,
    /// Prepared first-factor keys by slot rotation, possibly none.
    pub first_factor_keys: HashMap<i64, GLWESwitchingKeyPrepared<D, BE>>,
    /// Prepared trace and embedding keys, with EvalMod on the conjugate-invariant ring.
    pub conversion_keys: Option<(GLWECITraceKeyPrepared<D, BE>, GLWECIEmbedKeyPrepared<D, BE>)>,
    /// The layouts of these keys, read once at preparation: every step checks its arena against them, and a key
    /// replaced later does not update them.
    pub(crate) keys_layout: Option<BootstrappingKeysLayout>,
}

impl<D: Data, BE: Backend> BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    /// The layouts of the keys of this bundle, `None` without a tensor key.
    fn read_keys_layout(&self) -> Option<BootstrappingKeysLayout> {
        let tensor_key = match (&self.tensor_key, &self.ci_tensor_key) {
            (Some(key), _) => key.gglwe_layout(),
            (None, Some(key)) => key.gglwe_layout(),
            (None, None) => return None,
        };
        let rotation_keys = self
            .rotation_keys
            .iter()
            .map(|(&p, key)| (p, key.gglwe_layout().into()))
            .collect();
        let switching = |key: &GLWESwitchingKeyPrepared<D, BE>| GLWESwitchingKeyLayout::from(key.gglwe_layout());
        Some(BootstrappingKeysLayout {
            rotation_keys,
            tensor_key: tensor_key.into(),
            encapsulation: self.encapsulation_keys.as_ref().map(|(d2s, s2d)| EncapsulationKeysLayout {
                dense_to_sparse: switching(d2s),
                sparse_to_dense: s2d.as_ref().map(switching),
            }),
            first_factor_key: self.first_factor_keys.values().next().map(switching),
            conversion_keys: self
                .conversion_keys
                .as_ref()
                .map(|(trace, embed)| (trace.gglwe_layout().into(), embed.gglwe_layout().into())),
        })
    }
}

impl<D: Data, BE: Backend> GetAutomorphismKey<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
{
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.rotation_keys.lookup_automorphism_key(p, k)
    }
}

impl<D: Data, BE: Backend> BootstrappingLinearKeys<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type SwitchingKey = GLWESwitchingKeyPrepared<D, BE>;

    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, Option<&Self::SwitchingKey>)> {
        self.encapsulation_keys.as_ref().map(|(d2s, s2d)| (d2s, s2d.as_ref()))
    }

    fn first_factor_key(&self, rotation: i64) -> Option<&Self::SwitchingKey> {
        self.first_factor_keys.get(&rotation)
    }

    fn keys_layout(&self) -> Option<&BootstrappingKeysLayout> {
        self.keys_layout.as_ref()
    }
}

impl<D: Data, BE: Backend> BootstrappingKeys<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<D, BE, ConjugateInvariant>: GLWETensorKeyPreparedToBackendRef<BE, ConjugateInvariant>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWECITraceKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    GLWECIEmbedKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type TensorKey = GLWETensorKeyPrepared<D, BE>;
    type CITensorKey = GLWETensorKeyPrepared<D, BE, ConjugateInvariant>;
    type TraceKey = GLWECITraceKeyPrepared<D, BE>;
    type EmbedKey = GLWECIEmbedKeyPrepared<D, BE>;

    fn tensor_key(&self) -> Option<&Self::TensorKey> {
        self.tensor_key.as_ref()
    }

    fn ci_tensor_key(&self) -> Option<&Self::CITensorKey> {
        self.ci_tensor_key.as_ref()
    }

    fn conversion_keys(&self) -> Option<(&Self::TraceKey, &Self::EmbedKey)> {
        self.conversion_keys.as_ref().map(|(trace, embed)| (trace, embed))
    }
}

impl<D: Data, W: ZnxWord> BootstrappingKeySet<D, W> {
    /// Preprocesses every key into a [`BootstrappingKeysPrepared`] bundle: the conversion
    /// keys at degree `2N` and the tensor key of the set's ring.
    ///
    /// Convenience for the CPU path that prepares the whole set up front; streaming
    /// callers (e.g. GPU) can instead prepare individual keys on the fly from the
    /// public fields. `scratch` holds [`BootstrappingKeysLayout::prepare_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// None: it always returns `Ok`.
    pub fn prepare<BE>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingKeysPrepared<BE::OwnedBuf, BE>>
    where
        D: HostDataRef,
        BE: Backend<ZnxWord = W>,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWETensorKey<D, W, ConjugateInvariant>: GGLWEToBackendRef<BE, ConjugateInvariant> + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWECIKeyPreparedFactory<BE>,
        Module<BE>: GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    {
        let mut rotation_keys = HashMap::with_capacity(self.rotation_keys.len());
        for (&p, atk) in &self.rotation_keys {
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(atk);
            module.glwe_automorphism_key_prepare(&mut prepared, atk, scratch);
            rotation_keys.insert(p, prepared);
        }

        let tensor_key = self.tensor_key.as_ref().map(|key| {
            let mut prepared = module.glwe_tensor_key_prepared_alloc_from_infos(key);
            module.glwe_tensor_key_prepare(&mut prepared, key, scratch);
            prepared
        });
        let ci_tensor_key = self.ci_tensor_key.as_ref().map(|key| {
            let mut prepared =
                <Module<BE> as GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>>::glwe_tensor_key_prepared_alloc_from_infos(
                    module, key,
                );
            module.glwe_tensor_key_prepare(&mut prepared, key, scratch);
            prepared
        });

        let encapsulation_keys = self.encapsulation_keys.as_ref().map(|(d2s, s2d)| {
            let mut d2s_prepared = module.glwe_switching_key_prepared_alloc_from_infos(d2s);
            module.glwe_switching_key_prepare(&mut d2s_prepared, d2s, scratch);
            let s2d_prepared = s2d.as_ref().map(|s2d| {
                let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(s2d);
                module.glwe_switching_key_prepare(&mut prepared, s2d, scratch);
                prepared
            });
            (d2s_prepared, s2d_prepared)
        });

        let mut first_factor_keys = HashMap::with_capacity(self.first_factor_keys.len());
        for (&rotation, key) in &self.first_factor_keys {
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
            module.glwe_switching_key_prepare(&mut prepared, key, scratch);
            first_factor_keys.insert(rotation, prepared);
        }

        let conversion_keys = self.conversion_keys.as_ref().map(|(trace, embed)| {
            let mut trace_key = module.glwe_ci_trace_key_prepared_alloc_from_infos(trace);
            module.glwe_ci_key_prepare(&mut trace_key, trace, scratch);
            let mut embed_key = module.glwe_ci_embed_key_prepared_alloc_from_infos(embed);
            module.glwe_ci_key_prepare(&mut embed_key, embed, scratch);
            (trace_key, embed_key)
        });

        let mut keys = BootstrappingKeysPrepared {
            rotation_keys,
            tensor_key,
            ci_tensor_key,
            encapsulation_keys,
            first_factor_keys,
            conversion_keys,
            keys_layout: None,
        };
        keys.keys_layout = keys.read_keys_layout();
        Ok(keys)
    }
}

/// Layout parameters for the evaluation keys produced by
/// [`BootstrappingContextPrepared::generate_keys`].
///
/// Key generation uses [`poulpy_core::Noise::ENCRYPTION`] at each
/// destination key's precision.
#[derive(Clone, Debug)]
pub struct BootstrappingKeysLayout {
    /// Layouts of the rotation keys, by Galois element: exactly the elements of
    /// [`BootstrappingPlan::key_widths`], those of [`BootstrappingPlan::galois_elements`]
    /// and, with EvalMod on the standard ring, the conjugation `-1`. Each must cover the
    /// widest ciphertext its key switches, which a key used on narrow ciphertexts only
    /// can do with a wider digit, hence fewer rows, under the same modulus bound.
    pub rotation_keys: BTreeMap<i64, GLWEAutomorphismKeyLayout>,
    /// Layout of the EvalMod relinearization (tensor) key, on the ring EvalMod
    /// runs on. It must cover the widest ciphertext EvalMod relinearizes.
    pub tensor_key: GLWETensorKeyLayout,
    /// Physical layouts for the sparse-secret encapsulation keys required by
    /// the recipe, or `None` when the recipe disables the technique.
    pub encapsulation: Option<EncapsulationKeysLayout>,
    /// Layout of the keys of a first `CoeffsToSlots` factor left to the
    /// modulus raise ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise)),
    /// or `None` when the recipe evaluates it with the others. Each switches a
    /// diagonal's term, a few limbs at the bottom of the raised width, from the
    /// rotated raise secret to the dense one: its rows sit under the term, `k_offset`
    /// bits lower, and reach the width the raise sums its terms at.
    pub first_factor_key: Option<GLWESwitchingKeyLayout>,
    /// Layouts of the trace and embedding keys of degree `2N`, with EvalMod on
    /// the conjugate-invariant ring: they cover what CoeffsToSlots and EvalMod leave.
    pub conversion_keys: Option<(GLWESwitchingKeyLayout, GLWESwitchingKeyLayout)>,
}

impl BootstrappingKeysLayout {
    /// The distinct rotation key layouts, in the order of their first Galois element.
    pub fn rotation_key_layouts(&self) -> impl Iterator<Item = &GLWEAutomorphismKeyLayout> {
        let layouts = &self.rotation_keys;
        layouts
            .values()
            .enumerate()
            .filter(move |&(index, layout)| !layouts.values().take(index).any(|seen| seen == layout))
            .map(|(_, layout)| layout)
    }

    /// The switching keys of the raise: the encapsulation keys and the first-factor key.
    fn raise_key_layouts(&self) -> Vec<GLWESwitchingKeyLayout> {
        let encapsulation = self
            .encapsulation
            .iter()
            .flat_map(|e| [Some(e.dense_to_sparse), e.sparse_to_dense]);
        encapsulation.chain([self.first_factor_key]).flatten().collect()
    }

    /// One secret of the keys' degree and rank, as key generation carves it.
    fn secret_bytes<BE: Backend>(&self) -> usize {
        let (n, rank) = (self.tensor_key.n().as_usize(), self.tensor_key.rank_out().as_usize());
        BE::scratch_aligned(BE::scalar_znx_bytes_of(n, rank))
    }

    /// The encryption of the rotation and raise keys: the encapsulation keys beside the sparse
    /// secret, a first-factor key beside the rotated secret it is carved with, and beside the
    /// sparse secret too under encapsulation.
    fn linear_keys_tmp_bytes<BE>(&self, module: &Module<BE>) -> usize
    where
        BE: Backend,
        Module<BE>: GLWEAutomorphismKeyEncryptSk<BE> + GLWESwitchingKeyEncryptSk<BE>,
    {
        let secret = self.secret_bytes::<BE>();
        let sparse = usize::from(self.encapsulation.is_some()) * secret;
        let mut bytes = 0;
        for layout in self.rotation_key_layouts() {
            bytes = bytes.max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&layout));
        }
        for layout in self.raise_key_layouts() {
            bytes = bytes.max(sparse + module.glwe_switching_key_encrypt_sk_tmp_bytes(&layout));
        }
        if let Some(layout) = &self.first_factor_key {
            bytes = bytes.max(sparse + secret + module.glwe_switching_key_encrypt_sk_tmp_bytes(layout));
        }
        bytes
    }

    /// Scratch of [`BootstrappingPlan::generate_keys`] for these layouts: the encryption of
    /// each key, beside the secrets carved for the raise keys. With EvalMod on the
    /// conjugate-invariant ring, the tensor and conversion keys sit beside the
    /// conjugate-invariant secret.
    pub fn generate_tmp_bytes<BE>(&self, module: &Module<BE>) -> usize
    where
        BE: Backend,
        Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BE>: GLWETensorKeyEncryptSk<BE, ConjugateInvariant>,
    {
        let linear = self.linear_keys_tmp_bytes(module);
        match &self.conversion_keys {
            Some((trace, embed)) => {
                let ci_keys =
                    <Module<BE> as GLWETensorKeyEncryptSk<BE, ConjugateInvariant>>::glwe_tensor_key_encrypt_sk_tmp_bytes(
                        module,
                        &self.tensor_key,
                    )
                    .max(module.glwe_ci_key_encrypt_sk_tmp_bytes(trace))
                    .max(module.glwe_ci_key_encrypt_sk_tmp_bytes(embed));
                linear.max(self.secret_bytes::<BE>() + ci_keys)
            }
            None => linear
                .max(<Module<BE> as GLWETensorKeyEncryptSk<BE>>::glwe_tensor_key_encrypt_sk_tmp_bytes(module, &self.tensor_key)),
        }
    }

    /// Scratch of [`BootstrappingKeySet::prepare`] and [`BootstrappingKeySet::transfer_and_prepare`]
    /// for these layouts: the preparation of each key, beside the copy of it the latter carves,
    /// which the former leaves unused.
    pub fn prepare_tmp_bytes<BE>(&self, module: &Module<BE>) -> usize
    where
        BE: Backend,
        Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWECIKeyPreparedFactory<BE>
            + GLWEBytesOf<BE>
            + GGLWEBytesOf<BE>,
        Module<BE>: GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    {
        // Each key's preparation, beside the copy [`BootstrappingKeySet::transfer_and_prepare`]
        // moves it into.
        let moved = |layout: &dyn GGLWEInfos| BE::scratch_aligned(module.gglwe_bytes_of_from_infos(&layout));
        let mut bytes = 0;
        for layout in self.rotation_key_layouts() {
            bytes = bytes.max(moved(&layout) + module.glwe_automorphism_key_prepare_tmp_bytes(&layout));
        }
        for layout in self.raise_key_layouts() {
            bytes = bytes.max(moved(&layout) + module.glwe_switching_key_prepare_tmp_bytes(&layout));
        }
        match &self.conversion_keys {
            Some((trace, embed)) => {
                bytes
                    .max(moved(trace) + module.glwe_ci_key_prepare_tmp_bytes(trace))
                    .max(moved(embed) + module.glwe_ci_key_prepare_tmp_bytes(embed))
                    + moved(&self.tensor_key)
                    + <Module<BE> as GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>>::glwe_tensor_key_prepare_tmp_bytes(
                        module,
                        &self.tensor_key,
                    )
            }
            None => bytes.max(
                moved(&self.tensor_key)
                    + <Module<BE> as GLWETensorKeyPreparedFactory<BE>>::glwe_tensor_key_prepare_tmp_bytes(
                        module,
                        &self.tensor_key,
                    ),
            ),
        }
    }
}

/// Layout of the sparse-secret encapsulation key-switching keys
/// (<https://eprint.iacr.org/2022/024>).
///
/// The compiled recipe is the source of truth for whether encapsulation is
/// enabled and for the ephemeral secret's Hamming weight. This type describes
/// only the physical key-switch layouts. Key generation rejects a layout whose
/// optional presence disagrees with the recipe, here or for `sparseToDense`.
#[derive(Clone, Copy, Debug)]
pub struct EncapsulationKeysLayout {
    /// `denseToSparse` key layout (sized at the input modulus).
    pub dense_to_sparse: GLWESwitchingKeyLayout,
    /// `sparseToDense` key layout, or `None` when the first `CoeffsToSlots`
    /// factor is applied on the raise: its keys switch each diagonal's term to
    /// the dense secret, and this key is unused. With a row offset, its switch
    /// is the raise itself: its rows sit under the raised ciphertext and end at
    /// the bootstrap modulus, so it covers only the raised ciphertext's
    /// [`BootstrappingPlan::raised_window`](crate::layouts::BootstrappingPlan::raised_window)
    /// bits. Without one, it switches the raised ciphertext at the whole bootstrap modulus.
    pub sparse_to_dense: Option<GLWESwitchingKeyLayout>,
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F> {
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
    /// With EvalMod on the conjugate-invariant ring, a conjugate-invariant secret of the
    /// dense secret's Hamming weight and degree `N`, sampled after the raise keys, has the
    /// tensor key; the trace and embedding keys switch between it and the dense secret
    /// lifted to `s(X²)`, of degree `2N`, the degree of `module`, which also serves the
    /// bootstrap's degree `N`.
    ///
    /// [`BootstrappingKeysLayout::generate_tmp_bytes`] sizes `scratch`, which holds the
    /// ephemeral sparse secret with encapsulation, the rotated raise secret with
    /// first-factor keys and the conjugate-invariant secret, and which is wiped before
    /// returning.
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) with EvalMod on the
    /// conjugate-invariant ring, for missing conversion key layouts, a dense secret not of
    /// fixed-weight ternary law, or a module of another degree than twice the secret's,
    /// before any draw.
    #[expect(
        clippy::too_many_arguments,
        reason = "key generation takes the module, the secret, the layout, three sources and a scratch arena"
    )]
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
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BE>: GLWETensorKeyEncryptSk<BE, ConjugateInvariant> + GLWESecretSampling<BE, ConjugateInvariant>,
    {
        self.plan
            .generate_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch)
    }
}

impl BootstrappingPlan {
    /// [`BootstrappingContextPrepared::generate_keys`] from the plan alone, for a key
    /// owner that does not compile the context: the client of an evaluation on
    /// another backend. The keys are those the compiled context generates.
    #[expect(
        clippy::too_many_arguments,
        reason = "key generation takes the module, the secret, the layout, three sources and a scratch arena"
    )]
    pub fn generate_keys<BE>(
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
        BE: Backend,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BE>: GLWETensorKeyEncryptSk<BE, ConjugateInvariant> + GLWESecretSampling<BE, ConjugateInvariant>,
    {
        // The cyclotomic order of the secret's degree, which the module may exceed.
        let order = crate::operand_cyclotomic_order::<_, Standard>("generate_keys", module, sk_dense.n().as_usize())?;
        self.key_recipe(sk_dense.n().as_usize(), order)?
            .generate_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch)
    }

    /// The keys' recipe, on a ring of degree `n`, whose slots must hold both transforms, as
    /// compiling the context checks.
    pub(crate) fn key_recipe(&self, n: usize, cyclotomic_order: i64) -> Result<KeyRecipe> {
        let log_n = n.ilog2() as usize;
        for dft in [self.coeffs_to_slots(), self.slots_to_coeffs()] {
            crate::layouts::dft::check_dft_fits(dft, log_n)?;
        }
        let first_factor_rotations = if self.coeffs_to_slots().first_factor_on_raise() {
            self.coeffs_to_slots().first_factor_rotations(log_n)
        } else {
            Vec::new()
        };
        Ok(KeyRecipe::new(
            self.galois_elements(log_n, cyclotomic_order).into_iter().collect(),
            first_factor_rotations,
            self.sparse_secret_hamming_weight(),
            self.eval_mod_ring(),
            cyclotomic_order,
        ))
    }
}

/// What the keys of a bootstrap depend on, read from its plan, by the client and the compiled
/// context alike.
pub(crate) struct KeyRecipe {
    /// Galois elements of the homomorphic DFTs and of the bypass, and
    /// conjugation with EvalMod on the standard ring.
    galois_elements: BTreeSet<i64>,
    /// Slot rotations of a first CoeffsToSlots factor left to the raise.
    first_factor_rotations: Vec<i64>,
    sparse_secret_hamming_weight: Option<usize>,
    eval_mod_ring: EvalModRing,
    cyclotomic_order: i64,
}

impl KeyRecipe {
    fn new(
        mut galois_elements: BTreeSet<i64>,
        first_factor_rotations: Vec<i64>,
        sparse_secret_hamming_weight: Option<usize>,
        eval_mod_ring: EvalModRing,
        cyclotomic_order: i64,
    ) -> Self {
        // Conjugation, for the split forward transform.
        if eval_mod_ring == EvalModRing::Standard {
            galois_elements.insert(-1);
        }
        Self {
            galois_elements,
            first_factor_rotations,
            sparse_secret_hamming_weight,
            eval_mod_ring,
            cyclotomic_order,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "key generation takes the module, the secret, the layout, three sources and a scratch arena"
    )]
    fn generate_keys<BE>(
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
        BE: Backend,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BE>: GLWETensorKeyEncryptSk<BE, ConjugateInvariant> + GLWESecretSampling<BE, ConjugateInvariant>,
    {
        if self.eval_mod_ring == EvalModRing::Standard {
            let keys = self.generate_linear_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch)?;
            scratch.wipe(layout.generate_tmp_bytes(module));
            return Ok(keys);
        }
        let refuse = |reason: &str| -> anyhow::Error {
            CKKSCompositionError::InvalidOperand {
                op: "generate_keys",
                reason: reason.into(),
            }
            .into()
        };
        let (Some((trace_layout, embed_layout)), Distribution::TernaryFixed(hamming_weight)) =
            (layout.conversion_keys, *sk_dense.dist())
        else {
            return Err(refuse(
                "EvalMod on the CI ring takes conversion key layouts and a fixed-weight ternary dense secret",
            ));
        };
        if 2 * sk_dense.n().as_usize() != module.n() {
            return Err(refuse(
                "EvalMod on the CI ring takes a module of twice the degree of the dense secret",
            ));
        }
        let mut keys = self.generate_linear_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch)?;
        let mut ci_tensor_key = module.glwe_tensor_key_alloc_from_infos::<_, ConjugateInvariant>(&layout.tensor_key);
        let conversion_keys = {
            let (mut sk_ci, mut scratch) = scratch
                .borrow()
                .take_glwe_secret_scratch::<ConjugateInvariant>(sk_dense.n(), sk_dense.rank());
            module.glwe_secret_fill_ternary_hw(&mut sk_ci, hamming_weight, source_xs);
            // Created here and wiped before returning: it backs no public key and is not serialized.
            *sk_ci.dist_mut() = Distribution::ENCAPSULATED("ci-eval-mod");
            module.glwe_tensor_key_encrypt_sk(&mut ci_tensor_key, &sk_ci, source_xe, source_xa, &mut scratch);
            let mut trace_key = module.glwe_ci_trace_key_alloc_from_infos(&trace_layout);
            module.glwe_ci_trace_key_encrypt_sk(&mut trace_key, &sk_ci, sk_dense, source_xe, source_xa, &mut scratch);
            let mut embed_key = module.glwe_ci_embed_key_alloc_from_infos(&embed_layout);
            module.glwe_ci_embed_key_encrypt_sk(&mut embed_key, &sk_ci, sk_dense, source_xe, source_xa, &mut scratch);
            (trace_key, embed_key)
        };
        scratch.wipe(layout.generate_tmp_bytes(module));
        keys.ci_tensor_key = Some(ci_tensor_key);
        keys.conversion_keys = Some(conversion_keys);
        Ok(keys)
    }

    /// The keys of [`Self::generate_keys`] common to both rings, and the tensor key of the standard ring.
    #[expect(
        clippy::too_many_arguments,
        reason = "key generation takes the secret, the layout, three sources and a scratch arena"
    )]
    fn generate_linear_keys<BE>(
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
        BE: Backend,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>,
    {
        let sparse_secret_hamming_weight = self.sparse_secret_hamming_weight;
        anyhow::ensure!(
            sparse_secret_hamming_weight.is_some() == layout.encapsulation.is_some(),
            "bootstrapping key layout encapsulation does not match the compiled recipe (expected {}, got {})",
            sparse_secret_hamming_weight.is_some(),
            layout.encapsulation.is_some()
        );

        let (order, first_factor_rotations) = (self.cyclotomic_order, &self.first_factor_rotations);
        anyhow::ensure!(
            first_factor_rotations.is_empty() == layout.first_factor_key.is_none(),
            "bootstrapping key layout has {} first-factor key layout, the compiled recipe has {} first-factor rotations",
            if layout.first_factor_key.is_some() { "a" } else { "no" },
            first_factor_rotations.len()
        );
        if let Some(encaps) = &layout.encapsulation {
            anyhow::ensure!(
                encaps.sparse_to_dense.is_some() == first_factor_rotations.is_empty(),
                "bootstrapping key layout {} a sparse-to-dense key, which the recipe uses only without the first factor on the raise",
                if encaps.sparse_to_dense.is_some() { "has" } else { "lacks" }
            );
        }

        anyhow::ensure!(
            layout.rotation_keys.keys().eq(&self.galois_elements),
            "bootstrapping key layout has rotation keys of Galois elements {:?}, the compiled recipe uses {:?}",
            layout.rotation_keys.keys().collect::<Vec<_>>(),
            self.galois_elements
        );

        let mut rotation_keys = HashMap::with_capacity(self.galois_elements.len());
        for (&p, atk_layout) in &layout.rotation_keys {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(atk_layout);
            module.glwe_automorphism_key_encrypt_sk(&mut atk, p, sk_dense, source_xe, source_xa, scratch);
            rotation_keys.insert(p, atk);
        }

        // Tensor (relinearization) key for EvalMod's ct×ct squaring.
        let tensor_key = (self.eval_mod_ring == EvalModRing::Standard).then(|| {
            let mut key = module.glwe_tensor_key_alloc_from_infos(&layout.tensor_key);
            module.glwe_tensor_key_encrypt_sk(&mut key, sk_dense, source_xe, source_xa, scratch);
            key
        });

        // First-factor keys: each diagonal's term is rotated with no key, which leaves
        // it under the rotated raise secret, then switched to the dense secret.
        let first_factor = |sk_raise: &GLWESecretBackendRef<'_, BE>,
                            source_xe: &mut Source,
                            source_xa: &mut Source,
                            scratch: &mut ScratchArena<'_, BE>| {
            let mut keys = HashMap::with_capacity(first_factor_rotations.len());
            if let Some(key_layout) = &layout.first_factor_key {
                let rank = sk_raise.rank();
                let (mut sk_rotated, mut scratch) = scratch.borrow().take_glwe_secret_scratch(sk_raise.n(), rank);
                *sk_rotated.dist_mut() = *sk_raise.dist();
                for &rotation in first_factor_rotations {
                    {
                        let sk = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE, Standard>(sk_raise.data());
                        let mut res = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE, Standard>(sk_rotated.data_mut());
                        for col in 0..rank.as_usize() {
                            module.vec_znx_automorphism(galois_element(rotation, order), &mut res, col, &sk, col);
                        }
                    }
                    let mut key = module.glwe_switching_key_alloc_from_infos(key_layout);
                    module.glwe_switching_key_encrypt_sk(&mut key, &sk_rotated, sk_dense, source_xe, source_xa, &mut scratch);
                    keys.insert(rotation, key);
                }
            }
            keys
        };
        let first_factor_keys;

        // Sparse-secret encapsulation key-switching keys.
        let encapsulation_keys = match (sparse_secret_hamming_weight, &layout.encapsulation) {
            (Some(hamming_weight), Some(encaps)) => {
                let keys = {
                    let (mut sk_sparse, mut scratch) = scratch.borrow().take_glwe_secret_scratch(sk_dense.n(), sk_dense.rank());
                    module.glwe_secret_fill_ternary_hw(&mut sk_sparse, hamming_weight, source_xs);
                    *sk_sparse.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");

                    let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&encaps.dense_to_sparse);
                    module.glwe_switching_key_encrypt_sk(
                        &mut dense_to_sparse,
                        sk_dense,
                        &sk_sparse,
                        source_xe,
                        source_xa,
                        &mut scratch,
                    );
                    let sparse_to_dense = encaps.sparse_to_dense.map(|s2d_layout| {
                        let mut key = module.glwe_switching_key_alloc_from_infos(&s2d_layout);
                        module.glwe_switching_key_encrypt_sk(&mut key, &sk_sparse, sk_dense, source_xe, source_xa, &mut scratch);
                        key
                    });
                    // The raise runs under the sparse secret: so do the first factor's products.
                    first_factor_keys = first_factor(
                        &GLWESecretToBackendRef::<BE>::to_backend_ref(&sk_sparse),
                        source_xe,
                        source_xa,
                        &mut scratch,
                    );
                    (dense_to_sparse, sparse_to_dense)
                };
                Some(keys)
            }
            (None, None) => {
                first_factor_keys = first_factor(
                    &GLWESecretToBackendRef::<BE>::to_backend_ref(sk_dense),
                    source_xe,
                    source_xa,
                    scratch,
                );
                None
            }
            _ => unreachable!("recipe/layout encapsulation mismatch validated above"),
        };

        Ok(BootstrappingKeySet {
            rotation_keys,
            tensor_key,
            ci_tensor_key: None,
            encapsulation_keys,
            first_factor_keys,
            conversion_keys: None,
        })
    }
}

impl<D: Data, W: ZnxWord> BootstrappingKeySet<D, W> {
    /// [`Self::prepare`] for keys held by another backend: each key is moved
    /// into `module`'s storage, prepared there, and its moved copy dropped
    /// before the next one, so that one unprepared key at most is resident.
    /// `scratch` holds [`BootstrappingKeysLayout::prepare_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// None: it always returns `Ok`.
    pub fn transfer_and_prepare<BE>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingKeysPrepared<BE::OwnedBuf, BE>>
    where
        D: CopyToHost,
        BE: Backend<ZnxWord = W>,
        Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWECIKeyPreparedFactory<BE>,
        Module<BE>: GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    {
        let mut rotation_keys = HashMap::with_capacity(self.rotation_keys.len());
        for (&p, atk) in &self.rotation_keys {
            let (matrix, mut rest) = scratch.borrow().take_gglwe_scratch(atk);
            let moved = MovedKey::of(atk, matrix, atk.p(), (atk.n(), atk.n()));
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(atk);
            module.glwe_automorphism_key_prepare(&mut prepared, &moved, &mut rest);
            rotation_keys.insert(p, prepared);
        }

        let tensor_key = self.tensor_key.as_ref().map(|key| {
            let (mut moved, mut scratch) = scratch.borrow().take_gglwe_scratch(key);
            key.transfer_into(&mut moved);
            let mut prepared = module.glwe_tensor_key_prepared_alloc_from_infos(key);
            module.glwe_tensor_key_prepare(&mut prepared, &moved, &mut scratch);
            prepared
        });
        let ci_tensor_key = self.ci_tensor_key.as_ref().map(|key| {
            let (mut moved, mut scratch) = scratch.borrow().take_gglwe_scratch::<_, ConjugateInvariant>(key);
            key.transfer_into(&mut moved);
            let mut prepared =
                <Module<BE> as GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>>::glwe_tensor_key_prepared_alloc_from_infos(
                    module, key,
                );
            module.glwe_tensor_key_prepare(&mut prepared, &moved, &mut scratch);
            prepared
        });

        let mut switching = |key: &GLWESwitchingKey<D, W>| {
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
            prepare_moved_switching_key(module, &mut prepared, key, scratch);
            prepared
        };
        let encapsulation_keys = self
            .encapsulation_keys
            .as_ref()
            .map(|(d2s, s2d)| (switching(d2s), s2d.as_ref().map(&mut switching)));
        let first_factor_keys = self
            .first_factor_keys
            .iter()
            .map(|(&rotation, key)| (rotation, switching(key)))
            .collect();

        let conversion_keys = self.conversion_keys.as_ref().map(|(trace, embed)| {
            let mut trace_key = module.glwe_ci_trace_key_prepared_alloc_from_infos(trace);
            prepare_moved_ci_key(module, &mut trace_key, trace, scratch);
            let mut embed_key = module.glwe_ci_embed_key_prepared_alloc_from_infos(embed);
            prepare_moved_ci_key(module, &mut embed_key, embed, scratch);
            (trace_key, embed_key)
        });

        let mut keys = BootstrappingKeysPrepared {
            rotation_keys,
            tensor_key,
            ci_tensor_key,
            encapsulation_keys,
            first_factor_keys,
            conversion_keys,
            keys_layout: None,
        };
        keys.keys_layout = keys.read_keys_layout();
        Ok(keys)
    }
}

/// A key's matrix moved into the scratch, with what its preparation reads besides: its Galois
/// element and its degrees.
struct MovedKey<'a, BE: Backend> {
    matrix: GGLWEViewMut<'a, BE>,
    p: i64,
    degrees: (Degree, Degree),
}

impl<'a, BE: Backend> MovedKey<'a, BE> {
    /// `key`'s matrix moved into `matrix`, carved for its layout.
    fn of<K: TransferInto<GGLWEViewMut<'a, BE>>>(
        key: &K,
        mut matrix: GGLWEViewMut<'a, BE>,
        p: i64,
        degrees: (Degree, Degree),
    ) -> Self {
        key.transfer_into(&mut matrix);
        Self { matrix, p, degrees }
    }
}

impl<BE: Backend> GGLWEToBackendRef<BE> for MovedKey<'_, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.matrix.to_backend_ref()
    }
}

impl<BE: Backend> GetGaloisElement for MovedKey<'_, BE> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<BE: Backend> GLWESwitchingKeyDegrees for MovedKey<'_, BE> {
    fn input_degree(&self) -> &Degree {
        &self.degrees.0
    }

    fn output_degree(&self) -> &Degree {
        &self.degrees.1
    }
}

/// Prepares the switching key `key`, held by another backend, into `prepared` through a copy
/// carved from `scratch`.
fn prepare_moved_switching_key<BE, K, R>(module: &Module<BE>, prepared: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    K: GGLWEInfos + GLWESwitchingKeyDegrees + for<'a> TransferInto<GGLWEViewMut<'a, BE>>,
    R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
    Module<BE>: GLWESwitchingKeyPreparedFactory<BE>,
{
    let (matrix, mut rest) = scratch.borrow().take_gglwe_scratch(key);
    let moved = MovedKey::of(key, matrix, 0, (*key.input_degree(), *key.output_degree()));
    module.glwe_switching_key_prepare(prepared, &moved, &mut rest);
}

/// [`GLWECIKeyPreparedFactory::glwe_ci_key_prepare`] of `key`, moved into a copy carved from `scratch`.
fn prepare_moved_ci_key<BE, K, M>(
    module: &Module<BE>,
    prepared: &mut GLWECIKeyPrepared<BE::OwnedBuf, BE, M>,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    K: GGLWEInfos + GLWESwitchingKeyDegrees + for<'a> TransferInto<GGLWEViewMut<'a, BE>>,
    Module<BE>: GLWECIKeyPreparedFactory<BE>,
{
    let (matrix, mut rest) = scratch.borrow().take_gglwe_scratch(key);
    let moved = MovedKey::of(key, matrix, 0, (*key.input_degree(), *key.output_degree()));
    module.glwe_ci_key_prepare(prepared, &moved, &mut rest);
}

/// Switching keys between an input secret of degree `n`, lifted to the bootstrap
/// degree `N` through `X -> X^(N/n)`, and the bootstrap secret. A conjugate-invariant
/// input secret enters embedded, as `a_0 + Σ a_k(X^k - X^(2n-k))` of degree `2n`.
#[derive(Clone, Copy, Debug)]
pub struct RingSwitchKeys<S> {
    /// From the input secret to the bootstrap secret.
    pub inbound: S,
    /// From the bootstrap secret to the input secret.
    pub outbound: S,
}

impl<S: GGLWEInfos> RingSwitchKeys<S> {
    /// Layouts of both keys.
    pub fn gglwe_layout(&self) -> RingSwitchKeys<GGLWELayout> {
        RingSwitchKeys {
            inbound: self.inbound.gglwe_layout(),
            outbound: self.outbound.gglwe_layout(),
        }
    }
}

/// Unprepared ring-switch keys.
pub type RingSwitchKeySet<D, W> = RingSwitchKeys<GLWESwitchingKey<D, W>>;

/// Prepared ring-switch keys.
pub type RingSwitchKeysPrepared<D, BE> = RingSwitchKeys<GLWESwitchingKeyPrepared<D, BE>>;

/// Ring-switch key layouts, at the bootstrap degree.
pub type RingSwitchKeysLayout = RingSwitchKeys<GLWESwitchingKeyLayout>;

impl<D: Data, W: ZnxWord> RingSwitchKeySet<D, W> {
    /// Prepares both keys under the bootstrap module.
    pub fn prepare<BE: Backend>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<RingSwitchKeysPrepared<BE::OwnedBuf, BE>>
    where
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: GLWESwitchingKeyPreparedFactory<BE>,
    {
        anyhow::ensure!(
            self.inbound.n().as_usize() == module.n() && self.outbound.n().as_usize() == module.n(),
            "invalid ring-switch key degree"
        );
        let mut prepare = |key: &GLWESwitchingKey<D, W>| {
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
            module.glwe_switching_key_prepare(&mut prepared, key, scratch);
            prepared
        };
        Ok(RingSwitchKeys {
            inbound: prepare(&self.inbound),
            outbound: prepare(&self.outbound),
        })
    }
}

impl RingSwitchKeysLayout {
    /// Generates both keys between `sk_in`, whose degree divides that of `module`,
    /// and the bootstrap secret `sk`.
    pub fn generate<BE: Backend>(
        &self,
        module: &Module<BE>,
        sk_in: &BackendGLWESecret<BE>,
        sk: &BackendGLWESecret<BE>,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<RingSwitchKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWESwitchingKeyEncryptSk<BE>,
    {
        anyhow::ensure!(sk.n().as_usize() == module.n(), "invalid bootstrap secret degree");
        anyhow::ensure!(
            module.n().is_multiple_of(sk_in.n().as_usize()),
            "the input secret degree must divide the bootstrap degree"
        );
        anyhow::ensure!(
            sk_in.rank().as_usize() == 1 && sk.rank().as_usize() == 1,
            "ring switching requires rank-1 secrets"
        );
        for key in [&self.inbound, &self.outbound] {
            anyhow::ensure!(
                key.n.as_usize() == module.n() && key.rank_in.as_usize() == 1 && key.rank_out.as_usize() == 1,
                "invalid ring-switch key layout"
            );
        }
        let mut encrypt = |layout: GLWESwitchingKeyLayout, sk_in: &BackendGLWESecret<BE>, sk_out: &BackendGLWESecret<BE>| {
            let mut key = module.glwe_switching_key_alloc_from_infos(&layout);
            module.glwe_switching_key_encrypt_sk(&mut key, sk_in, sk_out, source_xe, source_xa, scratch);
            key
        };
        Ok(RingSwitchKeys {
            inbound: encrypt(self.inbound, sk_in, sk),
            outbound: encrypt(self.outbound, sk, sk_in),
        })
    }
}

/// Layouts of the keys a fold takes, which size it before or without them.
#[derive(Clone, Copy, Debug)]
pub struct CKKSFoldKeysLayout {
    /// Ring-switch key layouts, or `None` when the inputs are under the bootstrap
    /// secret at its degree.
    pub ring_switch: Option<RingSwitchKeys<GGLWELayout>>,
    /// Layout of the automorphism keys of the input secret, which split real pairs
    /// and sparse inputs.
    pub automorphism: Option<GGLWELayout>,
}
