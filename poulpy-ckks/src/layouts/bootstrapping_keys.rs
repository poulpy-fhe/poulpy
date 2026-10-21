//! Evaluation keys for CKKS bootstrapping.
//!
//! The bootstrapping pipelines consume four kinds of key material:
//!
//! - **rotation (automorphism) keys** for the two homomorphic DFTs — the union
//!   of the Galois elements of `CoeffsToSlots` (incl. the optional high-precision
//!   bypass) and `SlotsToCoeffs`;
//! - a **conjugation key** (Galois element `−1`) for the split `CoeffsToSlots`;
//! - a **tensor (relinearization) key** for EvalMod's `ct×ct` squaring;
//! - optionally, one **first-factor key** per diagonal of the first
//!   `CoeffsToSlots` factor, when the plan leaves that factor to the modulus
//!   raise ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise));
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
//! [`BootstrappingKeys`], the pipeline-facing access trait, and its supertrait
//! [`BootstrappingLinearKeys`], which the linear stages take; a custom key manager
//! (lazy / streaming / on-the-fly-prepared) can implement both directly instead of
//! materializing the whole bundle.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, ScratchArenaTakeCore,
    layouts::{
        BackendGLWESecret, GGLWEInfos, GGLWELayout, GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWEAutomorphismKey,
        GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecretLayout,
        GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
        GetAutomorphismKey, GetGaloisElement, GetTensorKey, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedBackendRef,
            GLWETensorKeyPreparedToBackendRef,
        },
    },
};
use poulpy_hal::{
    api::VecZnxAutomorphism,
    layouts::{
        Backend, CyclotomicOrder, Data, HostDataMut, HostDataRef, Module, ScalarZnxAsVecZnxBackendRef, ScratchArena, Standard,
        ZnxWord, galois_element, scalar_znx_as_vec_znx_backend_mut_from_mut,
    },
    source::Source,
};

use crate::layouts::BootstrappingContext;
use poulpy_core::layouts::GLWESecretSampling;
use poulpy_core::{Distribution, GetDistribution, GetDistributionMut};

/// Pipeline-facing access to the **prepared** keys of the linear stages of a CKKS
/// bootstrap: the two homomorphic DFTs and ModUp.
///
/// The key types are associated (not fixed to a backend buffer), so an
/// implementation is free to back them with any data representation and to
/// prepare them lazily / on the fly. [`BootstrappingLinearKeysPrepared`] and
/// [`BootstrappingKeysPrepared`] are the eager in-memory implementations.
pub trait BootstrappingLinearKeys<BE: Backend> {
    /// The rotation-key collection passed to the homomorphic DFT stages.
    type RotationKeys: GetAutomorphismKey<BE>;

    /// The prepared key-switching key type for sparse-secret encapsulation.
    type SwitchingKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Rotation (automorphism) keys, Galois element `−1` among them when the
    /// pipeline splits CoeffsToSlots.
    fn rotation_keys(&self) -> &Self::RotationKeys;

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
}

/// Pipeline-facing access to the **prepared** evaluation keys a CKKS bootstrap
/// consumes: those of the linear stages and the tensor keys of EvalMod.
///
/// The bootstrap stages take their keys through this trait, so any key store that
/// can answer its queries can drive a bootstrap. [`BootstrappingKeysPrepared`] is
/// the eager in-memory implementation.
pub trait BootstrappingKeys<BE: Backend>: BootstrappingLinearKeys<BE> {
    /// The prepared tensor (relinearization) key type for EvalMod.
    type TensorKey: GetTensorKey<BE>;

    /// Relinearization (tensor) key for EvalMod's `ct×ct` squaring. It covers
    /// the widest ciphertext EvalMod relinearizes.
    fn tensor_key(&self) -> &Self::TensorKey;
}

/// A key source that refuses a key not covering the precision it is asked for.
///
/// Past its [`max_input_k`](poulpy_core::layouts::GGLWELayout::max_input_k), a
/// key switch drops the input limbs beyond the key's rows. The bootstrap reads
/// every key through this source, so a key laid out narrower than the width it
/// is used at fails instead of losing precision.
pub(crate) struct CoveringKeys<'a, K>(pub(crate) &'a K);

/// Refuses a key whose layout does not cover inputs of `k` bits.
pub(crate) fn check_covers(op: &'static str, layout: GGLWELayout, k: TorusPrecision) -> poulpy_core::Result<()> {
    let covered = layout.max_input_k();
    if k > covered {
        return Err(poulpy_core::CoreError::GGLWEKeyUse {
            op,
            detail: format!("the key covers inputs of {covered} bits, asked for {k}"),
        });
    }
    Ok(())
}

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for CoveringKeys<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.0.lookup_automorphism_key(p, k)?;
        check_covers("get_automorphism_key", key.gglwe_layout(), k)?;
        Ok(key)
    }
}

impl<BE: Backend, K: GetTensorKey<BE>> GetTensorKey<BE> for CoveringKeys<'_, K> {
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        let key = self.0.get_tensor_key(k)?;
        check_covers("get_tensor_key", key.gglwe_layout(), k)?;
        Ok(key)
    }
}

/// Sparse-secret encapsulation keys `(denseToSparse, sparseToDense)`, without
/// `sparseToDense` when the first `CoeffsToSlots` factor is applied on the raise.
pub type EncapsulationKeys<K> = (K, Option<K>);

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
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
    /// Relinearization (tensor) key for EvalMod, covering its widest ciphertext.
    pub tensor_key: GLWETensorKey<D, W>,
    /// `(denseToSparse, sparseToDense)` encapsulation keys, or `None`; no
    /// `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKey<D, W>>>,
    /// First-factor keys by slot rotation, possibly none: switching keys from
    /// the rotated raise secret (the sparse one under encapsulation) to the
    /// dense one, see [`BootstrappingKeysLayout::first_factor_key`].
    pub first_factor_keys: HashMap<i64, GLWESwitchingKey<D, W>>,
}

/// The **prepared** (preprocessed) bootstrapping keys, ready for the pipeline.
///
/// Generic over the key data buffer `D`; implements [`BootstrappingKeys`]. Built
/// eagerly by [`BootstrappingKeySet::prepare`].
pub struct BootstrappingKeysPrepared<D: Data, BE: Backend> {
    /// Prepared rotation keys indexed by Galois element, conjugation among them.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    /// Prepared relinearization (tensor) key for EvalMod, covering its widest ciphertext.
    pub tensor_key: GLWETensorKeyPrepared<D, BE>,
    /// Prepared `(denseToSparse, sparseToDense)` encapsulation keys, or `None`;
    /// no `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<D, BE>>>,
    /// Prepared first-factor keys by slot rotation, possibly none.
    pub first_factor_keys: HashMap<i64, GLWESwitchingKeyPrepared<D, BE>>,
}

/// The linear-stage keys of [`BootstrappingKeysPrepared`] and
/// [`BootstrappingLinearKeysPrepared`], which hold them in fields of the same names.
macro_rules! impl_bootstrapping_linear_keys {
    ($keys:ident) => {
        impl<D: Data, BE: Backend> BootstrappingLinearKeys<BE> for $keys<D, BE>
        where
            GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
            GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        {
            type RotationKeys = HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>;
            type SwitchingKey = GLWESwitchingKeyPrepared<D, BE>;

            fn rotation_keys(&self) -> &Self::RotationKeys {
                &self.rotation_keys
            }

            fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, Option<&Self::SwitchingKey>)> {
                self.encapsulation_keys.as_ref().map(|(d2s, s2d)| (d2s, s2d.as_ref()))
            }

            fn first_factor_key(&self, rotation: i64) -> Option<&Self::SwitchingKey> {
                self.first_factor_keys.get(&rotation)
            }
        }
    };
}

impl_bootstrapping_linear_keys!(BootstrappingKeysPrepared);
impl_bootstrapping_linear_keys!(BootstrappingLinearKeysPrepared);

impl<D: Data, BE: Backend> BootstrappingKeys<BE> for BootstrappingKeysPrepared<D, BE>
where
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type TensorKey = GLWETensorKeyPrepared<D, BE>;

    fn tensor_key(&self) -> &Self::TensorKey {
        &self.tensor_key
    }
}

/// The **unprepared** keys of the linear stages of a bootstrap
/// ([`BootstrappingLinearKeys`]): a [`BootstrappingKeySet`] without tensor keys,
/// for a pipeline that relinearizes EvalMod with keys of its own.
pub struct BootstrappingLinearKeySet<D: Data, W: ZnxWord> {
    /// Rotation keys indexed by Galois element.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
    /// `(denseToSparse, sparseToDense)` encapsulation keys, or `None`; no
    /// `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKey<D, W>>>,
    /// First-factor keys by slot rotation, possibly none.
    pub first_factor_keys: HashMap<i64, GLWESwitchingKey<D, W>>,
}

/// The **prepared** keys of the linear stages of a bootstrap; implements
/// [`BootstrappingLinearKeys`].
pub struct BootstrappingLinearKeysPrepared<D: Data, BE: Backend> {
    /// Prepared rotation keys indexed by Galois element.
    pub rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    /// Prepared `(denseToSparse, sparseToDense)` encapsulation keys, or `None`;
    /// no `sparseToDense` key with the first factor on the raise.
    pub encapsulation_keys: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<D, BE>>>,
    /// Prepared first-factor keys by slot rotation, possibly none.
    pub first_factor_keys: HashMap<i64, GLWESwitchingKeyPrepared<D, BE>>,
}

type PreparedRotationKeys<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;
type PreparedEncapsulationKeys<BE> = Option<EncapsulationKeys<GLWESwitchingKeyPrepared<<BE as Backend>::OwnedBuf, BE>>>;
type PreparedFirstFactorKeys<BE> = HashMap<i64, GLWESwitchingKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;

/// Prepares the keys of the linear stages, as [`BootstrappingKeySet`] and
/// [`BootstrappingLinearKeySet`] hold them.
fn prepare_linear_keys<D, W, BE>(
    module: &Module<BE>,
    rotation_keys: &HashMap<i64, GLWEAutomorphismKey<D, W>>,
    encapsulation_keys: Option<&EncapsulationKeys<GLWESwitchingKey<D, W>>>,
    first_factor_keys: &HashMap<i64, GLWESwitchingKey<D, W>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> (
    PreparedRotationKeys<BE>,
    PreparedEncapsulationKeys<BE>,
    PreparedFirstFactorKeys<BE>,
)
where
    D: HostDataRef,
    W: ZnxWord,
    BE: Backend,
    GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
{
    let mut rotations = HashMap::with_capacity(rotation_keys.len());
    for (&p, atk) in rotation_keys {
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(atk);
        module.glwe_automorphism_key_prepare(&mut prepared, atk, scratch);
        rotations.insert(p, prepared);
    }

    let encapsulation = encapsulation_keys.map(|(d2s, s2d)| {
        let mut d2s_prepared = module.glwe_switching_key_prepared_alloc_from_infos(d2s);
        module.glwe_switching_key_prepare(&mut d2s_prepared, d2s, scratch);
        let s2d_prepared = s2d.as_ref().map(|s2d| {
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(s2d);
            module.glwe_switching_key_prepare(&mut prepared, s2d, scratch);
            prepared
        });
        (d2s_prepared, s2d_prepared)
    });

    let mut first_factor = HashMap::with_capacity(first_factor_keys.len());
    for (&rotation, key) in first_factor_keys {
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
        module.glwe_switching_key_prepare(&mut prepared, key, scratch);
        first_factor.insert(rotation, prepared);
    }
    (rotations, encapsulation, first_factor)
}

impl<D: Data, W: ZnxWord> BootstrappingLinearKeySet<D, W> {
    /// Preprocesses every key into a [`BootstrappingLinearKeysPrepared`] bundle
    /// (see [`BootstrappingKeySet::prepare`]).
    pub fn prepare<BE: Backend>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> BootstrappingLinearKeysPrepared<BE::OwnedBuf, BE>
    where
        D: HostDataRef,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>,
    {
        let (rotation_keys, encapsulation_keys, first_factor_keys) = prepare_linear_keys(
            module,
            &self.rotation_keys,
            self.encapsulation_keys.as_ref(),
            &self.first_factor_keys,
            scratch,
        );
        BootstrappingLinearKeysPrepared {
            rotation_keys,
            encapsulation_keys,
            first_factor_keys,
        }
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
        let mut tensor_key = module.alloc_tensor_key_prepared_from_infos(&self.tensor_key);
        module.prepare_tensor_key(&mut tensor_key, &self.tensor_key, scratch);

        let (rotation_keys, encapsulation_keys, first_factor_keys) = prepare_linear_keys(
            module,
            &self.rotation_keys,
            self.encapsulation_keys.as_ref(),
            &self.first_factor_keys,
            scratch,
        );

        BootstrappingKeysPrepared {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
            first_factor_keys,
        }
    }
}

/// Layout parameters for the evaluation keys produced by
/// [`BootstrappingContext::generate_keys`].
///
/// Key generation uses [`poulpy_core::Noise::ENCRYPTION`] at each
/// destination key's precision. The Galois elements of the rotation keys are read from the compiled DFT matrices,
/// so this layout only gives their shapes.
///
/// A rotation key used on narrow ciphertexts only can take a wider digit under
/// the same modulus bound, hence fewer rows: `rotation_keys` holds such shapes.
/// Left empty, every rotation key has the shape of the widest use.
#[derive(Clone, Debug)]
pub struct BootstrappingKeysLayout {
    /// Layout of the rotation and conjugation keys `rotation_keys` does not
    /// list. It must cover the widest ciphertext of the pipeline.
    pub automorphism_key: GLWEAutomorphismKeyLayout,
    /// Layouts of single rotation keys, by Galois element. Each must cover the
    /// widest ciphertext its key switches.
    pub rotation_keys: BTreeMap<i64, GLWEAutomorphismKeyLayout>,
    /// Layout of the EvalMod relinearization (tensor) key. It must cover the
    /// widest ciphertext EvalMod relinearizes.
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
}

/// The distinct layouts of `rotation_keys` and of the shared `automorphism_key`, the
/// shared one first.
fn distinct_rotation_key_layouts(
    automorphism_key: GLWEAutomorphismKeyLayout,
    rotation_keys: &BTreeMap<i64, GLWEAutomorphismKeyLayout>,
) -> Vec<GLWEAutomorphismKeyLayout> {
    let mut layouts = vec![automorphism_key];
    for layout in rotation_keys.values() {
        if !layouts.contains(layout) {
            layouts.push(*layout);
        }
    }
    layouts
}

impl BootstrappingKeysLayout {
    /// Layout of the rotation key of Galois element `p`.
    pub fn rotation_key(&self, p: i64) -> GLWEAutomorphismKeyLayout {
        self.rotation_keys.get(&p).copied().unwrap_or(self.automorphism_key)
    }

    /// The distinct rotation key layouts, the shared one first.
    pub fn rotation_key_layouts(&self) -> Vec<GLWEAutomorphismKeyLayout> {
        distinct_rotation_key_layouts(self.automorphism_key, &self.rotation_keys)
    }

    /// The layouts of the keys of the linear stages.
    pub fn linear(&self) -> BootstrappingLinearKeysLayout {
        BootstrappingLinearKeysLayout {
            automorphism_key: self.automorphism_key,
            rotation_keys: self.rotation_keys.clone(),
            encapsulation: self.encapsulation,
            first_factor_key: self.first_factor_key,
        }
    }
}

/// Layout parameters for the keys of the linear stages
/// ([`BootstrappingLinearKeys`]): those of a [`BootstrappingKeysLayout`] without
/// its tensor keys.
#[derive(Clone, Debug)]
pub struct BootstrappingLinearKeysLayout {
    /// Layout of the rotation keys `rotation_keys` does not list.
    pub automorphism_key: GLWEAutomorphismKeyLayout,
    /// Layouts of single rotation keys, by Galois element.
    pub rotation_keys: BTreeMap<i64, GLWEAutomorphismKeyLayout>,
    /// Layouts of the sparse-secret encapsulation keys, or `None`.
    pub encapsulation: Option<EncapsulationKeysLayout>,
    /// Layout of the first-factor keys, or `None`.
    pub first_factor_key: Option<GLWESwitchingKeyLayout>,
}

impl BootstrappingLinearKeysLayout {
    /// Layout of the rotation key of Galois element `p`.
    pub fn rotation_key(&self, p: i64) -> GLWEAutomorphismKeyLayout {
        self.rotation_keys.get(&p).copied().unwrap_or(self.automorphism_key)
    }

    /// The distinct rotation key layouts, the shared one first.
    pub fn rotation_key_layouts(&self) -> Vec<GLWEAutomorphismKeyLayout> {
        distinct_rotation_key_layouts(self.automorphism_key, &self.rotation_keys)
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
    /// `scratch` must be large enough for the key encrypt operations and, with
    /// first-factor keys, a secret beside them: each rotated raise secret is
    /// carved from it, and the region it took is wiped before returning.
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
        BE: Backend<Ring = Standard>,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>,
    {
        let linear = layout.linear();
        let rotation_keys = self.generate_rotation_keys(module, sk_dense, &linear, true, source_xe, source_xa, scratch)?;

        // Tensor (relinearization) key for EvalMod's ct×ct squaring.
        let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&layout.tensor_key);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_key, sk_dense, source_xe, source_xa, scratch);

        let (encapsulation_keys, first_factor_keys) =
            self.generate_raise_keys(module, sk_dense, &linear, source_xs, source_xe, source_xa, scratch);

        Ok(BootstrappingKeySet {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
            first_factor_keys,
        })
    }

    /// Generates the **unprepared** keys of the linear stages for `sk_dense`, as
    /// [`Self::generate_keys`] does but without tensor keys, and with the
    /// conjugation key only when `conjugation` is set: a pipeline that does not
    /// split CoeffsToSlots does not use it.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_linear_keys(
        &self,
        module: &Module<BE>,
        sk_dense: &BackendGLWESecret<BE>,
        layout: &BootstrappingLinearKeysLayout,
        conjugation: bool,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingLinearKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE: Backend<Ring = Standard>,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>,
    {
        let rotation_keys = self.generate_rotation_keys(module, sk_dense, layout, conjugation, source_xe, source_xa, scratch)?;
        let (encapsulation_keys, first_factor_keys) =
            self.generate_raise_keys(module, sk_dense, layout, source_xs, source_xe, source_xa, scratch);
        Ok(BootstrappingLinearKeySet {
            rotation_keys,
            encapsulation_keys,
            first_factor_keys,
        })
    }

    /// Checks `layout` against the compiled recipe and generates the rotation keys:
    /// the union of both DFTs' (and the bypass') Galois elements, and conjugation
    /// when `conjugation` is set.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn generate_rotation_keys(
        &self,
        module: &Module<BE>,
        sk_dense: &BackendGLWESecret<BE>,
        layout: &BootstrappingLinearKeysLayout,
        conjugation: bool,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<HashMap<i64, GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord>>>
    where
        BE: Backend<Ring = Standard>,
        BE::OwnedBuf: HostDataMut,
        Module<BE>:
            ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CyclotomicOrder + GLWEAutomorphismKeyEncryptSk<BE>,
    {
        let sparse_secret_hamming_weight = self.sparse_secret_hamming_weight();
        anyhow::ensure!(
            sparse_secret_hamming_weight.is_some() == layout.encapsulation.is_some(),
            "bootstrapping key layout encapsulation does not match the compiled recipe (expected {}, got {})",
            sparse_secret_hamming_weight.is_some(),
            layout.encapsulation.is_some()
        );
        let first_factor_rotations = self.first_factor_rotations();
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

        let order = module.cyclotomic_order();
        let mut gal_set: BTreeSet<i64> = BTreeSet::new();
        gal_set.extend(self.coeffs_to_slots().galois_elements(order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            gal_set.extend(bypass.galois_elements(order));
        }
        gal_set.extend(self.slots_to_coeffs().galois_elements(order));
        // Conjugation, for the split forward transform.
        if conjugation {
            gal_set.insert(-1);
        }

        let mut rotation_keys = HashMap::with_capacity(gal_set.len());
        for p in gal_set {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(&layout.rotation_key(p));
            module.glwe_automorphism_key_encrypt_sk(&mut atk, p, sk_dense, source_xe, source_xa, scratch);
            rotation_keys.insert(p, atk);
        }
        Ok(rotation_keys)
    }

    fn first_factor_rotations(&self) -> Vec<i64> {
        self.coeffs_to_slots_first_factor()
            .map(|factor| factor.giant_steps.iter().map(|giant| giant.rot).collect())
            .unwrap_or_default()
    }

    /// The keys of the raise, on a layout [`Self::generate_rotation_keys`] checked:
    /// the encapsulation keys, under a fresh sparse secret, and the first-factor keys.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn generate_raise_keys(
        &self,
        module: &Module<BE>,
        sk_dense: &BackendGLWESecret<BE>,
        layout: &BootstrappingLinearKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> (
        Option<EncapsulationKeys<GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord>>>,
        HashMap<i64, GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord>>,
    )
    where
        BE: Backend<Ring = Standard>,
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>,
    {
        let order = module.cyclotomic_order();
        let first_factor_rotations = self.first_factor_rotations();

        // First-factor keys: each diagonal's term is rotated with no key, which leaves
        // it under the rotated raise secret, then switched to the dense secret.
        let first_factor = |sk_raise: &BackendGLWESecret<BE>,
                            source_xe: &mut Source,
                            source_xa: &mut Source,
                            scratch: &mut ScratchArena<'_, BE>| {
            let mut keys = HashMap::with_capacity(first_factor_rotations.len());
            if let Some(key_layout) = &layout.first_factor_key {
                let (n, rank) = (sk_raise.n(), sk_raise.rank());
                {
                    // The rotated secret is carved from scratch, which is wiped once the keys
                    // are encrypted.
                    let scratch = scratch.borrow();
                    let (mut sk_rotated, mut scratch) = scratch.take_glwe_secret_scratch(n, rank);
                    *sk_rotated.dist_mut() = *sk_raise.dist();
                    for &rotation in &first_factor_rotations {
                        {
                            let sk = ScalarZnxAsVecZnxBackendRef::<BE>::as_vec_znx_backend(sk_raise.data());
                            let mut res = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(sk_rotated.data_mut());
                            for col in 0..rank.as_usize() {
                                module.vec_znx_automorphism(galois_element(rotation, order), &mut res, col, &sk, col);
                            }
                        }
                        let mut key = module.glwe_switching_key_alloc_from_infos(key_layout);
                        module.glwe_switching_key_encrypt_sk(&mut key, &sk_rotated, sk_dense, source_xe, source_xa, &mut scratch);
                        keys.insert(rotation, key);
                    }
                }
                scratch.wipe(BE::bytes_of_scalar_znx(n.as_usize(), rank.as_usize()));
            }
            keys
        };

        // Sparse-secret encapsulation key-switching keys.
        match (self.sparse_secret_hamming_weight(), &layout.encapsulation) {
            (Some(hamming_weight), Some(encaps)) => {
                let sk_layout = GLWESecretLayout {
                    n: sk_dense.n(),
                    rank: sk_dense.rank(),
                };
                let mut sk_sparse = module.glwe_secret_alloc_from_infos(&sk_layout);
                module.glwe_secret_fill_ternary_hw(&mut sk_sparse, hamming_weight, source_xs);
                *sk_sparse.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");

                let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&encaps.dense_to_sparse);
                module.glwe_switching_key_encrypt_sk(&mut dense_to_sparse, sk_dense, &sk_sparse, source_xe, source_xa, scratch);
                let sparse_to_dense = encaps.sparse_to_dense.map(|s2d_layout| {
                    let mut key = module.glwe_switching_key_alloc_from_infos(&s2d_layout);
                    module.glwe_switching_key_encrypt_sk(&mut key, &sk_sparse, sk_dense, source_xe, source_xa, scratch);
                    key
                });
                // The raise runs under the sparse secret: so do the first factor's products.
                let first_factor_keys = first_factor(&sk_sparse, source_xe, source_xa, scratch);
                (Some((dense_to_sparse, sparse_to_dense)), first_factor_keys)
            }
            (None, None) => (None, first_factor(sk_dense, source_xe, source_xa, scratch)),
            _ => unreachable!("recipe/layout encapsulation mismatch validated with the rotation keys"),
        }
    }
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
    pub fn prepare<BE: Backend<Ring = Standard>>(
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
    pub fn generate<BE: Backend<Ring = Standard>>(
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
