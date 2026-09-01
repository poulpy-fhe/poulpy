//! PaCo evaluation-key storage and backend preparation.
//!
//! [`PaCoKeySet`] is the unprepared, serializable form. Its fallible
//! constructor validates the key material against a [`PaCoPlan`] before the
//! set can be used. [`PaCoKeySet::prepare`] eagerly preprocesses the gadget
//! keys and returns [`PaCoKeysPrepared`], which implements the pipeline-facing
//! [`PaCoKeys`] access trait. Custom key managers may implement [`PaCoKeys`]
//! directly to prepare or stream keys on demand.

use poulpy_hal::layouts::Normalized;
use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GGLWEToBackendRef, GLWEAutomorphismKey, GLWEAutomorphismKeyPrepared,
    GLWEAutomorphismKeyPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyPrepared,
    GLWESwitchingKeyPreparedFactory, GLWETensorKey, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
    GLWETensorKeyPreparedToBackendRef, GLWEToBackendRef, GetAutomorphismKey, GetGaloisElement, GetTensorKey, LWEInfos,
    ModuleCoreAlloc, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, HostDataRef, Module, ScratchArena, ZnxWord};

use super::plan::PaCoPlan;
pub(crate) use crate::layouts::validation::{
    validate_backend_storage_capacity, validate_gadget_backend_view, validate_gadget_key, validate_storage_capacity,
};
use crate::{CKKSCtBounds, layouts::CKKSCiphertext};

/// Plan dimensions that determine the structured PaCo secret and encrypted
/// bootstrapping-key meaning.
///
/// Evaluation schedules may vary while reusing the same keys, but a bundle
/// generated for a different `(N, h, C, q, Delta_bsk)` is not interchangeable
/// even when its low-level layouts happen to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaCoKeyParameters {
    log_n: usize,
    h: usize,
    c: usize,
    log_q: u32,
    log_delta_bsk: usize,
}

impl PaCoKeyParameters {
    /// Derives the semantic key fingerprint from a validated PaCo plan.
    ///
    /// Custom [`PaCoKeys`] stores should retain this value alongside their key
    /// material and return it from [`PaCoKeys::parameters`]. Evaluation
    /// schedules and plaintext budgets are intentionally absent because they
    /// may change while reusing keys with the same structured secret and
    /// bootstrapping-key scale.
    pub fn from_plan(plan: &PaCoPlan) -> Self {
        Self {
            log_n: plan.log_n(),
            h: plan.h(),
            c: plan.c(),
            log_q: plan.log_q(),
            log_delta_bsk: plan.log_delta_bsk(),
        }
    }

    /// Ring degree exponent (`N = 2^log_n`).
    pub fn log_n(self) -> usize {
        self.log_n
    }

    /// Structured-secret row count.
    pub fn h(self) -> usize {
        self.h
    }

    /// Structured-secret block width.
    pub fn c(self) -> usize {
        self.c
    }

    /// Exhausted input modulus exponent.
    pub fn log_q(self) -> u32 {
        self.log_q
    }

    /// Scale exponent of the four encrypted bootstrapping plaintexts.
    pub fn log_delta_bsk(self) -> usize {
        self.log_delta_bsk
    }
}

/// Pipeline-facing access to all key material consumed by PaCo.
///
/// Associated types allow eager stores such as [`PaCoKeysPrepared`] as well
/// as lazy, streamed, or on-the-fly-prepared key managers. Bootstrapping-key
/// ciphertexts are ordinary CKKS ciphertexts; the remaining associated types
/// are prepared gadget keys. A custom store's reported layouts must match the
/// backend-native views returned by its core traits; operation preflight checks
/// that correspondence. The store is also responsible for cryptographic
/// provenance: all four bootstrapping ciphertexts, rotation keys, tensor key,
/// and optional switch must have been generated from the mutually compatible
/// secrets documented in `docs/paco.md`, which structural metadata cannot
/// prove.
pub trait PaCoKeys<BE: Backend> {
    /// CKKS ciphertext type used for the four encrypted structured-secret
    /// packings `bsk_t = Enc(sigma_t)`.
    type BootstrappingKey: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds;

    /// Prepared automorphism-key type used by PaCo rotations and folds.
    type AutomorphismKey: GetAutomorphismKey<BE>;

    /// Collection that resolves automorphism keys by Galois element.
    type RotationKeys: GetAutomorphismKey<BE>;

    /// Prepared tensor (relinearization) key used by the product fold.
    type TensorKey: GetTensorKey<BE>;

    /// Prepared dense-to-PaCo switching key used by optional encapsulation.
    type SwitchingKey: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;

    /// Dimensions that give the key material its PaCo meaning.
    fn parameters(&self) -> PaCoKeyParameters;

    /// The four bootstrapping-key ciphertexts, in paper order `t = 0..4`.
    fn bootstrapping_keys(&self) -> &[Self::BootstrappingKey; 4];

    /// Automorphism keys for all Galois elements required by the plan.
    fn rotation_keys(&self) -> &Self::RotationKeys;

    /// Relinearization key for the ciphertext product fold.
    fn tensor_key(&self) -> &Self::TensorKey;

    /// Optional dense-to-PaCo encapsulation key.
    fn encapsulation_key(&self) -> Option<&Self::SwitchingKey>;
}

/// Validated, unprepared PaCo key material.
///
/// Construct with [`Self::new`]. The fields are private so the validated
/// degree, rank, radix, metadata, and required-key invariants cannot be
/// invalidated afterwards.
pub struct PaCoKeySet<D: Data, W: ZnxWord> {
    parameters: PaCoKeyParameters,
    bootstrapping_keys: [CKKSCiphertext<D, W>; 4],
    rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
    tensor_key: GLWETensorKey<D, W>,
    encapsulation_key: Option<GLWESwitchingKey<D, W>>,
}

/// Eagerly prepared PaCo key material ready for a backend pipeline.
///
/// This includes the backend-resident bootstrapping ciphertexts as well as
/// all preprocessed gadget keys, keeping key ownership in one bundle.
pub struct PaCoKeysPrepared<D: Data, BE: Backend> {
    parameters: PaCoKeyParameters,
    bootstrapping_keys: [CKKSCiphertext<D, BE::ZnxWord>; 4],
    rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    tensor_key: GLWETensorKeyPrepared<D, BE>,
    encapsulation_key: Option<GLWESwitchingKeyPrepared<D, BE>>,
}

/// Owned components returned by [`PaCoKeySet::into_parts`].
pub type PaCoKeySetParts<D, W> = (
    PaCoKeyParameters,
    [CKKSCiphertext<D, W>; 4],
    HashMap<i64, GLWEAutomorphismKey<D, W>>,
    GLWETensorKey<D, W>,
    Option<GLWESwitchingKey<D, W>>,
);

/// Owned components returned by [`PaCoKeysPrepared::into_parts`].
pub type PaCoKeysPreparedParts<D, BE> = (
    PaCoKeyParameters,
    [CKKSCiphertext<D, <BE as Backend>::ZnxWord>; 4],
    HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    GLWETensorKeyPrepared<D, BE>,
    Option<GLWESwitchingKeyPrepared<D, BE>>,
);

type PreparedGadgetKeys<D, BE> = (
    HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
    GLWETensorKeyPrepared<D, BE>,
    Option<GLWESwitchingKeyPrepared<D, BE>>,
);

trait ActiveCiphertextStorage {
    fn active_size(&self) -> usize;
}

impl<D: Data, W: ZnxWord> ActiveCiphertextStorage for CKKSCiphertext<D, W> {
    fn active_size(&self) -> usize {
        self.data().size()
    }
}

impl<D: Data, W: ZnxWord> PaCoKeySet<D, W> {
    /// Structured-secret parameters bound to this key material.
    pub fn parameters(&self) -> PaCoKeyParameters {
        self.parameters
    }

    /// Builds and validates an unprepared PaCo key set.
    ///
    /// Validation covers the plan and cyclotomic order, all four CKKS
    /// ciphertext layouts and metadata, sufficient bootstrap budget, every
    /// required Galois key, map-label/key-label agreement, gadget-key layout
    /// compatibility, and optional encapsulation-key degrees.
    pub fn new(
        plan: &PaCoPlan,
        bootstrapping_keys: [CKKSCiphertext<D, W>; 4],
        rotation_keys: HashMap<i64, GLWEAutomorphismKey<D, W>>,
        tensor_key: GLWETensorKey<D, W>,
        encapsulation_key: Option<GLWESwitchingKey<D, W>>,
    ) -> Result<Self>
    where
        GLWESwitchingKey<D, W>: GLWESwitchingKeyDegrees,
    {
        validate_material(
            plan,
            &bootstrapping_keys,
            &rotation_keys,
            &tensor_key,
            encapsulation_key.as_ref(),
            GLWEAutomorphismKey::p,
        )?;

        Ok(Self {
            parameters: PaCoKeyParameters::from_plan(plan),
            bootstrapping_keys,
            rotation_keys,
            tensor_key,
            encapsulation_key,
        })
    }

    /// Prepares every gadget key for `module` and clones the four
    /// bootstrapping ciphertexts into the resulting unified key bundle.
    ///
    /// Preparation is currently same-backend and host-accessible: `D` must be
    /// `BE::OwnedBuf` and implement [`HostDataRef`]. The material is
    /// revalidated against `plan` immediately before backend preprocessing,
    /// and insufficient scratch is reported as an error rather than reaching
    /// the lower-level preparation assertions.
    pub fn prepare<BE: Backend<OwnedBuf = D, ZnxWord = W>>(
        &self,
        plan: &PaCoPlan,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<PaCoKeysPrepared<D, BE>>
    where
        D: HostDataRef,
        CKKSCiphertext<D, W>: Clone,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>,
    {
        ensure!(
            self.parameters == PaCoKeyParameters::from_plan(plan),
            "PaCo key bundle parameters {:?} do not match preparation plan {:?}",
            self.parameters,
            PaCoKeyParameters::from_plan(plan),
        );
        ensure!(
            module.n() == plan.n(),
            "PaCo key-preparation module degree {} does not match plan degree {}",
            module.n(),
            plan.n(),
        );
        validate_material(
            plan,
            &self.bootstrapping_keys,
            &self.rotation_keys,
            &self.tensor_key,
            self.encapsulation_key.as_ref(),
            GLWEAutomorphismKey::p,
        )?;

        let (rotation_keys, tensor_key, encapsulation_key) = prepare_gadget_keys(
            module,
            &self.rotation_keys,
            &self.tensor_key,
            self.encapsulation_key.as_ref(),
            scratch,
        )?;

        Ok(PaCoKeysPrepared {
            parameters: self.parameters,
            bootstrapping_keys: self.bootstrapping_keys.clone(),
            rotation_keys,
            tensor_key,
            encapsulation_key,
        })
    }

    /// Consumes and prepares this key set without cloning the four large
    /// bootstrapping ciphertexts.
    ///
    /// This is the preferred eager-preparation path when the unprepared set is
    /// no longer needed. It applies the same full validation and scratch
    /// preflight as [`Self::prepare`].
    pub fn into_prepare<BE: Backend<OwnedBuf = D, ZnxWord = W>>(
        self,
        plan: &PaCoPlan,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<PaCoKeysPrepared<D, BE>>
    where
        D: HostDataRef,
        GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>,
    {
        let Self {
            parameters,
            bootstrapping_keys,
            rotation_keys,
            tensor_key,
            encapsulation_key,
        } = self;
        ensure!(
            parameters == PaCoKeyParameters::from_plan(plan),
            "PaCo key bundle parameters {parameters:?} do not match preparation plan {:?}",
            PaCoKeyParameters::from_plan(plan),
        );
        ensure!(
            module.n() == plan.n(),
            "PaCo key-preparation module degree {} does not match plan degree {}",
            module.n(),
            plan.n(),
        );
        validate_material(
            plan,
            &bootstrapping_keys,
            &rotation_keys,
            &tensor_key,
            encapsulation_key.as_ref(),
            GLWEAutomorphismKey::p,
        )?;
        let (rotation_keys, tensor_key, encapsulation_key) =
            prepare_gadget_keys(module, &rotation_keys, &tensor_key, encapsulation_key.as_ref(), scratch)?;
        Ok(PaCoKeysPrepared {
            parameters,
            bootstrapping_keys,
            rotation_keys,
            tensor_key,
            encapsulation_key,
        })
    }
}

fn prepare_gadget_keys<D, W, BE>(
    module: &Module<BE>,
    rotation_keys: &HashMap<i64, GLWEAutomorphismKey<D, W>>,
    tensor_key: &GLWETensorKey<D, W>,
    encapsulation_key: Option<&GLWESwitchingKey<D, W>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<PreparedGadgetKeys<D, BE>>
where
    D: Data,
    W: ZnxWord,
    BE: Backend<OwnedBuf = D, ZnxWord = W>,
    GLWEAutomorphismKey<D, W>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    GLWETensorKey<D, W>: GGLWEToBackendRef<BE> + GGLWEInfos,
    GLWESwitchingKey<D, W>: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees + GGLWEInfos,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
{
    let mut required_scratch = module.prepare_tensor_key_tmp_bytes(tensor_key);
    for key in rotation_keys.values() {
        required_scratch = required_scratch.max(module.glwe_automorphism_key_prepare_tmp_bytes(key));
    }
    if let Some(key) = encapsulation_key {
        required_scratch = required_scratch.max(module.glwe_switching_key_prepare_tmp_bytes(key));
    }
    ensure!(
        scratch.available() >= required_scratch,
        "PaCo key preparation needs {required_scratch} scratch bytes, but only {} are available",
        scratch.available(),
    );

    let mut prepared_rotations = HashMap::with_capacity(rotation_keys.len());
    for (&element, key) in rotation_keys {
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(key);
        module.glwe_automorphism_key_prepare(&mut prepared, key, scratch);
        prepared_rotations.insert(element, prepared);
    }

    let prepared_tensor = {
        let mut prepared = module.alloc_tensor_key_prepared_from_infos(tensor_key);
        module.prepare_tensor_key(&mut prepared, tensor_key, scratch);
        prepared
    };

    let prepared_encapsulation = encapsulation_key.map(|key| {
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
        module.glwe_switching_key_prepare(&mut prepared, key, scratch);
        prepared
    });

    Ok((prepared_rotations, prepared_tensor, prepared_encapsulation))
}

impl<D: Data, W: ZnxWord> PaCoKeySet<D, W> {
    /// Returns the four unprepared bootstrapping ciphertexts.
    pub fn bootstrapping_keys(&self) -> &[CKKSCiphertext<D, W>; 4] {
        &self.bootstrapping_keys
    }

    /// Returns the unprepared automorphism-key map.
    pub fn rotation_keys(&self) -> &HashMap<i64, GLWEAutomorphismKey<D, W>> {
        &self.rotation_keys
    }

    /// Returns the unprepared tensor key.
    pub fn tensor_key(&self) -> &GLWETensorKey<D, W> {
        &self.tensor_key
    }

    /// Returns the optional unprepared dense-to-PaCo switching key.
    pub fn encapsulation_key(&self) -> Option<&GLWESwitchingKey<D, W>> {
        self.encapsulation_key.as_ref()
    }

    /// Decomposes the set into its unprepared key material.
    pub fn into_parts(self) -> PaCoKeySetParts<D, W> {
        (
            self.parameters,
            self.bootstrapping_keys,
            self.rotation_keys,
            self.tensor_key,
            self.encapsulation_key,
        )
    }
}

impl<D: Data, BE: Backend> PaCoKeysPrepared<D, BE> {
    /// Builds a validated bundle from already-prepared backend material.
    ///
    /// This is the entry point for custom preparation or transfer paths. It
    /// applies the same layout, metadata, Galois-map, and switching-degree
    /// checks as [`PaCoKeySet::new`].
    pub fn new(
        plan: &PaCoPlan,
        bootstrapping_keys: [CKKSCiphertext<D, BE::ZnxWord>; 4],
        rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>,
        tensor_key: GLWETensorKeyPrepared<D, BE>,
        encapsulation_key: Option<GLWESwitchingKeyPrepared<D, BE>>,
    ) -> Result<Self> {
        validate_material(
            plan,
            &bootstrapping_keys,
            &rotation_keys,
            &tensor_key,
            encapsulation_key.as_ref(),
            GetGaloisElement::p,
        )?;

        Ok(Self {
            parameters: PaCoKeyParameters::from_plan(plan),
            bootstrapping_keys,
            rotation_keys,
            tensor_key,
            encapsulation_key,
        })
    }

    /// Structured-secret parameters bound to this key material.
    pub fn parameters(&self) -> PaCoKeyParameters {
        self.parameters
    }

    /// Returns the four backend-resident bootstrapping ciphertexts.
    pub fn bootstrapping_keys(&self) -> &[CKKSCiphertext<D, BE::ZnxWord>; 4] {
        &self.bootstrapping_keys
    }

    /// Returns the prepared automorphism-key map.
    pub fn rotation_keys(&self) -> &HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>> {
        &self.rotation_keys
    }

    /// Returns the prepared tensor key.
    pub fn tensor_key(&self) -> &GLWETensorKeyPrepared<D, BE> {
        &self.tensor_key
    }

    /// Returns the optional prepared dense-to-PaCo switching key.
    pub fn encapsulation_key(&self) -> Option<&GLWESwitchingKeyPrepared<D, BE>> {
        self.encapsulation_key.as_ref()
    }

    /// Decomposes the bundle into its prepared key material.
    pub fn into_parts(self) -> PaCoKeysPreparedParts<D, BE> {
        (
            self.parameters,
            self.bootstrapping_keys,
            self.rotation_keys,
            self.tensor_key,
            self.encapsulation_key,
        )
    }
}

impl<D: Data, BE: Backend> PaCoKeys<BE> for PaCoKeysPrepared<D, BE>
where
    CKKSCiphertext<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    GLWEAutomorphismKeyPrepared<D, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
    GLWESwitchingKeyPrepared<D, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    type BootstrappingKey = CKKSCiphertext<D, BE::ZnxWord>;
    type AutomorphismKey = GLWEAutomorphismKeyPrepared<D, BE>;
    type RotationKeys = HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>;
    type TensorKey = GLWETensorKeyPrepared<D, BE>;
    type SwitchingKey = GLWESwitchingKeyPrepared<D, BE>;

    fn parameters(&self) -> PaCoKeyParameters {
        self.parameters
    }

    fn bootstrapping_keys(&self) -> &[Self::BootstrappingKey; 4] {
        &self.bootstrapping_keys
    }

    fn rotation_keys(&self) -> &Self::RotationKeys {
        &self.rotation_keys
    }

    fn tensor_key(&self) -> &Self::TensorKey {
        &self.tensor_key
    }

    fn encapsulation_key(&self) -> Option<&Self::SwitchingKey> {
        self.encapsulation_key.as_ref()
    }
}

/// Validates the common unprepared/prepared PaCo key representation.
fn validate_material<B, A, T, S>(
    plan: &PaCoPlan,
    bootstrapping_keys: &[B; 4],
    rotation_keys: &HashMap<i64, A>,
    tensor_key: &T,
    encapsulation_key: Option<&S>,
    automorphism_element: impl Fn(&A) -> i64,
) -> Result<()>
where
    B: CKKSCtBounds + ActiveCiphertextStorage,
    A: GGLWEInfos,
    T: GGLWEInfos,
    S: GGLWEInfos + GLWESwitchingKeyDegrees,
{
    plan.check_evaluation()
        .context("invalid PaCo evaluation plan for key material")?;

    let n = plan.n();
    // The cyclotomic order is fully determined by the plan degree (`2N`); it
    // used to be a caller-supplied argument that only this value could satisfy.
    let cyclotomic_order = n
        .checked_mul(2)
        .context("PaCo cyclotomic order overflows usize")
        .and_then(|order| i64::try_from(order).context("PaCo cyclotomic order does not fit i64"))?;

    let base2k = bootstrapping_keys[0].base2k();
    let k = bootstrapping_keys[0].k();
    ensure!(
        (1..=63).contains(&base2k.as_usize()),
        "PaCo bootstrapping-key base2k must be in [1, 63], got {base2k}",
    );
    ensure!(
        k.as_usize() >= plan.max_plaintext_width(),
        "PaCo bootstrapping-key width {k} is smaller than the widest plaintext width {}",
        plan.max_plaintext_width(),
    );

    for (index, key) in bootstrapping_keys.iter().enumerate() {
        ensure!(
            key.n().as_usize() == n,
            "PaCo bootstrapping_keys[{index}] degree {} does not match plan degree {n}",
            key.n(),
        );
        ensure!(
            key.rank().as_usize() == 1,
            "PaCo bootstrapping_keys[{index}] must have rank 1, got {}",
            key.rank(),
        );
        ensure!(
            key.base2k() == base2k,
            "PaCo bootstrapping_keys[{index}] base2k {} does not match bootstrapping_keys[0] base2k {base2k}",
            key.base2k(),
        );
        ensure!(
            key.k() == k,
            "PaCo bootstrapping_keys[{index}] torus width {} does not match bootstrapping_keys[0] width {k}",
            key.k(),
        );
        ensure!(
            key.log_delta() == plan.log_delta_bsk(),
            "PaCo bootstrapping_keys[{index}] scale {} does not match plan scale {}",
            key.log_delta(),
            plan.log_delta_bsk(),
        );
        ensure!(
            key.log_sparsity() == 0,
            "PaCo bootstrapping_keys[{index}] must use dense CKKS metadata, got log_sparsity={}",
            key.log_sparsity(),
        );
        ensure!(
            key.log_delta() <= key.k().as_usize(),
            "PaCo bootstrapping_keys[{index}] scale {} exceeds torus width {}",
            key.log_delta(),
            key.k(),
        );
        validate_active_ciphertext_storage(&format!("PaCo bootstrapping_keys[{index}]"), key)?;
        ensure!(
            key.log_budget() >= plan.consumed_bits(),
            "PaCo bootstrapping_keys[{index}] has {} budget bits, but the plan consumes {}",
            key.log_budget(),
            plan.consumed_bits(),
        );
    }

    let order_u64 = u64::try_from(cyclotomic_order).context("PaCo cyclotomic order must be positive")?;
    for (&label, key) in rotation_keys {
        let key_element = automorphism_element(key);
        ensure!(
            label == key_element,
            "PaCo rotation-key map label {label} does not match key Galois element {key_element}",
        );
        ensure!(
            label != 0 && label.unsigned_abs() < order_u64 && label.unsigned_abs() % 2 == 1,
            "PaCo rotation-key Galois element {label} is invalid for cyclotomic order {cyclotomic_order}",
        );
        validate_gadget_key(
            &format!("PaCo rotation key {label}"),
            key,
            n,
            base2k,
            bootstrapping_keys[0].max_size(),
        )?;
    }

    for element in plan.galois_elements() {
        ensure!(
            rotation_keys.contains_key(&element),
            "PaCo rotation-key map is missing required Galois element {element}",
        );
    }

    validate_gadget_key("PaCo tensor key", tensor_key, n, base2k, bootstrapping_keys[0].max_size())?;

    if let Some(key) = encapsulation_key {
        let input_size = (plan.log_q() as usize).div_ceil(base2k.as_usize());
        validate_gadget_key("PaCo encapsulation key", key, n, base2k, input_size)?;
        ensure!(
            key.input_degree().as_usize() == n,
            "PaCo encapsulation-key input degree {} does not match plan degree {n}",
            key.input_degree(),
        );
        ensure!(
            key.output_degree().as_usize() == n,
            "PaCo encapsulation-key output degree {} does not match plan degree {n}",
            key.output_degree(),
        );
    }

    Ok(())
}

fn validate_active_ciphertext_storage<K: LWEInfos + ActiveCiphertextStorage + ?Sized>(name: &str, key: &K) -> Result<()> {
    validate_storage_capacity(name, key)?;
    ensure!(
        key.active_size() >= key.size(),
        "{name} exposes {} active limbs, but its torus width requires {}",
        key.active_size(),
        key.size(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, Rank, TorusPrecision};
    use poulpy_hal::layouts::HostBytesBackend;

    use crate::layouts::CKKSModuleAlloc;

    struct MisreportedCiphertext(CKKSCiphertext<Vec<u8>, i64>);

    impl LWEInfos for MisreportedCiphertext {
        fn n(&self) -> Degree {
            Degree(128)
        }

        fn base2k(&self) -> Base2K {
            self.0.base2k()
        }

        fn max_size(&self) -> usize {
            self.0.max_size()
        }

        fn k(&self) -> TorusPrecision {
            self.0.k()
        }
    }

    impl GLWEInfos for MisreportedCiphertext {
        fn rank(&self) -> Rank {
            self.0.rank()
        }
    }

    impl GLWEToBackendRef<HostBytesBackend> for MisreportedCiphertext {
        type State = Normalized;
        fn to_backend_ref(&self) -> GLWE<<HostBytesBackend as Backend>::BufRef<'_>, i64> {
            <CKKSCiphertext<Vec<u8>, i64> as GLWEToBackendRef<HostBytesBackend>>::to_backend_ref(&self.0)
        }
    }

    #[test]
    fn backend_view_must_match_reported_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let key = MisreportedCiphertext(module.ckks_ciphertext_alloc(Base2K(19), TorusPrecision(38)));
        let error = validate_backend_storage_capacity::<HostBytesBackend, _>("custom PaCo key", &key)
            .expect_err("a custom key must not misreport its backend layout");
        assert!(
            error.to_string().contains("backend-native view"),
            "unexpected error: {error:#}",
        );
    }
}
