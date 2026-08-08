//! Public PaCo bootstrapping operations.
//!
//! The API follows the crate's caller-allocated convention. A single
//! sequential entry point accepts `kappa`: `kappa = 1` evaluates seqPaCo and
//! recovers `C` coefficient classes, while larger powers of two evaluate the
//! deterministic Algorithm-5 branch composition. The `_direct` variants
//! expect an input already under the structured PaCo secret; the default
//! variants first use the optional dense-to-PaCo switching key in
//! [`PaCoKeys`](crate::layouts::PaCoKeys).

use crate::CKKSResult as Result;
use crate::layouts::CKKSPlaintextOwned;
use crate::{
    CKKSCtBounds,
    api::CKKSEncodingScalar,
    layouts::{CKKSCiphertextOwned, PaCoContext, PaCoKeys, PaCoWorker},
};

use poulpy_core::layouts::GLWEToBackendRef;
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, ScratchArena, ScratchOwned},
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for crate::Quad {}
}

/// Scalar precision contract for PaCo's unit-circle coefficient encoding.
///
/// PaCo reduces ciphertext coefficients modulo `q` and converts those exact
/// residues to a floating-point phase, so the scalar must represent every
/// residue exactly: contexts reject plans whose `log_q` reaches
/// [`MANTISSA_BITS`](Self::MANTISSA_BITS). The trait is sealed to the two
/// scalar formats whose precision contracts are known here: `f64` and the
/// crate's binary128 [`Quad`](crate::Quad). This intentionally excludes
/// `f32`, for which valid PaCo moduli would lose residue bits before the
/// exponential.
///
/// The trait exists only for that contract. Factor-matrix generation uses
/// ordinary CKKS scalar arithmetic, while coefficient encoding additionally
/// uses the byte-stable [`CKKSEncodingScalar`] contract for backend-resident
/// buffers. It is never constrained to a host round-trip: a backend implements
/// [`CKKSPaCoOps::ckks_paco_coeff_encodings`] entirely on device with no
/// obligations beyond this trait.
pub trait PaCoScalar: sealed::Sealed + CKKSEncodingScalar {
    /// Number of significant binary digits in the scalar representation.
    const MANTISSA_BITS: u32;
}

impl PaCoScalar for f64 {
    const MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS;
}

impl PaCoScalar for crate::Quad {
    const MANTISSA_BITS: u32 = 113;
}

/// Caller-allocated CKKS PaCo bootstrapping.
///
/// All methods validate ring degree, rank, radix, modulus, sparse metadata,
/// output capacity, evaluation-key layouts, required Galois elements, and
/// scale/budget arithmetic before evaluating the circuit. The output is under
/// the application key encrypting the four bootstrapping ciphertexts. Its
/// `log_sparsity` is `log2(N / (kappa*C))`; its scale is the validated PaCo
/// re-anchoring of the exhausted input scale.
///
/// The scalar is a trait parameter so the methods stay free of backend
/// bounds: the delegating impl on `Module<BE>` requires the
/// [`CKKSPaCoCoeffEncodingImpl`](crate::oep::CKKSPaCoCoeffEncodingImpl) and
/// [`CKKSEncodingImpl`](crate::oep::CKKSEncodingImpl) seams at the impl
/// level, and a backend overrides those seams independently of any bounds
/// the reference implementation carries.
pub trait CKKSPaCoOps<BE: Backend, F: PaCoScalar> {
    /// Scratch bytes required by direct sequential or parallel PaCo.
    ///
    /// The bound covers one branch on the caller module; every parallel
    /// worker arena must provide at least the returned size. The ciphertext,
    /// context, and key layouts are validated while computing the bound.
    fn ckks_paco_bootstrap_direct_tmp_bytes<K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Scratch bytes required by encapsulated sequential or parallel PaCo.
    ///
    /// This is the maximum of the direct branch bound and the one-time
    /// dense-to-PaCo key switch. It fails if the key bundle has no
    /// encapsulation key. Parallel worker arenas only need the direct bound
    /// returned by [`Self::ckks_paco_bootstrap_direct_tmp_bytes`].
    fn ckks_paco_bootstrap_tmp_bytes<K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Standard-arena bytes required to build the four input-dependent β
    /// plaintexts, including backend-native scalar/FFT workspace.
    fn ckks_paco_coeff_encodings_tmp_bytes(&self, context: &PaCoContext<BE, F>) -> Result<usize>;

    /// Builds the four input-dependent beta plaintexts used by PaCo's blind
    /// rotation. This is the only PaCo primitive intended for native backend
    /// specialization; the bootstrap itself composes existing CKKS/core ops.
    /// The signature imposes no FFT engine, encoder, or host codec: a backend
    /// may implement the whole step as one fused kernel from the ciphertext
    /// residues, as long as its β packing agrees numerically with the
    /// `2C`-block packing used at key generation.
    fn ckks_paco_coeff_encodings<Src>(
        &self,
        ciphertext: &Src,
        context: &PaCoContext<BE, F>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<[CKKSPlaintextOwned<BE>; 4]>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Bootstraps `kappa*C` coefficient classes sequentially when `input` is
    /// already under the structured PaCo secret.
    ///
    /// `kappa` must be a non-zero power of two and `kappa*C <= N`. Branches are
    /// recombined in increasing branch order, making the result deterministic.
    fn ckks_paco_bootstrap_direct_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        kappa: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Bootstraps `kappa*C` coefficient classes sequentially after switching a
    /// dense-key input to the structured PaCo secret.
    ///
    /// Fails if `keys` has no dense-to-PaCo switching key. No switch back is
    /// needed: the bootstrapping ciphertexts transfer the result directly to
    /// their application key.
    fn ckks_paco_bootstrap_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        kappa: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Parallel direct-mode PaCo with bounded, reusable workers.
    ///
    /// The caller thread evaluates one branch and `workers` supplies reusable
    /// module/scratch contexts for the remaining branches. At most
    /// `1 + workers.len()` branches execute concurrently; an empty slice is
    /// the deterministic sequential fallback. Results are recombined in
    /// branch order.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_direct_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        kappa: usize,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds + Sync;

    /// Parallel encapsulated PaCo. The dense-to-PaCo switch is evaluated once
    /// before the bounded direct-mode branch pool.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_into<K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        kappa: usize,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds + Sync;
}
