use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{
        BootstrappingKeys, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedTransformWorkspace,
    },
};

/// Conversions between CKKS and a replicated slot layout, sharing keys.
///
/// # Metadata
///
/// `ckks_to_replicated` reads a full-layout ciphertext holding `k` values with period `k` and writes them on the data
/// slots of `ctx`'s layout, at scale `ctx.log_delta`, rounded with shaping:
///
/// ```text
/// log_budget_out = a.log_budget - consumed_bits(to)
/// log_delta_out  = ctx.log_delta
/// ```
///
/// `ckks_from_replicated` reads a ciphertext in `ctx`'s layout, with `R | k`, and writes its `k` values with period
/// `k` on a full layout, averaging the replicas through a trace of `L = log2(N/(2k))` steps:
///
/// ```text
/// log_budget_out = a.log_budget - consumed_bits(from) - L
/// log_delta_out  = a.log_delta + extra_bits + L
/// ```
///
/// Both land at `res.k()` when it is narrower, through a modulus drop.
///
/// `keys` holds the automorphism keys of [`ReplicatedConversion::rotations`], each covering the width the run looks
/// it up at. Allocate the reusable workspace with [`ReplicatedConversion::alloc_to_workspace`] or
/// [`ReplicatedConversion::alloc_from_workspace`] for the source layout.
///
/// The conversions serve both rings, `Rg` that of the operands, `ctx` compiled for it.
pub trait CKKSReplicatedOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch of both conversions of `ctx` from `a` into `res`, with keys of layout `key_infos`.
    fn ckks_replicated_conversion_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        a: &A,
        key_infos: &K,
        ctx: &ReplicatedConversion<BE, Rg>,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// # Errors
    /// Before writing `res`: `InvalidOperand` for operands of another rank or degree than one another and `ctx`, a
    /// source off the full layout or at a scale below `ctx.log_delta - extra_bits`, or a workspace of another layout
    /// or transform; `InsufficientHomomorphicCapacity` when `a` lacks the budget of the transform or the result does
    /// not fit `res.k()`; `MissingAutomorphismKey` for a key the run looks up and `keys` lacks or does not cover; and an
    /// arena below the query.
    fn ckks_to_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<BE, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg>;

    /// # Errors
    /// Before writing `res`: `InvalidOperand` for operands of another rank or degree than one another and `ctx`, a
    /// source off `ctx`'s layout, a conversion without the gather, or a workspace of another layout or transform;
    /// `InsufficientHomomorphicCapacity` when `a` lacks the budget of the transform and the trace or the result does
    /// not fit `res.k()`; `MissingAutomorphismKey` for a key the run looks up and `keys` lacks or does not cover; and an
    /// arena below the query.
    fn ckks_from_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<BE, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg>;
}

/// Bootstrapping of a replicated slot layout of `k` values, `R | k`, in
/// `SlotsToCoeffs`-first order.
///
/// 1. The input, at the scale `Δ_b` or at the leveled scale, is brought to
///    `Δ_b` at width [`ReplicatedBootstrapPlan::k_in`](crate::layouts::ReplicatedBootstrapPlan::k_in).
/// 2. The normalized bottom trace averages the `N/(2kR)` copies of the
///    layout's period with shaped automorphisms.
/// 3. `SlotsToCoeffs` gathers the data slots onto the coefficients of the
///    subring of degree `2k`, `extra_bits` under the scale, and rounds plainly
///    to the message ratio `q_b/Δ_b` of the EvalMod plan.
/// 4. ModUp, with sparse-secret encapsulation when `keys` carries it, lifted
///    by `mod_up_extra_bits`.
/// 5. The normalized top trace onto period `k` removes the lifted multiples
///    off the subring, and `CoeffsToSlots` writes the real and imaginary
///    parts of the coefficients onto the data slots, side by side when
///    packed, each rounded with shaping.
/// 6. EvalMod, then the parts are recombined.
/// 7. The result is relabelled at `Δ_b` and rounded with shaping to the
///    leveled scale.
///
/// `dst.k()` is the output width; ModUp raises to
/// [`ReplicatedBootstrapPlan::k_boot`](crate::layouts::ReplicatedBootstrapPlan::k_boot)
/// of it. Allocate its reusable workspace with
/// [`ReplicatedBootstrapping::alloc_workspace`]. The keys cover [`ReplicatedBootstrapping::galois_elements`].
pub trait CKKSReplicatedBootstrapOps<BE: Backend> {
    fn ckks_replicated_bootstrap_tmp_bytes<C, I, F, A, T, S>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos;

    /// # Errors
    /// Incompatible source and destination degrees or ranks, or a workspace
    /// with incompatible input dimensions or rotations, return
    /// [`crate::CKKSCompositionError::InvalidOperand`] before writing `dst`.
    fn ckks_replicated_bootstrap<Dst, Src, F, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>;
}

/// Bootstrapping of a replicated layout on the conjugate-invariant ring of
/// degree `N`, through the standard ring of degree `2N` of this module, where
/// `ctx` is compiled: the input is embedded with `embed_key`, refreshed by
/// [`CKKSReplicatedBootstrapOps::ckks_replicated_bootstrap`] and traced
/// back with `trace_key`, each ring map computed at the shaping depth under
/// the scale and rounded with shaping. The conjugate-invariant input and
/// output, of degree `N`, live on this module of degree `2N`. The keys switch
/// between the conjugate-invariant secret and the standard secret of the
/// bootstrapping keys (`GLWECIKeyEncryptSk`).
///
/// # Metadata
///
/// `dst` ends at its width and the leveled scale, the standard bootstrapping
/// running one bit wider for the trace to spend:
///
/// ```text
/// log_delta_out = plan.level_log_delta, k_out = dst.k
/// ```
pub trait CKKSReplicatedCIBootstrapOps<BE: Backend> {
    /// Scratch of [`Self::ckks_replicated_ci_bootstrap`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the scratch covers the bootstrapping keys and the conversion keys"
    )]
    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos;

    /// # Errors
    /// A source outside the bootstrapping's layout, a module or keys of
    /// another degree than `2N` for ciphertexts of degree `N`, and incompatible
    /// destinations return [`crate::CKKSCompositionError::InvalidOperand`]
    /// before writing `dst`, and the standard bootstrapping's checks apply.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes its context, workspace and both key sets"
    )]
    fn ckks_replicated_ci_bootstrap<Dst, Src, F, K, E, T>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, ConjugateInvariant> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, ConjugateInvariant> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>;
}
