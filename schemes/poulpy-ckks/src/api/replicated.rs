use crate::CKKSResult as Result;
use crate::{
    api::{
        CKKSAddOps, CKKSCIRingMapOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey, GetTensorKey,
};
use poulpy_core::{GLWECIEmbed, GLWECITrace, GLWECopy, GLWEKeyswitch, GLWEShift, GLWETrace};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

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
/// `ckks_to_replicated_into` reads a full-layout ciphertext holding `k`
/// values with period `k` and writes them on the data slots of `conv`'s
/// layout, at scale `conv.log_delta`:
///
/// ```text
/// log_budget_out = src.log_budget − log_delta_diag
/// log_delta_out  = conv.log_delta
/// ```
///
/// `ckks_from_replicated_into` reads a ciphertext in `conv`'s layout, with
/// `R | k`, and writes its `k` values with period `k` on a full layout,
/// averaging the replicas:
///
/// ```text
/// log_budget_out = src.log_budget − log_delta_diag
/// log_delta_out  = src.log_delta + extra_bits + log2(N/(2k))
/// ```
///
/// `keys` holds the automorphism keys of
/// [`ReplicatedConversion::rotations`]. Allocate the reusable workspace with
/// [`ReplicatedConversion::alloc_to_workspace`] or
/// [`ReplicatedConversion::alloc_from_workspace`] for the source layout.
pub trait CKKSReplicatedOps<BE: Backend> {
    fn ckks_replicated_conversion_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K, conv: &ReplicatedConversion<BE>) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// # Errors
    /// Incompatible source and destination degrees or ranks, or a workspace
    /// with incompatible input dimensions or rotations, return
    /// [`crate::CKKSCompositionError::InvalidOperand`] before writing `dst`.
    fn ckks_to_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    /// # Errors
    /// Incompatible source and destination degrees or ranks, or a workspace
    /// with incompatible input dimensions or rotations, return
    /// [`crate::CKKSCompositionError::InvalidOperand`] before writing `dst`.
    fn ckks_from_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;
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
    fn ckks_replicated_bootstrap_into<Dst, Src, F, K>(
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

impl<BE: Backend> CKKSReplicatedOps<BE> for Module<BE>
where
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>,
{
    fn ckks_replicated_conversion_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K, conv: &ReplicatedConversion<BE>) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        crate::reference::replicated::ckks_replicated_conversion_tmp_bytes(self, ct_infos, key_infos, conv)
    }

    fn ckks_to_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        crate::reference::replicated::ckks_to_replicated_into(self, dst, src, conv, workspace, keys, scratch)
    }

    fn ckks_from_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        crate::reference::replicated::ckks_from_replicated_into(self, dst, src, conv, workspace, keys, scratch)
    }
}

impl<BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl> CKKSReplicatedBootstrapOps<BE> for Module<BE>
where
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSEvalModOps<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWETrace<BE>,
{
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
        S: GGLWEInfos,
    {
        crate::reference::replicated::ckks_replicated_bootstrap_tmp_bytes(
            self,
            ct_out,
            ct_in,
            ctx,
            atk_infos,
            tsk_infos,
            encapsulation_infos,
        )
    }

    fn ckks_replicated_bootstrap_into<Dst, Src, F, K>(
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
        K: BootstrappingKeys<BE>,
    {
        crate::reference::replicated::ckks_replicated_bootstrap_into(self, dst, src, ctx, workspace, keys, scratch)
    }
}

/// Bootstrapping of a replicated layout on the conjugate-invariant ring of
/// degree `N`, through the standard ring of degree `2N` of this module, where
/// `ctx` is compiled: the input is embedded with `embed_key`, refreshed by
/// [`CKKSReplicatedBootstrapOps::ckks_replicated_bootstrap_into`] and traced
/// back with `trace_key`, each ring map computed at the shaping depth under
/// the scale and rounded with shaping. `ci` is the conjugate-invariant module
/// of degree `N`, on a backend that shares this one's buffers. The keys switch
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
    /// Scratch of [`Self::ckks_replicated_ci_bootstrap_into`]: `(scratch, scratch_ci)`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scratch covers the bootstrapping keys and the conversion keys"
    )]
    fn ckks_replicated_ci_bootstrap_tmp_bytes<BCI, C, I, F, A, T, S, K>(
        &self,
        ci: &Module<BCI>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> (usize, usize)
    where
        BCI: Backend<Ring = ConjugateInvariant>,
        Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos;

    /// # Errors
    /// A source outside the bootstrapping's layout, modules, keys or
    /// ciphertexts of other degrees than `2N` and `N`, and incompatible
    /// destinations return [`crate::CKKSCompositionError::InvalidOperand`]
    /// before writing `dst`, and the standard bootstrapping's checks apply.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes a module, keys and a scratch arena per ring"
    )]
    fn ckks_replicated_ci_bootstrap_into<BCI, Dst, Src, F, K, E, T>(
        &self,
        ci: &Module<BCI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()>
    where
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + for<'a> Backend<BufRef<'a> = BE::BufRef<'a>, BufMut<'a> = BE::BufMut<'a>>
            + 'static,
        Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
        Dst: GLWEToBackendMut<BCI> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BCI> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend<Ring = Standard>> CKKSReplicatedCIBootstrapOps<BE> for Module<BE>
where
    Module<BE>: CKKSReplicatedBootstrapOps<BE>
        + CKKSCIRingMapOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRoundOps<BE>
        + GLWECIEmbed<BE>
        + GLWECITrace<BE>
        + GLWEKeyswitch<BE>,
{
    fn ckks_replicated_ci_bootstrap_tmp_bytes<BCI, C, I, F, A, T, S, K>(
        &self,
        ci: &Module<BCI>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> (usize, usize)
    where
        BCI: Backend<Ring = ConjugateInvariant>,
        Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos,
    {
        crate::reference::replicated::ckks_replicated_ci_bootstrap_tmp_bytes(
            self,
            ci,
            ct_out,
            ct_in,
            ctx,
            atk_infos,
            tsk_infos,
            encapsulation_infos,
            conversion_infos,
        )
    }

    fn ckks_replicated_ci_bootstrap_into<BCI, Dst, Src, F, K, E, T>(
        &self,
        ci: &Module<BCI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()>
    where
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + for<'a> Backend<BufRef<'a> = BE::BufRef<'a>, BufMut<'a> = BE::BufMut<'a>>
            + 'static,
        Module<BCI>: CKKSCopyOps<BCI> + CKKSRoundOps<BCI>,
        Dst: GLWEToBackendMut<BCI> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BCI> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        let (embed_key, trace_key) = conversion_keys;
        crate::reference::replicated::ckks_replicated_ci_bootstrap_into(
            self, ci, dst, src, ctx, workspace, keys, embed_key, trace_key, scratch, scratch_ci,
        )
    }
}
