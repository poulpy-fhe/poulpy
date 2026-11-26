use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{ReplicatedConversion, ReplicatedTransformWorkspace},
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
/// `ckks_from_replicated` reads a ciphertext in `ctx`'s layout, with `R | k` for a conversion of dense maps, and writes
/// its `k` values with period `k` on a full layout, averaging the replicas through a trace of `L = log2(S/k)` steps
/// for the `S` slots of the ring:
///
/// ```text
/// log_budget_out = a.log_budget - consumed_bits(from) - L
/// log_delta_out  = a.log_delta + extra_bits + L
/// ```
///
/// Both land at `res.k()` when it is narrower, through a modulus drop.
///
/// `keys` holds the automorphism keys of [`ReplicatedConversion::galois`], each at the operand's order, covering the width the run looks
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
