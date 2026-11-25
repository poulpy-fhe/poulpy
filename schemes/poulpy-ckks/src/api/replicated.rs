use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::ReplicatedConversionPrepared};

/// Conversions between CKKS and a replicated slot layout, sharing keys.
///
/// # Metadata
///
/// `ckks_to_replicated` reads a full-layout ciphertext holding `k` values with period `k` and writes them on the data
/// slots of `ctx`'s layout, at scale `ctx.log_delta`, rounded with shaping:
///
/// ```text
/// log_budget_out = a.log_budget - consumed(to)
/// log_delta_out  = ctx.log_delta
/// ```
///
/// `ckks_from_replicated` reads a ciphertext in `ctx`'s layout, with `R | k` for a conversion of dense maps, and writes
/// its `k` values with period `k` on a full layout, averaging the replicas through a trace of `L = log2(S/k)` steps
/// for the `S` slots of the ring:
///
/// ```text
/// log_budget_out = a.log_budget - consumed(from) - L
/// log_delta_out  = a.log_delta + extra_bits + L
/// ```
///
/// `ckks_to_replicated` keeps the slot kind of `a`, stamps the replicated layout of `ctx` and the sparsity
/// `L - log2(R)` of the period `kR`; `ckks_from_replicated` keeps the slot kind, stamps the full layout and the sparsity
/// `L` the trace closes. Both leave the result canonical, and land at `res.k()` when it is narrower, through a modulus
/// drop. A map consumes the scales of its factors:
/// `consumed(to) = consumed(from) = log_delta_diag` for [`ReplicatedConversionPrepared::new`], and for
/// [`ReplicatedConversionPrepared::new_factored`] of `p` placement factors `consumed(to) = (p - 1)·inner_log_delta_diag +
/// log_delta_diag` and `consumed(from) = p·log_delta_diag`.
///
/// `keys` holds the automorphism keys of [`ReplicatedConversionPrepared::galois_elements`], in a source of the bounds a
/// linear transformation takes: its tensor key is never asked for. A conversion works on a copy of `a` at
/// `a.k() + extra_bits` bits and looks each key up at that `k`, less the scales of the factors before its widest use:
/// a factor's baby rotations switch its input and its giant rotations its product, one factor scale lower, and the
/// trace of `ckks_from_replicated` switches what the factors leave. Keys covering `a.k() + extra_bits` serve both
/// conversions.
///
/// The conversions serve both rings, `Rg` that of the operands, `ctx` compiled for it.
pub trait CKKSReplicatedOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch of both conversions of `ctx` from `a` into `res`, with keys of layout `key_infos`, which a conversion
    /// switching no key does not read.
    fn ckks_replicated_conversion_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<BE, Rg>,
        key_infos: &K,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// # Errors
    /// Before writing `res`: `InvalidOperand` for operands of another rank or degree than one another and `ctx`, or a
    /// source off the full layout or at a scale below `ctx.log_delta - extra_bits`; `InsufficientHomomorphicCapacity`
    /// when `a` lacks the budget of the transform or the result does not fit `res.k()`; `MissingAutomorphismKey` or `AutomorphismKeyTooNarrow` for a
    /// key the run looks up and `keys` lacks or does not cover; and an arena below the query.
    fn ckks_to_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>;

    /// # Errors
    /// Before writing `res`: `InvalidOperand` for operands of another rank or degree than one another and `ctx`, a
    /// source off `ctx`'s layout, or a conversion without the gather; `InsufficientHomomorphicCapacity` when `a` lacks
    /// the budget of the transform and the trace or the result does not fit `res.k()`; `MissingAutomorphismKey` or `AutomorphismKeyTooNarrow` for a
    /// key the run looks up and `keys` lacks or does not cover; and an arena below the query.
    fn ckks_from_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>;
}
