use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{ReplicatedConversion, ReplicatedTransformWorkspace},
    oep::derived::replicated::{
        ckks_from_replicated_derived, ckks_replicated_conversion_tmp_bytes_derived, ckks_to_replicated_derived,
    },
};

/// Backend contract of [`CKKSReplicatedOps`](crate::api::CKKSReplicatedOps), the conversions between
/// CKKS and a replicated slot layout. Every method defaults to its derived composition of the CKKS
/// operations of its supertraits on the ring `Rg`; [`impl_ckks_replicated_derived_full`] selects them all.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the derived body does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedImpl<Rg: Ring = Standard>:
    Backend
    + super::CKKSLinearTransformationImpl<Rg>
    + super::CKKSRotateImpl<Rg>
    + super::CKKSAddImpl<Rg>
    + super::CKKSCopyImpl<Rg>
    + super::CKKSPow2Impl<Rg>
    + super::CKKSRoundImpl<Rg>
{
    /// Scratch bound of [`Self::ckks_to_replicated`] and [`Self::ckks_from_replicated`]; may report less than the
    /// derived body, never less than the body takes.
    fn ckks_replicated_conversion_tmp_bytes<R, A, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        key_infos: &K,
        ctx: &ReplicatedConversion<Self, Rg>,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_replicated_conversion_tmp_bytes_derived(module, res, a, key_infos, ctx)
    }

    /// Reproduces the derived conversion onto `ctx`'s replicated layout bit for bit.
    fn ckks_to_replicated<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<Self, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg>,
    {
        ckks_to_replicated_derived(module, res, a, ctx, workspace, keys, scratch)
    }

    /// Reproduces the derived conversion from `ctx`'s replicated layout bit for bit.
    fn ckks_from_replicated<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<Self, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg>,
    {
        ckks_from_replicated_derived(module, res, a, ctx, workspace, keys, scratch)
    }
}

/// Selects the derived defaults of [`CKKSReplicatedImpl`] for `$be`, on both rings from `($be)` and on `$ring` from
/// `($be, $ring)`.
#[macro_export]
macro_rules! impl_ckks_replicated_derived_full {
    ($be:ty) => {
        $crate::impl_ckks_replicated_derived_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_replicated_derived_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedImpl<$ring> for $be {}
    };
}

pub use crate::impl_ckks_replicated_derived_full;
