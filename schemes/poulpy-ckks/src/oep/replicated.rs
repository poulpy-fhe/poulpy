use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::ReplicatedConversionPrepared,
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
    + super::CKKSCopyImpl<Rg>
    + super::CKKSRoundImpl<Rg>
    + super::CKKSInnerSumImpl<Rg>
{
    /// Default body: `ckks_replicated_conversion_tmp_bytes_derived`. Scratch bound of [`Self::ckks_to_replicated`] and
    /// [`Self::ckks_from_replicated`]; may report less than the derived body, never less than the body takes.
    fn ckks_replicated_conversion_tmp_bytes<R, A, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<Self, Rg>,
        key_infos: &K,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_replicated_conversion_tmp_bytes_derived(module, res, a, ctx, key_infos)
    }

    /// Default body: `ckks_to_replicated_derived`, a `ckks_copy` of `a` `extra_bits` under its scale, the selected
    /// `ckks_linear_transformation_eval_self_assign` of each factor of `ctx`'s `to` map on it, and a `ckks_round` into
    /// `res` with shaping, through a `ckks_copy` for a narrower `res`. An override reproduces it bit for bit, refusals
    /// included, within [`Self::ckks_replicated_conversion_tmp_bytes`], free only in kernels and scratch use.
    fn ckks_to_replicated<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg>,
    {
        ckks_to_replicated_derived(module, res, a, ctx, keys, scratch)
    }

    /// Default body: `ckks_from_replicated_derived`, a `ckks_copy` of `a` `extra_bits` under its scale, the selected
    /// `ckks_linear_transformation_eval_self_assign` of each factor of `ctx`'s `from` map on it, the selected
    /// `ckks_inner_sum_assign` of its mean trace and a `ckks_copy` into `res`. An override reproduces it bit for bit,
    /// refusals included, within [`Self::ckks_replicated_conversion_tmp_bytes`], free only in kernels and scratch use.
    fn ckks_from_replicated<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversionPrepared<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg>,
    {
        ckks_from_replicated_derived(module, res, a, ctx, keys, scratch)
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
