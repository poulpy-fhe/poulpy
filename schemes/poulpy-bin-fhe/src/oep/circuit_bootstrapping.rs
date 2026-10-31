#![allow(clippy::too_many_arguments)]
use crate::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared, CircuitBootstrappingPlan,
        CircuitBootstrappingPlanLayout,
    },
};
use poulpy_core::{
    GetDistribution,
    layouts::{
        GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWESecretToBackendRef, LWEInfos,
        LWESecretToBackendRef, LWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena, Standard},
    source::Source,
};
/// Backend implementation contract for [`CircuitBootstrappingExecute`](crate::api::CircuitBootstrappingExecute), on the
/// standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CircuitBootstrappingExecuteImpl<BRA: BlindRotationAlgo>: Backend<ZnxWord = i64, Ring = Standard> {
    /// Default body: `circuit_bootstrapping_execute_to_constant_tmp_bytes_derived`, the selected
    /// [`Self::circuit_bootstrapping_execute_prepared_tmp_bytes`] at a plan layout over `res_infos` with no `log_domain`
    /// or gaps, which bounds every domain at that output layout.
    ///
    /// Scratch bound of [`Self::circuit_bootstrapping_execute_to_constant`]; may report less than the reference, never
    /// less than the body takes.
    fn circuit_bootstrapping_execute_to_constant_tmp_bytes<R, A>(
        module: &Module<Self>,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_constant_tmp_bytes_derived(
            module,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
    /// Default body: `circuit_bootstrapping_execute_to_exponent_tmp_bytes_derived`, the selected
    /// [`Self::circuit_bootstrapping_execute_prepared_tmp_bytes`] at a plan layout over `res_infos` with `log_domain`,
    /// `log_gap_out`, and the input gap `circuit_bootstrapping_log_gap_in` gives for `res_infos`, `log_domain` and
    /// `extension_factor`.
    ///
    /// Scratch bound of [`Self::circuit_bootstrapping_execute_to_exponent`]; may report less than the reference, never
    /// less than the body takes.
    fn circuit_bootstrapping_execute_to_exponent_tmp_bytes<R, A>(
        module: &Module<Self>,
        log_gap_out: usize,
        log_domain: usize,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_exponent_tmp_bytes_derived(
            module,
            log_gap_out,
            log_domain,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
    /// Default body: `circuit_bootstrapping_execute_to_constant_derived`, the selected
    /// [`Self::circuit_bootstrapping_prepare_to_constant`] followed by [`Self::circuit_bootstrapping_execute_prepared`]
    /// on the plan, within [`Self::circuit_bootstrapping_execute_to_constant_tmp_bytes`].
    fn circuit_bootstrapping_execute_to_constant<R, L>(
        module: &Module<Self>,
        log_domain: usize,
        extension_factor: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_constant_derived(
            module,
            log_domain,
            extension_factor,
            res,
            lwe,
            key,
            scratch,
        )
    }
    /// Default body: `circuit_bootstrapping_execute_to_exponent_derived`, the selected
    /// [`Self::circuit_bootstrapping_prepare_to_exponent`] followed by [`Self::circuit_bootstrapping_execute_prepared`]
    /// on the plan, within [`Self::circuit_bootstrapping_execute_to_exponent_tmp_bytes`].
    fn circuit_bootstrapping_execute_to_exponent<R, L>(
        module: &Module<Self>,
        log_gap_out: usize,
        log_domain: usize,
        extension_factor: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_exponent_derived(
            module,
            log_gap_out,
            log_domain,
            extension_factor,
            res,
            lwe,
            key,
            scratch,
        )
    }
    /// Reproduces
    /// [`circuit_bootstrapping_prepare_to_constant_reference`](crate::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_constant_reference)
    /// bit for bit; free in kernels. The plan records the output layout, key layouts and block size the executions assert
    /// against.
    fn circuit_bootstrapping_prepare_to_constant<R: GGSWInfos>(
        module: &Module<Self>,
        log_domain: usize,
        extension_factor: usize,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
    ) -> CircuitBootstrappingPlan<Self::OwnedBuf>;
    /// Reproduces
    /// [`circuit_bootstrapping_prepare_to_exponent_reference`](crate::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_exponent_reference)
    /// bit for bit; free in kernels. The plan records the output layout, key layouts and block size the executions assert
    /// against.
    fn circuit_bootstrapping_prepare_to_exponent<R: GGSWInfos>(
        module: &Module<Self>,
        log_gap_out: usize,
        log_domain: usize,
        extension_factor: usize,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
    ) -> CircuitBootstrappingPlan<Self::OwnedBuf>;
    /// Scratch bound of [`Self::circuit_bootstrapping_execute_prepared`] over every plan of the layout `plan`; may report
    /// less than the reference, never less than the body takes. `key` is read for its layouts only, and a `log_domain` of
    /// `None` bounds every domain at that output layout.
    fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: CircuitBootstrappingKeyInfos>(
        module: &Module<Self>,
        plan: &CircuitBootstrappingPlanLayout,
        key: &A,
    ) -> usize;
    /// Reproduces
    /// [`circuit_bootstrapping_execute_prepared_reference`](crate::reference::circuit_bootstrapping::circuit_bootstrapping_execute_prepared_reference)
    /// bit for bit, within [`Self::circuit_bootstrapping_execute_prepared_tmp_bytes`]; free in kernels and scratch use.
    fn circuit_bootstrapping_execute_prepared<R, L>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        plan: &CircuitBootstrappingPlan<Self::OwnedBuf>,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos;
}
/// Backend implementation contract for
/// [`CircuitBootstrappingKeyEncryptSk`](crate::api::CircuitBootstrappingKeyEncryptSk).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CircuitBootstrappingKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::circuit_bootstrapping_key_encrypt_sk`]; may report less than the reference, never less
    /// than the body takes.
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;
    /// Reproduces
    /// [`circuit_bootstrapping_key_encrypt_sk_reference`](crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::circuit_bootstrapping_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use
    /// before the wipe.
    fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut CircuitBootstrappingKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos + GetDistribution;
}
/// Backend implementation contract for
/// [`CircuitBootstrappingKeyPreparedFactory`](crate::api::CircuitBootstrappingKeyPreparedFactory).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CircuitBootstrappingKeyPreparedFactoryImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::circuit_bootstrapping_key_prepare`]; may report less than the reference, never less than
    /// the body takes.
    fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;
    /// Reproduces
    /// [`circuit_bootstrapping_key_prepare_reference`](crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_prepare_reference)
    /// bit for bit, within [`Self::circuit_bootstrapping_key_prepare_tmp_bytes`]; free in kernels and scratch use.
    fn circuit_bootstrapping_key_prepare(
        module: &Module<Self>,
        res: &mut CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &CircuitBootstrappingKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
/// Selects the reference implementation of `CircuitBootstrappingExecuteImpl<$algo>` for `$be`.
#[macro_export]
macro_rules! impl_circuit_bootstrapping_execute_reference_full {
    ($be:ty, $algo:ty) => {
        unsafe impl $crate::oep::CircuitBootstrappingExecuteImpl<$algo> for $be {
            fn circuit_bootstrapping_prepare_to_constant<R: ::poulpy_core::layouts::GGSWInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_domain: usize,
                extension_factor: usize,
                res_infos: &R,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
            ) -> $crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingExecuteReference<$algo, $be>>::circuit_bootstrapping_prepare_to_constant_reference(
                    module,
                    log_domain,
                    extension_factor,
                    res_infos,
                    key,
                )
            }
            fn circuit_bootstrapping_prepare_to_exponent<R: ::poulpy_core::layouts::GGSWInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_gap_out: usize,
                log_domain: usize,
                extension_factor: usize,
                res_infos: &R,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
            ) -> $crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingExecuteReference<$algo, $be>>::circuit_bootstrapping_prepare_to_exponent_reference(
                    module,
                    log_gap_out,
                    log_domain,
                    extension_factor,
                    res_infos,
                    key,
                )
            }
            fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                plan: &$crate::circuit_bootstrapping::CircuitBootstrappingPlanLayout,
                key: &A,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingExecuteReference<$algo, $be>>::circuit_bootstrapping_execute_prepared_tmp_bytes_reference(
                    module, plan, key,
                )
            }
            fn circuit_bootstrapping_execute_prepared<R, L>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                lwe: &L,
                plan: &$crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf>,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) where
                R: ::poulpy_core::layouts::GGSWToBackendMut<Self>
                    + ::poulpy_core::layouts::GGSWAtViewRef<Self>
                    + ::poulpy_core::layouts::GGSWAtViewMut<Self>
                    + ::poulpy_core::layouts::GGSWInfos,
                L: ::poulpy_core::layouts::LWEToBackendRef<Self> + ::poulpy_core::layouts::LWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingExecuteReference<$algo, $be>>::circuit_bootstrapping_execute_prepared_reference(
                    module, res, lwe, plan, key, scratch,
                )
            }
        }
    };
}
/// Selects the reference implementation of `CircuitBootstrappingKeyEncryptSkImpl<$algo>` for `$be`.
#[macro_export]
macro_rules! impl_circuit_bootstrapping_key_encrypt_sk_reference_full {
    ($be:ty, $algo:ty) => {
        unsafe impl $crate::oep::CircuitBootstrappingKeyEncryptSkImpl<$algo> for $be {
            fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingKeyEncryptSkReference<$algo, $be>>::circuit_bootstrapping_key_encrypt_sk_tmp_bytes_reference(module, infos)
            }
            fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::circuit_bootstrapping::CircuitBootstrappingKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                sk_lwe: &S0,
                sk_glwe: &S1,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) where
                S0: ::poulpy_core::layouts::LWESecretToBackendRef<Self>
                    + ::poulpy_core::GetDistribution
                    + ::poulpy_core::layouts::LWEInfos,
                S1: ::poulpy_core::layouts::GLWESecretToBackendRef<Self>
                    + ::poulpy_core::layouts::GLWEInfos
                    + ::poulpy_core::GetDistribution,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingKeyEncryptSkReference<$algo, $be>>::circuit_bootstrapping_key_encrypt_sk_reference(
                    module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                )
            }
        }
    };
}
/// Selects the reference implementation of `CircuitBootstrappingKeyPreparedFactoryImpl<$algo>` for `$be`.
#[macro_export]
macro_rules! impl_circuit_bootstrapping_key_prepared_factory_reference_full {
    ($be:ty, $algo:ty) => {
        unsafe impl $crate::oep::CircuitBootstrappingKeyPreparedFactoryImpl<$algo> for $be {
            fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingKeyPreparedFactoryReference<$algo, $be>>::circuit_bootstrapping_key_prepare_tmp_bytes_reference(module, infos)
            }
            fn circuit_bootstrapping_key_prepare(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                other: &$crate::circuit_bootstrapping::CircuitBootstrappingKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::circuit_bootstrapping::CircuitBootstrappingKeyPreparedFactoryReference<$algo, $be>>::circuit_bootstrapping_key_prepare_reference(module, res, other, scratch)
            }
        }
    };
}
/// Explicitly registers all canonical circuit-bootstrap operations.
#[macro_export]
macro_rules! impl_circuit_bootstrapping_reference_full {
    ($be:ty, $algo:ty) => {
        $crate::impl_circuit_bootstrapping_execute_reference_full!($be, $algo);
        $crate::impl_circuit_bootstrapping_key_encrypt_sk_reference_full!($be, $algo);
        $crate::impl_circuit_bootstrapping_key_prepared_factory_reference_full!($be, $algo);
    };
}
