#![allow(clippy::too_many_arguments)]
use crate::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyInfos,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingPlan, CircuitBootstrappingPlanLayout,
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
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};
/// Backend contract for circuit bootstrapping.
///
/// # Safety
/// Preserve the canonical circuit, metadata, and input invariants. Execution must
/// stay within the selected scratch query; validate overrides with caller-selected parity.
pub unsafe trait CircuitBootstrappingExecuteImpl<BRA: BlindRotationAlgo>: Backend<ZnxWord = i64> {
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
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
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_tmp_bytes_derived(
            module,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
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
    fn circuit_bootstrapping_execute_to_constant<R, L>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_constant_derived(
            module,
            res,
            lwe,
            key,
            log_domain,
            extension_factor,
            scratch,
        )
    }
    fn circuit_bootstrapping_execute_to_exponent<R, L>(
        module: &Module<Self>,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos,
    {
        crate::oep::derived::circuit_bootstrapping::circuit_bootstrapping_execute_to_exponent_derived(
            module,
            log_gap_out,
            res,
            lwe,
            key,
            log_domain,
            extension_factor,
            scratch,
        )
    }
    fn circuit_bootstrapping_prepare_to_constant<R: GGSWInfos>(
        module: &Module<Self>,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<Self::OwnedBuf>;
    fn circuit_bootstrapping_prepare_to_exponent<R: GGSWInfos>(
        module: &Module<Self>,
        log_gap_out: usize,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<Self::OwnedBuf>;
    fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: CircuitBootstrappingKeyInfos>(
        module: &Module<Self>,
        plan: &CircuitBootstrappingPlanLayout,
        key: &A,
    ) -> usize;
    fn circuit_bootstrapping_execute_prepared<R, L>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        plan: &CircuitBootstrappingPlan<Self::OwnedBuf>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewRef<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        L: LWEToBackendRef<Self> + LWEInfos;
}
/// Backend contract for circuit bootstrapping.
///
/// # Safety
/// Preserve the canonical circuit, metadata, and input invariants. Execution must
/// stay within the selected scratch query; validate overrides with caller-selected parity.
pub unsafe trait CircuitBootstrappingKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;
    fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut CircuitBootstrappingKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos + GetDistribution;
}
/// Backend contract for circuit bootstrapping.
///
/// # Safety
/// Preserve the canonical circuit, metadata, and input invariants. Execution must
/// stay within the selected scratch query; validate overrides with caller-selected parity.
pub unsafe trait CircuitBootstrappingKeyPreparedImpl<BRA: BlindRotationAlgo>: Backend {
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        module: &Module<Self>,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>
    where
        A: CircuitBootstrappingKeyInfos;
    fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;
    fn circuit_bootstrapping_key_prepare(
        module: &Module<Self>,
        res: &mut CircuitBootstrappingKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &CircuitBootstrappingKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
/// Explicitly selects the canonical circuit-bootstrap execute implementation.
#[macro_export]
macro_rules! impl_bin_fhe_circuit_bootstrapping_execute_reference {
 ($be:ty, $algo:ty) => {
unsafe impl $crate::oep::CircuitBootstrappingExecuteImpl<$algo> for $be {


            fn circuit_bootstrapping_prepare_to_constant<R: ::poulpy_core::layouts::GGSWInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res_infos: &R,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                log_domain: usize,
                extension_factor: usize,
            ) -> $crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_constant_reference::<R, _, $algo, Self>(
                    module,
                    res_infos,
                    key,
                    log_domain,
                    extension_factor,
                )
            }
            fn circuit_bootstrapping_prepare_to_exponent<R: ::poulpy_core::layouts::GGSWInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_gap_out: usize,
                res_infos: &R,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                log_domain: usize,
                extension_factor: usize,
            ) -> $crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf> {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_prepare_to_exponent_reference::<R, _, $algo, Self>(
                    module,
                    log_gap_out,
                    res_infos,
                    key,
                    log_domain,
                    extension_factor,
                )
            }
            fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos>(
 module: &::poulpy_hal::layouts::Module<Self>,
 plan: &$crate::circuit_bootstrapping::CircuitBootstrappingPlanLayout,
 key: &A,
 ) -> usize {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_execute_prepared_tmp_bytes_reference::<_, _, $algo, Self>(
                    module, plan, key,
                )
            }
            fn circuit_bootstrapping_execute_prepared<R, L>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                lwe: &L,
                key: &$crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                plan: &$crate::circuit_bootstrapping::CircuitBootstrappingPlan<Self::OwnedBuf>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) where
                R: ::poulpy_core::layouts::GGSWToBackendMut<Self>
                    + ::poulpy_core::layouts::GGSWAtViewRef<Self>
                    + ::poulpy_core::layouts::GGSWAtViewMut<Self>
                    + ::poulpy_core::layouts::GGSWInfos,
                L: ::poulpy_core::layouts::LWEToBackendRef<Self> + ::poulpy_core::layouts::LWEInfos,
            {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_execute_prepared_reference::<R, L, _, $algo, Self>(
                    module, res, lwe, key, plan, scratch,
                )
            }
        }
 };
}
/// Explicitly selects the canonical circuit-bootstrap key encrypt sk implementation.
#[macro_export]
macro_rules! impl_bin_fhe_circuit_bootstrapping_key_encrypt_sk_reference {
 ($be:ty, $algo:ty) => {
unsafe impl $crate::oep::CircuitBootstrappingKeyEncryptSkImpl<$algo> for $be {
            fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
            {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_tmp_bytes_reference::<A, _, $algo, Self>(module, infos)
            }
            fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::circuit_bootstrapping::CircuitBootstrappingKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                sk_lwe: &S0,
                sk_glwe: &S1,
                enc_infos: &$crate::circuit_bootstrapping::CircuitBootstrappingEncryptionInfos,
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
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_encrypt_sk_reference::<S0, S1, _, $algo, Self>(
                    module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
                )
            }
        }
 };
}
/// Explicitly selects the canonical circuit-bootstrap key prepared implementation.
#[macro_export]
macro_rules! impl_bin_fhe_circuit_bootstrapping_key_prepared_reference {
 ($be:ty, $algo:ty) => {
unsafe impl $crate::oep::CircuitBootstrappingKeyPreparedImpl<$algo> for $be {
            fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> $crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>
            where
                A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
            {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_prepared_alloc_from_infos_reference::<A, _, $algo, Self>(
                    module, infos,
                )
            }
            fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                A: $crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos,
            {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_prepare_tmp_bytes_reference::<A, _, $algo, Self>(module, infos)
            }
            fn circuit_bootstrapping_key_prepare(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::circuit_bootstrapping::CircuitBootstrappingKeyPrepared<Self::OwnedBuf, $algo, Self>,
                other: &$crate::circuit_bootstrapping::CircuitBootstrappingKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                $crate::reference::circuit_bootstrapping::circuit_bootstrapping_key_prepare_reference::<_, $algo, Self>(module, res, other, scratch)
            }
        }
 };
}
/// Explicitly registers all canonical circuit-bootstrap operations.
#[macro_export]
macro_rules! impl_bin_fhe_circuit_bootstrapping_reference {
    ($be:ty, $algo:ty) => {
        $crate::impl_bin_fhe_circuit_bootstrapping_execute_reference!($be, $algo);
        $crate::impl_bin_fhe_circuit_bootstrapping_key_encrypt_sk_reference!($be, $algo);
        $crate::impl_bin_fhe_circuit_bootstrapping_key_prepared_reference!($be, $algo);
    };
}
