#![allow(clippy::too_many_arguments)]
use crate::{api::*, oep::*};
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
    layouts::{Module, ScratchArena},
    source::Source,
};
impl<BRA: BlindRotationAlgo, BE: CircuitBootstrappingExecuteImpl<BRA>> CircuitBootstrappingExecute<BRA, BE> for Module<BE> {
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_tmp_bytes(
            self,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
    fn circuit_bootstrapping_execute_to_constant_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_to_constant_tmp_bytes(
            self,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
    fn circuit_bootstrapping_execute_to_exponent_tmp_bytes<R, A>(
        &self,
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
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_to_exponent_tmp_bytes(
            self,
            log_gap_out,
            log_domain,
            block_size,
            extension_factor,
            res_infos,
            cbt_infos,
        )
    }
    fn circuit_bootstrapping_execute_to_constant<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_to_constant(
            self,
            res,
            lwe,
            key,
            log_domain,
            extension_factor,
            scratch,
        )
    }
    fn circuit_bootstrapping_execute_to_exponent<R, L>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_to_exponent(
            self,
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
        &self,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf> {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_prepare_to_constant(
            self,
            res_infos,
            key,
            log_domain,
            extension_factor,
        )
    }
    fn circuit_bootstrapping_prepare_to_exponent<R: GGSWInfos>(
        &self,
        log_gap_out: usize,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf> {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_prepare_to_exponent(
            self,
            log_gap_out,
            res_infos,
            key,
            log_domain,
            extension_factor,
        )
    }
    fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: CircuitBootstrappingKeyInfos>(
        &self,
        plan: &CircuitBootstrappingPlanLayout,
        key: &A,
    ) -> usize {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_prepared_tmp_bytes(self, plan, key)
    }
    fn circuit_bootstrapping_execute_prepared<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        plan: &CircuitBootstrappingPlan<BE::OwnedBuf>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        <BE as CircuitBootstrappingExecuteImpl<BRA>>::circuit_bootstrapping_execute_prepared(self, res, lwe, key, plan, scratch)
    }
}
impl<BRA: BlindRotationAlgo, BE: CircuitBootstrappingKeyEncryptSkImpl<BRA>> CircuitBootstrappingKeyEncryptSk<BRA, BE>
    for Module<BE>
{
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
    {
        <BE as CircuitBootstrappingKeyEncryptSkImpl<BRA>>::circuit_bootstrapping_key_encrypt_sk_tmp_bytes(self, infos)
    }
    fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution,
    {
        <BE as CircuitBootstrappingKeyEncryptSkImpl<BRA>>::circuit_bootstrapping_key_encrypt_sk(
            self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
        )
    }
}
impl<BRA: BlindRotationAlgo, BE: CircuitBootstrappingKeyPreparedImpl<BRA>> CircuitBootstrappingKeyPreparedFactory<BRA, BE>
    for Module<BE>
{
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
    {
        <BE as CircuitBootstrappingKeyPreparedImpl<BRA>>::circuit_bootstrapping_key_prepared_alloc_from_infos(self, infos)
    }
    fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
    {
        <BE as CircuitBootstrappingKeyPreparedImpl<BRA>>::circuit_bootstrapping_key_prepare_tmp_bytes(self, infos)
    }
    fn circuit_bootstrapping_key_prepare(
        &self,
        res: &mut CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        <BE as CircuitBootstrappingKeyPreparedImpl<BRA>>::circuit_bootstrapping_key_prepare(self, res, other, scratch)
    }
}
