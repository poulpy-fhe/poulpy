#![allow(clippy::too_many_arguments)]
use crate::{
    api::CircuitBootstrappingExecute,
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared, CircuitBootstrappingPlanLayout},
};
use poulpy_core::layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};
pub(crate) fn circuit_bootstrapping_execute_to_constant_tmp_bytes_derived<R, A, M, BRA, BE>(
    module: &M,
    block_size: usize,
    extension_factor: usize,
    res_infos: &R,
    cbt_infos: &A,
) -> usize
where
    R: GGSWInfos,
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: CircuitBootstrappingExecute<BRA, BE>,
{
    module.circuit_bootstrapping_execute_tmp_bytes(block_size, extension_factor, res_infos, cbt_infos)
}
pub(crate) fn circuit_bootstrapping_execute_to_constant_derived<R, L, M, BRA, BE>(
    module: &M,
    res: &mut R,
    lwe: &L,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    log_domain: usize,
    extension_factor: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: CircuitBootstrappingExecute<BRA, BE>,
{
    let plan = key.prepare_to_constant(module, res, log_domain, extension_factor);
    plan.execute(module, res, lwe, key, scratch);
}
pub(crate) fn circuit_bootstrapping_execute_to_exponent_derived<R, L, M, BRA, BE>(
    module: &M,
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
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: CircuitBootstrappingExecute<BRA, BE>,
{
    let plan = key.prepare_to_exponent(module, log_gap_out, res, log_domain, extension_factor);
    plan.execute(module, res, lwe, key, scratch);
}

pub(crate) fn circuit_bootstrapping_execute_tmp_bytes_derived<R, A, M, BRA, BE>(
    module: &M,
    block_size: usize,
    extension_factor: usize,
    res_infos: &R,
    cbt_infos: &A,
) -> usize
where
    R: GGSWInfos,
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: CircuitBootstrappingExecute<BRA, BE>,
{
    module.circuit_bootstrapping_execute_prepared_tmp_bytes(
        &CircuitBootstrappingPlanLayout {
            output_layout: res_infos.ggsw_layout(),
            log_domain: None,
            log_gap_in: None,
            log_gap_out: None,
            extension_factor,
            block_size,
        },
        cbt_infos,
    )
}
pub(crate) fn circuit_bootstrapping_execute_to_exponent_tmp_bytes_derived<R, A, M, BRA, BE>(
    module: &M,
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
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: CircuitBootstrappingExecute<BRA, BE>,
{
    module.circuit_bootstrapping_execute_prepared_tmp_bytes(
        &CircuitBootstrappingPlanLayout {
            output_layout: res_infos.ggsw_layout(),
            log_domain: Some(log_domain),
            log_gap_in: Some(crate::reference::circuit_bootstrapping::circuit_bootstrapping_log_gap_in(
                res_infos,
                log_domain,
                extension_factor,
            )),
            log_gap_out: Some(log_gap_out),
            extension_factor,
            block_size,
        },
        cbt_infos,
    )
}
