#![allow(clippy::too_many_arguments)]
use crate::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared, CircuitBootstrappingPlan, CircuitBootstrappingPlanLayout,
    },
};
use poulpy_core::layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Public dispatch for circuit execution, reusable plans, and their scratch queries.
/// Backends opt in through the corresponding OEP contract.
pub trait CircuitBootstrappingExecute<BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend,
{
    /// Returns the minimum scratch-space size for constant-encoding circuit
    /// bootstrapping.
    ///
    /// This compatibility estimator predates exponent-mode parameters. Use
    /// [`Self::circuit_bootstrapping_execute_to_exponent_tmp_bytes`] for that
    /// mode.
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos;

    /// Returns the scratch-space size for constant-encoding execution.
    fn circuit_bootstrapping_execute_to_constant_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos;

    /// Returns the scratch-space size for exponent-encoding execution.
    #[allow(clippy::too_many_arguments)]
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
        A: CircuitBootstrappingKeyInfos;

    /// Bootstraps `lwe` into `res`, encoding the plaintext as the constant
    /// term of each GGSW row polynomial.
    ///
    /// `log_domain` controls the number of discrete values representable (the
    /// LUT has `2^log_domain` entries).
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
        L: LWEToBackendRef<BE> + LWEInfos;

    /// Bootstraps `lwe` into `res`, encoding the plaintext in the exponent of
    /// the polynomial variable.
    ///
    /// `log_gap_out` controls the spacing of output coefficients (used in
    /// post-processing to adjust the gap for downstream operations).
    /// Allocate scratch with
    /// [`Self::circuit_bootstrapping_execute_to_exponent_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
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
        L: LWEToBackendRef<BE> + LWEInfos;

    /// Builds the constant-mode LUT on the host and uploads it once.
    fn circuit_bootstrapping_prepare_to_constant<R: GGSWInfos>(
        &self,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>;
    /// Builds the exponent-mode LUT on the host and uploads it once.
    fn circuit_bootstrapping_prepare_to_exponent<R: GGSWInfos>(
        &self,
        log_gap_out: usize,
        res_infos: &R,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>;
    /// Workspace for repeated execution of a prepared plan.
    fn circuit_bootstrapping_execute_prepared_tmp_bytes<A: CircuitBootstrappingKeyInfos>(
        &self,
        plan: &CircuitBootstrappingPlanLayout,
        key: &A,
    ) -> usize;
    /// Executes without rebuilding or transferring its LUT.
    fn circuit_bootstrapping_execute_prepared<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        plan: &CircuitBootstrappingPlan<BE::OwnedBuf>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos;
}
