use crate::{
    blind_rotation::{BlindRotationAlgo, LookupTable, LookupTableInfos},
    circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared},
};
use poulpy_core::layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWLayout, GGSWToBackendMut, LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CircuitBootstrappingOutput {
    Constant,
    Exponent { log_gap_out: usize },
}

#[derive(Clone, Copy)]
pub(crate) struct CircuitBootstrappingExecutionConfig {
    pub(crate) output: CircuitBootstrappingOutput,
    pub(crate) log_domain: usize,
    pub(crate) log_gap_in: Option<usize>,
    pub(crate) extension_factor: usize,
}

/// Metadata consumed by a prepared circuit-bootstrap scratch query.
///
/// `log_gap_out = None` selects constant output. Its legacy size query has no
/// plaintext-domain argument, so `log_domain = None` requests a bound for every
/// valid domain at the given output layout. An implementation must honor that
/// conservative request. Exponent queries always provide both domain and gaps.
#[derive(Clone, Copy, Debug)]
pub struct CircuitBootstrappingPlanLayout {
    pub output_layout: GGSWLayout,
    pub log_domain: Option<usize>,
    pub log_gap_in: Option<usize>,
    pub log_gap_out: Option<usize>,
    pub extension_factor: usize,
    pub block_size: usize,
}

/// LUT and dimensional state prepared for repeated circuit bootstrapping.
///
/// Preparing a plan performs the host-side LUT construction and uploads it to
/// the selected backend. Executing it only uses the plan, the prepared key,
/// the input/output ciphertexts, and caller-owned scratch space.
pub struct CircuitBootstrappingPlan<D: Data> {
    pub(crate) lut: LookupTable<D, i64>,
    pub(crate) output_layout: GGSWLayout,
    pub(crate) output: CircuitBootstrappingOutput,
    pub(crate) log_domain: usize,
    pub(crate) log_gap_in: usize,
    pub(crate) extension_factor: usize,
    pub(crate) key_layout: crate::circuit_bootstrapping::CircuitBootstrappingKeyLayout,
    pub(crate) block_size: usize,
}

impl<D: Data> CircuitBootstrappingPlan<D> {
    /// Builds a plan from an already uploaded LUT and its circuit metadata.
    ///
    /// `log_gap_out = None` selects constant output; `Some(gap)` selects exponent
    /// output. This constructor allows backend plan-preparation overrides to
    /// reuse their own LUT upload path. The LUT and layout must describe the
    /// canonical circuit for these parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn from_lut(
        lut: LookupTable<D, i64>,
        output_layout: GGSWLayout,
        log_domain: usize,
        log_gap_in: usize,
        log_gap_out: Option<usize>,
        key_layout: crate::circuit_bootstrapping::CircuitBootstrappingKeyLayout,
        block_size: usize,
    ) -> Self {
        use poulpy_core::layouts::LWEInfos;
        assert_eq!(lut.n(), output_layout.n());
        assert_eq!(key_layout.brk_layout.n_glwe, output_layout.n());
        assert_eq!(key_layout.atk_layout.n, output_layout.n());
        assert_eq!(key_layout.tsk_layout.n, output_layout.n());
        assert!(block_size != 0, "circuit-bootstrap block size must be nonzero");
        let extension_factor = lut.extension_factor();
        assert!(extension_factor.is_power_of_two());
        Self {
            lut,
            output_layout,
            output: log_gap_out.map_or(CircuitBootstrappingOutput::Constant, |log_gap_out| {
                CircuitBootstrappingOutput::Exponent { log_gap_out }
            }),
            log_domain,
            log_gap_in,
            extension_factor,
            key_layout,
            block_size,
        }
    }

    /// Uploaded LUT consumed by prepared execution.
    pub fn lut(&self) -> &LookupTable<D, i64> {
        &self.lut
    }
    /// Logarithm of the plaintext domain size.
    pub fn log_domain(&self) -> usize {
        self.log_domain
    }
    /// Input coefficient spacing after blind rotation, in logarithmic form.
    pub fn log_gap_in(&self) -> usize {
        self.log_gap_in
    }
    /// Exponent-mode output spacing, or `None` for constant output.
    pub fn log_gap_out(&self) -> Option<usize> {
        match self.output {
            CircuitBootstrappingOutput::Constant => None,
            CircuitBootstrappingOutput::Exponent { log_gap_out } => Some(log_gap_out),
        }
    }
    /// Blind-rotation extended-domain factor.
    pub fn extension_factor(&self) -> usize {
        self.extension_factor
    }
    /// Expected component key layouts.
    pub fn key_layout(&self) -> crate::circuit_bootstrapping::CircuitBootstrappingKeyLayout {
        self.key_layout
    }
    /// Number of LWE coefficients processed together.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the execution metadata used by backend scratch queries.
    pub fn layout(&self) -> CircuitBootstrappingPlanLayout {
        CircuitBootstrappingPlanLayout {
            output_layout: self.output_layout,
            log_domain: Some(self.log_domain),
            log_gap_in: Some(self.log_gap_in),
            log_gap_out: self.log_gap_out(),
            extension_factor: self.extension_factor,
            block_size: self.block_size,
        }
    }

    /// Returns the output layout this plan was prepared for.
    pub fn output_layout(&self) -> GGSWLayout {
        self.output_layout
    }

    /// Returns the scratch-space size selected by the backend for this plan.
    pub fn execute_tmp_bytes<M, BRA, BE>(&self, module: &M, key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>) -> usize
    where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: CircuitBootstrappingExecute<BRA, BE>,
    {
        self.assert_key_compatible(key);
        module.circuit_bootstrapping_execute_prepared_tmp_bytes(&self.layout(), key)
    }

    /// Executes a previously prepared plan through the selected backend.
    pub fn execute<M, R, L, BRA, BE>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.circuit_bootstrapping_execute_prepared(res, lwe, key, self, scratch);
    }

    pub(crate) fn assert_key_compatible<BRA, BE>(&self, key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>)
    where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
    {
        assert_eq!(
            key.brk_infos(),
            self.key_layout.brk_layout,
            "circuit-bootstrapping plan/BRK mismatch"
        );
        assert_eq!(
            key.atk_infos(),
            self.key_layout.atk_layout,
            "circuit-bootstrapping plan/ATK mismatch"
        );
        assert_eq!(
            key.tsk_infos(),
            self.key_layout.tsk_layout,
            "circuit-bootstrapping plan/TSK mismatch"
        );
        assert_eq!(
            key.block_size(),
            self.block_size,
            "circuit-bootstrapping plan/block-size mismatch"
        );
    }
}

/// Trait for evaluating a complete circuit bootstrapping.
///
/// Re-export of the public API. A module implements it when its backend opts
/// into [`CircuitBootstrappingExecuteImpl`](crate::oep::CircuitBootstrappingExecuteImpl).
pub use crate::api::CircuitBootstrappingExecute;

impl<BRA, BE> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
{
    /// Prepares a reusable constant-encoding circuit-bootstrap plan.
    pub fn prepare_to_constant<M, R>(
        &self,
        module: &M,
        res_infos: &R,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>
    where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWInfos,
    {
        module.circuit_bootstrapping_prepare_to_constant(res_infos, self, log_domain, extension_factor)
    }

    /// Prepares a reusable exponent-encoding circuit-bootstrap plan.
    pub fn prepare_to_exponent<M, R>(
        &self,
        module: &M,
        log_gap_out: usize,
        res_infos: &R,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>
    where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWInfos,
    {
        module.circuit_bootstrapping_prepare_to_exponent(log_gap_out, res_infos, self, log_domain, extension_factor)
    }

    /// Convenience method: bootstraps `lwe` into the GGSW ciphertext `res`
    /// using the constant-term encoding.
    ///
    /// See [`CircuitBootstrappingExecute::circuit_bootstrapping_execute_to_constant`].
    pub fn execute_to_constant<M, L, R>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_constant(res, lwe, self, log_domain, extension_factor, scratch);
    }

    /// Convenience method: bootstraps `lwe` into `res` using the exponent
    /// encoding.
    ///
    /// See [`CircuitBootstrappingExecute::circuit_bootstrapping_execute_to_exponent`]
    /// and its mode-specific scratch estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_to_exponent<R, L, M>(
        &self,
        module: &M,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_exponent(log_gap_out, res, lwe, self, log_domain, extension_factor, scratch);
    }
}
