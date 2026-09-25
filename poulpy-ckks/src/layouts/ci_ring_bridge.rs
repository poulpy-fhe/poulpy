use poulpy_core::{GLWEKeyswitch, GLWENormalize, layouts::GLWETensorKeyPrepared};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, ScratchArena, Standard};

use crate::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSBootstrappingOps, CKKSImagOps},
    layouts::{BootstrappingKeys, CKKSCiphertextOwned, CKKSModuleAlloc},
};

/// A CI module of degree N and its standard-ring bridge of degree 2N.
/// Coefficient buffers share storage; prepared keys and transforms stay on
/// their original backend.
pub struct CIRingBridge<'a, CI: Backend, BE: Backend> {
    ci: &'a Module<CI>,
    standard: &'a Module<BE>,
}

impl<'a, CI, BE> CIRingBridge<'a, CI, BE>
where
    BE: Backend<Ring = Standard>,
    CI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    /// Validates the degree-doubling relation.
    pub fn new(ci: &'a Module<CI>, standard: &'a Module<BE>) -> Result<Self> {
        crate::ckks_ensure!(
            ci.n().checked_mul(2) == Some(standard.n()),
            "bridge requires standard degree 2N"
        );
        Ok(Self { ci, standard })
    }

    /// The CI module used for ciphertext inputs and outputs.
    pub fn ci(&self) -> &'a Module<CI> {
        self.ci
    }

    /// The standard module used for the bootstrapping context and keys.
    pub fn standard(&self) -> &'a Module<BE> {
        self.standard
    }
}

impl<CI, BE> CIRingBridge<'_, CI, BE>
where
    BE: Backend<ZnxWord = i64>,
    CI: Backend<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    Module<BE>:
        GLWEKeyswitch<BE> + GLWENormalize<BE> + CKKSAddOps<BE> + CKKSImagOps<BE> + CKKSBootstrappingOps<BE> + CKKSModuleAlloc<BE>,
{
    /// Scratch bound for single and pair CI bootstraps, allocated on the standard backend.
    pub fn bootstrap_tmp_bytes<F>(
        &self,
        ct_out: &CKKSCiphertextOwned<CI>,
        ct_in: &CKKSCiphertextOwned<CI>,
        ctx: &crate::layouts::CIBootstrappingContext<BE, F>,
        keys_layout: &crate::layouts::CIBootstrappingKeysLayout,
    ) -> usize
    where
        Module<CI>: GLWENormalize<CI>,
    {
        crate::reference::ci_bootstrapping::ckks_ci_bootstrap_tmp_bytes_reference(
            self.standard,
            self.ci,
            ct_out,
            ct_in,
            ctx,
            keys_layout,
        )
    }

    /// Refreshes one CI ciphertext through the degree-doubled standard ring.
    /// S2C-first without EvalRound+ evaluates EvalMod once. Inputs and outputs
    /// use the CI module; the context and evaluation keys use the standard backend.
    /// The context must use full-slot standard transforms, including for sparse inputs.
    /// Conversion is internal, and the input scale and slot count are preserved.
    /// Reserve `plan.bootstrap_k(output_k + 1, input.log_delta())` bits in the
    /// output allocation. The return key must also cover the retained output
    /// scale: `output_k + 1 + c2s_guard_bits` for S2C-first, or
    /// `output_k + 1 + f_mod_log_delta - input.log_delta()` for C2S-first.
    pub fn bootstrap<F, K, S>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<CI>,
        ct_in: &CKKSCiphertextOwned<CI>,
        ctx: &crate::layouts::CIBootstrappingContext<BE, F>,
        keys: &crate::layouts::CIBootstrappingKeys<K, S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        for<'a> CI::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        for<'a> CI::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
        Module<CI>: GLWENormalize<CI> + CKKSModuleAlloc<CI>,
        for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
        F: Sync,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
        S: poulpy_core::layouts::GGLWEPreparedToBackendRef<BE> + poulpy_core::layouts::GGLWEInfos,
    {
        crate::reference::ci_bootstrapping::ckks_ci_bootstrap_reference(
            self.standard,
            self.ci,
            ct_out,
            None,
            ct_in,
            None,
            ctx,
            keys,
            scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Explicitly packs two CI ciphertexts into one standard bootstrap and
    /// extracts the two real results. Inputs must share their layout and metadata;
    /// outputs must share their layout. This evaluates both nonlinear branches.
    pub fn bootstrap_pair<F, K, S>(
        &self,
        left_out: &mut CKKSCiphertextOwned<CI>,
        right_out: &mut CKKSCiphertextOwned<CI>,
        left_in: &CKKSCiphertextOwned<CI>,
        right_in: &CKKSCiphertextOwned<CI>,
        ctx: &crate::layouts::CIBootstrappingContext<BE, F>,
        keys: &crate::layouts::CIBootstrappingKeys<K, S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        for<'a> CI::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        for<'a> CI::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
        Module<CI>: GLWENormalize<CI> + CKKSModuleAlloc<CI>,
        for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
        F: Sync,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
        S: poulpy_core::layouts::GGLWEPreparedToBackendRef<BE> + poulpy_core::layouts::GGLWEInfos,
    {
        crate::reference::ci_bootstrapping::ckks_ci_bootstrap_reference(
            self.standard,
            self.ci,
            left_out,
            Some(right_out),
            left_in,
            Some(right_in),
            ctx,
            keys,
            scratch,
        )
    }
}
