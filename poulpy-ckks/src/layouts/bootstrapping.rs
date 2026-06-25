//! Bootstrapping parameterization.
//!
//! [`BootstrappingPlan`] bundles the parameters of the three homomorphic
//! sub-circuits that make up CKKS bootstrapping — CoeffsToSlots (an `Encode`
//! [`DFTPlan`]), EvalMod (an [`EvalModPlan`]) and SlotsToCoeffs (a `Decode`
//! [`DFTPlan`]).
//!
//! Only the per-stage circuit descriptions live here. The torus widths
//! (`base2k`, the encoding scale `log_delta`, and the input/raised moduli) are
//! per-ciphertext metadata supplied at call time — ModUp reads them from its
//! source and destination ciphertexts — and the message ratio is already part of
//! the [`EvalModPlan`]. See
//! [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) for the ModUp
//! semantics (a digit shift in the base-`2^base2k` torus model, not an RNS
//! prime-basis extension).
//!
//! This crate provides no orchestrator: the plan is a parameter bundle the
//! caller reads while composing `ModUp → CoeffsToSlots → EvalMod →
//! SlotsToCoeffs` from the respective op traits, passing the relinearization and
//! rotation keys to the stages that need them.
//!
//! [`BootstrappingContext`] is the *compiled* form of a [`BootstrappingPlan`]:
//! the prepared, backend-resident homomorphic DFT matrices and the encoded,
//! uploaded EvalMod, built once and reused across bootstraps.

use anyhow::Result;
use poulpy_core::{
    ModuleTransfer,
    default::linear_transformation::DiagonalProd,
    layouts::{Base2K, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ModuleNew, NegacyclicFFT, NegacyclicFFTNew},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCtBounds, CKKSInfos,
    api::DFTOps,
    default::dft::matrices::DftScalar,
    encoding::reim::Encoder,
    layouts::{
        CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode,
        Encode, EvalMod, EvalModPlan, Split,
    },
};

/// End-to-end parameterization of CKKS bootstrapping.
///
/// See the [module docs](self) for the overall role.
#[derive(Clone, Debug)]
pub struct BootstrappingPlan {
    /// Hamming weight of the ephemeral **sparse** secret used to encapsulate the
    /// ModUp (sparse-secret encapsulation, <https://eprint.iacr.org/2022/024>).
    ///
    /// `0` disables the trick. Otherwise the pipeline key-switches the ciphertext
    /// from the dense secret to a sparse secret of this weight *before* ModUp and
    /// back to the dense secret *after* — `denseToSparse → ModUp → sparseToDense`.
    /// Under a sparse key the integer wrap-around `I·q` exposed by ModUp is bounded
    /// by this weight instead of the dense weight, so EvalMod can use a much
    /// smaller interval `K` (`f_mod_interval`) with negligible failure probability.
    /// The two key-switching keys are generated from the dense secret (see the
    /// reference composition); they are not part of the secret-independent
    /// [`BootstrappingContext`].
    pub ephemeral_secret_weight: usize,

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`](crate::layouts::DFTType)).
    pub coeffs_to_slots: DFTPlan,

    /// CoeffsToSlots high precision for bypass.
    pub coeffs_to_slots_bypass: Option<DFTPlan>,

    /// EvalMod: approximate `x mod 1`. EvalMod runs at its own
    /// ([`EvalModPlan::f_mod_log_delta`]) scale — `ckks_eval_mod` sets the
    /// ciphertext to it on entry and restores the input scale on exit (a pure,
    /// budget-neutral reinterpretation), so it can keep more `ct×ct` precision.
    pub eval_mod: EvalModPlan,

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`](crate::layouts::DFTType)).
    pub slots_to_coeffs: DFTPlan,
}

impl BootstrappingPlan {
    /// Total `log_budget` bits the pipeline consumes: the two DFT stages plus
    /// EvalMod (charged at its own `f_mod_log_delta` scale; the surrounding
    /// set-scale round-trip is budget-neutral).
    pub fn consumed_bits(&self) -> usize {
        self.coeffs_to_slots.consumed_bits() + self.eval_mod.consumed_bits() + self.slots_to_coeffs.consumed_bits()
    }

    /// Distinct Galois elements the pipeline's rotation keys must cover: the union
    /// of the CoeffsToSlots and SlotsToCoeffs transforms
    /// ([`DFTPlan::galois_elements`]), for a ring of degree `2^log_n` and the given
    /// `cyclotomic_order`. The split forward transform's conjugation key is
    /// separate (generate it from Galois element `−1`) and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
        set.extend(self.slots_to_coeffs.galois_elements(log_n, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// Compiled [`BootstrappingPlan`]: the resident DFT matrices and encoded EvalMod
/// the bootstrapping pipeline evaluates.
///
/// Built once by [`Self::compile`] and reused across bootstraps. The
/// `SplitRealAndImag` format is used for both transforms so the real and
/// imaginary coefficient halves can be reduced independently by EvalMod.
pub struct BootstrappingContext<BE: Backend, F> {
    /// Prepared CoeffsToSlots matrix (homomorphic encoding), pre-scaled by `1/K`.
    pub coeffs_to_slots: DFTMatrixPrepared<BE, Encode, Split>,

    /// Prepared bypass CoeffsToSlots matrix
    pub coeffs_to_slots_bypass: Option<DFTMatrixPrepared<BE, Encode, Split>>,

    /// Prepared SlotsToCoeffs matrix (homomorphic decoding).
    pub slots_to_coeffs: DFTMatrixPrepared<BE, Decode, Split>,

    /// Encoded, backend-resident EvalMod (`x mod 1`).
    pub eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>,
}

impl<BE: Backend, F> BootstrappingContext<BE, F>
where
    F: CKKSScalar + DftScalar,
{
    /// Compiles `plan` into resident matrices and an uploaded EvalMod.
    ///
    /// Each stage carries its own coefficient metadata (`DFTPlan::meta` /
    /// `EvalModPlan::meta`), read directly off the plan. The CoeffsToSlots matrix
    /// is scaled by `1/f_mod_interval` (see the struct docs).
    pub fn compile<E>(
        module: &Module<BE>,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        BE: TransferFrom<HostBytesBackend>,
        Module<BE>: DFTOps<BE> + ModuleTransfer<BE>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
        CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        let c2s_lt: DFTMatrix<BE, Encode, Split> =
            module.ckks_new_dft_matrix(host_module, encoder, base2k, &plan.coeffs_to_slots, scratch)?;
        let coeffs_to_slots = module.ckks_prepare_dft_matrix(&c2s_lt, scratch);

        let s2c_lt: DFTMatrix<BE, Decode, Split> =
            module.ckks_new_dft_matrix(host_module, encoder, base2k, &plan.slots_to_coeffs, scratch)?;
        let slots_to_coeffs = module.ckks_prepare_dft_matrix(&s2c_lt, scratch);

        let host_eval_mod = EvalMod::<F, _>::from_literal(base2k, plan.eval_mod, host_module)?;
        let eval_mod =
            host_eval_mod.map_plaintexts(|pt| CKKSPlaintext::from_inner(module.upload_glwe_plaintext(&pt.inner), pt.meta()));

        let coeffs_to_slots_bypass = if let Some(bypass) = &plan.coeffs_to_slots_bypass {
            let c2s_lt: DFTMatrix<BE, Encode, Split> =
                module.ckks_new_dft_matrix(host_module, encoder, base2k, bypass, scratch)?;

            Some(module.ckks_prepare_dft_matrix(&c2s_lt, scratch))
        } else {
            None
        };

        Ok(Self {
            coeffs_to_slots,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod,
        })
    }
}
