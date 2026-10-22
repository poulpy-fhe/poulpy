use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    layouts::{BootstrappingCIContext, BootstrappingCIKeysLayout, BootstrappingCIKeysPrepared, CKKSCiphertextOwned},
};

/// S2C-first bootstrapping with EvalMod on the conjugate-invariant ring (see
/// [`crate::layouts::bootstrapping_ci`]), on the standard module of degree `N`.
///
/// `wide` is a standard module of degree `2N`, for the conversions, and `ci` a
/// module of the CI ring of degree `N`, for EvalMod; the CI backend shares the
/// buffers of the standard one. `scratch` serves the standard stages and the
/// conversions, `scratch_ci` EvalMod.
///
/// # Metadata
///
/// As [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) on the
/// base plan: the output keeps the input scale, at the preset output width.
///
/// # Keys
///
/// `ct_out.k()` sets the working width. Before any work, the rotation keys of
/// the linear stages and the CI tensor key must cover the widest ciphertext the
/// bootstrap switches with them at that width, or it fails with
/// [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey)
/// or [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey),
/// whose `source` is the refusal; and the trace and embed keys must cover the
/// conversions, or it fails with
/// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand). The
/// first-factor keys, or a sparse-to-dense key with a row offset, are laid out
/// for one raise: one that does not fit the raise this bootstrap runs fails then
/// too, with an error that says which key and width. So does a working width
/// below what CoeffsToSlots, the conversion and EvalMod consume.
pub trait CKKSBootstrappingCIOps<BE: Backend, BCI: Backend> {
    /// Scratch of [`Self::ckks_bootstrap_ci`]: `(scratch, scratch_ci)`.
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingCIContext<BE, BCI, F>,
        keys_layout: &BootstrappingCIKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    #[expect(
        clippy::too_many_arguments,
        reason = "the CI bootstrap takes three modules and a scratch arena per ring"
    )]
    fn ckks_bootstrap_ci<F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingCIContext<BE, BCI, F>,
        keys: &BootstrappingCIKeysPrepared<BE::OwnedBuf, BE, BCI>,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()>;
}
