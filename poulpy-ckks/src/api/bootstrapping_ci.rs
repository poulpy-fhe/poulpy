use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    layouts::{BootstrappingContext, BootstrappingKeysLayout, BootstrappingKeysPrepared, CKKSCiphertextOwned},
};

/// S2C-first bootstrapping with EvalMod on the conjugate-invariant ring (see
/// [`crate::layouts::bootstrapping_ci`]), for a context compiled by
/// [`BootstrappingContext::compile_ci`].
///
/// `wide` is the standard module of degree `2N` of the conversions and `ci` the
/// CI module of degree `N` of EvalMod, whose backend shares the buffers of the
/// standard one; `scratch_ci` serves EvalMod. The output metadata and the key
/// checks at entry are those of
/// [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap), the
/// trace and embed keys included, which fail with
/// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand).
pub trait CKKSBootstrappingCIOps<BE: Backend, BCI: Backend> {
    /// Scratch of [`Self::ckks_bootstrap_ci`]: `(scratch, scratch_ci)`.
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        wide: &Module<BE>,
        ci: &Module<BCI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F, BCI>,
        keys_layout: &BootstrappingKeysLayout,
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
        ctx: &BootstrappingContext<BE, F, BCI>,
        keys: &BootstrappingKeysPrepared<BE::OwnedBuf, BE, BCI>,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<()>;
}
