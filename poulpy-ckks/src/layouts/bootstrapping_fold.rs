//! Fold strategies of the batched bootstrapping pipeline.
use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSLayout,
    layouts::{BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned},
};

/// Folds a batch of input ciphertexts into standard ciphertexts of the
/// bootstrapping module, refreshes them, and unfolds the results.
///
/// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap)
/// refreshes `ins` into `outs` through a fold, which decides how the inputs
/// are merged into the standard ciphertexts each bootstrap refreshes.
pub trait CKKSBootstrapFold<BE: Backend> {
    /// Input and output ciphertexts.
    type Ciphertext: CKKSCtBounds;

    /// Layouts `(refreshed, folded)` of the standard ciphertexts bootstrapped for
    /// outputs like `ct_out` and inputs like `ct_in`.
    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bound of [`Self::refresh`] for outputs like `ct_out` and inputs like
    /// `ct_in` under keys shaped like `keys_layout`, given the scratch bound
    /// `bootstrap_bytes` of one bootstrap.
    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Folds `ins` into standard ciphertexts, refreshes each with `bootstrap`, and
    /// unfolds the results into `outs`, which has the length of `ins`. `keys` are
    /// the bootstrap keys, available to the fold's own steps.
    #[allow(clippy::too_many_arguments)]
    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>;
}

/// Refreshes standard ciphertexts one per bootstrap.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardFold;

impl<BE: Backend> CKKSBootstrapFold<BE> for StandardFold
where
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
{
    type Ciphertext = CKKSCiphertextOwned<BE>;

    fn bootstrap_layouts<C1, C2>(&self, _module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        (
            CKKSLayout {
                glwe_layout: ct_out.glwe_layout(),
                meta: ct_out.meta(),
            },
            CKKSLayout {
                glwe_layout: ct_in.glwe_layout(),
                meta: ct_in.meta(),
            },
        )
    }

    fn tmp_bytes<C1, C2>(
        &self,
        _module: &Module<BE>,
        _ct_out: &C1,
        _ct_in: &C2,
        _keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        bootstrap_bytes
    }

    fn refresh<F, K, B>(
        &self,
        _module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        _ctx: &BootstrappingContext<BE, F>,
        _keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        for (out, input) in outs.iter_mut().zip(ins) {
            bootstrap(out, input, scratch)?;
        }
        Ok(())
    }
}
