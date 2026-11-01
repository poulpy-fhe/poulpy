use poulpy_hal::layouts::Module;

use crate::{
    CKKSCtBounds,
    layouts::{BootstrappingContextPrepared, BootstrappingKeysLayout},
    oep::CKKSBootstrappingImpl,
};

pub(crate) fn ckks_bootstrap_mod_up_tmp_bytes_derived<BE, C1, C2, F>(
    module: &Module<BE>,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSBootstrappingImpl,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    BE::ckks_bootstrap_tmp_bytes(module, ct_out, ct_in, ctx, keys_layout)
}
