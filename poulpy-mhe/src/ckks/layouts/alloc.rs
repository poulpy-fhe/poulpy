use poulpy_core::layouts::GLWEInfos;
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    ckks::layouts::{CKKSRefreshShare, CKKSRefreshShareOwned},
    layouts::MHEModuleAlloc,
};

/// CKKS share allocation on a backend module, default-bodied over
/// [`MHEModuleAlloc`] as it is over core.
pub trait MHECKKSModuleAlloc<BE: Backend>: MHEModuleAlloc<BE> {
    /// A CKKS refresh share of a ciphertext at `ct_infos` into an output at `res_infos`.
    fn ckks_refresh_share_alloc_from_infos<A: GLWEInfos, B: GLWEInfos>(
        &self,
        ct_infos: &A,
        res_infos: &B,
    ) -> CKKSRefreshShareOwned<BE> {
        CKKSRefreshShare {
            e2s: self.glwe_enc_to_share_share_alloc_from_infos(ct_infos),
            s2e: self.glwe_share_to_enc_share_alloc_from_infos(res_infos),
        }
    }
}

impl<BE: Backend> MHECKKSModuleAlloc<BE> for Module<BE> where Self: MHEModuleAlloc<BE> {}
