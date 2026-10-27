//! PAT and share allocation on a backend module.

use poulpy_core::layouts::{GGLWELayout, GGSWLayout, GLWELayout};
use poulpy_hal::{
    AlignedBuf,
    layouts::{HostBackend, Module},
};

use super::fixtures::{K, assert_panics_with, gglwe_layout, ggsw_layout, glwe_layout_at};
use crate::layouts::MHEModuleAlloc;

/// Every `MHEModuleAlloc::*_alloc_from_infos` refuses infos of half the module degree.
pub fn test_mhe_module_alloc_rejects_foreign_degree<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>,
{
    let n = (module.n() / 2).into();
    let glwe = GLWELayout {
        n,
        ..glwe_layout_at(module, K)
    };
    let gglwe = GGLWELayout {
        n,
        ..gglwe_layout(module)
    };
    let ggsw = GGSWLayout {
        n,
        ..ggsw_layout(module)
    };
    let allocators: [&dyn Fn(); 12] = [
        &|| drop(module.glwe_pat_compressed_alloc_from_infos(&glwe)),
        &|| drop(module.gglwe_pat_compressed_alloc_from_infos(&gglwe)),
        &|| drop(module.gglwe_pat_alloc_from_infos(&gglwe)),
        &|| drop(module.glwe_public_key_share_alloc_from_infos(&glwe)),
        &|| drop(module.glwe_switching_key_share_alloc_from_infos(&gglwe)),
        &|| drop(module.glwe_automorphism_key_share_alloc_from_infos(&gglwe)),
        &|| drop(module.glwe_tensor_key_share_alloc_from_infos(&gglwe)),
        &|| drop(module.ggsw_share_alloc_from_infos(&ggsw)),
        &|| drop(module.glwe_private_keyswitch_share_alloc_from_infos(&glwe)),
        &|| drop(module.glwe_public_keyswitch_share_alloc_from_infos(&glwe)),
        &|| drop(module.glwe_enc_to_share_share_alloc_from_infos(&glwe)),
        &|| drop(module.glwe_share_to_enc_share_alloc_from_infos(&glwe)),
    ];
    for allocate in allocators {
        assert_panics_with("invalid layout: degree differs from the module's", allocate);
    }
}
