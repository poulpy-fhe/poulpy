pub mod automorphism;
pub mod encryption;
pub mod external_product;
pub mod glwe_tensor;
pub mod keyswitch;

mod conversion;
mod glwe_packing;
mod rotate;
mod trace;

pub use conversion::*;
pub use glwe_packing::*;
pub use rotate::*;
pub use trace::*;

use crate::oep::{
    AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl, GGSWExternalProductImpl,
    GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEExternalProductImpl, GLWEKeyswitchImpl, GLWEMulConstImpl,
    GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl, GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl,
    GLWESubImpl, GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl,
};
use crate::{
    api::ModuleTransfer,
    layouts::{GGLWE, GGLWEToGGSWKey, GGSW, GLWE, GLWEAutomorphismKey, GLWEPlaintext, GLWESecret},
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{
        Backend, DataView, HostBackend, HostBytesBackend as HB, HostDataMut, HostDataRef, Module, ScalarZnx,
        ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef, ScratchArena, ScratchOwned, VecZnxBackendMut, VecZnxBackendRef,
    },
    oep::HalVecZnxImpl,
    test_suite::TestBackend as HalTestBackend,
    test_suite::{download_scalar_znx as hal_download_scalar_znx, upload_scalar_znx as hal_upload_scalar_znx},
};

use crate::ScratchArenaTakeCore;

pub trait TestBackend:
    HalTestBackend
    + GLWEKeyswitchImpl<Self>
    + GGLWEKeyswitchImpl<Self>
    + GGSWKeyswitchImpl<Self>
    + LWEKeyswitchImpl<Self>
    + GLWEAddImpl<Self>
    + GLWENegateImpl<Self>
    + GLWESubImpl<Self>
    + GLWECopyImpl<Self>
    + HalVecZnxImpl<Self>
    + GLWEExternalProductImpl<Self>
    + GGLWEExternalProductImpl<Self>
    + GGSWExternalProductImpl<Self>
    + GLWETensoringImpl<Self>
    + GLWEMulConstImpl<Self>
    + GLWEMulPlainImpl<Self>
    + GLWERotateImpl<Self>
    + GLWEMulXpMinusOneImpl<Self>
    + GLWEShiftImpl<Self>
    + GLWENormalizeImpl<Self>
    + GLWETraceImpl<Self>
    + GLWEPackImpl<Self>
    + GGSWRotateImpl<Self>
    + DecryptionImpl<Self>
    + ConversionImpl<Self>
    + AutomorphismImpl<Self>
where
    Self: HostBackend<OwnedBuf = Vec<u8>>,
    for<'a> Self::BufRef<'a>: HostDataRef,
    for<'a> Self::BufMut<'a>: HostDataMut,
    for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
{
}

impl<BE> TestBackend for BE
where
    BE: HalTestBackend
        + GLWEKeyswitchImpl<BE>
        + GGLWEKeyswitchImpl<BE>
        + GGSWKeyswitchImpl<BE>
        + LWEKeyswitchImpl<BE>
        + GLWEAddImpl<BE>
        + GLWENegateImpl<BE>
        + GLWESubImpl<BE>
        + GLWECopyImpl<BE>
        + HalVecZnxImpl<BE>
        + GLWEExternalProductImpl<BE>
        + GGLWEExternalProductImpl<BE>
        + GGSWExternalProductImpl<BE>
        + GLWETensoringImpl<BE>
        + GLWEMulConstImpl<BE>
        + GLWEMulPlainImpl<BE>
        + GLWERotateImpl<BE>
        + GLWEMulXpMinusOneImpl<BE>
        + GLWEShiftImpl<BE>
        + GLWENormalizeImpl<BE>
        + GLWETraceImpl<BE>
        + GLWEPackImpl<BE>
        + GGSWRotateImpl<BE>
        + DecryptionImpl<BE>
        + ConversionImpl<BE>
        + AutomorphismImpl<BE>,
    BE: HostBackend<OwnedBuf = Vec<u8>>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
}

pub fn scratch_host_arena<BE: Backend>(scratch: &mut ScratchOwned<BE>) -> ScratchArena<'_, BE>
where
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    scratch.borrow()
}

pub fn upload_scalar_znx<BE: Backend>(
    src: &poulpy_hal::layouts::ScalarZnx<Vec<u8>>,
) -> poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf> {
    hal_upload_scalar_znx::<BE>(src)
}

pub fn download_scalar_znx<BE: Backend>(
    src: &poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf>,
) -> poulpy_hal::layouts::ScalarZnx<Vec<u8>> {
    hal_download_scalar_znx::<BE>(src)
}

pub fn scalar_znx_as_vec_znx_backend_ref<BE: Backend>(src: &ScalarZnx<BE::OwnedBuf>) -> VecZnxBackendRef<'_, BE> {
    <ScalarZnx<BE::OwnedBuf> as ScalarZnxAsVecZnxBackendRef<BE>>::as_vec_znx_backend(src)
}

pub fn scalar_znx_as_vec_znx_backend_mut<BE: Backend>(src: &mut ScalarZnx<BE::OwnedBuf>) -> VecZnxBackendMut<'_, BE> {
    <ScalarZnx<BE::OwnedBuf> as ScalarZnxAsVecZnxBackendMut<BE>>::as_vec_znx_backend_mut(src)
}

pub fn upload_glwe<BE: HostBackend<OwnedBuf = Vec<u8>>>(module: &Module<BE>, src: &GLWE<Vec<u8>>) -> GLWE<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_glwe::<HB>(src)
}

pub fn download_glwe<BE: HostBackend<OwnedBuf = Vec<u8>>>(_module: &Module<BE>, src: &GLWE<BE::OwnedBuf>) -> GLWE<Vec<u8>> {
    let shape = src.data.shape();
    GLWE {
        data: poulpy_hal::layouts::VecZnx::from_data_with_max_size(
            BE::to_host_bytes(&src.data.data),
            shape.n(),
            shape.cols(),
            shape.size(),
            shape.max_size(),
        ),
        base2k: src.base2k,
    }
}

pub fn upload_glwe_plaintext<BE: HostBackend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    src: &GLWEPlaintext<Vec<u8>>,
) -> GLWEPlaintext<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_glwe_plaintext::<HB>(src)
}

pub fn download_glwe_plaintext<BE: HostBackend<OwnedBuf = Vec<u8>>>(
    _module: &Module<BE>,
    src: &GLWEPlaintext<BE::OwnedBuf>,
) -> GLWEPlaintext<Vec<u8>> {
    let shape = src.data.shape();
    GLWEPlaintext {
        data: poulpy_hal::layouts::VecZnx::from_data_with_max_size(
            BE::to_host_bytes(&src.data.data),
            shape.n(),
            shape.cols(),
            shape.size(),
            shape.max_size(),
        ),
        base2k: src.base2k,
    }
}

pub fn upload_glwe_secret<BE: HostBackend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    src: &GLWESecret<Vec<u8>>,
) -> GLWESecret<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_glwe_secret::<HB>(src)
}

pub fn upload_gglwe<BE: HostBackend<OwnedBuf = Vec<u8>>>(module: &Module<BE>, src: &GGLWE<Vec<u8>>) -> GGLWE<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_gglwe::<HB>(src)
}

pub fn upload_ggsw<BE: HostBackend<OwnedBuf = Vec<u8>>>(module: &Module<BE>, src: &GGSW<Vec<u8>>) -> GGSW<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    module.upload_ggsw::<HB>(src)
}

pub fn download_ggsw<BE: HostBackend<OwnedBuf = Vec<u8>>>(_module: &Module<BE>, src: &GGSW<BE::OwnedBuf>) -> GGSW<Vec<u8>> {
    GGSW {
        data: poulpy_hal::layouts::MatZnx::from_data(
            BE::to_host_bytes(src.data.data()),
            src.data.n(),
            src.data.rows(),
            src.data.cols_in(),
            src.data.cols_out(),
            src.data.size(),
        ),
        base2k: src.base2k,
        dsize: src.dsize,
    }
}

pub fn upload_glwe_automorphism_key<BE: HostBackend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    src: &GLWEAutomorphismKey<Vec<u8>>,
) -> GLWEAutomorphismKey<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    GLWEAutomorphismKey {
        key: upload_gglwe(module, &src.key),
        p: src.p,
    }
}

pub fn upload_gglwe_to_ggsw_key<BE: HostBackend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    src: &GGLWEToGGSWKey<Vec<u8>>,
) -> GGLWEToGGSWKey<BE::OwnedBuf>
where
    Module<BE>: ModuleTransfer<BE>,
{
    GGLWEToGGSWKey {
        keys: src.keys.iter().map(|key| upload_gglwe(module, key)).collect(),
    }
}

#[macro_export]
macro_rules! core_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr $(,)?
    ) => {
        poulpy_hal::backend_test_suite!(
            mod $modname,
            backend = $backend,
            params = $params,
            tests = {
                glwe_encrypt_sk => $crate::test_suite::encryption::test_glwe_encrypt_sk,
                glwe_compressed_encrypt_sk => $crate::test_suite::encryption::test_glwe_compressed_encrypt_sk,
                glwe_encrypt_zero_sk => $crate::test_suite::encryption::test_glwe_encrypt_zero_sk,
                glwe_encrypt_pk => $crate::test_suite::encryption::test_glwe_encrypt_pk,
                glwe_base2k_conv => $crate::test_suite::test_glwe_base2k_conversion,
                test_glwe_tensoring => $crate::test_suite::glwe_tensor::test_glwe_tensoring,
                test_glwe_tensor_square => $crate::test_suite::glwe_tensor::test_glwe_tensor_square,
                test_glwe_mul_plain => $crate::test_suite::glwe_tensor::test_glwe_mul_plain,
                test_glwe_mul_const => $crate::test_suite::glwe_tensor::test_glwe_mul_const,
                glwe_keyswitch => $crate::test_suite::keyswitch::test_glwe_keyswitch,
                glwe_keyswitch_assign => $crate::test_suite::keyswitch::test_glwe_keyswitch_assign,
                glwe_automorphism => $crate::test_suite::automorphism::test_glwe_automorphism,
                glwe_automorphism_assign => $crate::test_suite::automorphism::test_glwe_automorphism_assign,
                glwe_external_product => $crate::test_suite::external_product::test_glwe_external_product,
                glwe_external_product_assign => $crate::test_suite::external_product::test_glwe_external_product_assign,
                glwe_rotate => $crate::test_suite::test_glwe_rotate,
                glwe_trace_assign => $crate::test_suite::test_glwe_trace_assign,
                glwe_packing => $crate::test_suite::test_glwe_packing,
                gglwe_switching_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_switching_key_encrypt_sk,
                gglwe_switching_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_switching_key_compressed_encrypt_sk,
                gglwe_compressed_encrypt_sk => $crate::test_suite::encryption::test_gglwe_compressed_encrypt_sk,
                gglwe_automorphism_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_automorphism_key_encrypt_sk,
                gglwe_automorphism_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_automorphism_key_compressed_encrypt_sk,
                gglwe_tensor_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_tensor_key_encrypt_sk,
                gglwe_tensor_key_compressed_encrypt_sk =>
                    $crate::test_suite::encryption::test_gglwe_tensor_key_compressed_encrypt_sk,
                gglwe_to_ggsw_key_encrypt_sk => $crate::test_suite::encryption::test_gglwe_to_ggsw_key_encrypt_sk,
                gglwe_switching_key_keyswitch => $crate::test_suite::keyswitch::test_gglwe_switching_key_keyswitch,
                gglwe_switching_key_keyswitch_assign => $crate::test_suite::keyswitch::test_gglwe_switching_key_keyswitch_assign,
                gglwe_switching_key_external_product =>
                    $crate::test_suite::external_product::test_gglwe_switching_key_external_product,
                gglwe_switching_key_external_product_assign =>
                    $crate::test_suite::external_product::test_gglwe_switching_key_external_product_assign,
                gglwe_automorphism_key_automorphism =>
                    $crate::test_suite::automorphism::test_gglwe_automorphism_key_automorphism,
                gglwe_automorphism_key_automorphism_assign =>
                    $crate::test_suite::automorphism::test_gglwe_automorphism_key_automorphism_assign,
                ggsw_encrypt_sk => $crate::test_suite::encryption::test_ggsw_encrypt_sk,
                ggsw_compressed_encrypt_sk => $crate::test_suite::encryption::test_ggsw_compressed_encrypt_sk,
                ggsw_keyswitch => $crate::test_suite::keyswitch::test_ggsw_keyswitch,
                ggsw_keyswitch_assign => $crate::test_suite::keyswitch::test_ggsw_keyswitch_assign,
                ggsw_external_product => $crate::test_suite::external_product::test_ggsw_external_product,
                ggsw_external_product_assign => $crate::test_suite::external_product::test_ggsw_external_product_assign,
                ggsw_automorphism => $crate::test_suite::automorphism::test_ggsw_automorphism,
                ggsw_automorphism_assign => $crate::test_suite::automorphism::test_ggsw_automorphism_assign,
                lwe_keyswitch => $crate::test_suite::keyswitch::test_lwe_keyswitch,
                glwe_to_lwe => $crate::test_suite::test_glwe_to_lwe,
                lwe_to_glwe => $crate::test_suite::test_lwe_to_glwe,
            }
        );
    };
}

pub use crate::core_backend_test_suite;
