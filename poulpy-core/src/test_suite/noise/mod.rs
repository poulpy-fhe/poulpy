//! Scheme-correctness suite: each test encrypts, runs an operation, decrypts,
//! and checks the residual noise against the analytic bound for that operation.
//!
//! This judges one backend against a model. It is host-only by construction:
//! every test ends in [`crate::GLWENoise::glwe_noise`], which reads coefficients
//! to compute statistics and is bounded on [`HostBackend`] at the method.
//!
//! A noise bound is a weak oracle. For the complementary check, that a backend
//! agrees with a selected comparison backend byte-for-byte, see [`super::parity`].

pub mod automorphism;
pub mod encryption;
pub mod external_product;
pub mod glwe_tensor;
pub mod keyswitch;
pub mod linear_transformation;

mod conversion;
mod glwe_packing;
mod rotate;
mod shift;
mod trace;

pub use conversion::*;
pub use glwe_packing::*;
pub use rotate::*;
pub use shift::*;
pub use trace::*;

use crate::oep::{
    AutomorphismImpl, ConversionImpl, DecryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl,
    GGLWEProductDigitsStridedImpl, GGSWExternalProductImpl, GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl,
    GLWEExternalProductImpl, GLWEKeyswitchImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl,
    GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl,
    LWEKeyswitchImpl, SamplingImpl,
};
use crate::{
    GLWEDecrypt, GLWENoise,
    api::TransferInto,
    layouts::{
        GGLWE, GGLWEToGGSWKey, GGSW, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWEPlaintext, GLWESecret,
        GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, SetBase2k,
    },
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{
        Backend, DataView, HostBackend, HostDataMut, HostDataRef, HostStaged, Module, ScalarZnx, ScalarZnxAsVecZnxBackendMut,
        ScalarZnxAsVecZnxBackendRef, ScratchArena, ScratchOwned, Stats, VecZnxBackendMut, VecZnxBackendRef,
    },
    oep::HalVecZnxImpl,
    test_suite::TestBackend as HalTestBackend,
    test_suite::{download_scalar_znx as hal_download_scalar_znx, upload_scalar_znx as hal_upload_scalar_znx},
};

pub trait TestBackend:
    HalTestBackend
    + GLWEKeyswitchImpl
    + GGLWEKeyswitchImpl
    + GGSWKeyswitchImpl
    + LWEKeyswitchImpl
    + GLWEAddImpl
    + GLWENegateImpl
    + GLWESubImpl
    + GLWECopyImpl
    + HalVecZnxImpl
    + GLWEExternalProductImpl
    + GGLWEExternalProductImpl
    + GGSWExternalProductImpl
    + GLWETensoringImpl
    + GGLWEProductDigitsStridedImpl
    + GLWEMulConstImpl
    + GLWEMulPlainImpl
    + GLWERotateImpl
    + GLWEMulXpMinusOneImpl
    + GLWEShiftImpl
    + GLWENormalizeImpl
    + GLWETraceImpl
    + GLWEPackImpl
    + GGSWRotateImpl
    + DecryptionImpl
    + ConversionImpl
    + AutomorphismImpl
    + SamplingImpl
where
    Self: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> Self::BufRef<'a>: HostDataRef,
    for<'a> Self::BufMut<'a>: HostDataMut,
{
}

impl<BE> TestBackend for BE
where
    BE: HalTestBackend
        + GLWEKeyswitchImpl
        + GGLWEKeyswitchImpl
        + GGSWKeyswitchImpl
        + LWEKeyswitchImpl
        + GLWEAddImpl
        + GLWENegateImpl
        + GLWESubImpl
        + GLWECopyImpl
        + HalVecZnxImpl
        + GLWEExternalProductImpl
        + GGLWEExternalProductImpl
        + GGSWExternalProductImpl
        + GLWETensoringImpl
        + GGLWEProductDigitsStridedImpl
        + GLWEMulConstImpl
        + GLWEMulPlainImpl
        + GLWERotateImpl
        + GLWEMulXpMinusOneImpl
        + GLWEShiftImpl
        + GLWENormalizeImpl
        + GLWETraceImpl
        + GLWEPackImpl
        + GGSWRotateImpl
        + DecryptionImpl
        + ConversionImpl
        + AutomorphismImpl
        + SamplingImpl,
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
}

pub fn scratch_host_arena<BE: Backend>(scratch: &mut ScratchOwned<BE>) -> ScratchArena<'_, BE>
where
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    scratch.borrow()
}

/// Asserts that a GLWE flagged canonical is canonical at its `k`.
pub fn assert_glwe_flag_honest<BE, R>(res: &R)
where
    BE: Backend<ZnxWord = i64>,
    R: GLWEToBackendRef<BE>,
{
    let res = res.to_backend_ref();
    if !res.is_canonical() {
        return;
    }
    let (n, cols, size) = (res.data.n(), res.data.cols(), res.data.size());
    let mut host = vec![0i64; n * cols * size];
    BE::copy_view_to_host(res.data.data(), bytemuck::cast_slice_mut(&mut host));
    let base2k: usize = res.base2k().into();
    let k: usize = res.k().as_usize();
    let live: usize = k.div_ceil(base2k);
    let pad_mask: i64 = (1i64 << (live * base2k - k)) - 1;
    let half: i64 = 1i64 << (base2k - 1);
    let (mut in_range, mut no_bits_below_k, mut no_limbs_past_k) = (true, true, true);
    for col in 0..cols {
        for limb in 0..size {
            let digits = &host[(limb * cols + col) * n..][..n];
            if limb < live {
                in_range &= digits.iter().all(|digit| (-half..=half).contains(digit));
            } else {
                no_limbs_past_k &= digits.iter().all(|&digit| digit == 0);
            }
            if limb + 1 == live {
                no_bits_below_k &= digits.iter().all(|&digit| digit & pad_mask == 0);
            }
        }
    }
    assert!(in_range, "GLWE flagged canonical holds a digit outside the canonical range");
    assert!(no_bits_below_k, "GLWE flagged canonical holds bits below its k");
    assert!(no_limbs_past_k, "GLWE flagged canonical holds a non-zero limb past its k");
}

pub fn glwe_decrypt_checked<BE, M, R, P, S>(module: &M, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend<ZnxWord = i64>,
    M: GLWEDecrypt<BE>,
    R: GLWEToBackendRef<BE> + GLWEInfos,
    P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    assert_glwe_flag_honest::<BE, _>(res);
    module.glwe_decrypt(res, pt, sk, scratch);
}

pub fn glwe_noise_checked<BE, M, R, P, S>(module: &M, res: &R, pt_want: &P, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
where
    BE: HostBackend<ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    M: GLWENoise<BE>,
    R: GLWEToBackendRef<BE> + GLWEInfos,
    P: GLWEToBackendRef<BE>,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    assert_glwe_flag_honest::<BE, _>(res);
    module.glwe_noise(res, pt_want, sk, scratch)
}

pub fn upload_scalar_znx<BE: Backend>(
    src: &poulpy_hal::layouts::ScalarZnx<AlignedBuf, BE::ZnxWord>,
) -> poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf, BE::ZnxWord> {
    hal_upload_scalar_znx::<BE>(src)
}

pub fn download_scalar_znx<BE: Backend>(
    src: &poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> poulpy_hal::layouts::ScalarZnx<AlignedBuf, BE::ZnxWord> {
    hal_download_scalar_znx::<BE>(src)
}

pub fn scalar_znx_as_vec_znx_backend_ref<BE: Backend>(src: &ScalarZnx<BE::OwnedBuf, BE::ZnxWord>) -> VecZnxBackendRef<'_, BE> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxAsVecZnxBackendRef<BE>>::as_vec_znx_backend(src)
}

pub fn scalar_znx_as_vec_znx_backend_mut<BE: Backend>(
    src: &mut ScalarZnx<BE::OwnedBuf, BE::ZnxWord>,
) -> VecZnxBackendMut<'_, BE> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxAsVecZnxBackendMut<BE>>::as_vec_znx_backend_mut(src)
}

pub fn upload_glwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GLWE<AlignedBuf, i64>,
) -> GLWE<BE::OwnedBuf, BE::ZnxWord> {
    let mut dst = module.glwe_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_glwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    _module: &Module<BE>,
    src: &GLWE<BE::OwnedBuf, BE::ZnxWord>,
) -> GLWE<AlignedBuf, BE::ZnxWord> {
    let shape = src.data.shape();
    GLWE {
        data: poulpy_hal::layouts::VecZnx::from_shape(AlignedBuf::from(BE::to_host_bytes(src.data.data())), shape),
        k: src.k,
        base2k: src.base2k,
        canonical: src.canonical,
    }
}

pub fn upload_glwe_plaintext<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GLWEPlaintext<AlignedBuf, i64>,
) -> GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> {
    let mut dst = module.glwe_plaintext_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_glwe_plaintext<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    _module: &Module<BE>,
    src: &GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>,
) -> GLWEPlaintext<AlignedBuf, BE::ZnxWord> {
    let shape = src.data.shape();
    GLWEPlaintext {
        data: poulpy_hal::layouts::VecZnx::from_shape(AlignedBuf::from(BE::to_host_bytes(src.data.data())), shape),
        k: src.k,
        base2k: src.base2k,
    }
}

pub fn upload_glwe_secret<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GLWESecret<AlignedBuf, i64>,
) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord> {
    let mut dst = module.glwe_secret_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn upload_gglwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GGLWE<AlignedBuf, i64>,
) -> GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    let mut dst = module.gglwe_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn upload_ggsw<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GGSW<AlignedBuf, i64>,
) -> GGSW<BE::OwnedBuf, BE::ZnxWord> {
    let mut dst = module.ggsw_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_ggsw<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    _module: &Module<BE>,
    src: &GGSW<BE::OwnedBuf, BE::ZnxWord>,
) -> GGSW<AlignedBuf, BE::ZnxWord> {
    GGSW {
        data: poulpy_hal::layouts::MatZnx::from_data(
            AlignedBuf::from(BE::to_host_bytes(src.data.data())),
            src.data.n(),
            src.data.rows(),
            src.data.cols_in(),
            src.data.cols_out(),
            src.data.size(),
        ),
        k_aux: src.k_aux,
        base2k: src.base2k,
        dsize: src.dsize,
    }
}

pub fn upload_glwe_automorphism_key<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GLWEAutomorphismKey<AlignedBuf, i64>,
) -> GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> {
    GLWEAutomorphismKey {
        key: upload_gglwe(module, &src.key),
        p: src.p,
    }
}

pub fn upload_gglwe_to_ggsw_key<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged>(
    module: &Module<BE>,
    src: &GGLWEToGGSWKey<AlignedBuf, i64>,
) -> GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord> {
    GGLWEToGGSWKey {
        keys: src.keys.iter().map(|key| upload_gglwe(module, key)).collect(),
    }
}
