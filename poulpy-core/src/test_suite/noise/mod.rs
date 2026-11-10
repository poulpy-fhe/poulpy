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
mod gadget_copy;
mod glwe_packing;
mod k_digit;
mod rotate;
mod shift;
mod trace;

pub use conversion::*;
pub use gadget_copy::*;
pub use glwe_packing::*;
pub use k_digit::*;
pub use rotate::*;
pub use shift::*;
pub use trace::*;

use crate::oep::{
    AutomorphismImpl, DecryptionImpl, GGLWECopyImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl, GGLWEProductDigitsStridedImpl,
    GGSWConversionImpl, GGSWCopyImpl, GGSWExternalProductImpl, GGSWKeyswitchImpl, GLWEAddImpl, GLWECopyImpl,
    GLWEExternalProductImpl, GLWEKeyswitchImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWENegateImpl, GLWENormalizeImpl,
    GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, SamplingImpl, SecretImpl,
};
use crate::{
    GLWEDecrypt, GLWENoise,
    api::TransferInto,
    layouts::{
        GGLWE, GGLWEAlloc, GGLWEToGGSWKey, GGSW, GGSWAlloc, GLWE, GLWEAlloc, GLWEAutomorphismKey, GLWEInfos, GLWEPlaintext,
        GLWEPlaintextAlloc, GLWESecret, GLWESecretAlloc, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, SetBase2k,
    },
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{
        Backend, DataView, HostBackend, HostDataMut, HostDataRef, HostStaged, Module, Ring, ScalarZnx,
        ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef, ScratchArena, ScratchOwned, Standard, Stats, VecZnxBackendMut,
        VecZnxBackendRef,
    },
    test_suite::TestBackend as HalTestBackend,
    test_suite::{download_scalar_znx as hal_download_scalar_znx, upload_scalar_znx as hal_upload_scalar_znx},
};

pub trait TestBackend<Rg: Ring = Standard>:
    HalTestBackend<Rg>
    + GLWEKeyswitchImpl<Rg>
    + GGLWEKeyswitchImpl<Rg>
    + GGSWKeyswitchImpl<Rg>
    + GLWEAddImpl<Rg>
    + GLWENegateImpl<Rg>
    + GLWESubImpl<Rg>
    + GLWECopyImpl<Rg>
    + GGLWECopyImpl<Rg>
    + GGSWCopyImpl<Rg>
    + GLWEExternalProductImpl<Rg>
    + GGLWEExternalProductImpl<Rg>
    + GGSWExternalProductImpl<Rg>
    + GLWETensoringImpl<Rg>
    + GGLWEProductDigitsStridedImpl<Rg>
    + GLWEMulConstImpl<Rg>
    + GLWEMulPlainImpl<Rg>
    + GLWEShiftImpl<Rg>
    + GLWENormalizeImpl<Rg>
    + DecryptionImpl<Rg>
    + GGSWConversionImpl<Rg>
    + AutomorphismImpl<Rg>
    + SamplingImpl<Rg>
    + SecretImpl<Rg>
where
    Self: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> Self::BufRef<'a>: HostDataRef,
    for<'a> Self::BufMut<'a>: HostDataMut,
{
}

impl<BE, Rg: Ring> TestBackend<Rg> for BE
where
    BE: HalTestBackend<Rg>
        + GLWEKeyswitchImpl<Rg>
        + GGLWEKeyswitchImpl<Rg>
        + GGSWKeyswitchImpl<Rg>
        + GLWEAddImpl<Rg>
        + GLWENegateImpl<Rg>
        + GLWESubImpl<Rg>
        + GLWECopyImpl<Rg>
        + GGLWECopyImpl<Rg>
        + GGSWCopyImpl<Rg>
        + GLWEExternalProductImpl<Rg>
        + GGLWEExternalProductImpl<Rg>
        + GGSWExternalProductImpl<Rg>
        + GLWETensoringImpl<Rg>
        + GGLWEProductDigitsStridedImpl<Rg>
        + GLWEMulConstImpl<Rg>
        + GLWEMulPlainImpl<Rg>
        + GLWEShiftImpl<Rg>
        + GLWENormalizeImpl<Rg>
        + DecryptionImpl<Rg>
        + GGSWConversionImpl<Rg>
        + AutomorphismImpl<Rg>
        + SamplingImpl<Rg>
        + SecretImpl<Rg>,
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
pub fn assert_glwe_flag_honest<BE, R, Rg: Ring>(res: &R)
where
    BE: Backend<ZnxWord = i64>,
    R: GLWEToBackendRef<BE, Rg>,
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

pub fn glwe_decrypt_checked<BE, M, R, P, S, Rg: Ring>(module: &M, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend<ZnxWord = i64>,
    M: GLWEDecrypt<BE, Rg>,
    R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    P: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
    S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
{
    assert_glwe_flag_honest::<BE, _, Rg>(res);
    module.glwe_decrypt(res, pt, sk, scratch);
}

pub fn glwe_noise_checked<BE, M, R, P, S, Rg: Ring>(
    module: &M,
    res: &R,
    pt_want: &P,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) -> Stats
where
    BE: HostBackend<ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    M: GLWENoise<BE, Rg>,
    R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    P: GLWEToBackendRef<BE, Rg>,
    S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
{
    assert_glwe_flag_honest::<BE, _, Rg>(res);
    module.glwe_noise(res, pt_want, sk, scratch)
}

/// Checks that `op` refuses its input as wider than its key decomposes exactly.
///
/// # Panics
///
/// Unless `op` panics with that refusal.
pub fn assert_refuses_uncovered_input(what: &str, op: impl FnOnce()) {
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).expect_err(what);
    let message = crate::test_suite::decryption::panic_message(&*err);
    assert!(
        message.contains("wider than the key's max_input_k"),
        "{what}: panicked with {message:?}"
    );
}

pub fn upload_scalar_znx<BE: Backend, Rg: Ring>(
    src: &poulpy_hal::layouts::ScalarZnx<AlignedBuf, BE::ZnxWord, Rg>,
) -> poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> {
    hal_upload_scalar_znx::<BE, Rg>(src)
}

pub fn download_scalar_znx<BE: Backend, Rg: Ring>(
    src: &poulpy_hal::layouts::ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> poulpy_hal::layouts::ScalarZnx<AlignedBuf, BE::ZnxWord, Rg> {
    hal_download_scalar_znx::<BE, Rg>(src)
}

pub fn scalar_znx_as_vec_znx_backend_ref<BE: Backend, Rg: Ring>(
    src: &ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> VecZnxBackendRef<'_, BE, Rg> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendRef<BE, Rg>>::as_vec_znx_backend(src)
}

pub fn scalar_znx_as_vec_znx_backend_mut<BE: Backend, Rg: Ring>(
    src: &mut ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> VecZnxBackendMut<'_, BE, Rg> {
    <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxAsVecZnxBackendMut<BE, Rg>>::as_vec_znx_backend_mut(src)
}

pub fn upload_glwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GLWE<AlignedBuf, i64, Rg>,
) -> GLWE<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let mut dst = module.glwe_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_glwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>, Rg: Ring>(
    _module: &Module<BE>,
    src: &GLWE<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> GLWE<AlignedBuf, BE::ZnxWord, Rg> {
    let shape = src.data.shape();
    GLWE {
        noise: src.noise.clone(),
        data: poulpy_hal::layouts::VecZnx::from_shape(AlignedBuf::from(BE::to_host_bytes(src.data.data())), shape),
        k: src.k,
        base2k: src.base2k,
        canonical: src.canonical,
        plaintext_meta: src.plaintext_meta,
    }
}

pub fn upload_glwe_plaintext<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GLWEPlaintext<AlignedBuf, i64, Rg>,
) -> GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let mut dst = module.glwe_plaintext_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_glwe_plaintext<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>, Rg: Ring>(
    _module: &Module<BE>,
    src: &GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> GLWEPlaintext<AlignedBuf, BE::ZnxWord, Rg> {
    let shape = src.data.shape();
    GLWEPlaintext {
        data: poulpy_hal::layouts::VecZnx::from_shape(AlignedBuf::from(BE::to_host_bytes(src.data.data())), shape),
        k: src.k,
        plaintext_meta: src.plaintext_meta,
        base2k: src.base2k,
    }
}

pub fn upload_glwe_secret<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GLWESecret<AlignedBuf, i64, Rg>,
) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let mut dst = module.glwe_secret_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn upload_gglwe<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GGLWE<AlignedBuf, i64, Rg>,
) -> GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let mut dst = module.gglwe_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn upload_ggsw<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GGSW<AlignedBuf, i64, Rg>,
) -> GGSW<BE::OwnedBuf, BE::ZnxWord, Rg> {
    let mut dst = module.ggsw_alloc_from_infos(src);
    src.transfer_into(&mut dst);
    dst
}

pub fn download_ggsw<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>, Rg: Ring>(
    _module: &Module<BE>,
    src: &GGSW<BE::OwnedBuf, BE::ZnxWord, Rg>,
) -> GGSW<AlignedBuf, BE::ZnxWord, Rg> {
    GGSW {
        noise: src.noise.clone(),
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
        k_digit: src.k_digit,
    }
}

pub fn upload_glwe_automorphism_key<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GLWEAutomorphismKey<AlignedBuf, i64, Rg>,
) -> GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    GLWEAutomorphismKey {
        key: upload_gglwe(module, &src.key),
        p: src.p,
    }
}

pub fn upload_gglwe_to_ggsw_key<BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostStaged, Rg: Ring>(
    module: &Module<BE>,
    src: &GGLWEToGGSWKey<AlignedBuf, i64, Rg>,
) -> GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    GGLWEToGGSWKey {
        keys: src.keys.iter().map(|key| upload_gglwe(module, key)).collect(),
    }
}
