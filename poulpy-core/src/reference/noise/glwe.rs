use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmall, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxSubAssign,
    },
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, Stats},
};

use crate::{
    GLWENormalize, ScratchArenaTakeCore,
    api::GLWENoise,
    decryption::{GLWEDecrypt, glwe::glwe_decrypt_body_tmp_bytes, glwe_decrypt_backend_inner},
    layouts::{
        GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

// Coefficient-word fence: ends in `VecZnx::stats`, which is i64-only.
pub(crate) fn glwe_noise_backend_inner<M, BE>(
    module: &M,
    res_backend: &GLWEBackendRef<'_, BE>,
    pt_want_backend: &GLWEBackendRef<'_, BE>,
    sk_backend: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Stats
where
    M: GLWEBytesOf<BE>
        + GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxSubAssign<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    BE: HostBackend<ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert!(
        scratch.available() >= glwe_noise_body_tmp_bytes(module, res_backend),
        "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
        scratch.available(),
        glwe_noise_body_tmp_bytes(module, res_backend)
    );

    let (mut pt_have, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res_backend);
    {
        let mut pt_have_backend = pt_have.to_backend_mut();
        glwe_decrypt_backend_inner(module, res_backend, &mut pt_have_backend, sk_backend, &mut scratch_1);
    }
    {
        let mut pt_have_backend = pt_have.to_backend_mut();
        module.vec_znx_sub_assign(&mut pt_have_backend.data, 0, &pt_want_backend.data, 0);
    }
    let pt_base2k = pt_have.base2k();
    module.glwe_normalize_assign(&mut pt_have, &mut scratch_1);
    pt_have.data.stats(pt_base2k.into(), 0)
}

impl<BE: Backend + HostBackend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxSubAssign<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(infos)) + glwe_noise_body_tmp_bytes(self, infos)
    }

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= self.glwe_noise_tmp_bytes(res),
            "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
            scratch.available(),
            self.glwe_noise_tmp_bytes(res)
        );
        let (mut res_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(res);
        let res = if res.is_canonical() {
            res.to_backend_ref()
        } else {
            self.glwe_normalize(&mut res_tmp, res, &mut scratch.borrow());
            res_tmp.to_backend_ref()
        };
        let pt_want_backend = pt_want.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        glwe_noise_backend_inner(self, &res, &pt_want_backend, &sk_backend, &mut scratch)
    }
}

pub(crate) fn glwe_noise_body_tmp_bytes<M, BE, A>(module: &M, infos: &A) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + GLWENormalize<BE> + ModuleN + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    A: GLWEInfos,
{
    let lvl_0: usize = module.glwe_plaintext_bytes_of_from_infos(infos);
    let lvl_1: usize = module
        .glwe_normalize_tmp_bytes()
        .max(glwe_decrypt_body_tmp_bytes::<M, _>(module, infos));

    lvl_0 + lvl_1
}
