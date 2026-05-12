use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmallBackend, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxSubAssignBackend,
    },
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, Stats},
};

use crate::{
    GLWENormalize, ScratchArenaTakeCore,
    api::GLWENoise,
    decryption::{GLWEDecrypt, glwe_decrypt_backend_inner},
    layouts::{
        GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub(crate) fn glwe_noise_backend_inner<'s, M, BE>(
    module: &M,
    res_backend: &GLWEBackendRef<'_, BE>,
    pt_want_backend: &GLWEBackendRef<'_, BE>,
    sk_backend: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) -> Stats
where
    M: GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxSubAssignBackend<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    BE: HostBackend,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert!(
        scratch.available() >= module.glwe_noise_tmp_bytes(res_backend),
        "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
        scratch.available(),
        module.glwe_noise_tmp_bytes(res_backend)
    );

    let (mut pt_have, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res_backend);
    {
        let mut pt_have_backend = pt_have.to_backend_mut();
        glwe_decrypt_backend_inner(module, res_backend, &mut pt_have_backend, sk_backend, &mut scratch_1);
    }
    {
        let mut pt_have_backend = pt_have.to_backend_mut();
        module.vec_znx_sub_assign_backend(&mut pt_have_backend.data, 0, &pt_want_backend.data, 0);
    }
    let pt_base2k = pt_have.base2k();
    module.glwe_normalize_assign(&mut pt_have, &mut scratch_1);
    pt_have.data.stats(pt_base2k.into(), 0)
}

impl<BE: Backend + HostBackend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxSubAssignBackend<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_normalize_tmp_bytes().max(self.glwe_decrypt_tmp_bytes(infos));

        lvl_0 + lvl_1
    }

    fn glwe_noise<'s, R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'s, BE>) -> Stats
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_backend = res.to_backend_ref();
        let pt_want_backend = pt_want.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        glwe_noise_backend_inner(self, &res_backend, &pt_want_backend, &sk_backend, scratch)
    }
}
