use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxAddScalarAssign, VecZnxBigAddAssign, VecZnxBigBytesOf,
        VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA, VecZnxSubAssign,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats, ZnxView,
        ZnxZero,
    },
};

use crate::ScratchArenaTakeCore;
use crate::api::GLWEBytesOf;
use crate::noise::glwe::{glwe_noise_backend_inner, glwe_noise_body_tmp_bytes};
use crate::{
    GLWENormalize,
    api::{GGLWENoise, GLWENoise},
    decryption::GLWEDecrypt,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewRef,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

// Coefficient-word fence: noise measurement ends in `VecZnx::stats`, which is i64-only.
impl<D: HostDataRef> GGLWE<D, i64> {
    pub fn noise<M, S, BE>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        GGLWE<D, BE::ZnxWord>: GGLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GGLWENoise<BE>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssign<BE> + VecZnxSubAssign<BE> + GLWENoise<BE> + GLWEDecrypt<BE> + GLWENormalize<BE>,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        let lvl_1: usize = glwe_noise_body_tmp_bytes(self, infos);

        BE::scratch_aligned(BE::bytes_of_scalar_znx(self.n(), 1)) + BE::scratch_aligned(lvl_0) + lvl_1
    }

    fn gglwe_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let tmp_bytes = self.gglwe_noise_tmp_bytes(res);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GGLWENoise::gglwe_noise_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );

        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        let dsize: usize = res_backend.dsize().into();
        assert!(pt_want.n() <= self.n(), "invalid plaintext: degree exceeds the module's");
        let stats = {
            let (mut pt_want_backend, scratch_1) = scratch.borrow().take_scalar_znx_scratch(pt_want.n(), 1);
            BE::copy_host_to_view(&mut pt_want_backend.data, bytemuck::cast_slice(pt_want.at(res_col, 0)));
            let (mut pt, mut scratch_1) = scratch_1.take_glwe_plaintext_scratch(&res_backend);
            pt.data_mut().zero();
            {
                let mut pt_backend = pt.to_backend_mut();
                self.vec_znx_add_scalar_assign(
                    &mut pt_backend.data,
                    0,
                    (dsize - 1) + res_row * dsize,
                    &pt_want_backend.to_backend_ref(),
                    0,
                );
            }
            let res_at_backend: GLWEViewRef<'_, BE> = res_backend.at_view(res_row, res_col);
            let pt_backend = pt.to_backend_ref();
            glwe_noise_backend_inner(self, &res_at_backend, &pt_backend, &sk_backend, &mut scratch_1)
        };
        scratch.wipe(tmp_bytes);
        stats
    }
}
