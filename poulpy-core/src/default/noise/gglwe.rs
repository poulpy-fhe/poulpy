use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, VecZnxAddScalarAssignBackend, VecZnxBigAddAssign, VecZnxBigBytesOf,
        VecZnxBigFromSmallBackend, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA, VecZnxSubAssignBackend,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats, ZnxZero,
    },
};

use crate::noise::glwe::glwe_noise_backend_inner;
use crate::{
    GLWENormalize,
    api::{GGLWENoise, GLWENoise},
    decryption::GLWEDecrypt,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewRef,
        prepared::GLWESecretPreparedToBackendRef,
    },
};
use crate::{ScratchArenaTakeCore, layouts::GLWEPlaintext};

impl<D: HostDataRef> GGLWE<D> {
    pub fn noise<'s, M, S, BE>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        GGLWE<D>: GGLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GGLWENoise<BE>,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>:
        VecZnxAddScalarAssignBackend<BE> + VecZnxSubAssignBackend<BE> + GLWENoise<BE> + GLWEDecrypt<BE> + GLWENormalize<BE>,
    Module<BE>: ModuleN
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
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_noise_tmp_bytes(infos);

        lvl_0 + lvl_1
    }

    fn gglwe_noise<'s, R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        R: GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= self.gglwe_noise_tmp_bytes(res),
            "scratch.available(): {} < GGLWENoise::gglwe_noise_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_noise_tmp_bytes(res)
        );

        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        let dsize: usize = res_backend.dsize().into();
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&res_backend);
        pt.data_mut().zero();
        let pt_want_backend: ScalarZnx<BE::OwnedBuf> =
            ScalarZnx::from_data(BE::from_host_bytes(pt_want.data), pt_want.n(), pt_want.cols());
        {
            let mut pt_backend = pt.to_backend_mut();
            self.vec_znx_add_scalar_assign_backend(
                &mut pt_backend.data,
                0,
                (dsize - 1) + res_row * dsize,
                &<ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&pt_want_backend),
                res_col,
            );
        }
        let res_at_backend: GLWEViewRef<'_, BE> = res_backend.at_view(res_row, res_col);
        let pt_backend = pt.to_backend_ref();
        glwe_noise_backend_inner(self, &res_at_backend, &pt_backend, &sk_backend, &mut scratch_1)
    }
}
