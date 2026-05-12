use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxAddScalarAssignBackend, VecZnxBigAddAssign, VecZnxBigBytesOf,
        VecZnxBigFromSmallBackend, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA, VecZnxSubAssignBackend,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, ZnxZero,
    },
};

use crate::layouts::{GGSW, GGSWInfos, GGSWToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GLWEViewRef, LWEInfos};
use crate::noise::glwe::glwe_noise_backend_inner;
use crate::{
    GLWENormalize,
    api::{GGSWNoise, GLWENoise},
    decryption::GLWEDecrypt,
    layouts::prepared::GLWESecretPreparedToBackendRef,
};
use crate::{ScratchArenaTakeCore, layouts::GLWEPlaintext};

impl<D: HostDataRef> GGSW<D> {
    pub fn noise<'s, M, BE, S>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        GGSW<D>: GGSWToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
        M: GGSWNoise<BE>,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.ggsw_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGSWNoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssignBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSubAssignBackend<BE>
        + GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1_glwe_noise: usize = self.glwe_noise_tmp_bytes(infos);
        let lvl_1_mul: usize = self.bytes_of_vec_znx_dft(1, infos.size())
            + self.bytes_of_vec_znx_big(1, infos.size())
            + self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_glwe_noise.max(lvl_1_mul);

        lvl_0 + lvl_1
    }

    fn ggsw_noise<'s, R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        R: GGSWToBackendRef<BE> + GGSWInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();

        let base2k: usize = res_backend.base2k().into();
        let dsize: usize = res_backend.dsize().into();
        assert!(
            scratch.available() >= self.ggsw_noise_tmp_bytes(res),
            "scratch.available(): {} < GGSWNoise::ggsw_noise_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_noise_tmp_bytes(res)
        );

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
                0,
            );
        }

        // mul with sk[col_j-1]
        if res_col > 0 {
            let scratch_mul = scratch_1.borrow();
            let (mut pt_dft, scratch_2) = scratch_mul.take_vec_znx_dft_scratch(self, 1, res_backend.size());
            self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.to_backend_ref().data, 0);
            {
                let mut pt_dft_backend = pt_dft.to_backend_mut();
                self.svp_apply_dft_to_dft_assign(&mut pt_dft_backend, 0, &sk_backend.data, res_col - 1);
            }
            let (mut pt_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(self, 1, res_backend.size());
            {
                let mut pt_big_backend = pt_big.to_backend_mut();
                let mut pt_dft_backend = pt_dft.to_backend_mut();
                self.vec_znx_idft_apply_tmpa(&mut pt_big_backend, 0, &mut pt_dft_backend, 0);
            }
            {
                let mut pt_backend = pt.to_backend_mut();
                self.vec_znx_big_normalize(
                    &mut pt_backend.data,
                    base2k,
                    0,
                    0,
                    &pt_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch_3,
                );
            }
        }

        let res_at_backend: GLWEViewRef<'_, BE> = res_backend.at_view(res_row, res_col);
        let pt_backend = pt.to_backend_ref();
        glwe_noise_backend_inner(self, &res_at_backend, &pt_backend, &sk_backend, &mut scratch_1)
    }
}
