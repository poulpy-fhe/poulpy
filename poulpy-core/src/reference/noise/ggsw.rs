use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxAddScalarAssign, VecZnxBigAddAssign, VecZnxBigBytesOf,
        VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA, VecZnxSubAssign,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, ZnxView, ZnxZero,
    },
};

use crate::ScratchArenaTakeCore;
use crate::api::GLWEBytesOf;
use crate::layouts::{GGSW, GGSWInfos, GGSWToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GLWEViewRef, LWEInfos};
use crate::noise::glwe::{glwe_noise_backend_inner, glwe_noise_body_tmp_bytes};
use crate::{
    GLWENormalize,
    api::{GGSWNoise, GLWENoise},
    decryption::GLWEDecrypt,
    layouts::prepared::GLWESecretPreparedToBackendRef,
};

// Coefficient-word fence: noise measurement ends in `VecZnx::stats`, which is i64-only.
impl<D: HostDataRef> GGSW<D, i64> {
    pub fn noise<M, BE, S>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        GGSW<D, BE::ZnxWord>: GGSWToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
        M: GGSWNoise<BE>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.ggsw_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGSWNoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssign<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSubAssign<BE>
        + GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        let lvl_1_glwe_noise: usize = glwe_noise_body_tmp_bytes(self, infos);
        let lvl_1_mul: usize = self.bytes_of_vec_znx_dft(self.n(), 1, infos.size())
            + self.bytes_of_vec_znx_big(self.n(), 1, infos.size())
            + self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_glwe_noise.max(lvl_1_mul);

        BE::scratch_aligned(BE::bytes_of_scalar_znx(self.n(), 1)) + BE::scratch_aligned(lvl_0) + lvl_1
    }

    fn ggsw_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGSWToBackendRef<BE> + GGSWInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();

        let base2k: usize = res_backend.base2k().into();
        let res_k = res_backend.k().as_usize();
        let dsize: usize = res_backend.dsize().into();
        let tmp_bytes = self.ggsw_noise_tmp_bytes(res);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GGSWNoise::ggsw_noise_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );

        assert!(pt_want.n() <= self.n(), "invalid plaintext: degree exceeds the module's");
        let stats = {
            let (mut pt_want_backend, scratch_1) = scratch.borrow().take_scalar_znx_scratch(pt_want.n(), 1);
            BE::copy_host_to_view(&mut pt_want_backend.data, bytemuck::cast_slice(pt_want.at(0, 0)));
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

            // mul with sk[col_j-1]
            if res_col > 0 {
                let scratch_mul = scratch_1.borrow();
                let (mut pt_dft, scratch_2) = scratch_mul.take_vec_znx_dft_scratch(self.n(), 1, res_backend.size());
                self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.to_backend_ref().data, 0);
                {
                    let mut pt_dft_backend = pt_dft.to_backend_mut();
                    self.svp_apply_dft_to_dft_assign(&mut pt_dft_backend, 0, &sk_backend.data, res_col - 1);
                }
                let (mut pt_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(self.n(), 1, res_backend.size());
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
                        res_k,
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
        };
        scratch.wipe(tmp_bytes);
        stats
    }
}
