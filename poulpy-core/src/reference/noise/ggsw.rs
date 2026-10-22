use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSubAssign,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
};

use crate::ScratchArenaTakeCore;
use crate::api::GLWEBytesOf;
use crate::layouts::operand_degree;
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
    pub fn noise_stats<M, BE, S>(
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
    Module<BE>: VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
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
        let n: usize = operand_degree(self.n(), &[infos.n()]);

        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        let lvl_1_glwe_noise: usize = glwe_noise_body_tmp_bytes(self, infos);
        let lvl_1_mul: usize = self.bytes_of_vec_znx_dft(n, 1, infos.size())
            + self.bytes_of_vec_znx_big(n, 1, infos.size())
            + self
                .vec_znx_idft_apply_tmp_bytes()
                .max(self.vec_znx_big_normalize_tmp_bytes());
        let lvl_1: usize = lvl_1_glwe_noise.max(lvl_1_mul).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
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
        let k_digit: usize = res_backend.k_digit().as_usize();
        assert!(
            scratch.available() >= self.ggsw_noise_tmp_bytes(res),
            "scratch.available(): {} < GGSWNoise::ggsw_noise_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_noise_tmp_bytes(res)
        );

        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&res_backend);
        let pt_k: usize = pt.k().as_usize();
        let pt_want_backend: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> =
            ScalarZnx::from_data(BE::from_host_bytes(pt_want.data), pt_want.n(), pt_want.cols());
        {
            let mut pt_backend = pt.to_backend_mut();
            crate::reference::encryption::gadget_row_plaintext(
                self,
                &mut pt_backend.data,
                0,
                base2k,
                pt_k,
                res_row,
                k_digit,
                0,
                &<ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&pt_want_backend),
                0,
                &mut scratch_1.borrow(),
            );
        }

        // mul with sk[col_j-1]
        if res_col > 0 {
            let scratch_mul = scratch_1.borrow();
            let (mut pt_dft, scratch_2) = scratch_mul.take_vec_znx_dft_scratch(res.n().as_usize(), 1, res_backend.size());
            self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.to_backend_ref().data, 0);
            {
                let mut pt_dft_backend = pt_dft.to_backend_mut();
                self.svp_apply_dft_to_dft_assign(&mut pt_dft_backend, 0, &sk_backend.data, res_col - 1);
            }
            let (mut pt_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(res.n().as_usize(), 1, res_backend.size());
            self.vec_znx_idft_apply(&mut pt_big, 0, &pt_dft.to_backend_ref(), 0, &mut scratch_3.borrow());
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
    }
}
