use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmall, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VecZnxNormalizeTmpBytes, VecZnxSubAssign,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, Ring, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats,
    },
    oep::HalVecZnxImpl,
};

use crate::ScratchArenaTakeGLWEPlaintext;
use crate::api::GLWEPlaintextBytesOf;
use crate::noise::glwe::{glwe_noise_backend_inner, glwe_noise_body_tmp_bytes};
use crate::{
    GLWENormalize,
    api::{GGLWENoise, GLWENoise},
    decryption::GLWEDecrypt,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GLWEViewRef, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

// Coefficient-word fence: noise measurement ends in `VecZnx::stats`, which is i64-only.
impl<D: HostDataRef, Rg: Ring> GGLWE<D, i64, Rg> {
    pub fn noise_stats<M, S, BE>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8], i64, Rg>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        GGLWE<D, BE::ZnxWord, Rg>: GGLWEToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
        M: GGLWENoise<BE, Rg>,
        BE: HostBackend<ZnxWord = i64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend + HalVecZnxImpl<Rg>, Rg: Ring> GGLWENoise<BE, Rg> for Module<BE>
where
    Module<BE>:
        VecZnxNormalizeTmpBytes + VecZnxSubAssign<BE, Rg> + GLWENoise<BE, Rg> + GLWEDecrypt<BE, Rg> + GLWENormalize<BE, Rg>,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE, Rg>
        + VecZnxDftApply<BE, Rg>
        + SvpApplyDftToDftAssign<BE, Rg>
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAddAssign<BE, Rg>
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        let lvl_1: usize = glwe_noise_body_tmp_bytes(self, infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    fn gglwe_noise<R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8], i64, Rg>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        R: GGLWEToBackendRef<BE, Rg> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
        BE: HostBackend<ZnxWord = i64>,
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
        let base2k: usize = res_backend.base2k().into();
        let k_digit: usize = res_backend.k_digit().as_usize();
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&res_backend);
        let pt_k: usize = pt.k().as_usize();
        let pt_want_backend: ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> =
            ScalarZnx::from_data(BE::from_host_bytes(pt_want.data), pt_want.n(), pt_want.cols());
        {
            let mut pt_backend = pt.to_backend_mut();
            crate::gadget::gglwe_gadget_row_plaintext(
                self,
                &mut pt_backend.data,
                0,
                base2k,
                pt_k,
                res_row,
                k_digit,
                res_backend.k_offset().as_usize(),
                &<ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendRef<BE, Rg>>::to_backend_ref(&pt_want_backend),
                res_col,
                &mut scratch_1.borrow(),
            );
        }
        let res_at_backend: GLWEViewRef<'_, BE, Rg> = res_backend.at_view(res_row, res_col);
        let pt_backend = pt.to_backend_ref();
        glwe_noise_backend_inner(self, &res_at_backend, &pt_backend, &sk_backend, &mut scratch_1)
    }
}
