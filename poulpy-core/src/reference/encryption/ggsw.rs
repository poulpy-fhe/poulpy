use poulpy_hal::{
    api::{ModuleN, VecZnxDftBytesOf, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::api::GLWEPlaintextBytesOf;
use crate::layouts::operand_degree;
use crate::{
    GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeGLWEPlaintext,
    api::GLWEFill,
    layouts::{
        GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendRef, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GGSWEncryptSkReference<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos;
}

impl<BE: Backend + HalVecZnxImpl> GGSWEncryptSkReference<BE> for Module<BE>
where
    Self: ModuleN + GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE> + GLWEFill<BE> + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes,
{
    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        operand_degree(self.n(), &[infos.n()]);

        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        lvl_0 + self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ggsw_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos,
    {
        let metadata = crate::component_noise::fresh_sk_noise(sk.to_backend_ref().dist, &res.to_backend_ref());
        {
            assert!(res.rank() == sk.rank(), "the ciphertext rank differs from the secret's rank");
            operand_degree(self.n(), &[res.n(), pt.n().into(), sk.n()]);
            crate::layouts::assert_secret_set(&sk.to_backend_ref().dist);
            assert!(
                scratch.available() >= self.ggsw_encrypt_sk_tmp_bytes_reference(res),
                "scratch.available(): {} < GGSWEncryptSk::ggsw_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.ggsw_encrypt_sk_tmp_bytes_reference(res)
            );

            let base2k: usize = res.base2k().into();
            let rank: usize = res.rank().into();
            let k_digit: usize = res.k_digit().as_usize();
            let (mut tmp_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res);
            let tmp_pt_k = tmp_pt.k().as_usize();

            for row_i in 0..res.dnum().into() {
                crate::gadget::gglwe_gadget_row_plaintext(
                    self,
                    &mut tmp_pt.data,
                    0,
                    base2k,
                    tmp_pt_k,
                    row_i,
                    k_digit,
                    0,
                    &pt.to_backend_ref(),
                    0,
                    &mut scratch_1.borrow(),
                );
                for col_j in 0..rank + 1 {
                    let mut ct = res.at_view_mut(row_i, col_j);
                    self.glwe_fill_mask_from_source(&mut ct, source_xa);
                    self.glwe_encrypt_sk_internal(
                        base2k,
                        ct.k().as_usize(),
                        &mut ct.data,
                        Some((tmp_pt.to_backend_ref(), col_j)),
                        sk,
                        source_xe,
                        &mut scratch_1.borrow(),
                    );
                }
            }
            drop(tmp_pt);
            scratch.wipe(self.ggsw_encrypt_sk_tmp_bytes_reference(res));
        }
        res.set_noise(metadata);
    }
}
