use poulpy_hal::{
    api::{
        ModuleN, VecZnxAddScalarAssignBackend, VecZnxDftBytesOf, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
        VecZnxZeroBackend,
    },
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::api::GLWEBytesOf;
use crate::{
    EncryptionInfos, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeCore,
    encryption::glwe::GLWEMaskFillDefault,
    layouts::{
        GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GGSWEncryptSkDefault<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_default<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos;
}

impl<BE: Backend> GGSWEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + GLWEMaskFillDefault<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxZeroBackend<BE>,
{
    fn ggsw_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        lvl_0 + self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ggsw_encrypt_sk_default<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos,
    {
        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n());
        assert_eq!(sk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.ggsw_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GGSWEncryptSk::ggsw_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_encrypt_sk_tmp_bytes_default(res)
        );

        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let (mut tmp_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res);

        for row_i in 0..res.dnum().into() {
            self.vec_znx_zero_backend(&mut tmp_pt.data, 0);
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            {
                let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                self.vec_znx_add_scalar_assign_backend(
                    &mut tmp_pt_backend.data,
                    0,
                    (dsize - 1) + row_i * dsize,
                    &pt.to_backend_ref(),
                    0,
                );
            }

            self.vec_znx_normalize_assign_backend(base2k, &mut tmp_pt.data, 0, &mut scratch_1.borrow());
            for col_j in 0..rank + 1 {
                let mut ct = res.at_view_mut(row_i, col_j);
                self.fill_glwe_mask_from_source_default(base2k, &mut ct, 1, rank, source_xa);
                self.glwe_encrypt_sk_internal(
                    base2k,
                    &mut ct.data,
                    Some((tmp_pt.to_backend_ref(), col_j)),
                    sk,
                    enc_infos,
                    source_xe,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}
