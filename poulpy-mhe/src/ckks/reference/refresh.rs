use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWECopy, GLWEMaskInnerProduct, GLWENormalize, GLWEShift, GLWESub,
    ScratchArenaTakeCore, SmudgingNoise, VecZnxAddSmudging,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxAdd, VecZnxFillUniformSource},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    ckks::layouts::CKKSRefreshShareOwned,
    reference::{GLWEEncToShareMHEProtocolReference, GLWEPatCompressedReference, GLWEShareToEncMHEProtocolReference},
};

pub trait CKKSRefreshMHEProtocolReference<BE: Backend> {
    fn mhe_ckks_refresh_share_gen_tmp_bytes_reference<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_ckks_refresh_share_gen_reference<C, S, E>(
        &self,
        res: &mut CKKSRefreshShareOwned<BE>,
        mask: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: SmudgingNoise,
        enc_infos: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    fn mhe_ckks_refresh_share_aggregate_reference(&self, res: &mut CKKSRefreshShareOwned<BE>, a: &CKKSRefreshShareOwned<BE>);

    fn mhe_ckks_refresh_share_finalize_tmp_bytes_reference<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    fn mhe_ckks_refresh_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &CKKSRefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> CKKSRefreshMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEEncToShareMHEProtocolReference<BE>
        + GLWEShareToEncMHEProtocolReference<BE>
        + GLWEPatCompressedReference<BE>
        + GLWEMaskInnerProduct<BE>
        + GLWESub<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxAddSmudging<BE>
        + VecZnxAdd<BE>,
{
    fn mhe_ckks_refresh_share_gen_tmp_bytes_reference<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        2 * BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(res_infos))
            + self
                .glwe_mask_inner_product_tmp_bytes(ct_infos)
                .max(self.glwe_shift_tmp_bytes(ct_infos.size()))
                .max(self.glwe_shift_tmp_bytes(res_infos.size()))
                .max(self.glwe_normalize_tmp_bytes())
                .max(self.glwe_copy_tmp_bytes(res_infos, ct_infos))
                .max(self.mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference(res_infos, res_infos))
    }

    fn mhe_ckks_refresh_share_gen_reference<C, S, E>(
        &self,
        res: &mut CKKSRefreshShareOwned<BE>,
        mask: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: SmudgingNoise,
        enc_infos: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        let e2s = &mut res.e2s.inner;
        let k = mask.k().as_usize();
        assert!(
            mask.n().as_usize() == self.n() && sk.n() == mask.n() && sk.rank() == mask.rank(),
            "invalid share: secret and mask layouts differ"
        );
        assert!(
            e2s.rank().as_usize() == 0
                && e2s.n() == mask.n()
                && e2s.base2k() == mask.base2k()
                && e2s.k() == mask.k()
                && res.s2e.inner.base2k() == mask.base2k(),
            "invalid share: share and mask layouts differ"
        );
        assert!(
            mask.k() <= res.s2e.inner.k(),
            "invalid share: ciphertext more precise than the output"
        );
        assert!(
            log_bound > 0 && log_bound < k,
            "invalid share: bound outside the ciphertext precision"
        );
        let base2k = e2s.base2k().as_usize();
        flood.assert_valid_for(base2k, k);
        let s2e_infos = res.s2e.inner.glwe_layout();
        let tmp_bytes = self.mhe_ckks_refresh_share_gen_tmp_bytes_reference(mask, &s2e_infos);
        {
            let (mut m, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(mask);
            let (mut pt, scratch_2) = scratch_1.take_glwe_plaintext_scratch(mask);
            let (mut raised, mut scratch_3) = scratch_2.take_glwe_plaintext_scratch(&s2e_infos);
            self.vec_znx_fill_uniform_source(base2k, log_bound, m.data_mut(), 0, source_xm);
            self.glwe_rsh(k - log_bound, &mut m, &mut scratch_3);
            self.glwe_mask_inner_product(&mut pt, mask, sk, &mut scratch_3);
            self.glwe_sub(e2s, &pt, &m);
            self.glwe_normalize_assign(e2s, &mut scratch_3);
            self.vec_znx_add_smudging(
                base2k,
                k,
                GLWEToBackendMut::<BE>::to_backend_mut(e2s).data_mut(),
                0,
                flood,
                source_smudge,
            );
            self.glwe_normalize_assign(e2s, &mut scratch_3);
            self.glwe_copy(&mut raised, &m, &mut scratch_3);
            self.glwe_rsh(s2e_infos.k().as_usize() - k, &mut raised, &mut scratch_3);
            self.mhe_glwe_share_to_enc_share_gen_reference(&mut res.s2e, &raised, sk, seed, enc_infos, source_xe, &mut scratch_3);
        }
        scratch.wipe(tmp_bytes);
    }

    fn mhe_ckks_refresh_share_aggregate_reference(&self, res: &mut CKKSRefreshShareOwned<BE>, a: &CKKSRefreshShareOwned<BE>) {
        self.mhe_glwe_enc_to_share_share_aggregate_reference(&mut res.e2s, &a.e2s);
        self.glwe_pat_compressed_aggregate_assign_reference(&mut res.s2e.inner, &a.s2e.inner);
    }

    fn mhe_ckks_refresh_share_finalize_tmp_bytes_reference<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let raised = BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(res_infos))
            + self
                .glwe_normalize_tmp_bytes()
                .max(self.glwe_shift_tmp_bytes(res_infos.size()));
        raised.max(self.glwe_pat_compressed_finalize_tmp_bytes_reference())
    }

    fn mhe_ckks_refresh_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &CKKSRefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let (e2s, s2e) = (&share.e2s.inner, &share.s2e.inner);
        assert!(
            res.n().as_usize() == self.n()
                && ct.n() == res.n()
                && ct.rank() == res.rank()
                && e2s.rank().as_usize() == 0
                && e2s.n() == res.n()
                && ct.base2k() == res.base2k()
                && e2s.base2k() == res.base2k()
                && e2s.k() == ct.k(),
            "invalid finalization: ciphertext, share and output layouts differ"
        );
        assert!(
            ct.k() <= res.k(),
            "invalid finalization: ciphertext more precise than the output"
        );
        self.glwe_pat_compressed_finalize_reference(res, s2e, scratch);
        let infos = res.glwe_layout();
        let (mut raised, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&infos);
        self.vec_znx_add(
            raised.data_mut(),
            0,
            ct.to_backend_ref().data(),
            0,
            GLWEToBackendRef::<BE>::to_backend_ref(e2s).data(),
            0,
        );
        // The raise keeps the integer, so the sum must first be its centered representative.
        self.glwe_normalize_assign(&mut raised, &mut scratch_1);
        self.glwe_rsh(res.k().as_usize() - ct.k().as_usize(), &mut raised, &mut scratch_1);
        self.glwe_add_assign(res, &raised);
        self.glwe_normalize_assign(res, &mut scratch_1);
    }
}
