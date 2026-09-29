use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWENormalize, GLWEShift, ScratchArenaTakeCore, SmudgingInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::VecZnxAdd,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    layouts::GLWERefreshShareOwned,
    reference::{GLWEEncToShareMHEProtocolReference, GLWEPatCompressedReference, GLWEShareToEncMHEProtocolReference},
};

pub trait GLWERefreshMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_refresh_share_gen_tmp_bytes_reference<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_refresh_share_gen_reference<C, S, E1, E2>(
        &self,
        res: &mut GLWERefreshShareOwned<BE>,
        ct: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: &E1,
        enc_infos: &E2,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    fn mhe_glwe_refresh_share_aggregate_reference(&self, res: &mut GLWERefreshShareOwned<BE>, a: &GLWERefreshShareOwned<BE>);

    fn mhe_glwe_refresh_share_finalize_tmp_bytes_reference<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    fn mhe_glwe_refresh_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWERefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWERefreshMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEEncToShareMHEProtocolReference<BE>
        + GLWEShareToEncMHEProtocolReference<BE>
        + GLWEPatCompressedReference<BE>
        + GLWEAdd<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
        + VecZnxAdd<BE>,
{
    fn mhe_glwe_refresh_share_gen_tmp_bytes_reference<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + self
                .mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference(ct_infos)
                .max(self.mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference(res_infos, ct_infos))
    }

    fn mhe_glwe_refresh_share_gen_reference<C, S, E1, E2>(
        &self,
        res: &mut GLWERefreshShareOwned<BE>,
        ct: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: &E1,
        enc_infos: &E2,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos,
    {
        let (mut mask, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.mhe_glwe_enc_to_share_share_gen_reference(
            &mut res.e2s,
            &mut mask,
            ct,
            sk,
            log_bound,
            flood,
            source_xm,
            source_xe,
            &mut scratch_1,
        );
        self.mhe_glwe_share_to_enc_share_gen_reference(&mut res.s2e, &mask, sk, seed, enc_infos, source_xe, &mut scratch_1);
    }

    fn mhe_glwe_refresh_share_aggregate_reference(&self, res: &mut GLWERefreshShareOwned<BE>, a: &GLWERefreshShareOwned<BE>) {
        self.mhe_glwe_enc_to_share_share_aggregate_reference(&mut res.e2s, &a.e2s);
        self.glwe_pat_compressed_aggregate_assign_reference(&mut res.s2e.inner, &a.s2e.inner);
    }

    fn mhe_glwe_refresh_share_finalize_tmp_bytes_reference<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let raised = BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(res_infos))
            + self
                .glwe_normalize_tmp_bytes()
                .max(self.glwe_shift_tmp_bytes(res_infos.size()));
        raised.max(self.glwe_pat_compressed_finalize_tmp_bytes_reference())
    }

    fn mhe_glwe_refresh_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWERefreshShareOwned<BE>,
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
