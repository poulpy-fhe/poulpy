use poulpy_core::{
    EncryptionInfos, GGLWECompressedEncryptSk, GLWEBytesOf, GLWEKeyswitch, GLWEMaskFill, GLWENormalize, GetDistribution,
    ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeed, GGLWECompressedToBackendRef, GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWELayout,
        LWEInfos, Rank,
        prepared::{
            GGLWEPreparedToBackendRef, GLWESecretPreparedExtract, GLWESecretPreparedFactory, GLWESecretPreparedToBackendRef,
        },
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxCopy, VecZnxNegate, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxZero,
    },
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut},
    source::Source,
};

use crate::layouts::{GGSWShareOwned, ggsw_share_part_layout};

pub trait GGSWMHEProtocolReference<BE: Backend> {
    fn mhe_ggsw_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_ggsw_share_gen_reference<P, S, U, E>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    fn mhe_ggsw_share_finalize_tmp_bytes_reference<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos;

    fn mhe_ggsw_share_finalize_reference<R, K>(
        &self,
        res: &mut R,
        share: &GGSWShareOwned<BE>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GGSWMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWESecretPreparedFactory<BE>
        + GLWESecretPreparedExtract<BE>
        + GGLWECompressedEncryptSk<BE>
        + GLWEBytesOf<BE>
        + GLWEKeyswitch<BE>
        + GLWEMaskFill<BE>
        + GLWENormalize<BE>
        + VecZnxAddAssign<BE>
        + VecZnxCopy<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxNegate<BE>
        + VecZnxZero<BE>,
{
    fn mhe_ggsw_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        let col0 = ggsw_share_part_layout(infos, infos.rank());
        let circ = ggsw_share_part_layout(infos, Rank(1));
        BE::scratch_aligned(self.glwe_secret_prepared_bytes_of(Rank(1)))
            + BE::scratch_aligned(BE::bytes_of_scalar_znx(infos.n().as_usize(), 1))
            + self
                .gglwe_compressed_encrypt_sk_tmp_bytes(&col0)
                .max(self.gglwe_compressed_encrypt_sk_tmp_bytes(&circ))
    }

    fn mhe_ggsw_share_gen_reference<P, S, U, E>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        assert!(
            res.n().as_usize() == self.n(),
            "invalid share: degree differs from the module's"
        );
        assert!(
            sk.n() == res.n(),
            "invalid share: secret degree differs from the transcript's"
        );
        assert!(
            u.n() == res.n(),
            "invalid share: ephemeral secret degree differs from the transcript's"
        );
        let pt_ref = pt.to_backend_ref();
        assert!(
            pt_ref.n() == res.n().as_usize(),
            "invalid share: message degree differs from the transcript's"
        );
        assert!(pt_ref.cols() == 1, "invalid share: message must have one column");
        let rank = res.rank();
        assert!(sk.rank() == rank, "invalid share: secret rank differs from the transcript's");
        assert!(u.rank() == Rank(1), "invalid share: ephemeral secret rank differs from 1");
        let n = res.n();
        let mut seeds = Source::new(seed);
        let (mut sk_j, scratch_1) = scratch.borrow().take_glwe_secret_prepared_scratch(self, Rank(1));
        let (mut zero, mut scratch_2) = scratch_1.take_scalar_znx_scratch(n.as_usize(), 1);
        self.vec_znx_zero(&mut scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut zero), 0);
        self.gglwe_compressed_encrypt_sk(&mut res.col0, pt, sk, seeds.new_seed(), enc_infos, source_xe, &mut scratch_2);
        for j in 0..rank.as_usize() {
            // The two halves of column j + 1 share their masks, hence the seed.
            let seed_j = seeds.new_seed();
            self.gglwe_compressed_encrypt_sk(&mut res.circ_u[j], pt, u, seed_j, enc_infos, source_xe, &mut scratch_2);
            self.glwe_secret_prepared_extract(&mut sk_j, sk, j);
            self.gglwe_compressed_encrypt_sk(&mut res.circ_s[j], &zero, &sk_j, seed_j, enc_infos, source_xe, &mut scratch_2);
        }
    }

    fn mhe_ggsw_share_finalize_tmp_bytes_reference<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos,
    {
        assert!(
            res_infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        assert!(key_infos.n() == res_infos.n(), "invalid layout: key and GGSW degrees differ");
        let tmp = rank_one_layout(res_infos);
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&tmp))
            + self
                .glwe_keyswitch_tmp_bytes(res_infos, &tmp, key_infos)
                .max(self.glwe_normalize_tmp_bytes())
                .max(self.vec_znx_normalize_tmp_bytes())
    }

    fn mhe_ggsw_share_finalize_reference<R, K>(
        &self,
        res: &mut R,
        share: &GGSWShareOwned<BE>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        assert!(
            res.n().as_usize() == self.n(),
            "invalid finalization: degree differs from the module's"
        );
        assert!(key.n() == share.n(), "invalid finalization: key and GGSW degrees differ");
        assert!(
            res.ggsw_layout() == share.ggsw_layout(),
            "invalid finalization: layouts differ"
        );
        assert!(
            key.rank_in() == Rank(1),
            "invalid finalization: key input rank differs from 1"
        );
        assert!(
            key.rank_out() == share.rank(),
            "invalid finalization: key output rank differs from the GGSW's"
        );
        assert!(
            key.dnum().as_usize() * key.dsize().as_usize() * key.base2k().as_usize() >= share.k().as_usize(),
            "invalid finalization: key does not cover the GGSW precision"
        );
        let (dnum, rank) = (share.dnum().as_usize(), share.rank().as_usize());
        let key = key.to_backend_ref();
        let mut res = res.to_backend_mut();
        let (mut tmp, mut scratch_1) = scratch.borrow().take_glwe_scratch(&rank_one_layout(share));
        {
            let col0 = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.col0);
            let (base2k, k): (usize, usize) = (share.base2k().into(), share.k().into());
            for row in 0..dnum {
                let mut cell = res.at_view_mut(row, 0);
                cell.set_canonical(true);
                self.vec_znx_normalize(
                    cell.data_mut(),
                    base2k,
                    k,
                    0,
                    0,
                    col0.at_view(row, 0).data(),
                    base2k,
                    0,
                    &mut scratch_1,
                );
                // Seeded masks are uniform digits, already canonical.
                self.fill_glwe_mask_from_seed(&mut cell, share.col0.seed()[row]);
            }
        }
        // tmp's mask column stays zero for the whole loop; only the body is rewritten per row.
        self.vec_znx_zero(tmp.data_mut(), 0);
        for j in 1..=rank {
            let circ_u = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.circ_u[j - 1]);
            let circ_s = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.circ_s[j - 1]);
            for row in 0..dnum {
                // (0, -b2) decrypts under u to u * (a s_j - e2).
                self.vec_znx_negate(tmp.data_mut(), 1, circ_s.at_view(row, 0).data(), 0);
                let mut cell = res.at_view_mut(row, j);
                self.glwe_keyswitch(&mut cell, &tmp, &key, &mut scratch_1);
                self.vec_znx_add_assign(cell.data_mut(), j, circ_u.at_view(row, 0).data(), 0);
                self.glwe_normalize_assign(&mut cell, &mut scratch_1);
            }
        }
    }
}

fn rank_one_layout<A: GLWEInfos>(infos: &A) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: infos.k(),
        rank: Rank(1),
    }
}
