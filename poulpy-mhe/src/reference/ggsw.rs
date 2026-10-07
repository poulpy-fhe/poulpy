use poulpy_core::{
    ComponentNoise, FreshNoiseEstimate, GGLWECompressedEncryptSk, GLWEBytesOf, GLWEEncryptSk, GLWEKeyswitch, GLWEMaskFill,
    GLWENormalize, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeed, GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos,
        GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWELayout, LWEInfos, Rank, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWESecretPreparedFactory, GLWESecretPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxCopy, VecZnxNegate, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxZero,
    },
    layouts::{
        Backend, Module, ScalarZnxToBackendRef, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut,
        vec_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::layouts::{GGSWShareOwned, ggsw_share_part_layout};

pub trait GGSWMHEProtocolReference<BE: Backend> {
    fn mhe_ggsw_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_ggsw_share_gen_reference<P, S, U>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

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
        + GLWEEncryptSk<BE>
        + GGLWECompressedEncryptSk<BE>
        + GLWEBytesOf<BE>
        + GLWEKeyswitch<BE>
        + GLWEMaskFill<BE>
        + GLWENormalize<BE>
        + VecZnxAddAssign<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxNormalizeAssign<BE>
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
        let rank = infos.rank();
        let col0 = ggsw_share_part_layout(infos, Rank(1), rank);
        let circ = ggsw_share_part_layout(infos, rank, rank);
        let n = infos.n().as_usize();
        // The zero plaintext of `circ_s`, then the `rank + 1` mask holders and the row plaintext of `circ_u`.
        BE::scratch_aligned(BE::bytes_of_scalar_znx(n, rank.as_usize()))
            + (rank.as_usize() + 1) * BE::scratch_aligned(self.glwe_bytes_of_from_infos(&circ))
            + BE::scratch_aligned(BE::bytes_of_vec_znx(n, 1, circ.size()))
            + self
                .gglwe_compressed_encrypt_sk_tmp_bytes(&col0)
                .max(self.gglwe_compressed_encrypt_sk_tmp_bytes(&circ))
                .max(self.glwe_encrypt_sk_tmp_bytes(&circ))
                .max(self.vec_znx_normalize_tmp_bytes())
    }

    fn mhe_ggsw_share_gen_reference<P, S, U>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
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
        assert!(
            u.rank() == rank,
            "invalid share: ephemeral secret rank differs from the secret's"
        );
        let tmp_bytes: usize = self.mhe_ggsw_share_gen_tmp_bytes_reference(&*res);
        {
            let (n, r) = (res.n().as_usize(), rank.as_usize());
            let (dnum, dsize, base2k) = (res.dnum().as_usize(), res.dsize().as_usize(), res.base2k().as_usize());
            let circ = ggsw_share_part_layout(&*res, rank, rank);
            let k = circ.k().as_usize();
            let mut seeds = Source::new(seed);
            let (mut zero, scratch_1) = scratch.borrow().take_scalar_znx_scratch(n, r);
            let (mut mask, mut rest) = scratch_1.take_glwe_scratch(&circ);
            let mut cols = Vec::with_capacity(r);
            for _ in 0..r {
                let (ct, scratch) = rest.take_glwe_scratch(&circ);
                cols.push(ct);
                rest = scratch;
            }
            let (mut row_pt, mut scratch_2) = rest.take_glwe_plaintext_scratch(&circ);
            for col in 0..r {
                self.vec_znx_zero(&mut scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut zero), col);
            }
            self.gglwe_compressed_encrypt_sk(&mut res.col0, pt, sk, seeds.new_seed(), source_xe, &mut scratch_2);
            for j in 0..r {
                // Entry (row, i) of `circ_s[j]` encrypts zero under `sk` over row `i` of
                // the gadget row's common mask matrix; entry (row, l) of `circ_u[j]`
                // encrypts the message (at `l == j`, zero elsewhere) under `u` over its column `l`.
                self.gglwe_compressed_encrypt_sk(&mut res.circ_s[j], &zero, sk, seeds.new_seed(), source_xe, &mut scratch_2);
                let (circ_u, circ_s) = (&mut res.circ_u[j], &res.circ_s[j]);
                circ_u.seed_mut().copy_from_slice(circ_s.seed());
                GGLWECompressedToBackendMut::<BE>::set_noise(
                    circ_u,
                    Some(ComponentNoise::from_secret_at(*u.to_backend_ref().dist(), circ_u.k(), r)),
                );
                for row in 0..dnum {
                    for i in 0..r {
                        self.fill_glwe_mask_from_seed(&mut mask, circ_s.seed()[row * r + i]);
                        for (l, col) in cols.iter_mut().enumerate() {
                            self.vec_znx_copy(col.data_mut(), i + 1, &vec_znx_backend_ref_from_mut::<BE>(mask.data()), l + 1);
                        }
                    }
                    for (l, col) in cols.iter_mut().enumerate() {
                        self.vec_znx_zero(row_pt.data_mut(), 0);
                        if l == j {
                            self.vec_znx_add_scalar_assign(row_pt.data_mut(), 0, (dsize - 1) + row * dsize, &pt_ref, 0);
                            self.vec_znx_normalize_assign(base2k, k, 0, row_pt.data_mut(), 0, &mut scratch_2);
                        }
                        self.glwe_encrypt_sk_with_mask(col, &row_pt, u, source_xe, &mut scratch_2);
                        let mut dst = GGLWECompressedToBackendMut::<BE>::to_backend_mut(circ_u);
                        self.vec_znx_copy(
                            dst.at_view_mut(row, l).data_mut(),
                            0,
                            &vec_znx_backend_ref_from_mut::<BE>(col.data()),
                            0,
                        );
                    }
                }
            }
        }
        scratch.wipe(tmp_bytes);
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
        let tmp = glwe_layout(res_infos);
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
            key.rank_in() == share.rank(),
            "invalid finalization: key input rank differs from the GGSW's"
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
        assert!(
            match (share.noise(), key.noise()) {
                (Some(share), Some(key)) => share.same_secret(&key),
                _ => true,
            },
            "invalid finalization: output key provenance differs"
        );
        let metadata = fresh_ggsw_metadata::<BE, K>(share, key);
        let key = key.to_backend_ref();
        let mut res_be = res.to_backend_mut();
        let (mut tmp, mut scratch_1) = scratch.borrow().take_glwe_scratch(&glwe_layout(share));
        {
            let col0 = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.col0);
            let (base2k, k): (usize, usize) = (share.base2k().into(), share.k().into());
            for row in 0..dnum {
                let mut cell = res_be.at_view_mut(row, 0);
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
        // tmp's body stays zero; its masks are rewritten per entry.
        self.vec_znx_zero(tmp.data_mut(), 0);
        for j in 1..=rank {
            let circ_u = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.circ_u[j - 1]);
            let circ_s = GGLWECompressedToBackendRef::<BE>::to_backend_ref(&share.circ_s[j - 1]);
            for row in 0..dnum {
                // (0, -b_1, .., -b_r) over the `circ_s` bodies decrypts under `u` to the
                // cross term that the `circ_u` bodies in the mask columns cancel.
                for i in 0..rank {
                    self.vec_znx_negate(tmp.data_mut(), i + 1, circ_s.at_view(row, i).data(), 0);
                }
                let mut cell = res_be.at_view_mut(row, j);
                self.glwe_keyswitch(&mut cell, &tmp, &key, &mut scratch_1);
                for l in 0..rank {
                    self.vec_znx_add_assign(cell.data_mut(), l + 1, circ_u.at_view(row, l).data(), 0);
                }
                self.glwe_normalize_assign(&mut cell, &mut scratch_1);
            }
        }
        drop(res_be);
        res.set_noise(metadata);
    }
}

fn glwe_layout<A: GLWEInfos>(infos: &A) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: infos.k(),
        rank: infos.rank(),
    }
}

/// Retain each component's largest variance across gadget columns. The first
/// column carries direct body noise. Other columns carry E_s * U in the body
/// and E_u in each mask, followed by switching-key and rounding errors.
fn fresh_ggsw_metadata<BE: Backend, K: GGLWEInfos>(share: &GGSWShareOwned<BE>, key: &K) -> Option<ComponentNoise> {
    let metadata = share.noise()?;
    let k = share.k();
    let rank = share.rank().as_usize();
    let mut components: Vec<_> = metadata
        .components()
        .iter()
        .map(|noise| FreshNoiseEstimate::new(noise.variance_at(k), k))
        .collect();
    if rank == 0 {
        return Some(metadata.with_components(components));
    }
    let n = share.n().as_usize();
    let digit_bits = key.dsize().as_usize() * key.base2k().as_usize();
    let digit_factor = (1.0 - (-(digit_bits as f64)).exp2()) / (1.0 - (-(key.base2k().as_usize() as f64)).exp2());
    let digits = k.as_usize().div_ceil(digit_bits).min(key.dnum().as_usize());
    // Bounded balanced digits require no uniform-digit assumption. Fold their
    // squared magnitude into the precision rescaling to avoid a vanishing
    // key variance times an overflowing digit bound when the product is finite.
    // Full key coverage means there is no gadget truncation residue.
    let key_noise = key.noise();
    let switching_precision = k
        .as_usize()
        .checked_add(digit_bits - 1)
        .and_then(|precision| u32::try_from(precision).ok())
        .map(TorusPrecision);
    let switching_components: Vec<_> = (0..=rank)
        .map(|component| {
            let variance = key_noise.as_ref().map_or(f64::INFINITY, |noise| {
                let estimate = noise.components()[component];
                if estimate.variance() == 0.0 {
                    0.0
                } else {
                    switching_precision.map_or(f64::INFINITY, |precision| estimate.variance_at(precision))
                }
            });
            rank as f64 * n as f64 * digits as f64 * variance * digit_factor.powi(2)
        })
        .collect();
    let rounding = if key.k() > k { 0.25 } else { 0.0 };
    for (circ_s, circ_u) in share.circ_s.iter().zip(&share.circ_u) {
        let (body, mask) = match (circ_s.noise(), circ_u.noise()) {
            (Some(s), Some(u)) => {
                let ephemeral_second_moment = u.secret_distribution().coefficient_second_moment(n).unwrap_or(f64::INFINITY);
                let v_s = s.phase_noise(n).variance_at(k);
                let v_u = u.phase_noise(n).variance_at(k);
                let e_s_times_u = if v_s == 0.0 || ephemeral_second_moment == 0.0 {
                    0.0
                } else {
                    rank as f64 * n as f64 * v_s * ephemeral_second_moment
                };
                (e_s_times_u, v_u)
            }
            _ => (f64::INFINITY, f64::INFINITY),
        };
        for (component, (noise, switching)) in components.iter_mut().zip(&switching_components).enumerate() {
            let circular = if component == 0 { body } else { mask };
            *noise = FreshNoiseEstimate::new(noise.variance().max(circular + switching + rounding), k);
        }
    }
    Some(metadata.with_components(components))
}
