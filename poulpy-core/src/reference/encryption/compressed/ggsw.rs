#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarAssign, VecZnxCopy, VecZnxNormalizeAssign, VecZnxZero},
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

use crate::{
    EncryptionInfos, GGSWNoise, ScratchArenaTakeCore,
    encryption::{GGSWEncryptSk, GLWEEncryptSkInternal, glwe::GLWEMaskFillReference},
    layouts::{
        GGSWCompressedSeedMut, GGSWInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, compressed::GGSWCompressedToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GGSWCompressedEncryptSkReference<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

/// Independently callable portable composition for [`GGSWCompressedEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GGSWCompressedEncryptSkComposition<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

impl<BE: Backend> GGSWCompressedEncryptSkComposition<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GGSWEncryptSk<BE>
        + GGSWNoise<BE>
        + GLWEMaskFillReference<BE>
        + VecZnxCopy<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxZero<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let full_ct = self.bytes_of_vec_znx(self.n(), infos.rank().as_usize() + 1, infos.size());
        self.ggsw_encrypt_sk_tmp_bytes(infos) + full_ct
    }

    fn ggsw_compressed_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;
        let dsize: usize = res.dsize().into();

        let sk_ref = sk.to_backend_ref();
        let pt_backend = pt.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(pt_backend.n(), self.n());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.ggsw_compressed_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < GGSWCompressedEncryptSk::ggsw_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_compressed_encrypt_sk_tmp_bytes_composition(res)
        );

        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.dnum().as_usize() * (res.rank().as_usize() + 1)];

        {
            let mut res = res.to_backend_mut();

            let scratch = scratch.borrow();
            let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext_scratch(&res);
            let tmp_pt_k = tmp_pt.k().as_usize();

            let mut source = Source::new(seed_xa);

            for row_i in 0..res.dnum().into() {
                self.vec_znx_zero(&mut tmp_pt.data, 0);

                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                {
                    let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                    self.vec_znx_add_scalar_assign(&mut tmp_pt_backend.data, 0, (dsize - 1) + row_i * dsize, &pt_backend, 0);
                }
                scratch_1 = scratch_1.apply_mut(|scratch| {
                    let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                    self.vec_znx_normalize_assign(base2k, tmp_pt_k, 0, &mut tmp_pt_backend.data, 0, scratch)
                });

                for col_j in 0..rank + 1 {
                    // GLWE encrypt of vec_znx_pt into vec_znx_ct

                    let (seed, _) = source.branch();

                    seeds[row_i * cols + col_j] = seed;

                    let tmp_pt_backend = tmp_pt.to_backend_ref();
                    let base2k = res.base2k().into();
                    let scratch_full = scratch_1.borrow();
                    let (mut full_ct, mut scratch_2) = scratch_full.take_glwe_scratch(&res);
                    self.fill_glwe_mask_from_seed_reference(base2k, &mut full_ct, 1, rank, seed);
                    self.glwe_encrypt_sk_internal(
                        base2k,
                        &mut full_ct.data,
                        Some((tmp_pt_backend, col_j)),
                        sk,
                        enc_infos,
                        source_xe,
                        &mut scratch_2,
                    );
                    let full_ct_ref = full_ct.to_backend_ref();
                    let mut ct = res.at_view_mut(row_i, col_j);
                    self.vec_znx_copy(&mut ct.data, 0, &full_ct_ref.data, 0);
                }
            }
        };

        res.seed_mut().copy_from_slice(&seeds);
    }
}

/// Forwards every method of [`GGSWCompressedEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_ggsw_compressed_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkComposition<$be>>::ggsw_compressed_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
        }

    fn ggsw_compressed_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWCompressedToBackendMut<$be> + $crate::layouts::GGSWCompressedSeedMut + $crate::layouts::GGSWInfos,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkComposition<$be>>::ggsw_compressed_encrypt_sk_composition::<R, P, S, E>(self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
        }
        }
    };
}
