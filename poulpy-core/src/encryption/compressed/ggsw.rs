#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarAssign, VecZnxNormalizeAssign},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

pub use crate::api::GGSWCompressedEncryptSk;
use crate::{
    EncryptionInfos, ScratchTakeCore,
    encryption::{GGSWEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GGSWCompressedSeedMut, GGSWInfos, LWEInfos,
        compressed::{GGSWCompressed, GGSWCompressedToMut},
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GGSWCompressedEncryptSkDefault<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGSWCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GLWEEncryptSkInternal<BE> + GGSWEncryptSk<BE> + VecZnxAddScalarAssign + VecZnxNormalizeAssign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.ggsw_encrypt_sk_tmp_bytes(infos);
        lvl_0
    }

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;
        let dsize: usize = res.dsize().into();

        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();

        assert_eq!(res.rank(), sk.rank());
        assert_eq!(pt.n(), self.n());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.ggsw_compressed_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GGSWCompressedEncryptSk::ggsw_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_compressed_encrypt_sk_tmp_bytes(res)
        );

        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.dnum().as_usize() * (res.rank().as_usize() + 1)];

        {
            let res: &mut GGSWCompressed<&mut [u8]> = &mut res.to_mut();

            let (mut tmp_pt, scratch_1) = scratch.take_glwe_plaintext(res);

            let mut source = Source::new(seed_xa);

            for row_i in 0..res.dnum().into() {
                tmp_pt.data.zero();

                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                self.vec_znx_add_scalar_assign(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
                self.vec_znx_normalize_assign(base2k, &mut tmp_pt.data, 0, scratch_1);

                for col_j in 0..rank + 1 {
                    // rlwe encrypt of vec_znx_pt into vec_znx_ct

                    let (seed, mut source_xa_tmp) = source.branch();

                    seeds[row_i * cols + col_j] = seed;

                    self.glwe_encrypt_sk_internal(
                        res.base2k().into(),
                        &mut res.at_mut(row_i, col_j).data,
                        cols,
                        true,
                        Some((&tmp_pt, col_j)),
                        sk,
                        enc_infos,
                        source_xe,
                        &mut source_xa_tmp,
                        scratch_1,
                    );
                }
            }
        }

        res.seed_mut().copy_from_slice(&seeds);
    }
}
