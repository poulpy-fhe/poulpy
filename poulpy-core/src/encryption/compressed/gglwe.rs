#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarAssign, VecZnxDftBytesOf, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

pub use crate::api::GGLWECompressedEncryptSk;
use crate::{
    EncryptionInfos, ScratchTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GGLWECompressedSeedMut, GGLWEInfos, GLWEPlaintext, GLWESecretPrepared, LWEInfos,
        compressed::{GGLWECompressed, GGLWECompressedToMut},
        prepared::GLWESecretPreparedToRef,
    },
};

#[doc(hidden)]
pub trait GGLWECompressedEncryptSkDefault<BE: Backend> {
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGLWECompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeAssign<BE>
        + VecZnxAddScalarAssign
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.seed_mut().len()];

        {
            let res: &mut GGLWECompressed<&mut [u8]> = &mut res.to_mut();
            let pt: &ScalarZnx<&[u8]> = &pt.to_ref();
            let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

            assert_eq!(
                res.rank_in(),
                pt.cols() as u32,
                "res.rank_in(): {} != pt.cols(): {}",
                res.rank_in(),
                pt.cols()
            );
            assert_eq!(
                res.rank_out(),
                sk.rank(),
                "res.rank_out(): {} != sk.rank(): {}",
                res.rank_out(),
                sk.rank()
            );
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n() as u32, sk.n());
            assert!(
                scratch.available() >= self.gglwe_compressed_encrypt_sk_tmp_bytes(res),
                "scratch.available(): {} < GGLWECompressedEncryptSk::gglwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.gglwe_compressed_encrypt_sk_tmp_bytes(res)
            );
            assert!(
                res.dnum().0 * res.dsize().0 * res.base2k().0 <= res.max_k().0,
                "res.dnum() : {} * res.dsize() : {} * res.base2k() : {} = {} >= res.k() = {}",
                res.dnum(),
                res.dsize(),
                res.base2k(),
                res.dnum().0 * res.dsize().0 * res.base2k().0,
                res.max_k()
            );

            let dnum: usize = res.dnum().into();
            let dsize: usize = res.dsize().into();
            let base2k: usize = res.base2k().into();
            let rank_in: usize = res.rank_in().into();
            let cols: usize = (res.rank_out() + 1).into();

            let mut source_xa = Source::new(seed);

            let (mut tmp_pt, scrach_1) = scratch.take_glwe_plaintext(res);

            for col_j in 0..rank_in {
                for row_i in 0..dnum {
                    // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                    tmp_pt.data.zero(); // zeroes for next iteration
                    self.vec_znx_add_scalar_assign(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, col_j);
                    self.vec_znx_normalize_assign(base2k, &mut tmp_pt.data, 0, scrach_1);

                    let (seed, mut source_xa_tmp) = source_xa.branch();
                    seeds[row_i * rank_in + col_j] = seed;

                    self.glwe_encrypt_sk_internal(
                        res.base2k().into(),
                        &mut res.at_mut(row_i, col_j).data,
                        cols,
                        true,
                        Some((&tmp_pt, 0)),
                        sk,
                        enc_infos,
                        source_xe,
                        &mut source_xa_tmp,
                        scrach_1,
                    );
                }
            }
        }

        res.seed_mut().copy_from_slice(&seeds);
    }
}
