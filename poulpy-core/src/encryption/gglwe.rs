use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarAssign, VecZnxDftBytesOf, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

pub use crate::api::GGLWEEncryptSk;
use crate::{
    EncryptionInfos, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GLWEPlaintext, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GGLWEEncryptSkDefault<BE: Backend> {
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,

        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGLWEEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSk<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxAddScalarAssign
        + VecZnxNormalizeAssign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn gglwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,

        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
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
            scratch.available() >= self.gglwe_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GGLWEEncryptSk::gglwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_encrypt_sk_tmp_bytes(res)
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

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_plaintext(res);
        // For each input column (i.e. rank) produces a GGLWE of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s1', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                self.vec_znx_add_scalar_assign(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, col_i);
                self.vec_znx_normalize_assign(base2k, &mut tmp_pt.data, 0, scratch_1);
                self.glwe_encrypt_sk(
                    &mut res.at_mut(row_i, col_i),
                    &tmp_pt,
                    sk,
                    enc_infos,
                    source_xe,
                    source_xa,
                    scratch_1,
                );
            }
        }
    }
}
