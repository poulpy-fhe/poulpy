use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarAssign, VecZnxDftBytesOf, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

pub use crate::api::GGSWEncryptSk;
use crate::{
    EncryptionInfos, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchTakeCore,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GLWEInfos, GLWEPlaintext, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GGSWEncryptSkDefault<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGSWEncryptSkDefault<BE> for Module<BE>
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
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn ggsw_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n());
        assert_eq!(sk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.ggsw_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GGSWEncryptSk::ggsw_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_encrypt_sk_tmp_bytes(res)
        );

        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let cols: usize = rank + 1;

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_plaintext(res);

        for row_i in 0..res.dnum().into() {
            tmp_pt.data.zero();
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            self.vec_znx_add_scalar_assign(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            self.vec_znx_normalize_assign(base2k, &mut tmp_pt.data, 0, scratch_1);
            for col_j in 0..rank + 1 {
                self.glwe_encrypt_sk_internal(
                    base2k,
                    res.at_mut(row_i, col_j).data_mut(),
                    cols,
                    false,
                    Some((&tmp_pt, col_j)),
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
