use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchTakeBasic, VecZnxAddNormal, VecZnxFillUniform, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

pub use crate::api::LWEEncryptSk;
use crate::{
    EncryptionInfos, ScratchTakeCore,
    layouts::{LWE, LWEInfos, LWEPlaintext, LWEPlaintextToRef, LWESecret, LWESecretToRef, LWEToMut},
};

#[doc(hidden)]
pub trait LWEEncryptSkDefault<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        E: EncryptionInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> LWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized + VecZnxFillUniform + VecZnxAddNormal + VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        let size: usize = infos.size();

        let lvl_0: usize = LWEPlaintext::bytes_of(size);
        let lvl_1: usize = self.vec_znx_normalize_tmp_bytes();

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        E: EncryptionInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let pt: &LWEPlaintext<&[u8]> = &pt.to_ref();
        let sk: &LWESecret<&[u8]> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n())
        }

        assert!(
            scratch.available() >= self.lwe_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < LWEEncryptSk::lwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_encrypt_sk_tmp_bytes(res)
        );

        let base2k: usize = res.base2k().into();

        self.vec_znx_fill_uniform(base2k, &mut res.data, 0, source_xa);

        let (mut tmp_znx, scratch_1) = scratch.take_vec_znx(1, 1, res.size());
        tmp_znx.zero();

        let min_size: usize = res.size().min(pt.size());

        (0..min_size).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] = pt.data.at(0, i)[0]
                - res.data.at(0, i)[1..]
                    .iter()
                    .zip(sk.data.at(0, 0))
                    .map(|(x, y)| x * y)
                    .sum::<i64>();
        });

        (min_size..res.size()).for_each(|i| {
            tmp_znx.at_mut(0, i)[0] -= res.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
        });

        self.vec_znx_add_normal(base2k, &mut tmp_znx, 0, enc_infos.noise_infos(), source_xe);

        self.vec_znx_normalize_assign(base2k, &mut tmp_znx, 0, scratch_1);

        (0..res.size()).for_each(|i| {
            res.data.at_mut(0, i)[0] = tmp_znx.at(0, i)[0];
        });
    }
}
