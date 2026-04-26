use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataViewMut, Module, Scratch},
};

pub use crate::api::GLWEDecrypt;
use crate::layouts::{
    GLWE, GLWEInfos, GLWEPlaintextToMut, GLWESecretPrepared, GLWEToRef, LWEInfos, SetLWEInfos, prepared::GLWESecretPreparedToRef,
};

pub(crate) trait GLWEDecryptDefault<BE: Backend>:
    Sized
    + ModuleN
    + VecZnxDftBytesOf
    + VecZnxNormalizeTmpBytes
    + VecZnxBigBytesOf
    + VecZnxDftApply<BE>
    + SvpApplyDftToDftAssign<BE>
    + VecZnxIdftApplyConsume<BE>
    + VecZnxBigAddAssign<BE>
    + VecZnxBigAddSmallAssign<BE>
    + VecZnxBigNormalize<BE>
where
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
    fn glwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.bytes_of_vec_znx_big(1, size);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(1, size).max(self.vec_znx_normalize_tmp_bytes());

        lvl_0 + lvl_1
    }

    fn glwe_decrypt_default<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        let res: &GLWE<&[u8]> = &res.to_ref();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.rank(), sk.rank()); //NOTE: res.rank() != res.to_ref().rank() if res is of type GLWETensor
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
        }
        assert!(
            scratch.available() >= self.glwe_decrypt_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
            scratch.available(),
            self.glwe_decrypt_tmp_bytes_default(res)
        );

        let cols: usize = (res.rank() + 1).into();

        let (mut c0_big, scratch_1) = scratch.take_vec_znx_big(self, 1, res.size()); // TODO optimize size when pt << ct
        c0_big.data_mut().fill(0);

        (1..cols).for_each(|i| {
            // ci_dft = DFT(a[i]) * DFT(s[i])
            let (mut ci_dft, _) = scratch_1.take_vec_znx_dft(self, 1, res.size()); // TODO optimize size when pt << ct
            self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, res.data(), i);
            self.svp_apply_dft_to_dft_assign(&mut ci_dft, 0, &sk.data, i - 1);
            let ci_big = self.vec_znx_idft_apply_consume(ci_dft);

            // c0_big += a[i] * s[i]
            self.vec_znx_big_add_assign(&mut c0_big, 0, &ci_big, 0);
        });

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        self.vec_znx_big_add_small_assign(&mut c0_big, 0, res.data(), 0);

        let pt_base2k: usize = pt.base2k().into();

        // pt = norm(BIG(m + e))
        self.vec_znx_big_normalize(
            pt.to_mut().data_mut(),
            pt_base2k,
            0,
            0,
            &c0_big,
            res.base2k().into(),
            0,
            scratch_1,
        );
    }
}

impl<BE: Backend> GLWEDecryptDefault<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxNormalizeTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>,
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
}
