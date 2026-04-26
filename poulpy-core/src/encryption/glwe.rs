use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolBytesOf, SvpPrepare,
        VecZnxAddAssign, VecZnxAddNormal, VecZnxBigAddNormal, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubAssign,
    },
    layouts::{Backend, Module, ScalarZnx, Scratch, VecZnx, VecZnxBig, VecZnxToMut, ZnxInfos, ZnxZero},
    source::Source,
};

pub use crate::api::{GLWEEncryptPk, GLWEEncryptSk};
use crate::{
    EncryptionInfos, GetDistribution,
    dist::Distribution,
    layouts::{
        GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, GLWEPrepared, GLWEPreparedToRef, GLWEToMut, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

#[doc(hidden)]
pub trait GLWEEncryptSkDefault<BE: Backend> {
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn glwe_encrypt_zero_sk<R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GLWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized + ModuleN + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxDftBytesOf + GLWEEncryptSkInternal<BE>,
    Scratch<BE>: ScratchAvailable,
{
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = VecZnx::bytes_of(self.n(), 1, size).max(self.vec_znx_normalize_tmp_bytes());
        let lvl_1: usize = VecZnx::bytes_of(self.n(), 1, size);
        let lvl_2: usize = self.bytes_of_vec_znx_dft(1, size);
        let lvl_3: usize = self.vec_znx_normalize_tmp_bytes().max(self.vec_znx_big_normalize_tmp_bytes());

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let pt: &GLWEPlaintext<&[u8]> = &pt.to_ref();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes(res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.data_mut(),
            cols,
            false,
            Some((pt, 0)),
            sk,
            enc_infos,
            source_xe,
            source_xa,
            scratch,
        );
    }

    fn glwe_encrypt_zero_sk<R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes(res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.data_mut(),
            cols,
            false,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk,
            enc_infos,
            source_xe,
            source_xa,
            scratch,
        );
    }
}

#[doc(hidden)]
pub trait GLWEEncryptPkDefault<BE: Backend> {
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk<R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_encrypt_zero_pk<R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWEEncryptPkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptPkInternal<BE> + VecZnxDftBytesOf + SvpPPolBytesOf + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchAvailable,
{
    fn glwe_encrypt_pk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.bytes_of_svp_ppol(1);
        let lvl_1: usize =
            (self.bytes_of_vec_znx_dft(1, size) + self.bytes_of_vec_znx_big(1, size)).max(ScalarZnx::bytes_of(self.n(), 1));
        let lvl_2: usize = self.vec_znx_normalize_tmp_bytes();

        lvl_0 + lvl_1 + lvl_2
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk<R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes(res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_pk_tmp_bytes(res)
        );
        self.glwe_encrypt_pk_internal(res, Some((pt, 0)), pk, enc_infos, source_xu, source_xe, scratch);
    }

    fn glwe_encrypt_zero_pk<R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes(res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_pk_tmp_bytes(res)
        );
        self.glwe_encrypt_pk_internal(
            res,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            enc_infos,
            source_xu,
            source_xe,
            scratch,
        );
    }
}

pub(crate) trait GLWEEncryptPkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<R, P, K, E>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWEEncryptPkInternal<BE> for Module<BE>
where
    Self: SvpPrepare<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + SvpPPolBytesOf
        + ModuleN
        + VecZnxDftBytesOf,
    Scratch<BE>: ScratchTakeBasic,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<R, P, K, E>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.base2k(), pk.base2k());
        assert_eq!(res.n(), pk.n());
        assert_eq!(res.rank(), pk.rank());
        if let Some((pt, _)) = pt {
            assert_eq!(pt.base2k(), pk.base2k());
            assert_eq!(pt.n(), pk.n());
        }

        let base2k: usize = pk.base2k().into();
        let size_pk: usize = pk.size();
        let cols: usize = (res.rank() + 1).into();

        // Generates u according to the underlying secret distribution.
        let (mut u_dft, scratch_1) = scratch.take_svp_ppol(self, 1);

        {
            let (mut u, _) = scratch_1.take_scalar_znx(self.n(), 1);
            match pk.dist() {
                Distribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                Distribution::TernaryFixed(hw) => u.fill_ternary_hw(0, *hw, source_xu),
                Distribution::TernaryProb(prob) => u.fill_ternary_prob(0, *prob, source_xu),
                Distribution::BinaryFixed(hw) => u.fill_binary_hw(0, *hw, source_xu),
                Distribution::BinaryProb(prob) => u.fill_binary_prob(0, *prob, source_xu),
                Distribution::BinaryBlock(block_size) => u.fill_binary_block(0, *block_size, source_xu),
                Distribution::ZERO => {}
            }

            self.svp_prepare(&mut u_dft, 0, &u, 0);
        }

        {
            let pk: &GLWEPrepared<&[u8], BE> = &pk.to_ref();

            // ct[i] = pk[i] * u + ei (+ m if col = i)
            for i in 0..cols {
                let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, 1, size_pk);
                // ci_dft = DFT(u) * DFT(pk[i])
                self.svp_apply_dft_to_dft(&mut ci_dft, 0, &u_dft, 0, &pk.data, i);

                // ci_big = u * p[i]
                let mut ci_big = self.vec_znx_idft_apply_consume(ci_dft);

                // ci_big = u * pk[i] + e
                self.vec_znx_big_add_normal(base2k, &mut ci_big, 0, enc_infos.noise_infos(), source_xe);

                // ci_big = u * pk[i] + e + m (if col = i)
                if let Some((pt, col)) = pt
                    && col == i
                {
                    self.vec_znx_big_add_small_assign(&mut ci_big, 0, &pt.to_ref().data, 0);
                }

                // ct[i] = norm(ci_big)
                self.vec_znx_big_normalize(&mut res.data, base2k, 0, i, &ci_big, base2k, 0, scratch_2);
            }
        }
    }
}

pub(crate) trait GLWEEncryptSkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_internal<R, P, S, E>(
        &self,
        base2k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GLWEEncryptSkInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubAssign
        + VecZnxAddAssign
        + VecZnxNormalizeAssign<BE>
        + VecZnxAddNormal
        + VecZnxNormalize<BE>
        + VecZnxSub
        + VecZnxBigNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeBasic + ScratchAvailable,
{
    fn glwe_encrypt_sk_internal<R, P, S, E>(
        &self,
        base2k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>,
    {
        let ct: &mut VecZnx<&mut [u8]> = &mut res.to_mut();
        let sk: GLWESecretPrepared<&[u8], BE> = sk.to_ref();

        if compressed {
            assert_eq!(ct.cols(), 1, "invalid glwe: compressed tag=true but #cols={} != 1", ct.cols())
        }

        assert!(
            sk.dist != Distribution::NONE,
            "glwe secret distribution is NONE (have you prepared the key?)"
        );

        let size: usize = ct.size();

        let (mut c0, scratch_1) = scratch.take_vec_znx(self.n(), 1, size);
        c0.zero();

        {
            let (mut ci, scratch_2) = scratch_1.take_vec_znx(self.n(), 1, size);

            // ct[i] = uniform
            // ct[0] -= c[i] * s[i],
            (1..cols).for_each(|i| {
                let col_ct: usize = if compressed { 0 } else { i };

                // ct[i] = uniform (+ pt)
                self.vec_znx_fill_uniform(base2k, ct, col_ct, source_xa);

                let (mut ci_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, 1, size);

                // ci = ct[i] - pt
                // i.e. we act as we sample ct[i] already as uniform + pt
                // and if there is a pt, then we subtract it before applying DFT
                if let Some((pt, col)) = pt {
                    if i == col {
                        self.vec_znx_sub(&mut ci, 0, ct, col_ct, &pt.to_ref().data, 0);
                        self.vec_znx_normalize_assign(base2k, &mut ci, 0, scratch_3);
                        self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &ci, 0);
                    } else {
                        self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, ct, col_ct);
                    }
                } else {
                    self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, ct, col_ct);
                }

                self.svp_apply_dft_to_dft_assign(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big: VecZnxBig<&mut [u8], BE> = self.vec_znx_idft_apply_consume(ci_dft);

                // use c[0] as buffer, which is overwritten later by the normalization step
                self.vec_znx_big_normalize(&mut ci, base2k, 0, 0, &ci_big, base2k, 0, scratch_3);

                // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
                self.vec_znx_sub_assign(&mut c0, 0, &ci, 0);
            });
        }

        // c[0] += e
        self.vec_znx_add_normal(base2k, &mut c0, 0, enc_infos.noise_infos(), source_xe);

        // c[0] += m if col = 0
        if let Some((pt, col)) = pt
            && col == 0
        {
            self.vec_znx_add_assign(&mut c0, 0, &pt.to_ref().data, 0);
        }

        // c[0] = norm(c[0])
        self.vec_znx_normalize(ct, base2k, 0, 0, &c0, base2k, 0, scratch_1);
    }
}
