use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxAddAssign, VecZnxBigAddSmallAssign, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxFillUniformSource, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
        VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxZero, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Module, ScalarZnxToBackendMut, ScratchArena, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_mut, vec_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GLWEMaskFill, GetDistribution, ScalarZnxFillDistribution, VecZnxAddNormal, VecZnxBigAddNormal,
    dist::Distribution,
    layouts::{
        GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GLWEMaskFillReference<BE: Backend> {
    fn fill_glwe_mask_from_source_reference<R>(&self, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<BE>;

    fn fill_glwe_from_source_reference<R>(&self, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<BE>;
}

impl<BE: Backend> GLWEMaskFillReference<BE> for Module<BE>
where
    Self: VecZnxFillUniformSource<BE>,
{
    fn fill_glwe_mask_from_source_reference<R>(&self, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();
        let (base2k, k) = (res.base2k().as_usize(), res.k().as_usize());
        for col in 1..res.data.cols() {
            self.vec_znx_fill_uniform_source(base2k, k, &mut res.data, col, source_xa);
        }
    }

    fn fill_glwe_from_source_reference<R>(&self, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<BE>,
    {
        res.set_canonical(true);
        let mut res = res.to_backend_mut();
        let (base2k, k) = (res.base2k().as_usize(), res.k().as_usize());
        for col in 0..res.data.cols() {
            self.vec_znx_fill_uniform_source(base2k, k, &mut res.data, col, source);
        }
    }
}

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GLWEEncryptSkReference<BE: Backend> {
    fn glwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;

    fn glwe_encrypt_zero_sk_reference<R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

impl<BE: Backend> GLWEEncryptSkReference<BE> for Module<BE>
where
    Self: Sized
        + ModuleN
        + VecZnxNormalizeTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + GLWEMaskFill<BE>
        + GLWEEncryptSkInternal<BE>,
{
    fn glwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = BE::bytes_of_vec_znx(self.n(), 1, size);
        let lvl_1: usize = BE::bytes_of_vec_znx(self.n(), 1, size);
        let lvl_2: usize = self.vec_znx_normalize_tmp_bytes().max(
            self.bytes_of_vec_znx_dft(self.n(), 1, size)
                + self.bytes_of_vec_znx_big(self.n(), 1, size)
                + self.vec_znx_big_normalize_tmp_bytes(),
        );

        lvl_0 + lvl_1 + lvl_2
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        res.set_canonical(true);
        let res = &mut res.to_backend_mut();
        let pt_backend = pt.to_backend_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert_eq!(pt_backend.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes_reference(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes_reference(res)
        );

        {
            let mut res_ref = &mut *res;
            self.fill_glwe_mask_from_source(&mut res_ref, source_xa);
        }
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            &mut res.data,
            Some((pt_backend, 0)),
            sk,
            enc_infos,
            source_xe,
            scratch,
        );
    }

    fn glwe_encrypt_zero_sk_reference<R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        res.set_canonical(false);
        let res = &mut res.to_backend_mut();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes_reference(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes_reference(res)
        );

        {
            let mut res_ref = &mut *res;
            self.fill_glwe_mask_from_source(&mut res_ref, source_xa);
        }
        self.glwe_encrypt_sk_internal(res.base2k().into(), &mut res.data, None, sk, enc_infos, source_xe, scratch);
    }
}

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GLWEEncryptPkReference<BE: Backend> {
    fn glwe_encrypt_pk_tmp_bytes_reference<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos;

    fn glwe_encrypt_pk_reference<R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

    fn glwe_encrypt_zero_pk_reference<R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEEncryptPkReference<BE> for Module<BE>
where
    Self: GLWEEncryptPkInternal<BE> + VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
{
    fn glwe_encrypt_pk_tmp_bytes_reference<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos,
    {
        let size: usize = res_infos.size().max(pk_infos.size());
        let rank: usize = pk_infos.rank().into();
        let n: usize = self.n();
        assert_eq!(n as u32, res_infos.n());
        let lvl_0: usize = BE::bytes_of_scalar_znx(n, rank);
        let lvl_1: usize = self.bytes_of_vec_znx_dft(n, rank, 1);
        let lvl_2: usize = self.bytes_of_vec_znx_dft(n, rank + 1, size);
        let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(size, 1, 1, rank, rank + 1, size);
        let lvl_3: usize = BE::bytes_of_vec_znx(n, 1, vmp.div_ceil(BE::bytes_of_vec_znx(n, 1, 1)))
            .max(self.bytes_of_vec_znx_big(n, 1, size) + self.vec_znx_big_normalize_tmp_bytes());

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_reference<R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes_reference(res, pk),
            "insufficient scratch for GLWE public-key encryption"
        );
        self.glwe_encrypt_pk_internal(
            res,
            Some((pt.to_backend_ref(), 0)),
            pk,
            enc_infos,
            source_xu,
            source_xe,
            scratch,
        );
    }

    fn glwe_encrypt_zero_pk_reference<R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes_reference(res, pk),
            "insufficient scratch for GLWE public-key encryption"
        );
        self.glwe_encrypt_pk_internal(res, None, pk, enc_infos, source_xu, source_xe, scratch);
    }
}

pub(crate) trait GLWEEncryptPkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<R, K, E>(
        &self,
        res: &mut R,
        pt: Option<(GLWEBackendRef<'_, BE>, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEEncryptPkInternal<BE> for Module<BE>
where
    Self: VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes
        + VecZnxDftZero<BE>
        + VecZnxZero<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigAddSmallAssign<BE>
        + ModuleN
        + ScalarZnxFillDistribution<BE>,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<R, K, E>(
        &self,
        res: &mut R,
        pt: Option<(GLWEBackendRef<'_, BE>, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        res.set_canonical(true);
        let res = &mut res.to_backend_mut();

        assert_eq!(res.base2k(), pk.base2k());
        assert_eq!(res.n(), pk.n());
        assert_eq!(res.rank(), pk.rank());
        assert!(pk.k() >= res.k(), "invalid public key: less precise than the output");
        if let Some((pt, _)) = &pt {
            assert_eq!(pt.base2k(), pk.base2k());
            assert_eq!(pt.n(), pk.n());
        }

        let pk = <K as GLWEPublicKeyPreparedToBackendRef<BE>>::to_backend_ref(pk);
        assert!(
            pk.data.cols_out() == pk.data.cols_in() + 1,
            "invalid public key: entry count differs from its rank"
        );
        let n: usize = self.n();
        let base2k: usize = pk.base2k().into();
        let noise_infos = enc_infos.noise_infos();
        let size_pk: usize = pk.size();
        let res_k: usize = res.k().as_usize();
        let rank: usize = pk.data.cols_in();

        // One ephemeral per entry, drawn like the secret: a single one leaves the masks rank-1 in u.
        let dist: Distribution = match pk.dist() {
            Distribution::NONE => panic!(
                "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                 Self::generate"
            ),
            Distribution::ENCAPSULATED(_) => panic!("invalid public key: secret is tagged for encapsulation"),
            // A zero ephemeral leaves the ciphertext as the message plus fresh noise.
            Distribution::ZERO | Distribution::TernaryFixed(0) | Distribution::BinaryFixed(0) => {
                panic!("invalid public key: zero ephemeral distribution")
            }
            Distribution::TernaryProb(p) | Distribution::BinaryProb(p) if p.is_nan() || *p <= 0.0 => {
                panic!("invalid public key: zero ephemeral distribution")
            }
            dist => *dist,
        };

        let scratch = scratch.borrow();
        let (mut u, scratch_1) = scratch.take_scalar_znx_scratch(n, rank);
        let (mut u_dft, scratch_1) = scratch_1.take_vec_znx_dft_scratch(n, rank, 1);
        for l in 0..rank {
            self.scalar_znx_fill_distribution(&mut u.to_backend_mut(), l, dist, source_xu);
            self.vec_znx_dft_apply(
                1,
                0,
                &mut u_dft.to_backend_mut(),
                l,
                &scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(&u),
                l,
            );
        }

        let (mut res_dft, mut scratch_1) = scratch_1.take_vec_znx_dft_scratch(n, rank + 1, size_pk);
        self.vmp_apply_dft_to_dft(
            &mut res_dft.to_backend_mut(),
            &u_dft.to_backend_ref(),
            &pk.data,
            0,
            &mut scratch_1.borrow(),
        );

        {
            let (mut ci_big, mut scratch_2) = scratch_1.borrow().take_vec_znx_big_scratch(n, 1, size_pk);
            for i in 0..rank + 1 {
                self.vec_znx_idft_apply_tmpa(&mut ci_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), i);
                self.vec_znx_big_add_normal(base2k, &mut ci_big, 0, noise_infos, source_xe);

                if let Some((pt, col)) = &pt
                    && *col == i
                {
                    self.vec_znx_big_add_small_assign(&mut ci_big.to_backend_mut(), 0, &pt.data, 0);
                }

                self.vec_znx_big_normalize(
                    &mut res.data,
                    base2k,
                    res_k,
                    0,
                    i,
                    &ci_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch_2,
                );
            }
        }

        // The ephemerals and the products would decrypt the ciphertext from the caller's scratch.
        for l in 0..rank {
            self.vec_znx_zero(&mut scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut u), l);
            self.vec_znx_dft_zero(&mut u_dft.to_backend_mut(), l);
        }
        for i in 0..rank + 1 {
            self.vec_znx_dft_zero(&mut res_dft.to_backend_mut(), i);
        }
        let vmp: usize = self.vmp_apply_dft_to_dft_tmp_bytes(size_pk, 1, 1, rank, rank + 1, size_pk);
        let (mut vmp_tmp, _) = scratch_1.take_vec_znx_scratch(n, 1, vmp.div_ceil(BE::bytes_of_vec_znx(n, 1, 1)));
        self.vec_znx_zero(&mut vmp_tmp, 0);
    }
}

pub(crate) trait GLWEEncryptSkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_internal<'pt, S, E>(
        &self,
        base2k: usize,
        res: &mut VecZnx<BE::BufMut<'_>, BE::ZnxWord>,
        pt: GLWEEncryptSkPlaintext<'pt, BE>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

type GLWEEncryptSkPlaintext<'a, BE> = Option<(GLWEBackendRef<'a, BE>, usize)>;

impl<BE: Backend> GLWEEncryptSkInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniformSource<BE>
        + VecZnxAddAssign<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxAddNormal<BE>
        + VecZnxNormalize<BE>
        + VecZnxSubAssign<BE>
        + VecZnxSubNegateAssign<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    fn glwe_encrypt_sk_internal<'pt, S, E>(
        &self,
        base2k: usize,
        res: &mut VecZnx<BE::BufMut<'_>, BE::ZnxWord>,
        pt: GLWEEncryptSkPlaintext<'pt, BE>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let sk = sk.to_backend_ref();
        let noise_infos = enc_infos.noise_infos();

        assert!(
            sk.dist != Distribution::NONE,
            "glwe secret distribution is NONE (have you prepared the key?)"
        );
        assert_eq!(
            res.cols(),
            sk.rank().as_usize() + 1,
            "GLWE encryption expects a full ciphertext with pre-sampled mask columns"
        );

        let size: usize = res.size();

        let scratch_local = scratch.borrow();
        let (mut c0, scratch_1) = scratch_local.take_vec_znx_scratch(self.n(), 1, size);
        let (mut ci, scratch_2) = scratch_1.take_vec_znx_scratch(self.n(), 1, size);
        let mut scratch_2 = scratch_2;
        self.vec_znx_zero(&mut c0, 0);

        for i in 1..res.cols() {
            if let Some((pt, col)) = pt.as_ref() {
                if i == *col {
                    self.vec_znx_copy(&mut ci, 0, &pt.data, 0);
                    let ct_ref = vec_znx_backend_ref_from_mut::<BE>(res);
                    self.vec_znx_sub_negate_assign(&mut ci, 0, &ct_ref, i);
                    self.vec_znx_normalize_assign(base2k, size * base2k, 0, &mut ci.to_backend_mut(), 0, &mut scratch_2.borrow());
                } else {
                    let ct_ref = vec_znx_backend_ref_from_mut::<BE>(res);
                    self.vec_znx_copy(&mut ci, 0, &ct_ref, i);
                }
            } else {
                let ct_ref = vec_znx_backend_ref_from_mut::<BE>(res);
                self.vec_znx_copy(&mut ci, 0, &ct_ref, i);
            }

            {
                let (mut ci_dft, scratch_3) = scratch_2.borrow().take_vec_znx_dft_scratch(self.n(), 1, size);
                self.vec_znx_dft_apply(1, 0, &mut ci_dft.to_backend_mut(), 0, &ci.to_backend_ref(), 0);
                self.svp_apply_dft_to_dft_assign(&mut ci_dft.to_backend_mut(), 0, &sk.data, i - 1);
                let (mut ci_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self.n(), 1, size);
                self.vec_znx_idft_apply_tmpa(&mut ci_big.to_backend_mut(), 0, &mut ci_dft.to_backend_mut(), 0);
                self.vec_znx_big_normalize(
                    &mut ci.to_backend_mut(),
                    base2k,
                    size * base2k,
                    0,
                    0,
                    &ci_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch_4.borrow(),
                );
            }

            self.vec_znx_sub_assign(&mut c0, 0, &ci.to_backend_ref(), 0);
        }

        // c[0] += e
        self.vec_znx_add_normal(base2k, &mut c0.to_backend_mut(), 0, noise_infos, source_xe);

        // c[0] += m if col = 0
        if let Some((pt, col)) = &pt
            && *col == 0
        {
            self.vec_znx_add_assign(&mut c0.to_backend_mut(), 0, &pt.data, 0);
            self.vec_znx_normalize_assign(base2k, size * base2k, 0, &mut c0.to_backend_mut(), 0, &mut scratch_2.borrow());
        }
        self.vec_znx_copy(res, 0, &c0.to_backend_ref(), 0);
    }
}
