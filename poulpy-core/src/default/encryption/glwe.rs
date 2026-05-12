use poulpy_hal::{
    api::{
        ModuleN, ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwSourceBackend, ScalarZnxFillBinaryProbSourceBackend,
        ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbSourceBackend, ScratchArenaTakeBasic, SvpApplyDftToDft,
        SvpApplyDftToDftAssign, SvpPPolBytesOf, SvpPrepare, VecZnxAddAssignBackend, VecZnxAddNormalSourceBackend,
        VecZnxBigAddNormal, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxFillUniformSourceBackend, VecZnxIdftApplyTmpA, VecZnxNormalize, VecZnxNormalizeAssignBackend,
        VecZnxNormalizeTmpBytes, VecZnxSubAssignBackend, VecZnxSubNegateAssignBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Module, ScalarZnx, ScratchArena, SvpPPolToBackendRef, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxDftToBackendMut, VecZnxToBackendMut, VecZnxToBackendRef, scalar_znx_as_vec_znx_backend_mut_from_mut,
        vec_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, ScratchArenaTakeCore,
    dist::Distribution,
    layouts::{
        GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::{GLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
};
pub(crate) fn normalize_scratch_vec_znx<'a, BE: Backend + 'a>(
    module: &Module<BE>,
    base2k: usize,
    vec: &mut VecZnx<BE::BufMut<'a>>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxNormalizeAssignBackend<BE>,
{
    scratch.scope(|mut scratch| {
        let mut vec_ref = vec;
        let mut vec_mut = <&mut VecZnx<BE::BufMut<'a>> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut vec_ref);
        <Module<BE> as VecZnxNormalizeAssignBackend<BE>>::vec_znx_normalize_assign_backend(
            module,
            base2k,
            &mut vec_mut,
            0,
            &mut scratch,
        );
    });
}

#[doc(hidden)]
pub trait GLWEEncryptSkDefault<BE: Backend> {
    fn glwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_encrypt_zero_sk_default<'s, R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized + ModuleN + VecZnxNormalizeTmpBytes + VecZnxBigNormalizeTmpBytes + VecZnxDftBytesOf + GLWEEncryptSkInternal<BE>,
{
    fn glwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = VecZnx::bytes_of(self.n(), 1, size);
        let lvl_1: usize = VecZnx::bytes_of(self.n(), 1, size);
        let lvl_2: usize = self.vec_znx_normalize_tmp_bytes().max(
            self.bytes_of_vec_znx_dft(1, size) + self.bytes_of_vec_znx_big(1, size) + self.vec_znx_big_normalize_tmp_bytes(),
        );

        lvl_0 + lvl_1 + lvl_2
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let pt_backend = pt.to_backend_ref();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert_eq!(pt_backend.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes_default(res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            &mut res.data,
            cols,
            false,
            Some((pt_backend, 0)),
            sk,
            enc_infos,
            source_xe,
            source_xa,
            scratch,
        );
    }

    fn glwe_encrypt_zero_sk_default<'s, R, E, S>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let sk_ref = sk.to_backend_ref();

        assert_eq!(res.rank(), sk_ref.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk_ref.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWE::encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_sk_tmp_bytes_default(res)
        );

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            &mut res.data,
            cols,
            false,
            None,
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
    fn glwe_encrypt_pk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk_default<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_encrypt_zero_pk_default<'s, R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWEEncryptPkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptPkInternal<BE>
        + VecZnxDftBytesOf
        + SvpPPolBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxZeroBackend<BE>,
{
    fn glwe_encrypt_pk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let size: usize = infos.size();
        let cols: usize = (infos.rank() + 1).into();
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.bytes_of_svp_ppol(1);
        let lvl_1: usize = ScalarZnx::bytes_of(self.n(), 1);
        let lvl_2: usize = cols
            * (self.bytes_of_vec_znx_dft(1, size) + self.bytes_of_vec_znx_big(1, size) + VecZnx::bytes_of(self.n(), 1, size));
        let lvl_3: usize = self.vec_znx_big_normalize_tmp_bytes();

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_default<'s, R, P, K, E>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_pk_tmp_bytes_default(res)
        );
        self.glwe_encrypt_pk_internal(
            res,
            Some((pt.to_backend_ref(), 0)),
            pk,
            enc_infos,
            source_xu,
            source_xe,
            scratch,
        )
    }

    fn glwe_encrypt_zero_pk_default<'s, R, K, E>(
        &self,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        assert!(
            scratch.available() >= self.glwe_encrypt_pk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEEncryptPk::glwe_encrypt_pk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_encrypt_pk_tmp_bytes_default(res)
        );
        self.glwe_encrypt_pk_internal(res, None, pk, enc_infos, source_xu, source_xe, scratch)
    }
}

pub(crate) trait GLWEEncryptPkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<'s, R, K, E>(
        &self,
        res: &mut R,
        pt: Option<(GLWEBackendRef<'_, BE>, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWEEncryptPkInternal<BE> for Module<BE>
where
    Self: SvpPrepare<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + ScalarZnxFillTernaryHwSourceBackend<BE>
        + ScalarZnxFillTernaryProbSourceBackend<BE>
        + ScalarZnxFillBinaryHwSourceBackend<BE>
        + ScalarZnxFillBinaryProbSourceBackend<BE>
        + ScalarZnxFillBinaryBlockSourceBackend<BE>
        + SvpPPolBytesOf
        + ModuleN
        + VecZnxDftBytesOf,
{
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_internal<'s, R, K, E>(
        &self,
        res: &mut R,
        pt: Option<(GLWEBackendRef<'_, BE>, usize)>,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();

        assert_eq!(res.base2k(), pk.base2k());
        assert_eq!(res.n(), pk.n());
        assert_eq!(res.rank(), pk.rank());
        if let Some((pt, _)) = &pt {
            assert_eq!(pt.base2k(), pk.base2k());
            assert_eq!(pt.n(), pk.n());
        }

        let base2k: usize = pk.base2k().into();
        let size_pk: usize = pk.size();
        let cols: usize = (res.rank() + 1).into();

        // Generates u according to the underlying secret distribution.
        let scratch = scratch.borrow();
        let (mut u_dft, mut scratch_1) = scratch.take_svp_ppol_scratch(self, 1);

        {
            let (mut u_backend, scratch_2) = scratch_1.take_scalar_znx_scratch(self.n(), 1);
            match pk.dist() {
                Distribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                Distribution::TernaryFixed(hw) => {
                    self.scalar_znx_fill_ternary_hw_source_backend(&mut u_backend, 0, *hw, source_xu)
                }
                Distribution::TernaryProb(prob) => {
                    self.scalar_znx_fill_ternary_prob_source_backend(&mut u_backend, 0, *prob, source_xu)
                }
                Distribution::BinaryFixed(hw) => self.scalar_znx_fill_binary_hw_source_backend(&mut u_backend, 0, *hw, source_xu),
                Distribution::BinaryProb(prob) => {
                    self.scalar_znx_fill_binary_prob_source_backend(&mut u_backend, 0, *prob, source_xu)
                }
                Distribution::BinaryBlock(block_size) => {
                    self.scalar_znx_fill_binary_block_source_backend(&mut u_backend, 0, *block_size, source_xu)
                }
                Distribution::ZERO => {
                    let mut u_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut u_backend);
                    self.vec_znx_zero_backend(&mut u_vec, 0);
                }
            }

            let u_backend_ref = ScalarZnx::from_data(BE::view_ref_mut(&u_backend.data), u_backend.n(), u_backend.cols());
            self.svp_prepare(&mut u_dft, 0, &u_backend_ref, 0);
            scratch_1 = scratch_2;
        }

        {
            let pk = <K as GLWEPreparedToBackendRef<BE>>::to_backend_ref(pk);

            // ct[i] = pk[i] * u + ei (+ m if col = i)
            for i in 0..cols {
                let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft_scratch(self, 1, size_pk);
                // ci_dft = DFT(u) * DFT(pk[i])
                let u_dft_ref = u_dft.to_backend_ref();
                {
                    let mut ci_dft_backend = ci_dft.to_backend_mut();
                    self.svp_apply_dft_to_dft(&mut ci_dft_backend, 0, &u_dft_ref, 0, &pk.data, i);
                }

                // ci_big = u * p[i]
                let (mut ci_big, scratch_3) = scratch_2.take_vec_znx_big_scratch(self, 1, size_pk);
                {
                    let mut ci_big_backend = ci_big.to_backend_mut();
                    let mut ci_dft_backend = ci_dft.to_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut ci_big_backend, 0, &mut ci_dft_backend, 0);
                }

                // ci_big = u * pk[i] + e
                self.vec_znx_big_add_normal(base2k, &mut ci_big, 0, enc_infos.noise_infos(), source_xe);

                let (mut ci, scratch_4) = scratch_3.take_vec_znx_scratch(self.n(), 1, size_pk);
                let scratch_next = {
                    let ci_big_ref = ci_big.to_backend_ref();
                    scratch_4
                        .apply_mut(|scratch| self.vec_znx_big_normalize(&mut ci, base2k, 0, 0, &ci_big_ref, base2k, 0, scratch))
                };
                scratch_1 = scratch_next;

                if let Some((pt, col)) = &pt
                    && *col == i
                {
                    let mut ci_mut = ci.to_backend_mut();
                    self.vec_znx_add_assign_backend(&mut ci_mut, 0, &pt.data, 0);
                }

                let ci_ref = ci.to_backend_ref();
                self.vec_znx_copy_backend(&mut res.data, i, &ci_ref, 0);
            }
        }
    }
}

pub(crate) trait GLWEEncryptSkInternal<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_sk_internal<'s, 'pt, S, E>(
        &self,
        base2k: usize,
        res: &mut VecZnx<BE::BufMut<'_>>,
        cols: usize,
        compressed: bool,
        pt: GLWEEncryptSkPlaintext<'pt, BE>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
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
        + VecZnxFillUniformSourceBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxAddNormalSourceBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxSubNegateAssignBackend<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    fn glwe_encrypt_sk_internal<'s, 'pt, S, E>(
        &self,
        base2k: usize,
        res: &mut VecZnx<BE::BufMut<'_>>,
        cols: usize,
        compressed: bool,
        pt: GLWEEncryptSkPlaintext<'pt, BE>,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let mut res_ref = res;
        let mut ct = <&mut VecZnx<BE::BufMut<'_>> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut res_ref);
        let sk = sk.to_backend_ref();

        if compressed {
            assert_eq!(ct.cols(), 1, "invalid glwe: compressed tag=true but #cols={} != 1", ct.cols())
        }

        assert!(
            sk.dist != Distribution::NONE,
            "glwe secret distribution is NONE (have you prepared the key?)"
        );

        let size: usize = ct.size();

        let scratch_local = scratch.borrow();
        let (mut c0, scratch_1) = scratch_local.take_vec_znx_scratch(self.n(), 1, size);
        let (mut ci, scratch_2) = scratch_1.take_vec_znx_scratch(self.n(), 1, size);
        let mut scratch_2 = scratch_2;
        self.vec_znx_zero_backend(&mut c0, 0);

        for i in 1..cols {
            let col_ct: usize = if compressed { 0 } else { i };
            self.vec_znx_fill_uniform_source_backend(base2k, &mut ct, col_ct, source_xa);

            if let Some((pt, col)) = pt.as_ref() {
                if i == *col {
                    self.vec_znx_copy_backend(&mut ci, 0, &pt.data, 0);
                    let ct_ref = vec_znx_backend_ref_from_mut::<BE>(&ct);
                    self.vec_znx_sub_negate_assign_backend(&mut ci, 0, &ct_ref, col_ct);
                    {
                        let mut scratch_norm = scratch_2.borrow();
                        let mut ci_mut = ci.to_backend_mut();
                        <Module<BE> as VecZnxNormalizeAssignBackend<BE>>::vec_znx_normalize_assign_backend(
                            self,
                            base2k,
                            &mut ci_mut,
                            0,
                            &mut scratch_norm,
                        );
                    }
                } else {
                    let ct_ref = vec_znx_backend_ref_from_mut::<BE>(&ct);
                    self.vec_znx_copy_backend(&mut ci, 0, &ct_ref, col_ct);
                }
            } else {
                let ct_ref = vec_znx_backend_ref_from_mut::<BE>(&ct);
                self.vec_znx_copy_backend(&mut ci, 0, &ct_ref, col_ct);
            }

            {
                let scratch_dft = scratch_2.borrow();
                let (mut ci_dft, scratch_3) = scratch_dft.take_vec_znx_dft_scratch(self, 1, size);
                {
                    let ci_ref = ci.to_backend_ref();
                    let mut ci_dft_mut = ci_dft.to_backend_mut();
                    <Module<BE> as VecZnxDftApply<BE>>::vec_znx_dft_apply(self, 1, 0, &mut ci_dft_mut, 0, &ci_ref, 0);
                }

                {
                    let mut ci_dft_backend = ci_dft.to_backend_mut();
                    self.svp_apply_dft_to_dft_assign(&mut ci_dft_backend, 0, &sk.data, i - 1);
                }
                let (mut ci_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, size);
                {
                    let mut ci_big_backend = ci_big.to_backend_mut();
                    let mut ci_dft_backend = ci_dft.to_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut ci_big_backend, 0, &mut ci_dft_backend, 0);
                }
                {
                    let mut scratch_norm = scratch_4.borrow();
                    let ci_big_ref = ci_big.to_backend_ref();
                    let mut ci_mut = ci.to_backend_mut();
                    <Module<BE> as VecZnxBigNormalize<BE>>::vec_znx_big_normalize(
                        self,
                        &mut ci_mut,
                        base2k,
                        0,
                        0,
                        &ci_big_ref,
                        base2k,
                        0,
                        &mut scratch_norm,
                    );
                }
            }

            let ci_ref = ci.to_backend_ref();
            self.vec_znx_sub_assign_backend(&mut c0, 0, &ci_ref, 0);
        }

        // c[0] += e
        {
            let mut c0_mut = c0.to_backend_mut();
            self.vec_znx_add_normal_source_backend(base2k, &mut c0_mut, 0, enc_infos.noise_infos(), source_xe);
        }

        // c[0] += m if col = 0
        if let Some((pt, col)) = &pt
            && *col == 0
        {
            let mut c0_mut = c0.to_backend_mut();
            self.vec_znx_add_assign_backend(&mut c0_mut, 0, &pt.data, 0);
        }

        {
            let mut scratch_norm = scratch_2.borrow();
            let mut c0_mut = c0.to_backend_mut();
            <Module<BE> as VecZnxNormalizeAssignBackend<BE>>::vec_znx_normalize_assign_backend(
                self,
                base2k,
                &mut c0_mut,
                0,
                &mut scratch_norm,
            );
        }
        let c0_ref = c0.to_backend_ref();
        self.vec_znx_copy_backend(&mut ct, 0, &c0_ref, 0);
    }
}
