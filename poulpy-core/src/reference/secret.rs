//! Zero fills, preparation, tensoring and LWE/GLWE conversion of secrets.

use poulpy_hal::{
    api::{
        ScalarZnxAutomorphism, ScratchArenaTakeBasic, SvpApplyDftToDft, SvpPrepare, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VecZnxZero,
    },
    layouts::{
        Backend, Module, ScratchArena, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref, vec_znx_backend_mut_from_mut,
        vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    GetDistribution, GetDistributionMut, ScratchArenaTakeCore,
    api::GLWESecretPreparedFactory,
    dist::Distribution,
    layouts::{
        Degree, GLWEInfos, GLWESecret, GLWESecretPreparedToBackendMut, GLWESecretTensorPreparedToBackendMut,
        GLWESecretTensorToBackendMut, GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, LWEInfos,
        LWESecret, LWESecretToBackendMut, LWESecretToBackendRef, ModuleCoreAlloc, Rank, operand_degree,
    },
};

/// Reference secret bodies. Backend dispatch is selected independently through
/// [`SecretImpl`](crate::oep::SecretImpl).
pub trait SecretReference<BE: Backend> {
    fn glwe_secret_fill_zero_reference<S>(&self, sk: &mut S)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    fn lwe_secret_fill_zero_reference<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    fn glwe_secret_prepare_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretToBackendRef<BE> + GetDistribution;

    fn glwe_secret_tensor_prepared_prepare_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<BE> + GetDistribution;

    fn glwe_secret_tensor_prepare_tmp_bytes_reference(&self, rank: Rank) -> usize;

    fn glwe_secret_tensor_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_secret_from_lwe_secret_reference<S>(&self, src: &S) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: LWESecretToBackendRef<BE>;

    fn lwe_secret_from_glwe_secret_reference<S>(&self, src: &S, lwe_n: Degree) -> LWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: GLWESecretToBackendRef<BE>;
}

// Coefficient-word fence: `scalar_znx_automorphism` is delegated by
// poulpy-hal for i64 backends only.
impl<BE: Backend<ZnxWord = i64>> SecretReference<BE> for Module<BE>
where
    Module<BE>: GLWESecretPreparedFactory<BE>
        + ScalarZnxAutomorphism<BE>
        + SvpApplyDftToDft<BE>
        + SvpPrepare<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxZero<BE>,
{
    fn glwe_secret_fill_zero_reference<S>(&self, sk: &mut S)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
            for i in 0..rank {
                self.vec_znx_zero(&mut sk_vec, i);
            }
        }
        *sk.dist_mut() = Distribution::ZERO;
    }

    fn lwe_secret_fill_zero_reference<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
            self.vec_znx_zero(&mut sk_vec, 0);
        }
        *sk.dist_mut() = Distribution::ZERO;
    }

    fn glwe_secret_prepare_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretToBackendRef<BE> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            for i in 0..res.rank().into() {
                self.svp_prepare(&mut res.data, i, &other.data, i);
            }
        }

        *res.dist_mut() = *other.dist();
    }

    fn glwe_secret_tensor_prepared_prepare_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<BE> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            assert_eq!(
                res.rank, other.rank,
                "GLWESecretTensorPrepared rank must equal the source tensor's rank"
            );
            let cols: usize = other.data.cols();
            assert_eq!(res.data.cols(), cols);
            for i in 0..cols {
                self.svp_prepare(&mut res.data, i, &other.data, i);
            }
        }

        *res.dist_mut() = *other.dist();
    }

    fn glwe_secret_tensor_prepare_tmp_bytes_reference(&self, rank: Rank) -> usize {
        let n: usize = self.n();
        BE::scratch_aligned(self.glwe_secret_prepared_bytes_of(rank))
            + BE::scratch_aligned(self.bytes_of_vec_znx_dft(n, rank.into(), 1))
            + BE::scratch_aligned(self.bytes_of_vec_znx_dft(n, 1, 1))
            + BE::scratch_aligned(self.bytes_of_vec_znx_big(n, 1, 1))
            + self
                .vec_znx_big_normalize_tmp_bytes()
                .max(self.vec_znx_idft_apply_tmp_bytes())
    }

    fn glwe_secret_tensor_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let distribution = *other.dist();
        let tmp_bytes: usize = self.glwe_secret_tensor_prepare_tmp_bytes_reference(other.rank());
        {
            let res = &mut res.to_backend_mut();
            let a = other.to_backend_ref();

            // `res.rank()` is the rank of the base secret the tensor is derived
            // from; its column count is `pairs(rank)`.
            assert_eq!(res.rank(), a.rank());
            let n: usize = operand_degree(self.n(), &[res.n(), a.n()]);
            assert!(
                scratch.available() >= tmp_bytes,
                "insufficient scratch for GLWE secret tensor preparation"
            );

            let rank: usize = a.rank().into();

            let scratch = scratch.borrow();
            let (mut a_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(a.n(), rank.into());
            let (mut a_dft, scratch_2) = scratch_1.take_vec_znx_dft_scratch(n, rank, 1);
            let (mut a_ij_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(n, 1, 1);
            let (mut a_ij_big_backend, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(n, 1, 1);
            {
                let mut a_prepared_data = a_prepared.data.reborrow_backend_mut();
                for i in 0..rank {
                    self.svp_prepare(&mut a_prepared_data, i, a.data(), i);
                }
            }
            a_prepared.dist = *a.dist();

            let base2k: usize = 17;

            let a_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(a.data());
            for i in 0..rank {
                let mut a_dft_backend = a_dft.to_backend_mut();
                self.vec_znx_dft_apply(1, 0, &mut a_dft_backend, i, &a_backend_vec, i);
            }

            let a_prepared_backend_ref = a_prepared.data.reborrow_backend_ref();
            // Tag of the base secret `a`, carried over as-is: the products below
            // are not distributed like `a`, but their statistics derive from it.
            res.dist = *a.dist();
            let mut res_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(res.data_mut());

            // sk_tensor = sk (x) sk
            // For example: (s0, s1) (x) (s0, s1) = (s0^2, s0s1, s1^2)
            for i in 0..rank {
                for j in i..rank {
                    let idx: usize = i * rank + j - (i * (i + 1) / 2);
                    let a_dft_ref = a_dft.to_backend_ref();
                    {
                        let mut a_ij_dft_backend = a_ij_dft.to_backend_mut();
                        self.svp_apply_dft_to_dft(&mut a_ij_dft_backend, 0, &a_prepared_backend_ref, j, &a_dft_ref, i);
                    }
                    self.vec_znx_idft_apply(
                        &mut a_ij_big_backend.to_backend_mut(),
                        0,
                        &a_ij_dft.to_backend_ref(),
                        0,
                        &mut scratch_4.borrow(),
                    );
                    {
                        let a_ij_big = a_ij_big_backend.to_backend_ref();
                        let res_k = res_backend.size() * base2k;
                        self.vec_znx_big_normalize(
                            &mut res_backend,
                            base2k,
                            res_k,
                            0,
                            idx,
                            &a_ij_big,
                            base2k,
                            0,
                            &mut scratch_4.borrow(),
                        );
                    }
                }
            }
        }
        scratch.wipe(tmp_bytes);
        // Backend views carry copied metadata; commit to the owning object.
        *res.dist_mut() = distribution;
    }

    fn glwe_secret_from_lwe_secret_reference<S>(&self, src: &S) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: LWESecretToBackendRef<BE>,
    {
        let src = src.to_backend_ref();
        assert_eq!(src.n().as_usize(), self.n(), "LWE secret degree must equal module degree");
        let mut res = self.glwe_secret_alloc(Rank(1));
        res.dist = src.dist;
        {
            let mut res_ref = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut res);
            self.scalar_znx_automorphism(-1, res_ref.data_mut(), 0, src.data(), 0);
        }
        res
    }

    fn lwe_secret_from_glwe_secret_reference<S>(&self, src: &S, lwe_n: Degree) -> LWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: GLWESecretToBackendRef<BE>,
    {
        let src = src.to_backend_ref();
        let n: usize = self.n();
        let rank: usize = src.rank().into();
        let target: usize = lwe_n.as_usize();
        assert_eq!(src.n().as_usize(), n, "GLWE secret degree must equal module degree");
        assert!(
            target <= rank * n,
            "lwe_secret_from_glwe_secret: requested LWE dim {} > rank * N ({})",
            target,
            rank * n
        );

        // Scratch buffer shaped like the source (rank cols, ring N) so the
        // per-column automorphism dimensions match. The result is then packed
        // into the flat LWE secret via the backend copy op, truncating the
        // last block when `target % n != 0`.
        let mut tmp: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = self.glwe_secret_alloc(src.rank());
        {
            let mut tmp_ref = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut tmp);
            for j in 0..rank {
                self.scalar_znx_automorphism(-1, tmp_ref.data_mut(), j, src.data(), j);
            }
        }

        let mut res: LWESecret<BE::OwnedBuf, BE::ZnxWord> = self.lwe_secret_alloc(lwe_n);
        res.dist = src.dist;
        {
            let tmp_ref = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut tmp);
            let mut res_ref = LWESecretToBackendMut::<BE>::to_backend_mut(&mut res);
            let tmp_vz = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(tmp_ref.data());
            let mut res_vz = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(res_ref.data_mut());

            let mut written: usize = 0;
            for j in 0..rank {
                if written >= target {
                    break;
                }
                let take: usize = (target - written).min(n);
                self.vec_znx_copy(
                    &mut vec_znx_backend_mut_from_mut::<BE>(&mut res_vz).window_coeffs(written, take),
                    0,
                    &vec_znx_backend_ref_from_ref::<BE>(&tmp_vz).window_coeffs(0, take),
                    j,
                );
                written += take;
            }
        }
        res
    }
}
