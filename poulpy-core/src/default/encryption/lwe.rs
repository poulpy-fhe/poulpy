use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddNormal, VecZnxBigBytesOf, VecZnxBigInnerSumBackend, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateAssign, VecZnxFillUniformSourceBackend, VecZnxScalarProduct,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef},
    source::Source,
};

use crate::{
    EncryptionInfos,
    layouts::{LWEInfos, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut},
};

#[doc(hidden)]
pub trait LWEEncryptSkDefault<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_default<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
        E: EncryptionInfos;
}

impl<BE: Backend> LWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized
        + VecZnxFillUniformSourceBackend<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigBytesOf
        + VecZnxBigInnerSumBackend<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxScalarProduct<BE>
        + VecZnxBigSubSmallNegateAssign<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        let n: usize = infos.n().into();
        let size: usize = infos.size();
        let tmp_hadamard: usize = self.bytes_of_vec_znx_big_n(n, 1, size);
        let tmp_scalar: usize = self.bytes_of_vec_znx_big_n(1, 1, size);
        let normalize: usize = self.vec_znx_big_normalize_tmp_bytes();
        (tmp_hadamard + tmp_scalar).next_multiple_of(BE::SCRATCH_ALIGN) + normalize
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_encrypt_sk_default<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
        E: EncryptionInfos,
    {
        let pt = pt.to_backend_ref();
        let sk = sk.to_backend_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n())
        }

        assert!(
            scratch.available() >= self.lwe_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < LWEEncryptSk::lwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_encrypt_sk_tmp_bytes_default(res)
        );

        let base2k: usize = res.base2k().into();
        let res_n: usize = res.n().into();
        let res_size = res.size();
        {
            // Sample the mask directly into res.mask.
            let mut res_mut = res.to_backend_mut();
            self.vec_znx_fill_uniform_source_backend(base2k, &mut res_mut.mask, 0, source_xa);
        }

        // tmp_hadamard[limb][k] = mask[limb][k] * sk[k]  (element-wise, BigScalar)
        let (mut tmp_hadamard, scratch_1) = scratch.borrow().take_vec_znx_big_scratch_n(res_n, 1, res_size);
        {
            let res_ref = res.to_backend_ref();
            self.vec_znx_scalar_product(&mut tmp_hadamard, 0, &res_ref.mask, 0, &sk.data, 0);
        }

        // tmp_scalar[limb][0] = sum_k tmp_hadamard[limb][k] = <mask, sk>
        let (mut tmp_scalar, mut scratch_2) = scratch_1.take_vec_znx_big_scratch_n(1, 1, res_size);
        self.vec_znx_big_inner_sum_backend(&mut tmp_scalar, 0, 0, &tmp_hadamard.to_backend_ref(), 0);

        // tmp_scalar = m - <mask, sk>
        self.vec_znx_big_sub_small_negate_assign(&mut tmp_scalar, 0, &pt.data, 0);

        // tmp_scalar = m - <mask, sk> + e
        self.vec_znx_big_add_normal(base2k, &mut tmp_scalar, 0, enc_infos.noise_infos(), source_xe);

        // Normalize into res.body.
        {
            let mut res_mut = res.to_backend_mut();
            self.vec_znx_big_normalize(
                &mut res_mut.body,
                base2k,
                0,
                0,
                &tmp_scalar.to_backend_ref(),
                base2k,
                0,
                &mut scratch_2.borrow(),
            )
        }
    }
}
