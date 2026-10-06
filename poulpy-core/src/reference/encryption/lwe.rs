use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigInnerSum, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSubSmallNegateAssign, VecZnxFillUniformSource, VecZnxScalarProduct,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef},
    source::Source,
};

use crate::{
    LWEFillMask, Noise, VecZnxBigAddNoise,
    layouts::{LWEInfos, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut},
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait LWEFillMaskReference<BE: Backend> {
    fn fill_lwe_mask_from_source_reference<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>;
}

impl<BE: Backend> LWEFillMaskReference<BE> for Module<BE>
where
    Self: VecZnxFillUniformSource<BE>,
{
    fn fill_lwe_mask_from_source_reference<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>,
    {
        {
            let mut res = res.to_backend_mut();
            assert_eq!(res.mask.cols(), 1, "fill_lwe_mask_from_source: LWE mask cols must be 1");
            self.vec_znx_fill_uniform_source(base2k, res.k().as_usize(), &mut res.mask, 0, source_xa);
        }
        res.set_noise(None);
    }
}

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait LWEEncryptSkReference<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>;
}

impl<BE: Backend> LWEEncryptSkReference<BE> for Module<BE>
where
    Self: Sized
        + LWEFillMask<BE>
        + VecZnxBigAddNoise<BE>
        + VecZnxBigBytesOf
        + VecZnxBigInnerSum<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxScalarProduct<BE>
        + VecZnxBigSubSmallNegateAssign<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        let n: usize = infos.n().into();
        let size: usize = infos.size();
        let tmp_hadamard: usize = self.bytes_of_vec_znx_big(n, 1, size);
        let tmp_scalar: usize = self.bytes_of_vec_znx_big(1, 1, size);
        let normalize: usize = self.vec_znx_big_normalize_tmp_bytes();
        (tmp_hadamard + tmp_scalar).next_multiple_of(BE::SCRATCH_ALIGN) + normalize
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
    {
        let metadata = Some(crate::ComponentNoise::from_secret_at(
            sk.to_backend_ref().dist,
            crate::layouts::LWEInfos::k(&res.to_backend_ref()),
            crate::layouts::LWEInfos::n(&res.to_backend_ref()).as_usize(),
        ));
        {
            let pt = pt.to_backend_ref();
            let sk = sk.to_backend_ref();

            #[cfg(debug_assertions)]
            {
                assert_eq!(res.n(), sk.n())
            }

            assert!(
                scratch.available() >= self.lwe_encrypt_sk_tmp_bytes_reference(res),
                "scratch.available(): {} < LWEEncryptSk::lwe_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.lwe_encrypt_sk_tmp_bytes_reference(res)
            );
            let tmp_bytes: usize = self.lwe_encrypt_sk_tmp_bytes_reference(res);
            {
                let base2k: usize = res.base2k().into();
                let res_n: usize = res.n().into();
                let res_size = res.size();
                self.fill_lwe_mask_from_source(base2k, res, source_xa);

                // tmp_hadamard[limb][k] = mask[limb][k] * sk[k]  (element-wise, BigScalar)
                let (mut tmp_hadamard, scratch_1) = scratch.borrow().take_vec_znx_big_scratch(res_n, 1, res_size);
                {
                    let res_ref = res.to_backend_ref();
                    self.vec_znx_scalar_product(&mut tmp_hadamard, 0, &res_ref.mask, 0, &sk.data, 0);
                }

                // tmp_scalar[limb][0] = sum_k tmp_hadamard[limb][k] = <mask, sk>
                let (mut tmp_scalar, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(1, 1, res_size);
                self.vec_znx_big_inner_sum(&mut tmp_scalar, 0, 0, &tmp_hadamard.to_backend_ref(), 0);

                // tmp_scalar = m - <mask, sk>
                self.vec_znx_big_sub_small_negate_assign(&mut tmp_scalar, 0, &pt.data, 0);

                // tmp_scalar = m - <mask, sk> + e
                self.vec_znx_big_add_noise(base2k, res.k().as_usize(), &mut tmp_scalar, 0, Noise::ENCRYPTION, source_xe);

                // Normalize into res.body.
                {
                    let res_k = res.k().as_usize();
                    let mut res_mut = res.to_backend_mut();
                    self.vec_znx_big_normalize(
                        &mut res_mut.body,
                        base2k,
                        res_k,
                        0,
                        0,
                        &tmp_scalar.to_backend_ref(),
                        base2k,
                        0,
                        &mut scratch_2.borrow(),
                    )
                }
            }
            scratch.wipe(tmp_bytes);
        }
        res.set_noise(metadata);
    }
}
