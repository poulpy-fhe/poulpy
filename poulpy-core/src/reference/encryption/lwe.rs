use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigInnerSum, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSubSmallNegateAssign, VecZnxFillUniformSource, VecZnxScalarProduct,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef},
    source::Source,
};

use crate::{
    EncryptionInfos, VecZnxBigAddNormal,
    layouts::{LWEInfos, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut},
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait LWEFillMaskReference<BE: Backend> {
    fn fill_lwe_mask_from_source_reference<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>;

    fn fill_lwe_mask_from_seed_reference<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>;
}

/// Independently callable portable composition for [`LWEFillMaskReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait LWEFillMaskComposition<BE: Backend> {
    fn fill_lwe_mask_from_source_composition<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>;

    fn fill_lwe_mask_from_seed_composition<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>;
}

impl<BE: Backend> LWEFillMaskComposition<BE> for Module<BE>
where
    Self: VecZnxFillUniformSource<BE>,
{
    fn fill_lwe_mask_from_source_composition<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();
        assert_eq!(res.mask.cols(), 1, "fill_lwe_mask_from_source: LWE mask cols must be 1");
        self.vec_znx_fill_uniform_source(base2k, res.k().as_usize(), &mut res.mask, 0, source_xa);
    }

    fn fill_lwe_mask_from_seed_composition<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>,
    {
        let mut source_xa = Source::new(seed_xa);
        self.fill_lwe_mask_from_source_composition(base2k, res, &mut source_xa);
    }
}

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait LWEEncryptSkReference<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_reference<R, P, S, E>(
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

/// Independently callable portable composition for [`LWEEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait LWEEncryptSkComposition<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_composition<R, P, S, E>(
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

impl<BE: Backend> LWEEncryptSkComposition<BE> for Module<BE>
where
    Self: Sized
        + LWEFillMaskReference<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigBytesOf
        + VecZnxBigInnerSum<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxScalarProduct<BE>
        + VecZnxBigSubSmallNegateAssign<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
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
    fn lwe_encrypt_sk_composition<R, P, S, E>(
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
            scratch.available() >= self.lwe_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < LWEEncryptSk::lwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_encrypt_sk_tmp_bytes_composition(res)
        );

        let base2k: usize = res.base2k().into();
        let res_n: usize = res.n().into();
        let res_size = res.size();
        self.fill_lwe_mask_from_source_reference(base2k, res, source_xa);

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
        self.vec_znx_big_add_normal(base2k, &mut tmp_scalar, 0, enc_infos.noise_infos(), source_xe);

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
}

/// Forwards every method of [`LWEFillMaskReference`] to its portable composition.
#[macro_export]
macro_rules! impl_lwe_mask_fill_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::LWEFillMaskReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn fill_lwe_mask_from_source_reference<R>(&self, base2k: usize, res: &mut R, source_xa: &mut ::poulpy_hal::source::Source)
    where
        R: $crate::layouts::LWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEFillMaskComposition<$be>>::fill_lwe_mask_from_source_composition::<R>(self, base2k, res, source_xa)
        }

    fn fill_lwe_mask_from_seed_reference<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: $crate::layouts::LWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEFillMaskComposition<$be>>::fill_lwe_mask_from_seed_composition::<R>(self, base2k, res, seed_xa)
        }
        }
    };
}

/// Forwards every method of [`LWEEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_lwe_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::LWEEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn lwe_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: $crate::layouts::LWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkComposition<$be>>::lwe_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
        }

    fn lwe_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
        P: $crate::layouts::LWEPlaintextToBackendRef<$be>,
        S: $crate::layouts::LWESecretToBackendRef<$be>,
        E: $crate::api::EncryptionInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkComposition<$be>>::lwe_encrypt_sk_composition::<R, P, S, E>(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }
        }
    };
}
