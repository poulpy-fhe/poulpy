use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarAssign, VecZnxDftBytesOf, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxZero},
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::api::GLWEBytesOf;
use crate::{
    EncryptionInfos, GLWEEncryptSk, GLWEEncryptSkInternal, ScratchArenaTakeCore,
    encryption::glwe::GLWEMaskFillReference,
    layouts::{
        GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GGSWEncryptSkReference<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos;
}

/// Independently callable portable composition for [`GGSWEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GGSWEncryptSkComposition<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos;
}

impl<BE: Backend> GGSWEncryptSkComposition<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + GLWEMaskFillReference<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeAssign<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxZero<BE>,
{
    fn ggsw_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_plaintext_bytes_of_from_infos(infos);
        lvl_0 + self.glwe_encrypt_sk_tmp_bytes(infos).max(self.vec_znx_normalize_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ggsw_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos,
    {
        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n());
        assert_eq!(sk.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.ggsw_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < GGSWEncryptSk::ggsw_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_encrypt_sk_tmp_bytes_composition(res)
        );

        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let (mut tmp_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(res);
        let tmp_pt_k = tmp_pt.k().as_usize();

        for row_i in 0..res.dnum().into() {
            self.vec_znx_zero(&mut tmp_pt.data, 0);
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            {
                let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                self.vec_znx_add_scalar_assign(
                    &mut tmp_pt_backend.data,
                    0,
                    (dsize - 1) + row_i * dsize,
                    &pt.to_backend_ref(),
                    0,
                );
            }

            self.vec_znx_normalize_assign(base2k, tmp_pt_k, 0, &mut tmp_pt.data, 0, &mut scratch_1.borrow());
            for col_j in 0..rank + 1 {
                let mut ct = res.at_view_mut(row_i, col_j);
                self.fill_glwe_mask_from_source_reference(base2k, &mut ct, 1, rank, source_xa);
                self.glwe_encrypt_sk_internal(
                    base2k,
                    &mut ct.data,
                    Some((tmp_pt.to_backend_ref(), col_j)),
                    sk,
                    enc_infos,
                    source_xe,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}

/// Forwards every method of [`GGSWEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_ggsw_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GGSWEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkComposition<$be>>::ggsw_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
        }

    fn ggsw_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos + $crate::layouts::GGSWAtViewMut<$be>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be> + ::poulpy_hal::layouts::ZnxInfos,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::LWEInfos + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkComposition<$be>>::ggsw_encrypt_sk_composition::<R, P, S, E>(self, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }
        }
    };
}
