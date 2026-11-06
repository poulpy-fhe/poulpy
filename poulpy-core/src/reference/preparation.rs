//! Preparation of stored GLWE, GGLWE, GGSW, public-key and GGLWE-to-GGSW-key layouts.

use poulpy_hal::{
    api::{VecZnxDftApply, VmpPMatPrepare, VmpPMatPrepareTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    GetDistribution, GetDistributionMut, ScratchArenaTakeGLWE,
    api::{GLWEBytesOf, GLWENormalize},
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef,
        GGSWInfos, GGSWPreparedToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEPreparedToBackendMut,
        GLWEPublicKeyPreparedToBackendMut, GLWEPublicKeyToBackendRef, GLWEToBackendRef, GetDegree, LWEInfos, operand_degree,
    },
};

/// Reference bodies of the required [`PreparationImpl`](crate::oep::PreparationImpl) methods, which
/// [`impl_preparation_reference_full!`](crate::impl_preparation_reference_full) forwards to.
pub trait PreparationReference<BE: Backend> {
    fn glwe_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPreparedToBackendMut<BE>,
        O: GLWEToBackendRef<BE> + GLWEInfos;

    fn gglwe_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>;

    fn ggsw_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWPreparedToBackendMut<BE>,
        O: GGSWToBackendRef<BE>;

    fn glwe_public_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_public_key_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<BE> + GetDistribution;

    fn gglwe_to_ggsw_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>;
}

impl<BE: Backend> PreparationReference<BE> for Module<BE>
where
    Module<BE>: GLWENormalize<BE> + VecZnxDftApply<BE> + VmpPMatPrepare<BE> + VmpPMatPrepareTmpBytes,
{
    fn glwe_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(infos)) + self.glwe_normalize_tmp_bytes()
    }

    fn glwe_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPreparedToBackendMut<BE>,
        O: GLWEToBackendRef<BE> + GLWEInfos,
    {
        {
            let res = res.to_backend_mut();
            assert_eq!(res.rank(), other.rank(), "glwe_prepare: rank mismatch");
            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
            assert_eq!(res.size(), other.size());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.base2k(), other.base2k());
        }
        let tmp_bytes: usize = self.glwe_prepare_tmp_bytes_reference(other);
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available(): {} < GLWEPreparedFactory::glwe_prepare_tmp_bytes: {}",
            scratch.available(),
            tmp_bytes
        );
        let noise = other.to_backend_ref().noise();
        {
            let (mut other_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(other);
            let other = if other.is_canonical() {
                other.to_backend_ref()
            } else {
                self.glwe_normalize(&mut other_tmp, other, &mut scratch.borrow());
                other_tmp.to_backend_ref()
            };
            let mut res = res.to_backend_mut();

            for i in 0..(res.rank() + 1).into() {
                self.vec_znx_dft_apply(1, 0, &mut res.data, i, &other.data, i);
            }
        }
        res.set_noise(noise);
    }

    fn gglwe_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.vmp_pmat_prepare_tmp_bytes(
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        lvl_0
    }

    fn gglwe_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>,
    {
        let noise = other.to_backend_ref().noise();
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();

            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
            assert_eq!(res.base2k, other.base2k);
            assert_eq!(res.size(), other.size());
            assert_eq!(res.k_digit, other.k_digit);
            assert_eq!(res.k_offset, other.k_offset, "gglwe_prepare: row offsets differ");
            assert!(
                scratch.available() >= self.gglwe_prepare_tmp_bytes_reference(&res),
                "scratch.available(): {} < GGLWEPreparedFactory::gglwe_prepare_tmp_bytes: {}",
                scratch.available(),
                self.gglwe_prepare_tmp_bytes_reference(&res)
            );
            self.vmp_pmat_prepare(&mut res.data, &other.data, scratch);
        }
        res.set_noise(noise);
    }

    fn ggsw_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        let lvl_0: usize = self.vmp_pmat_prepare_tmp_bytes(
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        lvl_0
    }

    fn ggsw_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWPreparedToBackendMut<BE>,
        O: GGSWToBackendRef<BE>,
    {
        let noise = other.to_backend_ref().noise();
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
            assert_eq!(res.base2k, other.base2k);
            assert_eq!(res.k_digit, other.k_digit);
            assert!(
                scratch.available() >= self.ggsw_prepare_tmp_bytes_reference(&res),
                "scratch.available(): {} < GGSWPreparedFactory::ggsw_prepare_tmp_bytes: {}",
                scratch.available(),
                self.ggsw_prepare_tmp_bytes_reference(&res)
            );
            self.vmp_pmat_prepare(&mut res.data, &other.data, scratch);
        }
        res.set_noise(noise);
    }

    fn glwe_public_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let rank: usize = infos.rank().into();
        self.vmp_pmat_prepare_tmp_bytes(1, rank, rank + 1, infos.size())
    }

    fn glwe_public_key_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<BE> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            assert!(
                res.data.cols_in() == other.data.cols_in(),
                "glwe_public_key_prepare: public key and prepared public key have different entry counts"
            );
            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
            assert_eq!(res.base2k(), other.base2k());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.size(), other.size());

            self.vmp_pmat_prepare(&mut res.data, &other.data, scratch);
        }
        res.set_noise(other.to_backend_ref().noise());
        *res.dist_mut() = *other.dist();
    }

    fn gglwe_to_ggsw_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.vmp_pmat_prepare_tmp_bytes(
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        lvl_0
    }

    fn gglwe_to_ggsw_key_prepare_reference<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>,
    {
        let noise = other.to_backend_ref().noise();
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            assert_eq!(res.keys.len(), other.keys.len());
            let needed: usize = self.gglwe_to_ggsw_key_prepare_tmp_bytes_reference(&res);
            assert!(
                scratch.available() >= needed,
                "scratch.available(): {} < GGLWEToGGSWKeyPreparedFactory::gglwe_to_ggsw_key_prepare_tmp_bytes: {}",
                scratch.available(),
                needed
            );
            for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
                self.vmp_pmat_prepare(&mut a.data, &b.data, &mut scratch.borrow());
            }
        }
        res.set_noise(noise);
    }
}
