//! Preparation of a stored GLWE layout into its DFT form.

use poulpy_hal::{
    api::VecZnxDftApply,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    api::{GLWEBytesOf, GLWENormalize},
    layouts::{GLWEInfos, GLWEPreparedToBackendMut, GLWEToBackendRef, LWEInfos, operand_degree},
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
}

impl<BE: Backend> PreparationReference<BE> for Module<BE>
where
    Module<BE>: GLWENormalize<BE> + VecZnxDftApply<BE>,
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
            operand_degree(self.n(), &[res.n(), other.n()]);
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
}
