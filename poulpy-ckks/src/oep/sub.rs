use crate::default::sub::CKKSSubDefault;

use anyhow::Result;
use poulpy_core::{GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore, layouts::LWEInfos};
use poulpy_hal::{
    api::{VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSSubImpl<BE: Backend>: Backend {
    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_sub_into<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_sub_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos;
    fn ckks_sub_one_assign<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos;
    fn ckks_sub_pt_vec_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_sub_pt_vec_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_sub_pt_vec_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_sub_pt_const_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_sub_pt_const_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> CKKSSubImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::sub::CKKSSubDefault<BE>
        + GLWENormalize<BE>
        + GLWEShift<BE>
        + GLWESub<BE>
        + VecZnxRshSubBackend<BE>
        + VecZnxRshSubCoeffIntoBackend<BE>
        + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_sub_tmp_bytes_default()
    }

    fn ckks_sub_into<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_sub_into_default(dst, a, b, scratch)
    }

    fn ckks_sub_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        module.ckks_sub_assign_default(dst, a, scratch)
    }

    fn ckks_sub_one_assign<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_sub_one_assign_default(dst, scratch)
    }

    fn ckks_sub_pt_vec_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_sub_pt_vec_tmp_bytes_default()
    }

    fn ckks_sub_pt_vec_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        crate::default::sub::CKKSSubDefault::ckks_sub_pt_vec_into_default(module, dst, a, pt, scratch)
    }

    fn ckks_sub_pt_vec_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_sub_pt_vec_assign_default(dst, pt, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_sub_pt_const_tmp_bytes_default()
    }

    fn ckks_sub_pt_const_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_sub_pt_const_into_default(dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_sub_pt_const_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_sub_pt_const_assign_default(dst, dst_coeff, pt, pt_coeff, scratch)
    }
}
