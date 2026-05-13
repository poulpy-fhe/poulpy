use crate::default::add::CKKSAddDefault;

use anyhow::Result;
use poulpy_core::{GLWEAdd, GLWENormalize, GLWEShift, ScratchArenaTakeCore, layouts::LWEInfos};
use poulpy_hal::{
    api::{VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena},
    oep::HalVecZnxImpl,
};

use crate::{
    CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    default::plaintext::CKKSPlaintextDefault,
    layouts::{CKKSCiphertext, UnnormalizedCKKSCiphertext, ciphertext::UnnormalizedCKKSCiphertextRefMut},
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSAddImpl<BE: Backend>: Backend {
    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_add_into<Dst, A, B>(
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
    fn ckks_add_into_unnormalized<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos;
    fn ckks_add_assign_unnormalized<Dst, A>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos;
    fn ckks_add_assign_unnormalized_ref<Dst, A>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertextRefMut<'_, Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos;
    fn ckks_add_one_assign<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_pt_vec_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_add_pt_vec_into<Dst, A, P>(
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
    fn ckks_add_pt_vec_into_unnormalized<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_pt_vec_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_pt_vec_assign_unnormalized<Dst, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_add_pt_const_into<Dst, A, P>(
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
    fn ckks_add_pt_const_into_unnormalized<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
    fn ckks_add_pt_const_assign<Dst, P>(
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
    fn ckks_add_pt_const_assign_unnormalized<Dst, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> CKKSAddImpl<BE> for BE
where
    BE: HalVecZnxImpl<BE>,
    Module<BE>: CKKSAddDefault<BE>
        + CKKSPlaintextDefault<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWEShift<BE>
        + VecZnxRshAddCoeffIntoBackend<BE>
        + VecZnxRshAddIntoBackend<BE>
        + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_add_tmp_bytes_default()
    }

    fn ckks_add_into<Dst, A, B>(
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
        module.ckks_add_into_default(dst, a, b, scratch)
    }

    fn ckks_add_into_unnormalized<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_into_unsafe_default(dst, a, b, scratch)
    }

    fn ckks_add_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        module.ckks_add_assign_default(dst, a, scratch)
    }

    fn ckks_add_assign_unnormalized<Dst, A>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        module.ckks_add_assign_unsafe_default(dst, a, scratch)
    }

    fn ckks_add_assign_unnormalized_ref<Dst, A>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertextRefMut<'_, Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos,
    {
        module.ckks_add_assign_unsafe_default(dst.inner, a, scratch)
    }

    fn ckks_add_one_assign<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_one_assign_default(dst, scratch)
    }

    fn ckks_add_pt_vec_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_add_pt_vec_tmp_bytes_default()
    }

    fn ckks_add_pt_vec_into<Dst, A, P>(
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
        crate::default::add::CKKSAddDefault::ckks_add_pt_vec_into_default(module, dst, a, pt, scratch)
    }

    fn ckks_add_pt_vec_into_unnormalized<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_pt_vec_into_unsafe_default(dst, a, pt, scratch)
    }

    fn ckks_add_pt_vec_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_pt_vec_assign_default(dst, pt, scratch)
    }

    fn ckks_add_pt_vec_assign_unnormalized<Dst, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_pt_vec_assign_unsafe_default(dst, pt, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_add_pt_const_tmp_bytes_default()
    }

    fn ckks_add_pt_const_into<Dst, A, P>(
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
        crate::default::add::CKKSAddDefault::ckks_add_pt_const_into_default(module, dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_into_unnormalized<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_pt_const_into_unsafe_default(dst, a, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_assign<Dst, P>(
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
        module.ckks_add_pt_const_assign_default(dst, dst_coeff, pt, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_assign_unnormalized<Dst, P>(
        module: &Module<BE>,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_add_pt_const_assign_unsafe_default(dst, dst_coeff, pt, pt_coeff, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_add_defaults {
    ($be:ty) => {
        impl $crate::default::add::CKKSAddDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_add_defaults;
