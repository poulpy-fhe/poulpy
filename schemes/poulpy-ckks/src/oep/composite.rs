//! Sums, multiply-accumulate and affine forms: derived families over the selected products and additions.
use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos},
    oep::GLWENormalizeImpl,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    oep::{
        CKKSAddImpl, CKKSCopyImpl, CKKSMulImpl, CKKSSubImpl,
        derived::composite::{
            ckks_add_many_derived, ckks_add_many_tmp_bytes_derived, ckks_affine_pt_const_assign_derived,
            ckks_affine_pt_const_into_derived, ckks_affine_pt_const_tmp_bytes_derived, ckks_affine_pt_vec_assign_derived,
            ckks_affine_pt_vec_into_derived, ckks_affine_pt_vec_tmp_bytes_derived, ckks_mul_add_ct_into_derived,
            ckks_mul_add_ct_tmp_bytes_derived, ckks_mul_add_pt_const_into_derived, ckks_mul_add_pt_const_tmp_bytes_derived,
            ckks_mul_add_pt_vec_into_derived, ckks_mul_add_pt_vec_tmp_bytes_derived, ckks_mul_sub_ct_into_derived,
            ckks_mul_sub_ct_tmp_bytes_derived, ckks_mul_sub_pt_const_into_derived, ckks_mul_sub_pt_const_tmp_bytes_derived,
            ckks_mul_sub_pt_vec_into_derived, ckks_mul_sub_pt_vec_tmp_bytes_derived,
        },
    },
};

/// Backend contract of [`CKKSAddManyOps`](crate::api::CKKSAddManyOps).
///
/// Every method defaults to the composition of the selected additions and copy; an
/// override reproduces it bit for bit within its own query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSAddManyImpl: Backend + CKKSAddImpl + CKKSCopyImpl + GLWENormalizeImpl {
    /// Scratch of [`Self::ckks_add_many`]: the largest of the selected `ckks_add_tmp_bytes` at `res`, poulpy-core's
    /// `glwe_normalize_tmp_bytes` and the selected `ckks_copy_tmp_bytes` from `a` to `res`.
    fn ckks_add_many_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        ckks_add_many_tmp_bytes_derived(module, res, a)
    }

    /// `ckks_copy` of a single term, else `ckks_add_into` of the first two terms then `ckks_add_assign` of each later
    /// one, poulpy-core's `glwe_normalize_assign` of `res` running before every addition that follows a multiple of
    /// `normalize_after` additions, the noise record cleared.
    fn ckks_add_many<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &[&A],
        normalize_after: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_add_many_derived(module, res, a, normalize_after, scratch)
    }
}

/// Backend contract of [`CKKSMulAddOps`](crate::api::CKKSMulAddOps).
///
/// Every method defaults to the selected product into a scratch temporary shaped
/// like `dst`, then `ckks_add_assign`; an override reproduces it bit for bit within
/// its own query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSMulAddImpl: Backend + CKKSMulImpl + CKKSAddImpl {
    /// Scratch of [`Self::ckks_mul_add_ct_into`]: the temporary plus the larger of the product and addition queries.
    fn ckks_mul_add_ct_tmp_bytes<R, A, B, T>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_mul_add_ct_tmp_bytes_derived(module, res, a, b, tsk)
    }

    /// Scratch of [`Self::ckks_mul_add_pt_vec_into`]: the temporary plus the larger of the product and addition queries.
    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_add_pt_vec_tmp_bytes_derived(module, res, a, b)
    }

    /// Scratch of [`Self::ckks_mul_add_pt_const_into`]: the temporary plus the larger of the product and addition queries.
    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_add_pt_const_tmp_bytes_derived(module, res, a, b)
    }

    /// `ckks_mul_into` into the temporary, then `ckks_add_assign` into `dst`.
    fn ckks_mul_add_ct_into<Dst, A, B, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        B: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetTensorKey<Self>,
    {
        ckks_mul_add_ct_into_derived(module, dst, a, b, tsk, scratch)
    }

    /// `ckks_mul_pt_vec_into` into the temporary, then `ckks_add_assign` into `dst`.
    fn ckks_mul_add_pt_vec_into<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_add_pt_vec_into_derived(module, dst, a, pt, scratch)
    }

    /// `ckks_mul_pt_const_into` into the temporary, then `ckks_add_assign` into `dst`.
    fn ckks_mul_add_pt_const_into<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_add_pt_const_into_derived(module, dst, a, pt, pt_coeff, scratch)
    }
}

/// Backend contract of [`CKKSAffineOps`](crate::api::CKKSAffineOps).
///
/// Every method defaults to the selected plaintext product then the selected
/// plaintext addition in `dst`; an override reproduces it bit for bit within its own
/// query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSAffineImpl: Backend + CKKSMulImpl + CKKSAddImpl {
    /// Scratch of the constant affine forms: the larger of the product and addition queries.
    fn ckks_affine_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, affine_const: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_affine_pt_const_tmp_bytes_derived(module, res, a, affine_const)
    }

    /// `ckks_mul_pt_const_into` at `scale_coeff`, then `ckks_add_pt_const_assign` at `offset_coeff`.
    fn ckks_affine_pt_const_into<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_affine_pt_const_into_derived(module, dst, a, affine_const, offset_coeff, scale_coeff, scratch)
    }

    /// `ckks_mul_pt_const_assign` at `scale_coeff`, then `ckks_add_pt_const_assign` at `offset_coeff`.
    fn ckks_affine_pt_const_assign<Dst, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_affine_pt_const_assign_derived(module, dst, affine_const, offset_coeff, scale_coeff, scratch)
    }

    /// Scratch of the vector affine forms: the larger of the product and addition queries.
    fn ckks_affine_pt_vec_tmp_bytes<R, A, S>(module: &Module<Self>, res: &R, a: &A, scale: &S) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        S: CKKSInfos,
    {
        ckks_affine_pt_vec_tmp_bytes_derived(module, res, a, scale)
    }

    /// `ckks_mul_pt_vec_into` by `scale`, then `ckks_add_pt_vec_assign` of `offset`.
    fn ckks_affine_pt_vec_into<Dst, A, S, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_affine_pt_vec_into_derived(module, dst, a, scale, offset, scratch)
    }

    /// `ckks_mul_pt_vec_assign` by `scale`, then `ckks_add_pt_vec_assign` of `offset`.
    fn ckks_affine_pt_vec_assign<Dst, S, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_affine_pt_vec_assign_derived(module, dst, scale, offset, scratch)
    }
}

/// Backend contract of [`CKKSMulSubOps`](crate::api::CKKSMulSubOps).
///
/// Every method defaults to the selected product into a scratch temporary shaped
/// like `dst`, then `ckks_sub_assign`; an override reproduces it bit for bit within
/// its own query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSMulSubImpl: Backend + CKKSMulImpl + CKKSSubImpl {
    /// Scratch of [`Self::ckks_mul_sub_ct_into`]: the temporary plus the larger of the product and subtraction queries.
    fn ckks_mul_sub_ct_tmp_bytes<R, A, B, T>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_mul_sub_ct_tmp_bytes_derived(module, res, a, b, tsk)
    }

    /// Scratch of [`Self::ckks_mul_sub_pt_vec_into`]: the temporary plus the larger of the product and subtraction queries.
    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_sub_pt_vec_tmp_bytes_derived(module, res, a, b)
    }

    /// Scratch of [`Self::ckks_mul_sub_pt_const_into`]: the temporary plus the larger of the product and subtraction
    /// queries.
    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_sub_pt_const_tmp_bytes_derived(module, res, a, b)
    }

    /// `ckks_mul_into` into the temporary, then `ckks_sub_assign` from `dst`.
    fn ckks_mul_sub_ct_into<Dst, A, B, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        B: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetTensorKey<Self>,
    {
        ckks_mul_sub_ct_into_derived(module, dst, a, b, tsk, scratch)
    }

    /// `ckks_mul_pt_vec_into` into the temporary, then `ckks_sub_assign` from `dst`.
    fn ckks_mul_sub_pt_vec_into<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_sub_pt_vec_into_derived(module, dst, a, pt, scratch)
    }

    /// `ckks_mul_pt_const_into` into the temporary, then `ckks_sub_assign` from `dst`.
    fn ckks_mul_sub_pt_const_into<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_sub_pt_const_into_derived(module, dst, a, pt, pt_coeff, scratch)
    }
}

/// Selects the derived defaults of `CKKSAddManyImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_add_many_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSAddManyImpl for $be {}
    };
}

/// Selects the derived defaults of `CKKSMulAddImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_mul_add_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSMulAddImpl for $be {}
    };
}

/// Selects the derived defaults of `CKKSAffineImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_affine_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSAffineImpl for $be {}
    };
}

/// Selects the derived defaults of `CKKSMulSubImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_mul_sub_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSMulSubImpl for $be {}
    };
}

pub use crate::{
    impl_ckks_add_many_derived_full, impl_ckks_affine_derived_full, impl_ckks_mul_add_derived_full,
    impl_ckks_mul_sub_derived_full,
};
