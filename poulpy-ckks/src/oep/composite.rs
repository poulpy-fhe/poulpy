//! Sums, multiply-accumulate and affine forms: derived families over the selected products and additions.
use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
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
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSAddManyImpl: Backend + CKKSAddImpl + CKKSCopyImpl {
    /// Default body: `ckks_add_many_tmp_bytes_derived`, the larger of the selected `ckks_add_tmp_bytes` at
    /// `res.max_size()` and the selected `ckks_copy_tmp_bytes` from `input` to `res`, the copy serving a single input.
    /// Scratch bound of [`Self::ckks_add_many`]; may report less than the reference, never less than the body takes.
    fn ckks_add_many_tmp_bytes<R, A>(module: &Module<Self>, res: &R, input: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        ckks_add_many_tmp_bytes_derived(module, res, input)
    }

    /// Default body: `ckks_add_many_derived`. No input is a refusal. One input is the selected `ckks_copy`. Two or more
    /// are `ckks_add_into` of the first two inputs, then `ckks_add_assign` of each later one. The noise record is
    /// cleared in every case.
    fn ckks_add_many<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        inputs: &[&Src],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_add_many_derived(module, dst, inputs, scratch)
    }
}

/// Backend contract of [`CKKSMulAddOps`](crate::api::CKKSMulAddOps).
///
/// Every method defaults to the selected product into a scratch temporary shaped
/// like `dst`, then `ckks_add_assign`; an override reproduces it bit for bit within
/// its own query.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSMulAddImpl: Backend + CKKSMulImpl + CKKSAddImpl {
    /// Default body: `ckks_mul_add_ct_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger of
    /// the selected `ckks_mul_tmp_bytes` and `ckks_add_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_add_ct_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_add_ct_tmp_bytes<R, A, B, T>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_mul_add_ct_tmp_bytes_derived(module, res, a, b, tsk)
    }

    /// Default body: `ckks_mul_add_pt_vec_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger
    /// of the selected `ckks_mul_pt_vec_tmp_bytes` and `ckks_add_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_add_pt_vec_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_add_pt_vec_tmp_bytes_derived(module, res, a, b)
    }

    /// Default body: `ckks_mul_add_pt_const_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger
    /// of the selected `ckks_mul_pt_const_tmp_bytes` and `ckks_add_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_add_pt_const_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_add_pt_const_tmp_bytes_derived(module, res, a, b)
    }

    /// Default body: `ckks_mul_add_ct_into_derived`, the tensor key at the wider of `a.k()` and `b.k()`
    /// (`MissingRelinearizationKey`) and an arena below [`Self::ckks_mul_add_ct_tmp_bytes`] at its layout
    /// (`CKKSError::Internal`) refused first, then `ckks_mul_into` into a `dst`-shaped scratch temporary and
    /// `ckks_add_assign` of the temporary into `dst`.
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

    /// Default body: `ckks_mul_add_pt_vec_into_derived`, an arena below [`Self::ckks_mul_add_pt_vec_tmp_bytes`]
    /// refused first (`CKKSError::Internal`), then `ckks_mul_pt_vec_into` into a `dst`-shaped scratch temporary and
    /// `ckks_add_assign` of the temporary into `dst`.
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

    /// Default body: `ckks_mul_add_pt_const_into_derived`, an arena below
    /// [`Self::ckks_mul_add_pt_const_tmp_bytes`] refused first (`CKKSError::Internal`), then `ckks_mul_pt_const_into`
    /// into a `dst`-shaped scratch temporary and `ckks_add_assign` of the temporary into `dst`.
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
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSAffineImpl: Backend + CKKSMulImpl + CKKSAddImpl {
    /// Default body: `ckks_affine_pt_const_tmp_bytes_derived`, the larger of the selected `ckks_mul_pt_const_tmp_bytes`
    /// and `ckks_add_pt_const_tmp_bytes` at `res.size()`. Scratch bound of [`Self::ckks_affine_pt_const_into`] and
    /// [`Self::ckks_affine_pt_const_assign`]; may report less than the reference, never less than the body takes.
    fn ckks_affine_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, affine_const: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_affine_pt_const_tmp_bytes_derived(module, res, a, affine_const)
    }

    /// Default body: `ckks_affine_pt_const_into_derived`, `ckks_mul_pt_const_into` at `scale_coeff`, then
    /// `ckks_add_pt_const_assign` of the same constant at `offset_coeff` into coefficient 0 of `dst`.
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

    /// Default body: `ckks_affine_pt_const_assign_derived`, `ckks_mul_pt_const_assign` at `scale_coeff`, then
    /// `ckks_add_pt_const_assign` of the same constant at `offset_coeff` into coefficient 0 of `dst`.
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

    /// Default body: `ckks_affine_pt_vec_tmp_bytes_derived`, the larger of the selected `ckks_mul_pt_vec_tmp_bytes` and
    /// `ckks_add_pt_vec_tmp_bytes` at `res.size()`. Scratch bound of [`Self::ckks_affine_pt_vec_into`] and
    /// [`Self::ckks_affine_pt_vec_assign`]; may report less than the reference, never less than the body takes.
    fn ckks_affine_pt_vec_tmp_bytes<R, A, S>(module: &Module<Self>, res: &R, a: &A, scale: &S) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        S: CKKSInfos,
    {
        ckks_affine_pt_vec_tmp_bytes_derived(module, res, a, scale)
    }

    /// Default body: `ckks_affine_pt_vec_into_derived`, `ckks_mul_pt_vec_into` by `scale`, then
    /// `ckks_add_pt_vec_assign` of `offset`.
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

    /// Default body: `ckks_affine_pt_vec_assign_derived`, `ckks_mul_pt_vec_assign` by `scale`, then
    /// `ckks_add_pt_vec_assign` of `offset`.
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
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSMulSubImpl: Backend + CKKSMulImpl + CKKSSubImpl {
    /// Default body: `ckks_mul_sub_ct_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger of
    /// the selected `ckks_mul_tmp_bytes` and `ckks_sub_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_sub_ct_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_sub_ct_tmp_bytes<R, A, B, T>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_mul_sub_ct_tmp_bytes_derived(module, res, a, b, tsk)
    }

    /// Default body: `ckks_mul_sub_pt_vec_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger
    /// of the selected `ckks_mul_pt_vec_tmp_bytes` and `ckks_sub_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_sub_pt_vec_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_sub_pt_vec_tmp_bytes_derived(module, res, a, b)
    }

    /// Default body: `ckks_mul_sub_pt_const_tmp_bytes_derived`, the bytes of the `dst`-shaped temporary plus the larger
    /// of the selected `ckks_mul_pt_const_tmp_bytes` and `ckks_sub_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_mul_sub_pt_const_into`]; may report less than the reference, never less than the body takes.
    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        ckks_mul_sub_pt_const_tmp_bytes_derived(module, res, a, b)
    }

    /// Default body: `ckks_mul_sub_ct_into_derived`, the tensor key at the wider of `a.k()` and `b.k()`
    /// (`MissingRelinearizationKey`) and an arena below [`Self::ckks_mul_sub_ct_tmp_bytes`] at its layout
    /// (`CKKSError::Internal`) refused first, then `ckks_mul_into` into a `dst`-shaped scratch temporary and
    /// `ckks_sub_assign` of the temporary from `dst`.
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

    /// Default body: `ckks_mul_sub_pt_vec_into_derived`, an arena below [`Self::ckks_mul_sub_pt_vec_tmp_bytes`]
    /// refused first (`CKKSError::Internal`), then `ckks_mul_pt_vec_into` into a `dst`-shaped scratch temporary and
    /// `ckks_sub_assign` of the temporary from `dst`.
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

    /// Default body: `ckks_mul_sub_pt_const_into_derived`, an arena below
    /// [`Self::ckks_mul_sub_pt_const_tmp_bytes`] refused first (`CKKSError::Internal`), then `ckks_mul_pt_const_into`
    /// into a `dst`-shaped scratch temporary and `ckks_sub_assign` of the temporary from `dst`.
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
