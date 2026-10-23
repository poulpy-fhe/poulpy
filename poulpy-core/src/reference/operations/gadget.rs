use poulpy_hal::{
    api::{MatZnxCopy, MatZnxNormalize, MatZnxNormalizeTmpBytes, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, LWEInfos, operand_degree,
};

/// Rows copy limb by limb when nothing is lost; rows of a GGLWE or a GGSW are canonical.
fn copies_raw<R: LWEInfos, A: LWEInfos>(res: &R, a: &A) -> bool {
    res.base2k() == a.base2k() && res.k() >= a.k()
}

/// One `mat_znx_copy` or `mat_znx_normalize` per output column: the first `res` rows of `a` at `res`'s radix and
/// precision.
#[expect(clippy::too_many_arguments, reason = "the radix and precision of both operands")]
fn mat_znx_rows_copy<BE: Backend>(
    module: &Module<BE>,
    res: &mut poulpy_hal::layouts::MatZnxBackendMut<'_, BE>,
    res_base2k: usize,
    res_k: usize,
    a: &poulpy_hal::layouts::MatZnxBackendRef<'_, BE>,
    a_base2k: usize,
    raw: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: MatZnxCopy<BE> + MatZnxNormalize<BE>,
{
    for col in 0..res.cols_out() {
        if raw {
            module.mat_znx_copy(res, col, a, col);
        } else {
            module.mat_znx_normalize(res, res_base2k, res_k, 0, col, a, a_base2k, col, &mut scratch.borrow());
        }
    }
}

/// HAL-based reference implementation, independently callable from backend overrides.
pub trait GGLWECopyReference<BE: Backend> {
    fn gglwe_copy_tmp_bytes_reference<R: GGLWEInfos, A: GGLWEInfos>(&self, res: &R, a: &A) -> usize;

    fn gglwe_copy_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GGLWECopyReference<BE> for Module<BE>
where
    Self: ModuleN + MatZnxCopy<BE> + MatZnxNormalize<BE> + MatZnxNormalizeTmpBytes,
{
    fn gglwe_copy_tmp_bytes_reference<R: GGLWEInfos, A: GGLWEInfos>(&self, res: &R, a: &A) -> usize {
        if copies_raw(res, a) {
            0
        } else {
            self.mat_znx_normalize_tmp_bytes()
        }
    }

    fn gglwe_copy_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        operand_degree(self.n(), &[res.n(), a.n()]);
        assert!(
            (res.rank_in(), res.rank_out()) == (a.rank_in(), a.rank_out()),
            "gglwe_copy: res and a differ in rank"
        );
        assert!(
            (res.k_digit(), res.k_offset()) == (a.k_digit(), a.k_offset()),
            "gglwe_copy: res and a differ in gadget"
        );
        assert!(res.dnum() <= a.dnum(), "gglwe_copy: res has more digits than a");
        assert!(
            scratch.available() >= self.gglwe_copy_tmp_bytes_reference(res, a),
            "scratch.available(): {} < GGLWECopy::gglwe_copy_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_copy_tmp_bytes_reference(res, a)
        );

        let raw: bool = copies_raw(res, a);
        let noise = if res.k() >= a.k() { a.noise() } else { None };
        let (res_base2k, res_k, a_base2k) = (res.base2k().as_usize(), res.k().as_usize(), a.base2k().as_usize());
        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            mat_znx_rows_copy(self, &mut res.data, res_base2k, res_k, &a.data, a_base2k, raw, scratch);
        }
        res.set_noise(noise);
    }
}

/// HAL-based reference implementation, independently callable from backend overrides.
pub trait GGSWCopyReference<BE: Backend> {
    fn ggsw_copy_tmp_bytes_reference<R: GGSWInfos, A: GGSWInfos>(&self, res: &R, a: &A) -> usize;

    fn ggsw_copy_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos;
}

impl<BE: Backend> GGSWCopyReference<BE> for Module<BE>
where
    Self: ModuleN + MatZnxCopy<BE> + MatZnxNormalize<BE> + MatZnxNormalizeTmpBytes,
{
    fn ggsw_copy_tmp_bytes_reference<R: GGSWInfos, A: GGSWInfos>(&self, res: &R, a: &A) -> usize {
        if copies_raw(res, a) {
            0
        } else {
            self.mat_znx_normalize_tmp_bytes()
        }
    }

    fn ggsw_copy_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
    {
        operand_degree(self.n(), &[res.n(), a.n()]);
        assert!(res.rank() == a.rank(), "ggsw_copy: res and a differ in rank");
        assert!(res.k_digit() == a.k_digit(), "ggsw_copy: res and a differ in gadget");
        assert!(res.dnum() <= a.dnum(), "ggsw_copy: res has more digits than a");
        assert!(
            scratch.available() >= self.ggsw_copy_tmp_bytes_reference(res, a),
            "scratch.available(): {} < GGSWCopy::ggsw_copy_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_copy_tmp_bytes_reference(res, a)
        );

        let raw: bool = copies_raw(res, a);
        let noise = if res.k() >= a.k() { a.noise() } else { None };
        let (res_base2k, res_k, a_base2k) = (res.base2k().as_usize(), res.k().as_usize(), a.base2k().as_usize());
        {
            let mut res = res.to_backend_mut();
            let a = a.to_backend_ref();
            mat_znx_rows_copy(self, &mut res.data, res_base2k, res_k, &a.data, a_base2k, raw, scratch);
        }
        res.set_noise(noise);
    }
}

/// Forwards every method of [`GGLWECopyReference`] to its HAL-based reference implementation.
#[macro_export]
macro_rules! impl_gglwe_copy_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWECopyImpl for $be {
            fn gglwe_copy_tmp_bytes<R: $crate::layouts::GGLWEInfos, A: $crate::layouts::GGLWEInfos>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GGLWECopyReference<$be>>::gglwe_copy_tmp_bytes_reference::<R, A>(module, res, a)
            }

            fn gglwe_copy<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GGLWECopyReference<$be>>::gglwe_copy_reference::<R, A>(module, res, a, scratch)
            }
        }
    };
}

/// Forwards every method of [`GGSWCopyReference`] to its HAL-based reference implementation.
#[macro_export]
macro_rules! impl_ggsw_copy_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWCopyImpl for $be {
            fn ggsw_copy_tmp_bytes<R: $crate::layouts::GGSWInfos, A: $crate::layouts::GGSWInfos>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GGSWCopyReference<$be>>::ggsw_copy_tmp_bytes_reference::<R, A>(module, res, a)
            }

            fn ggsw_copy<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWToBackendRef<$be> + $crate::layouts::GGSWInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GGSWCopyReference<$be>>::ggsw_copy_reference::<R, A>(module, res, a, scratch)
            }
        }
    };
}
