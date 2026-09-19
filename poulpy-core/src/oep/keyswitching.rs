#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::GGLWEPreparedBackendRef,
};

/// Output limbs computed by gadget digit `di`.
///
/// The first digit overwrites the full result. Later digits can omit limbs
/// below the supplied product spill window.
#[inline]
pub fn gglwe_product_digit_output_size(res_size: usize, key_size: usize, dsize: usize, di: usize, product_limbs: usize) -> usize {
    assert!(di < dsize);
    if di == 0 {
        res_size
    } else {
        let omitted_limbs = dsize.saturating_sub(di.saturating_add(product_limbs));
        res_size.min(key_size.saturating_sub(omitted_limbs))
    }
}

/// Backend implementation of the interleaved-digit GGLWE product.
///
/// For `dsize >= 2`, it must reproduce
/// [`gglwe_product_digits_strided_reference`](crate::reference::keyswitching::glwe::gglwe_product_digits_strided_reference)
/// bit for bit. `product_limbs` is the caller-derived spill width for the full
/// coefficient-product accumulation.
///
/// # Safety
/// Implementations must honor the supplied layouts and return a scratch bound
/// sufficient for [`Self::gglwe_product_digits_strided`].
pub unsafe trait GGLWEProductDigitsStridedImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_digits_strided_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        pmat_cols_out: usize,
        pmat_size: usize,
    ) -> usize;

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// Opts a backend into the canonical GGLWE interleaved-digit product.
#[macro_export]
macro_rules! impl_gglwe_product_digits_strided_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWEProductDigitsStridedImpl for $be {
            fn gglwe_product_digits_strided_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_size: usize,
                a_cols: usize,
                a_size: usize,
                dsize: usize,
                pmat_rows: usize,
                pmat_cols_in: usize,
                pmat_cols_out: usize,
                pmat_size: usize,
            ) -> usize {
                $crate::reference::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_reference(
                    module,
                    res_size,
                    a_cols,
                    a_size,
                    dsize,
                    pmat_rows,
                    pmat_cols_in,
                    pmat_cols_out,
                    pmat_size,
                )
            }

            fn gglwe_product_digits_strided(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxDftBackendRef<'_, $be>,
                dsize: usize,
                product_limbs: usize,
                pmat: &::poulpy_hal::layouts::VmpPMatBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                $crate::reference::keyswitching::glwe::gglwe_product_digits_strided_reference(
                    module,
                    res,
                    a,
                    dsize,
                    product_limbs,
                    pmat,
                    scratch,
                )
            }
        }
    };
}

/// Backend-provided GLWE key-switching operations.
///
/// # Safety
/// Implementations must satisfy the documented key-switch semantics, honor layout metadata and
/// prepared-key interpretation, and keep all reads and writes within the described backend buffers.
pub unsafe trait GLWEKeyswitchImpl: Backend {
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_keyswitch_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;
}

/// Backend-provided LWE key-switching operations.
///
/// # Safety
/// Implementations must only access the ciphertext and key regions described by the layouts and
/// must produce results matching the logical key-switch operation for the backend.
pub unsafe trait LWEKeyswitchImpl: Backend {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: LWEToBackendRef<Self> + LWEInfos;
}

/// Override surface for the GLWE key-switching sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`glwe_keyswitch_reference`]
/// for reference algorithms a backend may forward to.
///
/// # Gadget-digit width contract
///
/// An override that fuses the digit loop (rather than forwarding to the reference
/// body) must reproduce its output widths exactly, because the choice is not
/// noise-visible: an accumulator one limb too narrow still passes the keyswitch
/// noise sweep. With `key_size = key.size()` and `res_size = res.size()`, digit
/// `di` of `0..dsize` contributes to output limb `c` iff
///
/// ```text
/// c < min(key_size - di, compute_size(di))
/// compute_size(0)  = res_size
/// compute_size(di) = min(res_size,
///                        key_size - max(dsize - di - product_limbs, 0))
///                    for di > 0
/// ```
///
/// Two properties are load-bearing:
///
/// - `di == 0` runs at **full** width and is the overwriting pass. On CPU it is
///   also what zeroes the limbs the accumulating digits add into, so the digits
///   cannot be walked in reverse to widen the first pass. An implementation that
///   writes each output limb exactly once needs no zeroing but must still match
///   the arithmetic.
/// - `product_limbs` is the two-limb elementary product plus the coefficient
///   accumulation growth. Pass `di` consumes `a`'s limbs at offset
///   `dsize - di - 1`; the product spill reaches further down according to the
///   ring and matrix shape. Treating this as a constant is silent at small
///   shapes but truncates live limbs once the accumulation needs a third or
///   fourth limb.
///
/// Assert parity against a reference backend, not only the noise bound.
pub trait GLWEKeyswitchReference<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_assign_reference<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

/// Override surface for the LWE key-switching sub-family.
pub trait LWEKeyswitchReference<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;
}

unsafe impl<BE: Backend> GLWEKeyswitchImpl for BE
where
    Module<BE>: GLWEKeyswitchReference<BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_keyswitch_tmp_bytes_reference(res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_keyswitch_reference(res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        module.glwe_keyswitch_assign_reference(res, key, scratch)
    }
}

unsafe impl<BE: Backend> LWEKeyswitchImpl for BE
where
    Module<BE>: LWEKeyswitchReference<BE>,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_keyswitch_tmp_bytes_reference(res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
    {
        module.lwe_keyswitch_reference(res, a, ksk, scratch)
    }
}

// === Convenience macros for full-reference opt-in ===

/// Implements [`GLWEKeyswitchReference`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_keyswitch_reference`] free function.
#[macro_export]
macro_rules! impl_glwe_keyswitch_reference_full {
    ($be:ty) => {
        impl $crate::oep::GLWEKeyswitchReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::glwe::glwe_keyswitch_tmp_bytes_reference::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch_reference<R, A>(
                &self,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::glwe::glwe_keyswitch_reference::<$be, _, _, _>(self, res, a, key, scratch)
            }

            fn glwe_keyswitch_assign_reference<R>(
                &self,
                res: &mut R,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::glwe::glwe_keyswitch_assign_reference::<$be, _, _>(self, res, key, scratch)
            }
        }
    };
}

/// Implements [`LWEKeyswitchReference`] for `Module<$be>` by forwarding every method to
/// the corresponding [`lwe_keyswitch_reference`] free function.
#[macro_export]
macro_rules! impl_lwe_keyswitch_reference_full {
    ($be:ty) => {
        impl $crate::oep::LWEKeyswitchReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_keyswitch_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::lwe::lwe_keyswitch_tmp_bytes_reference::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn lwe_keyswitch_reference<R, A>(
                &self,
                res: &mut R,
                a: &A,
                ksk: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::keyswitching::lwe::lwe_keyswitch_reference::<$be, _, _, _>(self, res, a, ksk, scratch)
            }
        }
    };
}

#[cfg(test)]
mod digit_output_size_tests {
    use super::gglwe_product_digit_output_size;

    #[test]
    fn product_spill_controls_later_digit_widths() {
        assert_eq!(gglwe_product_digit_output_size(12, 12, 7, 0, 2), 12);
        assert_eq!(gglwe_product_digit_output_size(12, 12, 7, 1, 2), 8);
        assert_eq!(gglwe_product_digit_output_size(12, 12, 7, 1, 4), 10);
        assert_eq!(gglwe_product_digit_output_size(9, 12, 7, 1, 4), 9);
    }
}
