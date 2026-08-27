#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
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
/// [`gglwe_product_digits_strided_default`](crate::default::keyswitching::glwe::gglwe_product_digits_strided_default)
/// bit for bit. `product_limbs` is the caller-derived spill width for the full
/// coefficient-product accumulation.
///
/// # Safety
/// Implementations must honor the supplied layouts and return a scratch bound
/// sufficient for [`Self::gglwe_product_digits_strided`].
pub unsafe trait GGLWEProductDigitsStridedImpl<BE: Backend>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_digits_strided_tmp_bytes(
        module: &Module<BE>,
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
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Opts a backend into the canonical GGLWE interleaved-digit product.
#[macro_export]
macro_rules! impl_gglwe_product_digits_strided_default {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWEProductDigitsStridedImpl<$be> for $be {
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
                $crate::default::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_default(
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
                $crate::default::keyswitching::glwe::gglwe_product_digits_strided_default(
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
pub unsafe trait GLWEKeyswitchImpl<BE: Backend>: Backend {
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

/// Backend-provided GGLWE key-switching operations.
///
/// # Safety
/// Implementations must preserve ciphertext invariants, use scratch space according to the
/// advertised temporary-size contract, and uphold aliasing guarantees for backend-owned buffers.
pub unsafe trait GGLWEKeyswitchImpl<BE: Backend>: Backend {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Backend-provided GGSW key-switching operations.
///
/// # Safety
/// Implementations must correctly interpret prepared key material for the backend, respect all
/// layout-derived bounds, and avoid invalid aliasing or mutation through scratch-backed views.
pub unsafe trait GGSWKeyswitchImpl<BE: Backend>: Backend {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ggsw_keyswitch_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Backend-provided LWE key-switching operations.
///
/// # Safety
/// Implementations must only access the ciphertext and key regions described by the layouts and
/// must produce results matching the logical key-switch operation for the backend.
pub unsafe trait LWEKeyswitchImpl<BE: Backend>: Backend {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the GLWE key-switching sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`glwe_keyswitch_defaults`]
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
pub trait GLWEKeyswitchDefault<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_assign_default<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

/// Override surface for the GGLWE key-switching sub-family.
pub trait GGLWEKeyswitchDefault<BE: Backend> {
    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch_default<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the GGSW key-switching sub-family.
pub trait GGSWKeyswitchDefault<BE: Backend> {
    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch_default<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ggsw_keyswitch_assign_default<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the LWE key-switching sub-family.
pub trait LWEKeyswitchDefault<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_default<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

unsafe impl<BE: Backend> GLWEKeyswitchImpl<BE> for BE
where
    Module<BE>: GLWEKeyswitchDefault<BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
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
        module.glwe_keyswitch_default(res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        module.glwe_keyswitch_assign_default(res, key, scratch)
    }
}

unsafe impl<BE: Backend> GGLWEKeyswitchImpl<BE> for BE
where
    Module<BE>: GGLWEKeyswitchDefault<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        module.gglwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.gglwe_keyswitch_default(res, a, key, scratch)
    }

    fn gglwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.gglwe_keyswitch_assign_default(res, key, scratch)
    }
}

unsafe impl<BE: Backend> GGSWKeyswitchImpl<BE> for BE
where
    Module<BE>: GGSWKeyswitchDefault<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        module.ggsw_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_keyswitch<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_keyswitch_default(res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_keyswitch_assign_default(res, key, tsk, scratch)
    }
}

unsafe impl<BE: Backend> LWEKeyswitchImpl<BE> for BE
where
    Module<BE>: LWEKeyswitchDefault<BE>,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.lwe_keyswitch_default(res, a, ksk, scratch)
    }
}

// === Convenience macros for full-default opt-in ===

/// Implements [`GLWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GLWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_default::<$be, _, _, _>(self, res, a, key, scratch)
            }

            fn glwe_keyswitch_assign_default<R>(
                &self,
                res: &mut R,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_assign_default::<$be, _, _>(self, res, key, scratch)
            }
        }
    };
}

/// Implements [`GGLWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`gglwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_gglwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGLWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn gglwe_keyswitch_default<R, A, B>(
                &self,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                B: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_default::<$be, _, _, _, _>(self, res, a, b, scratch)
            }

            fn gglwe_keyswitch_assign_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_assign_default::<$be, _, _, _>(self, res, a, scratch)
            }
        }
    };
}

/// Implements [`GGSWKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`ggsw_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_ggsw_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGSWKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(
                &self,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
                tsk_infos: &T,
            ) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWInfos,
                K: $crate::layouts::GGLWEInfos,
                T: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_tmp_bytes_default::<$be, _, _, _, _, _>(
                    self, res_infos, a_infos, key_infos, tsk_infos,
                )
            }

            fn ggsw_keyswitch_default<R, A, K, T>(
                &self,
                res: &mut R,
                a: &A,
                key: &K,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWToBackendRef<$be> + $crate::layouts::GGSWInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_default::<$be, _, _, _, _, _>(self, res, a, key, tsk, scratch)
            }

            fn ggsw_keyswitch_assign_default<R, K, T>(
                &self,
                res: &mut R,
                key: &K,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_assign_default::<$be, _, _, _, _>(
                    self, res, key, tsk, scratch,
                )
            }
        }
    };
}

/// Implements [`LWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`lwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_lwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::LWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::lwe::lwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn lwe_keyswitch_default<R, A, K>(
                &self,
                res: &mut R,
                a: &A,
                ksk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::lwe::lwe_keyswitch_default::<$be, _, _, _, _>(self, res, a, ksk, scratch)
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
