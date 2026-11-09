#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{
    Backend, Module, Ring, ScratchArena, Standard, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
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
/// An `impl` written by hand forwards the step hooks it does not override to their references,
/// all at once with [`impl_gglwe_product_digits_strided_reference!`](crate::impl_gglwe_product_digits_strided_reference).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGLWEProductDigitsStridedImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::gglwe_product_digits_strided`]; may report less than the reference, never less than
    /// the body takes.
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

    /// Reproduces
    /// [`gglwe_product_digits_strided_reference`](crate::reference::keyswitching::GGLWEProductDigitsStridedReference::gglwe_product_digits_strided_reference)
    /// bit for bit for `dsize >= 2`, within [`Self::gglwe_product_digits_strided_tmp_bytes`]; free in kernels and
    /// scratch use.
    ///
    /// `product_limbs` is the caller-derived spill width of the full coefficient-product accumulation, and digit `di`
    /// computes the limbs of [`gglwe_product_digit_output_size`](crate::oep::gglwe_product_digit_output_size). The
    /// caller records no metadata.
    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bytes of [`Self::gglwe_product_fused`] for a destination of `res_size` limbs, a
    /// product of `product_size` limbs and an operand of `a_size` limbs.
    fn gglwe_product_fused_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    /// The gadget product of `a` by `key` with what follows it in a composite operation:
    ///
    /// `res = sigma_p(a x key + addends)`, or `res += ...` with `accumulate`.
    ///
    /// - The product is the one of `GGLWEProductReference::gglwe_product_dft_reference` over
    ///   `product_size` limbs, at most `res.size()`, for `term_count` accumulated products.
    /// - `addends[d]`, when present, is a column `(vector, column)` added to output column `d` over the
    ///   limbs both have. At most `res.cols()` entries.
    /// - The sum is stored through the automorphism `X -> X^p`, as `vec_znx_dft_automorphism` does.
    /// - The first `product_size` limbs of every column of `res` are written, the others zeroed;
    ///   with `accumulate`, the first `product_size` limbs are added to and the others unchanged.
    ///
    /// The reference, [`gglwe_product_fused_reference`](crate::reference::keyswitching::GGLWEProductDigitsStridedReference::gglwe_product_fused_reference),
    /// composes the product, `vec_znx_dft_add_assign` and the automorphism. Overrides must
    /// reproduce it bit for bit within [`Self::gglwe_product_fused_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_fused(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self, Rg>, usize)>],
        p: i64,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bytes of [`Self::gglwe_products_from_dft`] for `count` sources of rank `rank` and
    /// at most `source_size` limbs.
    fn gglwe_products_from_dft_tmp_bytes<K>(
        module: &Module<Self>,
        count: usize,
        rank: usize,
        source_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    /// The key switches of several sources in the transform domain against one key, each stored
    /// through the automorphism `X -> X^p`: item `i` takes `res[i]`, `sources[i]` over its first
    /// `source_sizes[i]` limbs and `accumulate[i]`, every other argument is shared, and no item
    /// reads the `res` of another.
    ///
    /// For a source `s`, `res = sigma_p(op(s) x key + s_0)`, or `res += ...` with `accumulate`,
    /// over `output_size` limbs, at most `key.size()`, as [`Self::gglwe_product_fused`] stores:
    /// `op(s)` is the gadget operand of the mask columns of `s`, each normalized from its inverse
    /// transform at radix `source_base2k` over `source_sizes[i].min(output_size)` limbs, and `s_0`
    /// is column 0 of `s`, added over the limbs both have. The mask columns of the sources are
    /// unspecified afterwards.
    ///
    /// The reference, [`gglwe_products_from_dft_reference`](crate::reference::keyswitching::GGLWEProductDigitsStridedReference::gglwe_products_from_dft_reference),
    /// lays out one operand at a time ([`gglwe_gadget_dft_apply_from_dft`](crate::gadget::gglwe_gadget_dft_apply_from_dft))
    /// and takes its product with [`Self::gglwe_product_fused`]. Overrides must reproduce it bit
    /// for bit within [`Self::gglwe_products_from_dft_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn gglwe_products_from_dft(
        module: &Module<Self>,
        p: i64,
        term_count: usize,
        res: &mut [VecZnxDftBackendMut<'_, Self, Rg>],
        output_size: usize,
        accumulate: &[bool],
        sources: &mut [VecZnxDftBackendMut<'_, Self, Rg>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bytes of [`Self::gglwe_product_idft_normalize`] for a coefficient output of
    /// `res_size` limbs, a product of `product_size` limbs and an operand of `a_size` limbs,
    /// `res_dft` aside.
    fn gglwe_product_idft_normalize_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos;

    /// The gadget product of `a` by `key` followed at once by its inverse transform and
    /// normalization:
    ///
    /// `res[col] = normalize(idft(a x key + addends)[col] + body)`, `body` on column 0 only.
    ///
    /// - `res_dft` is the caller's scratch for the product, of the width the product is taken at;
    ///   its content is unspecified afterwards.
    /// - `addends` are those of [`Self::gglwe_product_fused`]; `body` is the coefficient-domain
    ///   addend of `vec_znx_idft_normalize` on column 0.
    /// - `res_base2k`, `res_k`, `res_offset` and `a_base2k` are the arguments of
    ///   `vec_znx_idft_normalize`, the same for every column.
    ///
    /// The reference, [`gglwe_product_idft_normalize_reference`](crate::reference::keyswitching::GGLWEProductDigitsStridedReference::gglwe_product_idft_normalize_reference),
    /// takes the product into `res_dft`, adds the addends there and calls `vec_znx_idft_normalize`
    /// column by column. Overrides must reproduce it bit for bit within
    /// [`Self::gglwe_product_idft_normalize_tmp_bytes`]; the caller records the metadata of `res`.
    #[allow(clippy::too_many_arguments)]
    fn gglwe_product_idft_normalize(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self, Rg>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self, Rg>, usize)>],
        body: Option<(&poulpy_hal::layouts::VecZnxBackendRef<'_, Self, Rg>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// [`Self::gglwe_product_idft_normalize`] of several sources against one key: item `i` takes
    /// `res[i]`, `res_dft[i]`, `a[i]`, `addends[i]` (none when `addends` is empty) and `body[i]`,
    /// every other argument is shared.
    ///
    /// Defaults to [`Self::gglwe_product_idft_normalize`] on the items in order, every item and
    /// the scratch checked before the first write. Overrides must reproduce it bit for bit within
    /// [`Self::gglwe_product_idft_normalize_tmp_bytes`].
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn gglwe_products_idft_normalize(
        module: &Module<Self>,
        res: &mut [poulpy_hal::layouts::VecZnxBackendMut<'_, Self, Rg>],
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut [VecZnxDftBackendMut<'_, Self, Rg>],
        a: &[VecZnxDftBackendRef<'_, Self, Rg>],
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        term_count: usize,
        addends: &[&[Option<(&VecZnxDftBackendRef<'_, Self, Rg>, usize)>]],
        body: &[Option<(&poulpy_hal::layouts::VecZnxBackendRef<'_, Self, Rg>, usize)>],
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::oep::derived::keyswitching::gglwe_products_idft_normalize_derived(
            module, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, scratch,
        )
    }
}

/// The step hooks of [`GGLWEProductDigitsStridedImpl`] (`gglwe_product_fused`,
/// `gglwe_product_idft_normalize`, `gglwe_products_from_dft` and their queries) forwarding to
/// their reference bodies,
/// for an `impl` block that writes the rest of the family by hand.
///
/// `$ring` is the ring of the enclosing `GGLWEProductDigitsStridedImpl<$ring>` impl: `Rg` in a
/// ring-generic `unsafe impl<Rg: Ring> GGLWEProductDigitsStridedImpl<Rg> for MyBackend`.
#[macro_export]
macro_rules! impl_gglwe_product_digits_strided_reference {
    ($ring:ty) => {
        fn gglwe_product_fused_tmp_bytes<K>(
            module: &::poulpy_hal::layouts::Module<Self>,
            res_size: usize,
            product_size: usize,
            a_size: usize,
            key_infos: &K,
        ) -> usize
        where
            K: $crate::layouts::GGLWEInfos,
        {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_product_fused_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
        }

        fn gglwe_product_fused(
            module: &::poulpy_hal::layouts::Module<Self>,
            res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            product_size: usize,
            a: &::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>,
            key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, Self, $ring>,
            term_count: usize,
            addends: &[Option<(&::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>, usize)>],
            p: i64,
            accumulate: bool,
            scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_product_fused_reference(
                module,
                res,
                product_size,
                a,
                key,
                term_count,
                addends,
                p,
                accumulate,
                scratch,
            )
        }

        fn gglwe_product_idft_normalize_tmp_bytes<K>(
            module: &::poulpy_hal::layouts::Module<Self>,
            res_size: usize,
            product_size: usize,
            a_size: usize,
            key_infos: &K,
        ) -> usize
        where
            K: $crate::layouts::GGLWEInfos,
        {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_product_idft_normalize_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
        }

        fn gglwe_product_idft_normalize(
            module: &::poulpy_hal::layouts::Module<Self>,
            res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self, $ring>,
            res_base2k: usize,
            res_k: usize,
            res_offset: i64,
            res_dft: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
            a: &::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>,
            key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, Self, $ring>,
            term_count: usize,
            addends: &[Option<(&::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self, $ring>, usize)>],
            body: Option<(&::poulpy_hal::layouts::VecZnxBackendRef<'_, Self, $ring>, usize)>,
            a_base2k: usize,
            scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_product_idft_normalize_reference(
                module, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, scratch,
            )
        }

        fn gglwe_products_from_dft_tmp_bytes<K>(
            module: &::poulpy_hal::layouts::Module<Self>,
            count: usize,
            rank: usize,
            source_size: usize,
            key_infos: &K,
        ) -> usize
        where
            K: $crate::layouts::GGLWEInfos,
        {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_products_from_dft_tmp_bytes_reference(module, count, rank, source_size, key_infos)
        }

        fn gglwe_products_from_dft(
            module: &::poulpy_hal::layouts::Module<Self>,
            p: i64,
            term_count: usize,
            res: &mut [::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>],
            output_size: usize,
            accumulate: &[bool],
            sources: &mut [::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>],
            source_sizes: &[usize],
            source_base2k: usize,
            key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, Self, $ring>,
            scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            <::poulpy_hal::layouts::Module<Self> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                Self,
                $ring,
            >>::gglwe_products_from_dft_reference(
                module,
                p,
                term_count,
                res,
                output_size,
                accumulate,
                sources,
                source_sizes,
                source_base2k,
                key,
                scratch,
            )
        }
    };
}

/// Selects the reference implementation of `GGLWEProductDigitsStridedImpl` for `$be`.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_gglwe_product_digits_strided_reference_full {
    ($be:ty) => {
        $crate::impl_gglwe_product_digits_strided_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_gglwe_product_digits_strided_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::GGLWEProductDigitsStridedImpl<$ring> for $be {
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                    $be,
                    $ring,
                >>::gglwe_product_digits_strided_tmp_bytes_reference(
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
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be, $ring>,
                a: &::poulpy_hal::layouts::VecZnxDftBackendRef<'_, $be, $ring>,
                dsize: usize,
                product_limbs: usize,
                pmat: &::poulpy_hal::layouts::VmpPMatBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::keyswitching::GGLWEProductDigitsStridedReference<
                    $be,
                    $ring,
                >>::gglwe_product_digits_strided_reference(module, res, a, dsize, product_limbs, pmat, scratch)
            }

            $crate::impl_gglwe_product_digits_strided_reference!($ring);
        }
    };
}

/// Backend-provided GLWE key-switching operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEKeyswitchImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::glwe_keyswitch`] and [`Self::glwe_keyswitch_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn glwe_keyswitch_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    /// Reproduces [`glwe_keyswitch_reference`](crate::reference::keyswitching::glwe::glwe_keyswitch_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_keyswitch_tmp_bytes`]; free in kernels and scratch use, but a fused digit loop keeps the output
    /// widths of [`Self::glwe_keyswitch_dft`].
    fn glwe_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_keyswitch_assign_reference`](crate::reference::keyswitching::glwe::glwe_keyswitch_assign_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_keyswitch_tmp_bytes`]; free in kernels and scratch use, but a fused digit loop keeps the output
    /// widths of [`Self::glwe_keyswitch_dft`].
    fn glwe_keyswitch_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_keyswitch_dft`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_keyswitch_dft_tmp_bytes<A, K>(module: &Module<Self>, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`glwe_keyswitch_dft_reference`](crate::reference::keyswitching::GLWEKeyswitchReference::glwe_keyswitch_dft_reference) bit for
    /// bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_keyswitch_dft_tmp_bytes`]; free in kernels and scratch use.
    ///
    /// An override that fuses the digit loop, rather than forwarding to the reference body, reproduces its output
    /// widths exactly, because the choice is not noise-visible: an accumulator one limb too narrow still passes the
    /// keyswitch noise sweep. With `key_size = key.size()` and `res_size = res.size()`, digit `di` of `0..dsize`
    /// contributes to output limb `c` iff
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
    /// - `di == 0` runs at **full** width and is the overwriting pass. The reference initializes the output before
    ///   later digits accumulate into it, so reversing the digit order needs another initialization step. An
    ///   implementation that writes each output limb exactly once needs no zeroing but must still match the
    ///   arithmetic.
    /// - `product_limbs` is the two-limb elementary product plus the coefficient accumulation growth. Pass `di`
    ///   consumes `a`'s limbs at offset `dsize - di - 1`; the product spill reaches further down according to the
    ///   ring and matrix shape. Treating it as a constant is silent at small shapes but truncates live limbs once the
    ///   accumulation needs a third or fourth limb.
    ///
    /// Validate with parity against a caller-selected comparison backend as well as the noise bound.
    fn glwe_keyswitch_dft<A>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_keyswitch_dft_add`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_keyswitch_dft_add_tmp_bytes<A, K>(module: &Module<Self>, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// `res += Σ_i decomp(a_i) · key_i`: [`Self::glwe_keyswitch_dft`] added to `res`. Reproduces
    /// [`glwe_keyswitch_dft_add_reference`](crate::reference::keyswitching::GLWEKeyswitchReference::glwe_keyswitch_dft_add_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_keyswitch_dft_add_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_keyswitch_dft_add<A>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;
}

/// Backend-provided GGLWE key-switching operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGLWEKeyswitchImpl<Rg: Ring = Standard>: Backend + GLWEKeyswitchImpl<Rg> {
    /// Default body: `gglwe_keyswitch_tmp_bytes_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes).
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::keyswitching::gglwe_keyswitch_tmp_bytes_derived::<Self, _, _, _, _, Rg>(
            module, res_infos, a_infos, key_infos,
        )
    }

    /// Default body: `gglwe_keyswitch_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch) on each entry of `a`,
    /// within [`Self::gglwe_keyswitch_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn gglwe_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self, Rg> + GGLWEInfos,
        A: GGLWEToBackendRef<Self, Rg> + GGLWEInfos,
    {
        crate::oep::derived::keyswitching::gglwe_keyswitch_derived::<Self, _, _, _, Rg>(module, res, a, key, scratch)
    }

    /// Default body: `gglwe_keyswitch_assign_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_assign`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch_assign) on each
    /// entry of `res`, within [`Self::gglwe_keyswitch_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    fn gglwe_keyswitch_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self, Rg> + GGLWEInfos,
    {
        crate::oep::derived::keyswitching::gglwe_keyswitch_assign_derived::<Self, _, _, Rg>(module, res, key, scratch)
    }
}

/// Backend-provided GGSW key-switching operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWKeyswitchImpl<Rg: Ring = Standard>:
    Backend + GLWEKeyswitchImpl<Rg> + crate::oep::GGSWConversionImpl<Rg>
{
    /// Default body: `ggsw_keyswitch_tmp_bytes_derived`, the larger of
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes) and
    /// [`GGSWConversionImpl::ggsw_expand_rows_assign_tmp_bytes`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign_tmp_bytes).
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<Self>,
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
        crate::oep::derived::keyswitching::ggsw_keyswitch_tmp_bytes_derived::<Self, _, _, _, _, _, Rg>(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    /// Default body: `ggsw_keyswitch_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch) on the first column of
    /// each row, then [`GGSWConversionImpl::ggsw_expand_rows_assign`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign),
    /// within [`Self::ggsw_keyswitch_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos,
        A: GGSWToBackendRef<Self, Rg> + GGSWInfos,
    {
        crate::oep::derived::keyswitching::ggsw_keyswitch_derived::<Self, _, _, _, Rg>(module, res, a, key, tsk, scratch)
    }

    /// Default body: `ggsw_keyswitch_assign_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_assign`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch_assign) on the
    /// first column of each row, then
    /// [`GGSWConversionImpl::ggsw_expand_rows_assign`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign), within
    /// [`Self::ggsw_keyswitch_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_keyswitch_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos,
    {
        crate::oep::derived::keyswitching::ggsw_keyswitch_assign_derived::<Self, _, _, Rg>(module, res, key, tsk, scratch)
    }
}

/// Backend-provided LWE key-switching operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait LWEKeyswitchImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::lwe_keyswitch`]; may report less than the reference, never less than the body takes.
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    /// Reproduces [`lwe_keyswitch_reference`](crate::reference::keyswitching::lwe::lwe_keyswitch_reference) bit for
    /// bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within [`Self::lwe_keyswitch_tmp_bytes`];
    /// free in kernels and scratch use.
    fn lwe_keyswitch<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ksk: &GGLWEPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self, Rg> + LWEInfos,
        A: LWEToBackendRef<Self, Rg> + LWEInfos;
}

// === Convenience macros for full-reference opt-in ===

/// Selects the portable HAL algorithms for this backend operation family.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_glwe_keyswitch_reference_full {
    ($be:ty) => {
        $crate::impl_glwe_keyswitch_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_glwe_keyswitch_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::GLWEKeyswitchImpl<$ring> for $be {
            fn glwe_keyswitch_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_keyswitch_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_assign_reference(
                    module, res, key, scratch,
                )
            }

            fn glwe_keyswitch_dft_tmp_bytes<A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_size: usize,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_dft_tmp_bytes_reference(
                    module, res_size, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch_dft<A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be, $ring>,
                a: &A,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_dft_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_keyswitch_dft_add_tmp_bytes<A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_size: usize,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_dft_add_tmp_bytes_reference(
                    module, res_size, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch_dft_add<A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be, $ring>,
                a: &A,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::keyswitching::GLWEKeyswitchReference::<$be, $ring>::glwe_keyswitch_dft_add_reference(
                    module, res, a, key, scratch,
                )
            }
        }
    };
}

/// Selects the portable HAL algorithms for this backend operation family.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_lwe_keyswitch_reference_full {
    ($be:ty) => {
        $crate::impl_lwe_keyswitch_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_lwe_keyswitch_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::LWEKeyswitchImpl<$ring> for $be {
            fn lwe_keyswitch_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::keyswitching::LWEKeyswitchReference::<$be, $ring>::lwe_keyswitch_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn lwe_keyswitch<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                ksk: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be, $ring> + $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be, $ring> + $crate::layouts::LWEInfos,
            {
                $crate::reference::keyswitching::LWEKeyswitchReference::<$be, $ring>::lwe_keyswitch_reference(
                    module, res, a, ksk, scratch,
                )
            }
        }
    };
}

/// Selects the same-layer row composition for this backend.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_gglwe_keyswitch_derived_full {
    ($be:ty) => {
        $crate::impl_gglwe_keyswitch_derived_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_gglwe_keyswitch_derived_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::GGLWEKeyswitchImpl<$ring> for $be {}
    };
}

/// Selects the same-layer row composition for this backend.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_ggsw_keyswitch_derived_full {
    ($be:ty) => {
        $crate::impl_ggsw_keyswitch_derived_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ggsw_keyswitch_derived_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::GGSWKeyswitchImpl<$ring> for $be {}
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
