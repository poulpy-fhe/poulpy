//! Shared default bodies for the derived HAL operations.
//!
//! Every function here is the canonical decomposition of one derived
//! operation in terms of *other* OEP methods of the same backend. The OEP
//! trait method's default body is a one-line call into the matching
//! `<op>_derived` function, and its `_tmp_bytes` default calls
//! `<op>_tmp_bytes_derived`.
//!
//! Free functions in this module are generic over the implementing type `S`
//! and the backend `BE`; a trait default body passes `Self` for `S`, while a
//! test that wants to run the decomposition directly on a concrete backend
//! passes that backend for both `S` and `BE`. `S`'s only bound is the OEP
//! trait of the op being decomposed (e.g. `S: HalVmpImpl<BE>`); the
//! supertrait chain on `S` supplies whichever other families the
//! decomposition needs (e.g. `HalVecZnxDftImpl<BE>`), so every OEP call
//! inside is dispatched as `<S as HalXImpl<BE>>::method(...)`, never
//! `<BE as HalXImpl<BE>>::...`.
//!
//! Rules (spec section 5) every function in this module obeys:
//!
//! - only OEP methods of `S`/`BE`, backend-native views, [`ScratchArena`]
//!   carving and `Backend::bytes_of_*`;
//! - no `HostData*` bound, no family kernel trait, no knowledge of a prepared
//!   representation;
//! - `<op>_tmp_bytes_derived` reports exactly the scratch `<op>_derived` takes.
//!
//! These functions are `pub` so that `poulpy_hal::test_suite` can call the
//! decomposition directly and pin it against a hand-built oracle, which is
//! how the per-op parity tests are written. They are not part of the public
//! HAL surface; call the api traits instead.

#![allow(clippy::too_many_arguments)]

use crate::{
    api::ScratchArenaTakeBasic,
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, MatZnxInfos, Module,
        ScalarZnxBackendRef, ScratchArena, SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxBigBackendRef, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VecZnxInfos, VecZnxToBackendRef, VmpPMatBackendRef, ZnxInfos,
        scalar_znx_as_vec_znx_backend_ref_from_ref, vec_znx_backend_ref_from_mut, vec_znx_reborrow_backend_mut,
    },
    oep::{HalConvolutionImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

/// Scratch for [`vmp_apply_dft_derived`]: one `VecZnxDft` for the transformed
/// input plus whatever `vmp_apply_dft_to_dft` needs.
#[doc(hidden)]
pub fn vmp_apply_dft_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
    b_rows: usize,
    b_cols_in: usize,
    b_cols_out: usize,
    b_size: usize,
) -> usize
where
    S: HalVmpImpl<BE>,
    BE: Backend,
{
    let a_dft_size = a_size.min(b_rows);
    BE::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
        + <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft_tmp_bytes(
            module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
}

/// `res = dft(a) * pmat`: transform the `min(a.size(), pmat.rows())` limbs the
/// matrix consumes into scratch, aligning `a`'s trailing columns with
/// `pmat.cols_in()` and zeroing the leading ones, then apply in the DFT domain.
#[doc(hidden)]
pub fn vmp_apply_dft_derived<S, BE, R>(
    module: &Module<BE>,
    res: &mut R,
    a: &VecZnxBackendRef<'_, BE>,
    b: &VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVmpImpl<BE>,
    BE: Backend,
    R: VecZnxDftToBackendMut<BE>,
{
    let a_cols: usize = VecZnxInfos::cols(a);
    let a_size: usize = ZnxInfos::size(a);
    let b_rows: usize = MatZnxInfos::rows(b);
    let cols_to_copy: usize = a_cols.min(MatZnxInfos::cols_in(b));
    let a_start_col: usize = a_cols - cols_to_copy;
    let a_dft_size: usize = a_size.min(b_rows);
    let offset: usize = MatZnxInfos::cols_in(b) - cols_to_copy;

    let (mut a_dft, mut scratch) =
        ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, MatZnxInfos::cols_in(b), a_dft_size);

    for j in 0..offset {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_zero(module, &mut a_dft, j);
    }

    for j in 0..cols_to_copy {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_apply(module, 1, 0, &mut a_dft, offset + j, a, a_start_col + j);
    }

    let mut res_ref = res.to_backend_mut();
    <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft(module, &mut res_ref, &a_dft.to_backend_ref(), b, 0, &mut scratch);
}

/// Scratch for [`vmp_apply_dft_to_dft_add_derived`]: a full-width staging
/// accumulator plus the product's own scratch.
#[doc(hidden)]
pub fn vmp_apply_dft_to_dft_add_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
    b_rows: usize,
    b_cols_in: usize,
    b_cols_out: usize,
    b_size: usize,
) -> usize
where
    S: HalVmpImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_dft(module.n(), b_cols_out, res_size)
        + <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
}

/// `res += a * pmat` in the DFT domain: apply the product into a full-width
/// staging accumulator sized to `res`, then fold it into `res` column by
/// column. The staging accumulator spans exactly `res.size()` limbs, so it
/// accumulates over the same `limb_offset..limb_offset + res.size()` window
/// of `b` that `vmp_apply_dft_to_dft` itself reads.
#[doc(hidden)]
pub fn vmp_apply_dft_to_dft_add_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    b: &VmpPMatBackendRef<'_, BE>,
    limb_offset: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVmpImpl<BE>,
    BE: Backend,
{
    let cols_out: usize = VecZnxInfos::cols(res);
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, cols_out, res_size);
    for col in 0..cols_out {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_zero(module, &mut tmp, col);
    }
    <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft(module, &mut tmp, a, b, limb_offset, &mut scratch);
    let tmp_ref = tmp.to_backend_ref();
    for col in 0..cols_out {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_add_assign(module, res, col, &tmp_ref, col);
    }
}

/// Scratch for the whole left-shift family (ruling R1): `vec_znx_lsh`,
/// `vec_znx_lsh_add`, `vec_znx_lsh_sub` and `vec_znx_lsh_assign`. The largest
/// of the four default bodies takes one `res_size`-limb `VecZnx` plus the
/// normalization carry the shift itself is; the plain `vec_znx_lsh` default
/// needs only the carry, so this over-reports for it — never under-reports.
#[doc(hidden)]
pub fn vec_znx_lsh_tmp_bytes_derived<S, BE>(module: &Module<BE>, res_size: usize) -> usize
where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx(module.n(), 1, res_size) + <S as HalVecZnxImpl<BE>>::vec_znx_normalize_tmp_bytes(module)
}

/// `res = a * 2^k` at the same radix: a normalization with `offset = +k`.
#[doc(hidden)]
pub fn vec_znx_lsh_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_k: usize = ZnxInfos::size(res) * base2k;
    <S as HalVecZnxImpl<BE>>::vec_znx_normalize(module, res, base2k, res_k, k as i64, res_col, a, base2k, a_col, scratch);
}

/// Scratch for the whole right-shift family (ruling R1): `vec_znx_rsh`,
/// `vec_znx_rsh_add`, `vec_znx_rsh_sub` and `vec_znx_rsh_assign`.
#[doc(hidden)]
pub fn vec_znx_rsh_tmp_bytes_derived<S, BE>(module: &Module<BE>, res_size: usize) -> usize
where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx(module.n(), 1, res_size) + <S as HalVecZnxImpl<BE>>::vec_znx_normalize_tmp_bytes(module)
}

/// `res = a / 2^k` at the same radix: a normalization with `offset = -k`.
#[doc(hidden)]
pub fn vec_znx_rsh_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_k: usize = ZnxInfos::size(res) * base2k;
    <S as HalVecZnxImpl<BE>>::vec_znx_normalize(module, res, base2k, res_k, -(k as i64), res_col, a, base2k, a_col, scratch);
}

/// `res += a * 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_add_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary matches the destination's *visible* degree: the shift
    // family is also driven on `window_coeffs` views, where `res.n() < module.n()`.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, res_size);
    <S as HalVecZnxImpl<BE>>::vec_znx_lsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    <S as HalVecZnxImpl<BE>>::vec_znx_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res -= a * 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_sub_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary matches the destination's *visible* degree: the shift
    // family is also driven on `window_coeffs` views, where `res.n() < module.n()`.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, res_size);
    <S as HalVecZnxImpl<BE>>::vec_znx_lsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    <S as HalVecZnxImpl<BE>>::vec_znx_sub_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res += a / 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_add_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary matches the destination's *visible* degree: the shift
    // family is also driven on `window_coeffs` views, where `res.n() < module.n()`.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, res_size);
    <S as HalVecZnxImpl<BE>>::vec_znx_rsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    <S as HalVecZnxImpl<BE>>::vec_znx_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res -= a / 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_sub_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary matches the destination's *visible* degree: the shift
    // family is also driven on `window_coeffs` views, where `res.n() < module.n()`.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, res_size);
    <S as HalVecZnxImpl<BE>>::vec_znx_rsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    <S as HalVecZnxImpl<BE>>::vec_znx_sub_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `a *= 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_assign_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    a: &mut VecZnxBackendMut<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let size: usize = ZnxInfos::size(a);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, size);
    <S as HalVecZnxImpl<BE>>::vec_znx_lsh(
        module,
        base2k,
        k,
        &mut tmp,
        0,
        &vec_znx_backend_ref_from_mut::<BE>(a),
        a_col,
        &mut scratch,
    );
    <S as HalVecZnxImpl<BE>>::vec_znx_copy(module, a, a_col, &tmp.to_backend_ref(), 0);
}

/// `a /= 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_assign_derived<S, BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    a: &mut VecZnxBackendMut<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let size: usize = ZnxInfos::size(a);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, size);
    <S as HalVecZnxImpl<BE>>::vec_znx_rsh(
        module,
        base2k,
        k,
        &mut tmp,
        0,
        &vec_znx_backend_ref_from_mut::<BE>(a),
        a_col,
        &mut scratch,
    );
    <S as HalVecZnxImpl<BE>>::vec_znx_copy(module, a, a_col, &tmp.to_backend_ref(), 0);
}

/// `res = (X^p - 1) * a`: rotate, then subtract the unrotated value.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_derived<S, BE>(
    module: &Module<BE>,
    p: i64,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    <S as HalVecZnxImpl<BE>>::vec_znx_rotate(module, p, res, res_col, a, a_col);
    <S as HalVecZnxImpl<BE>>::vec_znx_sub_assign(module, res, res_col, a, a_col);
}

/// Scratch for [`vec_znx_mul_xp_minus_one_assign_derived`]: one `size`-limb
/// `VecZnx` for the rotated copy.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_derived<S, BE>(module: &Module<BE>, size: usize) -> usize
where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx(module.n(), 1, size)
}

/// `res = (X^p - 1) * res`: rotate into a full-width temporary, then copy back.
///
/// The temporary is a whole `res.size()`-limb `VecZnx` rather than a single
/// limb: `vec_znx_rotate` is a ring operation and rejects windowed views by
/// design (`assert_dense`), so the per-limb form is not available.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_assign_derived<S, BE>(
    module: &Module<BE>,
    p: i64,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let size: usize = ZnxInfos::size(res);
    let (mut tmp, _) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, size);
    vec_znx_mul_xp_minus_one_derived::<S, BE>(module, p, &mut tmp, 0, &vec_znx_backend_ref_from_mut::<BE>(res), res_col);
    <S as HalVecZnxImpl<BE>>::vec_znx_copy(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res[res_col][res_limb] += a[a_col]`: an `add_assign` on the one-limb
/// window of `res`, with the scalar seen as a one-limb `VecZnx`.
#[doc(hidden)]
pub fn vec_znx_add_scalar_assign_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    res_limb: usize,
    a: &ScalarZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    S: HalVecZnxImpl<BE>,
    BE: Backend,
{
    let mut window = vec_znx_reborrow_backend_mut::<BE>(res).window_limbs(res_limb, 1, 1);
    <S as HalVecZnxImpl<BE>>::vec_znx_add_assign(
        module,
        &mut window,
        res_col,
        &scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(a),
        a_col,
    );
}

/// `res = a + b` with `b` a coefficient-domain operand: promote `b` into the
/// destination, then fold `a` in place. Scratch-free; the limb windows match
/// the fused kernel exactly — limbs only `b` reaches keep `b`, limbs only `a`
/// reaches keep `a` (added to the zero `from_small` left there), limbs neither
/// reaches are zero.
#[doc(hidden)]
pub fn vec_znx_big_add_small_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
) where
    S: HalVecZnxBigImpl<BE>,
    BE: Backend,
{
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_from_small(res, res_col, b, b_col);
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_add_assign(module, res, res_col, a, a_col);
}

/// `res = a - b` with `a` the coefficient-domain operand.
#[doc(hidden)]
pub fn vec_znx_big_sub_small_a_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBigBackendRef<'_, BE>,
    b_col: usize,
) where
    S: HalVecZnxBigImpl<BE>,
    BE: Backend,
{
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_from_small(res, res_col, a, a_col);
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_sub_assign(module, res, res_col, b, b_col);
}

/// `res = a - b` with `b` the coefficient-domain operand.
#[doc(hidden)]
pub fn vec_znx_big_sub_small_b_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
) where
    S: HalVecZnxBigImpl<BE>,
    BE: Backend,
{
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_from_small(res, res_col, b, b_col);
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_sub_negate_assign(module, res, res_col, a, a_col);
}

/// Scratch for [`vec_znx_idft_normalize_consume_derived`]: the `VecZnxBig` the
/// inverse transform lands in, plus the big normalization's own carry.
#[doc(hidden)]
pub fn vec_znx_idft_normalize_consume_tmp_bytes_derived<S, BE>(module: &Module<BE>, _res_size: usize, a_size: usize) -> usize
where
    S: HalVecZnxDftImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_big(module.n(), 1, a_size) + <S as HalVecZnxBigImpl<BE>>::vec_znx_big_normalize_tmp_bytes(module)
}

/// `res = normalize(idft(a) + addend)`, clobbering `a`.
#[doc(hidden)]
pub fn vec_znx_idft_normalize_consume_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_base2k: usize,
    res_k: usize,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_col: usize,
    a_base2k: usize,
    addend: Option<(&VecZnxBackendRef<'_, BE>, usize)>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxDftImpl<BE>,
    BE: Backend,
{
    let a_size: usize = ZnxInfos::size(a);
    let (mut big, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, 1, a_size);
    <S as HalVecZnxDftImpl<BE>>::vec_znx_idft_apply_tmpa(module, &mut big, 0, a, a_col);
    if let Some((add, add_col)) = addend {
        <S as HalVecZnxBigImpl<BE>>::vec_znx_big_add_small_assign(module, &mut big, 0, add, add_col);
    }
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_normalize(
        module,
        res,
        res_base2k,
        res_k,
        0,
        res_col,
        &big.to_backend_ref(),
        a_base2k,
        0,
        &mut scratch,
    );
}

/// Scratch for [`vec_znx_dft_automorphism_add_with_plan_derived`]: the limbs
/// the accumulation touches, as one `VecZnxDft`.
#[doc(hidden)]
pub fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
) -> usize
where
    S: HalVecZnxDftImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
}

/// `res += automorphism(a)` over `min(res.size(), a.size())` limbs.
#[doc(hidden)]
pub fn vec_znx_dft_automorphism_add_with_plan_derived<S, BE>(
    module: &Module<BE>,
    plan: &<S as HalVecZnxDftImpl<BE>>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVecZnxDftImpl<BE>,
    BE: Backend,
{
    let size: usize = ZnxInfos::size(res).min(ZnxInfos::size(a));
    let (mut tmp, _) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, size);
    <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_automorphism_with_plan(module, plan, &mut tmp, 0, a, a_col);
    <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`svp_apply_dft_derived`]: the transformed right operand.
#[doc(hidden)]
pub fn svp_apply_dft_tmp_bytes_derived<S, BE>(module: &Module<BE>, b_size: usize) -> usize
where
    S: HalSvpImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_dft(module.n(), 1, b_size)
}

/// `res = ppol * dft(b)`: transform `b`, then apply in the DFT domain.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn svp_apply_dft_derived<S, BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalSvpImpl<BE>,
    BE: Backend,
{
    let b_size: usize = ZnxInfos::size(b);
    let (mut b_dft, _) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
    <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    <S as HalSvpImpl<BE>>::svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft.to_backend_ref(), 0);
}

/// Scratch for [`cnv_prepare_self_derived`]: the larger of the two prepares.
#[doc(hidden)]
pub fn cnv_prepare_self_tmp_bytes_derived<S, BE>(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    <S as HalConvolutionImpl<BE>>::cnv_prepare_left_tmp_bytes(module, res_size, a_size).max(
        <S as HalConvolutionImpl<BE>>::cnv_prepare_right_tmp_bytes(module, res_size, a_size),
    )
}

/// Prepares one operand as both a left and a right convolution factor.
#[doc(hidden)]
pub fn cnv_prepare_self_derived<S, BE>(
    module: &Module<BE>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    <S as HalConvolutionImpl<BE>>::cnv_prepare_left(module, left, a, mask, scratch);
    <S as HalConvolutionImpl<BE>>::cnv_prepare_right(module, right, a, mask, scratch);
}

/// Scratch for [`cnv_apply_dft_add_derived`]: one `res_size`-limb `VecZnxDft`
/// plus the convolution's own scratch.
#[doc(hidden)]
pub fn cnv_apply_dft_add_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_dft(module.n(), 1, res_size)
        + <S as HalConvolutionImpl<BE>>::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
}

/// `res += a (x) b`. The staging temporary spans exactly `res.size()` limbs and
/// `cnv_apply_dft` zero-fills the limbs past the convolution bound, so those
/// limbs of `res` receive zero and keep their previous value, as the contract
/// requires.
#[doc(hidden)]
pub fn cnv_apply_dft_add_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft(module, cnv_offset, &mut tmp, 0, a, a_col, b, b_col, &mut scratch);
    <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`cnv_by_const_apply_add_derived`]: one `res_size`-limb
/// `VecZnxBig` plus the product's own scratch.
#[doc(hidden)]
pub fn cnv_by_const_apply_add_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    BE::bytes_of_vec_znx_big(module.n(), 1, res_size)
        + <S as HalConvolutionImpl<BE>>::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
}

/// `res += a (x) b[b_coeff]`, with the same untouched-limb contract as
/// [`cnv_apply_dft_add_derived`].
#[doc(hidden)]
pub fn cnv_by_const_apply_add_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, 1, res_size);
    <S as HalConvolutionImpl<BE>>::cnv_by_const_apply(module, cnv_offset, &mut tmp, 0, a, a_col, b, b_col, b_coeff, &mut scratch);
    <S as HalVecZnxBigImpl<BE>>::vec_znx_big_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`cnv_pairwise_apply_dft_derived`]: the larger of the one
/// overwriting and the three accumulating products it chains.
#[doc(hidden)]
pub fn cnv_pairwise_apply_dft_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size).max(
        <S as HalConvolutionImpl<BE>>::cnv_apply_dft_add_tmp_bytes(module, cnv_offset, res_size, a_size, b_size),
    )
}

/// `res = (a[i] + a[j]) (x) (b[i] + b[j])`, expanded in the DFT domain where
/// the prepared operands are linear; `i == j` degenerates to one product.
#[doc(hidden)]
pub fn cnv_pairwise_apply_dft_derived<S, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    i: usize,
    j: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalConvolutionImpl<BE>,
    BE: Backend,
{
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft(module, cnv_offset, res, res_col, a, i, b, i, scratch);
    if i == j {
        return;
    }
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, i, b, j, scratch);
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, j, b, i, scratch);
    <S as HalConvolutionImpl<BE>>::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, j, b, j, scratch);
}
