//! Shared default bodies for the derived HAL operations.
//!
//! Every function here is the canonical decomposition of one derived
//! operation in terms of *other* OEP methods of the same backend. The OEP
//! trait method's default body is a one-line call into the matching
//! `<op>_derived` function, and its `_tmp_bytes` default calls
//! `<op>_tmp_bytes_derived`.
//!
//! Free functions in this module are generic over the backend `BE` alone,
//! bounded by the OEP trait of the op being decomposed (e.g.
//! `BE: HalVmpImpl`); the supertrait chain supplies whichever other families
//! the decomposition needs (e.g. `HalVecZnxDftImpl`), so every OEP call
//! inside is `BE::method(module, ...)`. A trait default body passes `Self`,
//! and a test that runs the decomposition directly on a concrete backend
//! passes that backend.
//!
//! Rules every function in this module obeys:
//!
//! - only OEP methods of `BE`, backend-native views, [`ScratchArena`]
//!   carving and `Backend::bytes_of_*`;
//! - no `HostData*` bound, no family kernel trait, no knowledge of a prepared
//!   representation;
//! - `<op>_tmp_bytes_derived` reports exactly the scratch `<op>_derived` takes;
//!   when it adds a nested call's scratch to a temporary of its own, the
//!   temporary is rounded with `Backend::scratch_aligned` first, since every
//!   carve from the arena realigns its start.
//!
//! These functions are `pub` so that `poulpy_hal::test_suite` can call the
//! decomposition directly and pin it against a hand-built oracle, which is
//! how the per-op parity tests are written. They are not part of the public
//! HAL surface; call the api traits instead.

#![allow(clippy::too_many_arguments)]

use crate::{
    api::ScratchArenaTakeBasic,
    layouts::{
        CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, MatZnxInfos, Module,
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
pub fn vmp_apply_dft_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
    b_rows: usize,
    b_cols_in: usize,
    b_cols_out: usize,
    b_size: usize,
) -> usize
where
    BE: HalVmpImpl,
{
    let a_dft_size = a_size.min(b_rows);
    BE::scratch_aligned(BE::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size))
        + BE::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size)
}

/// `res = dft(a) * pmat`: transform the `min(a.size(), pmat.rows())` limbs the
/// matrix consumes into scratch, aligning `a`'s trailing columns with
/// `pmat.cols_in()` and zeroing the leading ones, then apply in the DFT domain.
#[doc(hidden)]
pub fn vmp_apply_dft_derived<BE, R>(
    module: &Module<BE>,
    res: &mut R,
    a: &VecZnxBackendRef<'_, BE>,
    b: &VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVmpImpl,
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
        BE::vec_znx_dft_zero(module, &mut a_dft, j);
    }

    for j in 0..cols_to_copy {
        BE::vec_znx_dft_apply(module, 1, 0, &mut a_dft, offset + j, a, a_start_col + j);
    }

    let mut res_ref = res.to_backend_mut();
    BE::vmp_apply_dft_to_dft(module, &mut res_ref, &a_dft.to_backend_ref(), b, 0, &mut scratch);
}

/// Scratch for [`vmp_apply_dft_to_dft_add_derived`]: a full-width staging
/// accumulator plus the product's own scratch.
#[doc(hidden)]
pub fn vmp_apply_dft_to_dft_add_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
    b_rows: usize,
    b_cols_in: usize,
    b_cols_out: usize,
    b_size: usize,
) -> usize
where
    BE: HalVmpImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx_dft(module.n(), b_cols_out, res_size))
        + BE::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
}

/// `res += a * pmat` in the DFT domain: apply the product into a full-width
/// staging accumulator sized to `res`, then fold it into `res` column by
/// column. The staging accumulator spans exactly `res.size()` limbs, so it
/// accumulates over the same `limb_offset..limb_offset + res.size()` window
/// of `b` that `vmp_apply_dft_to_dft` itself reads.
///
/// The `vec_znx_dft_zero` loop over the accumulator is defensive:
/// `vmp_apply_dft_to_dft` zeroes the limbs past its bound by contract, and
/// the loop keeps the sum correct on a kernel that only writes the limbs the
/// product reaches.
#[doc(hidden)]
pub fn vmp_apply_dft_to_dft_add_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    b: &VmpPMatBackendRef<'_, BE>,
    limb_offset: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVmpImpl,
{
    let cols_out: usize = VecZnxInfos::cols(res);
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, cols_out, res_size);
    for col in 0..cols_out {
        BE::vec_znx_dft_zero(module, &mut tmp, col);
    }
    BE::vmp_apply_dft_to_dft(module, &mut tmp, a, b, limb_offset, &mut scratch);
    let tmp_ref = tmp.to_backend_ref();
    for col in 0..cols_out {
        BE::vec_znx_dft_add_assign(module, res, col, &tmp_ref, col);
    }
}

/// The shift `k` as a normalization offset. `bound` is a width past which the
/// shift has moved every bit of any source the normalization accepts out of
/// the destination, so a `k` at or past it yields the same zero as `bound`
/// itself and the clamp changes no result; it keeps the cast from wrapping for
/// `k >= 2^63`, which would reverse the shift's direction. For a left shift
/// the bound is the source width: every term is then an integer. For a right
/// shift it is the destination width plus [`RSH_DIGIT_BITS`]: a digit is at
/// most 63 bits, so the whole source then sits below half a unit of the
/// destination's last limb.
#[inline]
fn shift_offset(k: usize, bound: usize) -> i64 {
    k.min(bound) as i64
}

/// Bits a right shift adds to the destination width before it is sure to have
/// zeroed any accepted source; see [`shift_offset`].
const RSH_DIGIT_BITS: usize = 64;

/// Scratch for the whole left-shift family (ruling R1): `vec_znx_lsh`,
/// `vec_znx_lsh_add`, `vec_znx_lsh_sub` and `vec_znx_lsh_assign`. The largest
/// of the four default bodies takes one `res_size`-limb `VecZnx` plus the
/// normalization carry the shift itself is; the plain `vec_znx_lsh` default
/// needs only the carry, so this over-reports for it, never under-reports.
#[doc(hidden)]
pub fn vec_znx_lsh_tmp_bytes_derived<BE>(module: &Module<BE>, res_size: usize) -> usize
where
    BE: HalVecZnxImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx(module.n(), 1, res_size)) + BE::vec_znx_normalize_tmp_bytes(module)
}

/// `res = a * 2^k` at the same radix: a normalization with `offset = +k`.
#[doc(hidden)]
pub fn vec_znx_lsh_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_k: usize = ZnxInfos::size(res) * base2k;
    let offset: i64 = shift_offset(k, ZnxInfos::size(a) * base2k);
    BE::vec_znx_normalize(module, res, base2k, res_k, offset, res_col, a, base2k, a_col, scratch);
}

/// Scratch for the whole right-shift family (ruling R1): `vec_znx_rsh`,
/// `vec_znx_rsh_add`, `vec_znx_rsh_sub` and `vec_znx_rsh_assign`.
#[doc(hidden)]
pub fn vec_znx_rsh_tmp_bytes_derived<BE>(module: &Module<BE>, res_size: usize) -> usize
where
    BE: HalVecZnxImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx(module.n(), 1, res_size)) + BE::vec_znx_normalize_tmp_bytes(module)
}

/// `res = a / 2^k` at the same radix: a normalization with `offset = -k`.
#[doc(hidden)]
pub fn vec_znx_rsh_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_k: usize = ZnxInfos::size(res) * base2k;
    let offset: i64 = -shift_offset(k, res_k + RSH_DIGIT_BITS);
    BE::vec_znx_normalize(module, res, base2k, res_k, offset, res_col, a, base2k, a_col, scratch);
}

/// `res += a * 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_add_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary takes a's degree: for a window it is the destination's
    // visible degree, for a dense degree-n operand it is n, and the add or sub
    // that follows reads it through its sparse-capable slot.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, res_size);
    BE::vec_znx_lsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    BE::vec_znx_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res -= a * 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_sub_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary takes a's degree: for a window it is the destination's
    // visible degree, for a dense degree-n operand it is n, and the add or sub
    // that follows reads it through its sparse-capable slot.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, res_size);
    BE::vec_znx_lsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    BE::vec_znx_sub_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res += a / 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_add_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary takes a's degree: for a window it is the destination's
    // visible degree, for a dense degree-n operand it is n, and the add or sub
    // that follows reads it through its sparse-capable slot.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, res_size);
    BE::vec_znx_rsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    BE::vec_znx_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res -= a / 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_sub_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    // The temporary takes a's degree: for a window it is the destination's
    // visible degree, for a dense degree-n operand it is n, and the add or sub
    // that follows reads it through its sparse-capable slot.
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, res_size);
    BE::vec_znx_rsh(module, base2k, k, &mut tmp, 0, a, a_col, &mut scratch);
    BE::vec_znx_sub_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `a *= 2^k`. Sized by [`vec_znx_lsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_lsh_assign_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    a: &mut VecZnxBackendMut<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let size: usize = ZnxInfos::size(a);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(a), 1, size);
    BE::vec_znx_lsh(
        module,
        base2k,
        k,
        &mut tmp,
        0,
        &vec_znx_backend_ref_from_mut::<BE>(a),
        a_col,
        &mut scratch,
    );
    BE::vec_znx_copy(module, a, a_col, &tmp.to_backend_ref(), 0);
}

/// `a /= 2^k`. Sized by [`vec_znx_rsh_tmp_bytes_derived`].
#[doc(hidden)]
pub fn vec_znx_rsh_assign_derived<BE>(
    module: &Module<BE>,
    base2k: usize,
    k: usize,
    a: &mut VecZnxBackendMut<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let res_k: usize = ZnxInfos::size(a) * base2k;
    let offset: i64 = -shift_offset(k, res_k + RSH_DIGIT_BITS);
    BE::vec_znx_normalize_assign(module, base2k, res_k, offset, a, a_col, scratch);
}

/// `res = (X^p - 1) * a`: rotate, then subtract the unrotated value.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_derived<BE>(
    module: &Module<BE>,
    p: i64,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: HalVecZnxImpl,
{
    BE::vec_znx_rotate(module, p, res, res_col, a, a_col);
    BE::vec_znx_sub_assign(module, res, res_col, a, a_col);
}

/// Scratch for [`vec_znx_mul_xp_minus_one_assign_derived`]: one `size`-limb
/// `VecZnx` for the rotated copy.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_derived<BE>(module: &Module<BE>, size: usize) -> usize
where
    BE: HalVecZnxImpl,
{
    BE::bytes_of_vec_znx(module.n(), 1, size)
}

/// `res = (X^p - 1) * res`: rotate into a full-width temporary, then copy back.
///
/// The temporary is a whole `res.size()`-limb `VecZnx` rather than a single
/// limb: `vec_znx_rotate` is a ring operation and rejects windowed views by
/// design (`assert_dense`), so the per-limb form is not available.
#[doc(hidden)]
pub fn vec_znx_mul_xp_minus_one_assign_derived<BE>(
    module: &Module<BE>,
    p: i64,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxImpl,
{
    let size: usize = ZnxInfos::size(res);
    let (mut tmp, _) = ScratchArenaTakeBasic::take_vec_znx_scratch(scratch.borrow(), ZnxInfos::n(res), 1, size);
    vec_znx_mul_xp_minus_one_derived::<BE>(module, p, &mut tmp, 0, &vec_znx_backend_ref_from_mut::<BE>(res), res_col);
    BE::vec_znx_copy(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// `res[res_col][res_limb] += a[a_col]`: an `add_assign` on the one-limb
/// window of `res`, with the scalar seen as a one-limb `VecZnx`.
#[doc(hidden)]
pub fn vec_znx_add_scalar_assign_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    res_limb: usize,
    a: &ScalarZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: HalVecZnxImpl,
{
    let mut window = vec_znx_reborrow_backend_mut::<BE>(res).window_limbs(res_limb, 1, 1);
    BE::vec_znx_add_assign(
        module,
        &mut window,
        res_col,
        &scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(a),
        a_col,
    );
}

/// `res = a + b` with `b` a coefficient-domain operand: promote `b` into the
/// destination, then fold `a` in place. Scratch-free.
///
/// Window contract, over the whole of `res`: limbs only `b` reaches keep `b`,
/// limbs only `a` reaches keep `a` (added to the zero `vec_znx_big_from_small`
/// left there), limbs neither reaches are zero.
///
/// Caller precondition: `res` must not alias `a` or `b`. The body writes `res`
/// before reading them, and nothing checks it: the backend view types do not
/// enforce it (`vec_znx_big_backend_ref_from_mut` hands out a shared view of a
/// mutable one). This is new relative to the fused kernels that preceded the derived bodies, which
/// read both operands per limb before writing and so tolerated aliasing.
#[doc(hidden)]
pub fn vec_znx_big_add_small_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: HalVecZnxBigImpl,
{
    BE::vec_znx_big_from_small(res, res_col, b, b_col);
    BE::vec_znx_big_add_assign(module, res, res_col, a, a_col);
}

/// `res = a - b` with `a` the coefficient-domain operand: promote `a` into the
/// destination, then subtract `b` in place. Scratch-free.
///
/// Window contract, over the whole of `res`: limbs only `a` reaches keep `a`,
/// limbs only `b` reaches keep `-b` (subtracted from the zero
/// `vec_znx_big_from_small` left there), limbs neither reaches are zero.
///
/// Caller precondition: `res` must not alias `a` or `b`. The body writes `res`
/// before reading them, and nothing checks it: the backend view types do not
/// enforce it (`vec_znx_big_backend_ref_from_mut` hands out a shared view of a
/// mutable one). This is new relative to the fused kernels that preceded the derived bodies, which
/// read both operands per limb before writing and so tolerated aliasing.
#[doc(hidden)]
pub fn vec_znx_big_sub_small_a_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBigBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: HalVecZnxBigImpl,
{
    BE::vec_znx_big_from_small(res, res_col, a, a_col);
    BE::vec_znx_big_sub_assign(module, res, res_col, b, b_col);
}

/// `res = a - b` with `b` the coefficient-domain operand: promote `b` into the
/// destination, then negate it against `a` in place. Scratch-free.
///
/// Window contract, over the whole of `res`: limbs only `a` reaches keep `a`,
/// limbs only `b` reaches keep `-b` (the negation applied to the `b` that
/// `vec_znx_big_from_small` left there), limbs neither reaches are zero.
///
/// Caller precondition: `res` must not alias `a` or `b`. The body writes `res`
/// before reading them, and nothing checks it: the backend view types do not
/// enforce it (`vec_znx_big_backend_ref_from_mut` hands out a shared view of a
/// mutable one). This is new relative to the fused kernels that preceded the derived bodies, which
/// read both operands per limb before writing and so tolerated aliasing.
#[doc(hidden)]
pub fn vec_znx_big_sub_small_b_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBigBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: HalVecZnxBigImpl,
{
    BE::vec_znx_big_from_small(res, res_col, b, b_col);
    BE::vec_znx_big_sub_negate_assign(module, res, res_col, a, a_col);
}

/// Scratch for [`vec_znx_idft_normalize_consume_derived`]: the `VecZnxBig` the
/// inverse transform lands in, plus the big normalization's own carry.
#[doc(hidden)]
pub fn vec_znx_idft_normalize_consume_tmp_bytes_derived<BE>(module: &Module<BE>, _res_size: usize, a_size: usize) -> usize
where
    BE: HalVecZnxDftImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx_big(module.n(), 1, a_size)) + BE::vec_znx_big_normalize_tmp_bytes(module)
}

/// `res = normalize(idft(a) + addend)`, clobbering `a`.
#[doc(hidden)]
pub fn vec_znx_idft_normalize_consume_derived<BE>(
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
    BE: HalVecZnxDftImpl,
{
    let a_size: usize = ZnxInfos::size(a);
    let (mut big, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, 1, a_size);
    BE::vec_znx_idft_apply_tmpa(module, &mut big, 0, a, a_col);
    if let Some((add, add_col)) = addend {
        BE::vec_znx_big_add_small_assign(module, &mut big, 0, add, add_col);
    }
    BE::vec_znx_big_normalize(
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

/// `res = tau_p(a)`: build the plan for `p`, apply it, drop it.
#[doc(hidden)]
pub fn vec_znx_dft_automorphism_derived<BE>(
    module: &Module<BE>,
    p: i64,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: HalVecZnxDftImpl,
{
    let plan = BE::vec_znx_dft_automorphism_plan(module, p);
    BE::vec_znx_dft_automorphism_with_plan(module, &plan, res, res_col, a, a_col);
}

/// Scratch for [`vec_znx_dft_automorphism_add_with_plan_derived`]: the limbs
/// the accumulation touches, as one `VecZnxDft`.
#[doc(hidden)]
pub fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived<BE>(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
where
    BE: HalVecZnxDftImpl,
{
    BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
}

/// `res += automorphism(a)` over `min(res.size(), a.size())` limbs.
#[doc(hidden)]
pub fn vec_znx_dft_automorphism_add_with_plan_derived<BE>(
    module: &Module<BE>,
    plan: &BE::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalVecZnxDftImpl,
{
    let size: usize = ZnxInfos::size(res).min(ZnxInfos::size(a));
    let (mut tmp, _) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, size);
    BE::vec_znx_dft_automorphism_with_plan(module, plan, &mut tmp, 0, a, a_col);
    BE::vec_znx_dft_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`svp_apply_dft_derived`]: the transformed right operand.
#[doc(hidden)]
pub fn svp_apply_dft_tmp_bytes_derived<BE>(module: &Module<BE>, b_size: usize) -> usize
where
    BE: HalSvpImpl,
{
    BE::bytes_of_vec_znx_dft(module.n(), 1, b_size)
}

/// `res = ppol * dft(b)`: transform `b`, then apply in the DFT domain.
#[doc(hidden)]
pub fn svp_apply_dft_derived<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalSvpImpl,
{
    let b_size: usize = ZnxInfos::size(b);
    let (mut b_dft, _) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, b_size);
    BE::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    BE::svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft.to_backend_ref(), 0);
}

/// Scratch for [`cnv_prepare_self_derived`]: the larger of the two prepares.
#[doc(hidden)]
pub fn cnv_prepare_self_tmp_bytes_derived<BE>(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
where
    BE: HalConvolutionImpl,
{
    BE::cnv_prepare_left_tmp_bytes(module, res_size, a_size).max(BE::cnv_prepare_right_tmp_bytes(module, res_size, a_size))
}

/// Prepares one operand as both a left and a right convolution factor.
#[doc(hidden)]
pub fn cnv_prepare_self_derived<BE>(
    module: &Module<BE>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalConvolutionImpl,
{
    assert_eq!(right.size(), left.size());
    BE::cnv_prepare_left(module, left, a, scratch);
    BE::cnv_prepare_right(module, right, a, scratch);
}

/// Scratch for [`cnv_apply_dft_add_derived`]: one `res_size`-limb `VecZnxDft`
/// plus the convolution's own scratch.
#[doc(hidden)]
pub fn cnv_apply_dft_add_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    BE: HalConvolutionImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx_dft(module.n(), 1, res_size))
        + BE::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
}

/// `res += a (x) b`. The staging temporary spans exactly `res.size()` limbs and
/// `cnv_apply_dft` zero-fills the limbs past the convolution bound, so those
/// limbs of `res` receive zero and keep their previous value, as the contract
/// requires.
#[doc(hidden)]
pub fn cnv_apply_dft_add_derived<BE>(
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
    BE: HalConvolutionImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, 1, res_size);
    BE::cnv_apply_dft(module, cnv_offset, &mut tmp, 0, a, a_col, b, b_col, &mut scratch);
    BE::vec_znx_dft_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`cnv_apply_dft_sum_derived`]: the larger of the one
/// overwriting and the accumulating product the per-term fallback chains.
#[doc(hidden)]
pub fn cnv_apply_dft_sum_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    BE: HalConvolutionImpl,
{
    BE::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        .max(BE::cnv_apply_dft_add_tmp_bytes(module, cnv_offset, res_size, a_size, b_size))
}

/// `res[res_col] = Σ_t a_t (x) b_t` (overwriting): the first term overwrites
/// with `cnv_apply_dft`, which also zeroes the limbs past the convolution
/// bound, and the remaining terms fold in with `cnv_apply_dft_add`. An empty
/// `terms` slice zeroes the destination column.
#[doc(hidden)]
pub fn cnv_apply_dft_sum_derived<'a, BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[CnvDftAccTerm<'a, BE>],
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: HalConvolutionImpl + 'a,
{
    if terms.is_empty() {
        BE::vec_znx_dft_zero(module, res, res_col);
        return;
    }
    for (idx, term) in terms.iter().enumerate() {
        if idx == 0 {
            BE::cnv_apply_dft(
                module, cnv_offset, res, res_col, &term.a, term.a_col, &term.b, term.b_col, scratch,
            );
        } else {
            BE::cnv_apply_dft_add(
                module, cnv_offset, res, res_col, &term.a, term.a_col, &term.b, term.b_col, scratch,
            );
        }
    }
}

/// Scratch for [`cnv_by_const_apply_add_derived`]: one `res_size`-limb
/// `VecZnxBig` plus the product's own scratch.
#[doc(hidden)]
pub fn cnv_by_const_apply_add_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    BE: HalConvolutionImpl,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx_big(module.n(), 1, res_size))
        + BE::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
}

/// `res += a (x) b[b_coeff]`, with the same untouched-limb contract as
/// [`cnv_apply_dft_add_derived`].
#[doc(hidden)]
pub fn cnv_by_const_apply_add_derived<BE>(
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
    BE: HalConvolutionImpl,
{
    let res_size: usize = ZnxInfos::size(res);
    let (mut tmp, mut scratch) = ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), module, 1, res_size);
    BE::cnv_by_const_apply(module, cnv_offset, &mut tmp, 0, a, a_col, b, b_col, b_coeff, &mut scratch);
    BE::vec_znx_big_add_assign(module, res, res_col, &tmp.to_backend_ref(), 0);
}

/// Scratch for [`cnv_pairwise_apply_dft_derived`]: the larger of the one
/// overwriting and the three accumulating products it chains.
#[doc(hidden)]
pub fn cnv_pairwise_apply_dft_tmp_bytes_derived<BE>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    BE: HalConvolutionImpl,
{
    BE::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        .max(BE::cnv_apply_dft_add_tmp_bytes(module, cnv_offset, res_size, a_size, b_size))
}

/// `res = (a[i] + a[j]) (x) (b[i] + b[j])`, expanded in the DFT domain where
/// the prepared operands are linear; `i == j` degenerates to one product.
#[doc(hidden)]
pub fn cnv_pairwise_apply_dft_derived<BE>(
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
    BE: HalConvolutionImpl,
{
    BE::cnv_apply_dft(module, cnv_offset, res, res_col, a, i, b, i, scratch);
    if i == j {
        return;
    }
    BE::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, i, b, j, scratch);
    BE::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, j, b, i, scratch);
    BE::cnv_apply_dft_add(module, cnv_offset, res, res_col, a, j, b, j, scratch);
}
