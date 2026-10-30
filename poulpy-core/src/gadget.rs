//! Gadget operands, product windows and limb placement shared by the reference
//! bodies, the derived defaults and backend overrides.

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeVecZnx, VecZnxDftApply, VecZnxGadgetDecompose, VecZnxGadgetDecomposeTmpBytes,
        VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{Backend, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxToBackendRef, vec_znx_backend_ref_from_ref},
};

use crate::layouts::{
    Base2K, Dnum, GGLWEInfos, GadgetProductOutputSizeParams, LWEInfos, TorusPrecision, digits_aligned,
    gadget_product_output_size, prepared::GGLWEPreparedBackendRef,
};

/// Limbs of the DFT operand a gadget product with digits of `k_digit` bits
/// reads, for an input of precision `a_k` stored on `a_size` limbs.
///
/// Whole-limb digits read the input limbs. Other widths lay the input out as
/// `ceil(k_digit / base2k)` sub-limbs per digit, for the digits it reaches and
/// at most `dnum`.
pub fn gadget_input_size(base2k: Base2K, k_digit: TorusPrecision, dnum: Dnum, a_k: TorusPrecision, a_size: usize) -> usize {
    if digits_aligned(base2k, k_digit) {
        a_size
    } else {
        let digits: usize = a_k.as_usize().div_ceil(k_digit.as_usize()).min(dnum.as_usize());
        digits * k_digit.as_usize().div_ceil(base2k.as_usize())
    }
}

/// `a` restricted to the limbs that precision `a_k` covers at radix `base2k`,
/// or `None` when it covers none, as for an input of precision 0.
///
/// A buffer may hold more limbs than its precision, and operations sized from
/// that precision read only these: the digit layout rounds at the gadget
/// precision, which can reach past `a_k`, and the body of a key switch adds
/// the input back.
pub fn precision_limbs<'a, BE: Backend>(
    a: &'a VecZnxBackendRef<'_, BE>,
    base2k: Base2K,
    a_k: TorusPrecision,
) -> Option<VecZnxBackendRef<'a, BE>> {
    let live: usize = a_k.as_usize().div_ceil(base2k.as_usize()).min(a.size());
    (live > 0).then(|| vec_znx_backend_ref_from_ref::<BE>(a).window_limbs(0, 1, live))
}

/// Scratch of [`gadget_dft_apply`]: none for whole-limb digits.
pub fn gadget_dft_apply_tmp_bytes<BE, M>(
    module: &M,
    base2k: Base2K,
    k_digit: TorusPrecision,
    dnum: Dnum,
    a_k: TorusPrecision,
    a_size: usize,
) -> usize
where
    BE: Backend,
    M: ModuleN + VecZnxGadgetDecomposeTmpBytes,
{
    if digits_aligned(base2k, k_digit) {
        0
    } else {
        let size: usize = gadget_input_size(base2k, k_digit, dnum, a_k, a_size);
        BE::scratch_aligned(BE::vec_znx_bytes_of(module.n(), 1, size)) + module.vec_znx_gadget_decompose_tmp_bytes()
    }
}

/// DFT of `a[a_col]`, at radix `base2k` and precision `a_k`, into `res[res_col]`
/// as the operand of a gadget product with digits of `k_digit` bits; `res`
/// holds at least [`gadget_input_size`] limbs.
///
/// Whole-limb digits transform the input as is, exactly as before. Other
/// widths first lay the limbs `a_k` covers ([`precision_limbs`]) out on
/// the digit grid with `vec_znx_gadget_decompose`, rounded at the gadget
/// precision.
#[allow(clippy::too_many_arguments)]
pub fn gadget_dft_apply<BE, M>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    base2k: Base2K,
    k_digit: TorusPrecision,
    dnum: Dnum,
    a_k: TorusPrecision,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: VecZnxDftApply<BE> + VecZnxGadgetDecompose<BE>,
{
    if digits_aligned(base2k, k_digit) {
        module.vec_znx_dft_apply(1, 0, res, res_col, a, a_col);
        return;
    }
    let size: usize = gadget_input_size(base2k, k_digit, dnum, a_k, a.size());
    assert!(
        res.size() >= size,
        "gadget operand needs {size} limbs, res has {}",
        res.size()
    );
    // An input of precision 0 has no digit, and its operand no limb.
    let Some(a_live) = precision_limbs::<BE>(a, base2k, a_k).filter(|_| size > 0) else {
        return;
    };
    let (mut digits, mut scratch_1) = scratch.borrow().take_vec_znx_scratch(a.n(), 1, size);
    module.vec_znx_gadget_decompose(
        &mut digits,
        0,
        &a_live,
        a_col,
        base2k.as_usize(),
        k_digit.as_usize(),
        &mut scratch_1.borrow(),
    );
    module.vec_znx_dft_apply(1, 0, res, res_col, &digits.to_backend_ref(), 0);
}

/// Practical limb window used for an immediately normalized GGLWE/VMP product.
///
/// Any lower limb can affect rounding through a sufficiently long carry chain.
/// The retained window covers the live precision plus the worst-case norm
/// growth of the signed polynomial products and VMP accumulation, on every
/// backend; see `gadget_product_output_size`.
pub fn gglwe_product_output_size<BE, R, A, K>(res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    gglwe_product_accumulation_output_size::<BE, _, _, _>(res_infos, a_infos, key_infos, 1)
}

/// Number of limbs required when `term_count` GGLWE/VMP products are summed
/// before a single normalization.
///
/// Relative to one product, summing `term_count` values can amplify the tail
/// by that factor, which the product-norm window accounts for.
pub fn gglwe_product_accumulation_output_size<BE, R, A, K>(res_infos: &R, a_infos: &A, key_infos: &K, term_count: usize) -> usize
where
    BE: Backend,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    gglwe_product_accumulation_output_size_with_tail(res_infos, a_infos, key_infos, term_count, 0)
}

pub(crate) fn gglwe_product_accumulation_output_size_with_tail<R, A, K>(
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    term_count: usize,
    extra_live_limbs: usize,
) -> usize
where
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    let product_terms = key_infos
        .n()
        .as_usize()
        .saturating_mul(key_infos.dnum().as_usize())
        .saturating_mul(key_infos.dsize().as_usize())
        .saturating_mul(key_infos.rank_in().as_usize().max(1))
        .saturating_mul(term_count.max(1));
    gadget_product_output_size(GadgetProductOutputSizeParams {
        key_size: key_infos.size(),
        key_base2k: key_infos.base2k(),
        input_k: a_infos.k(),
        output_k: res_infos.k(),
        k_digit: key_infos.k_digit(),
        k_offset: key_infos.k_offset(),
        k_aux: key_infos.k_aux(),
        product_terms,
        extra_live_limbs,
    })
}

/// Limbs of the product of an `a_size`-limb operand by a `b_size`-limb operand, from limb `cnv_offset_hi` on.
///
/// Limb `k` of a product sums the pairs `i + j = k`, so it has `a_size + b_size - 1` limbs.
/// One limb more would be zero: it would be transformed and decomposed like the others, and counted as live.
pub fn prod_dft_size(a_size: usize, b_size: usize, cnv_offset_hi: usize) -> usize {
    (a_size + b_size - cnv_offset_hi).saturating_sub(1).max(1)
}

pub fn cnv_offset_to_limb_offset(cnv_offset: usize, base2k: usize) -> (usize, i64) {
    assert_ne!(base2k, 0);
    if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    }
}

#[inline]
pub(crate) fn normalize_input_limb_bound(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    offset_bits: usize,
) -> usize {
    full_size.min((res_size * res_base2k + offset_bits).div_ceil(in_base2k))
}

#[inline]
pub fn normalize_input_limb_bound_with_offset(
    full_size: usize,
    res_size: usize,
    res_base2k: usize,
    in_base2k: usize,
    res_offset: i64,
) -> usize {
    let mut offset_bits = res_offset % in_base2k as i64;
    if res_offset < 0 && offset_bits != 0 {
        offset_bits += in_base2k as i64;
    }
    normalize_input_limb_bound(full_size, res_size, res_base2k, in_base2k, offset_bits as usize)
}

/// Writes the plaintext of gadget row `row`, `pt[pt_col] * 2^-(k_offset + (row + 1) * k_digit)`,
/// into `res[res_col]`, canonical at radix `base2k` and precision `k`.
///
/// The scalar is read as a one-limb vector at radix 1, worth `pt / 2`, and
/// scaled by `2^(1 - k_offset - (row + 1) * k_digit)`, so the row lands on any
/// bit and not only on a limb boundary. Every limb of `res[res_col]` is written.
/// For `k_offset = 0` and `k_digit = dsize * base2k` this is the scalar added to
/// limb `(dsize - 1) + row * dsize` and normalized, bit for bit.
///
/// # Panics
///
/// Unless the row lies within the precision, `k_offset + (row + 1) * k_digit <= k`.
#[allow(clippy::too_many_arguments)]
pub fn gadget_row_plaintext<BE>(
    module: &poulpy_hal::layouts::Module<BE>,
    res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, BE>,
    res_col: usize,
    base2k: usize,
    k: usize,
    row: usize,
    k_digit: usize,
    k_offset: usize,
    pt: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, BE>,
    pt_col: usize,
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) where
    BE: poulpy_hal::oep::HalVecZnxImpl,
{
    let position: usize = k_offset + (row + 1) * k_digit;
    assert!(position <= k, "gadget row {row} at bit {position} is below the precision {k}");
    let src = poulpy_hal::layouts::scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(pt);
    BE::vec_znx_normalize(module, res, base2k, k, 1 - position as i64, res_col, &src, 1, pt_col, scratch);
}

/// Limbs of the gadget operand [`gadget_dft_apply_from_dft`] lays out for `key` from a source of
/// `a_size` limbs taken over `output_size`, at most `key.size()`.
pub fn gadget_dft_operand_size<K: GGLWEInfos>(key: &K, a_size: usize, output_size: usize) -> usize {
    let mask_small_size = a_size.min(output_size);
    let mask_k = TorusPrecision((mask_small_size * key.base2k().as_usize()) as u32);
    gadget_input_size(key.base2k(), key.k_digit(), key.dnum(), mask_k, mask_small_size)
}

/// The gadget operand of `key` for the mask columns of the DFT source `a`, into the
/// `a.cols() - 1` columns of `res`: each mask column normalized from its inverse transform over
/// `a.size().min(output_size)` limbs at radix `a_base2k`, then laid out in digits. The mask
/// columns of `a` are unspecified afterwards. The operand of each source of
/// [`GGLWEProductDigitsStridedImpl::gglwe_products_from_dft`](crate::oep::GGLWEProductDigitsStridedImpl::gglwe_products_from_dft).
pub fn gadget_dft_apply_from_dft<BE, M>(
    module: &M,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_base2k: usize,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    output_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + VecZnxDftApply<BE>
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxGadgetDecompose<BE>
        + VecZnxGadgetDecomposeTmpBytes,
{
    let rank = a.cols() - 1;
    let key_base2k = key.base2k().as_usize();
    assert!(
        a_base2k == key_base2k,
        "gadget_dft_apply_from_dft: the source's base2k differs from the key's"
    );
    assert!(
        scratch.available() >= gadget_dft_apply_from_dft_tmp_bytes::<BE, _, _>(module, a.size(), output_size, key),
        "gadget_dft_apply_from_dft: scratch below its query"
    );
    let mask_small_size = a.size().min(output_size);
    let (k_digit, dnum) = (key.k_digit(), key.dnum());
    let mask_k = TorusPrecision((mask_small_size * key_base2k) as u32);
    let (mut col_small, mut scratch_1) = scratch.borrow().take_vec_znx_scratch(a.n(), 1, mask_small_size);
    for c in 0..rank {
        module.vec_znx_idft_normalize(
            &mut col_small,
            key_base2k,
            mask_small_size * key_base2k,
            0,
            0,
            a,
            c + 1,
            a_base2k,
            None,
            &mut scratch_1.borrow(),
        );
        gadget_dft_apply(
            module,
            res,
            c,
            &col_small.to_backend_ref(),
            0,
            key.base2k(),
            k_digit,
            dnum,
            mask_k,
            &mut scratch_1.borrow(),
        );
    }
}

/// Scratch of [`gadget_dft_apply_from_dft`]: one normalized mask column beside its inverse
/// transform and its gadget layout.
pub fn gadget_dft_apply_from_dft_tmp_bytes<BE, M, K>(module: &M, a_size: usize, output_size: usize, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + VecZnxIdftNormalizeTmpBytes + VecZnxGadgetDecomposeTmpBytes,
    K: GGLWEInfos,
{
    let mask_small_size = a_size.min(output_size);
    let (base2k, k_digit, dnum) = (key_infos.base2k(), key_infos.k_digit(), key_infos.dnum());
    let mask_k = TorusPrecision((mask_small_size * base2k.as_usize()) as u32);
    BE::scratch_aligned(BE::vec_znx_bytes_of(module.n(), 1, mask_small_size))
        + module
            .vec_znx_idft_normalize_tmp_bytes(mask_small_size, a_size)
            .max(gadget_dft_apply_tmp_bytes::<BE, _>(
                module,
                base2k,
                k_digit,
                dnum,
                mask_k,
                mask_small_size,
            ))
}
