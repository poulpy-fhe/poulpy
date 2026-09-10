//! Shared normalization loops. Public inputs follow [`crate::api::VecZnxNormalize`]:
//! i64 coefficients in `[-2^62, 2^62]` and radix widths in `1..=62`.
//! Coefficient, in-place and disjoint-range entrypoints share these bounds.

use std::{marker::PhantomData, mem::size_of};

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxShape, ZnxView},
    reference::znx::{
        I64NormalizeOps, ZnxAddAssign, ZnxCopy, ZnxMulPowerOfTwoAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
    },
};

#[cfg(test)]
use crate::layouts::ZnxViewMut;

#[cfg(test)]
fn alloc_host_vec_znx(n: usize, cols: usize, size: usize) -> crate::layouts::VecZnx<Vec<u8>, i64> {
    use crate::layouts::VecZnx;

    crate::layouts::VecZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, i64>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<i64>()
}

#[inline]
pub(crate) fn normalize_needs_exact(a_size: usize, a_base2k: usize, res_size: usize, res_base2k: usize, offset: i64) -> bool {
    if a_size == 0 || res_size == 0 || a_base2k > 63 || res_base2k > 63 {
        return true;
    }
    let a_bits = (a_size * a_base2k) as i64;
    let res_bits = (res_size * res_base2k) as i64;
    // Dropping at most one same-base limb rounds only in the first carry step.
    let min_offset = if a_base2k == res_base2k {
        (a_bits - a_base2k as i64) - res_bits
    } else {
        a_bits - res_bits
    };
    offset < min_offset || offset >= a_bits
}

#[inline]
pub(crate) fn normalize_cross_needs_exact(a_size: usize, a_base2k: usize, res_base2k: usize, res_k: usize, offset: i64) -> bool {
    if a_size == 0 || !(1..=63).contains(&a_base2k) || !(1..=63).contains(&res_base2k) {
        return true;
    }
    let a_bits = a_size as i128 * a_base2k as i128;
    let aligned_offset = offset.div_euclid(a_base2k as i64) as i128 * a_base2k as i128;
    // The slice traversal needs a source limb at the rounding boundary.
    a_bits + res_k as i128 > i64::MAX as i128 || res_k as i128 + aligned_offset <= 0 || offset as i128 >= a_bits
}

pub(crate) struct VecZnxRangeMut<'a> {
    ptr: *mut i64,
    shape: VecZnxShape,
    col: usize,
    start: usize,
    len: usize,
    marker: PhantomData<&'a mut i64>,
}

impl<'a> VecZnxRangeMut<'a> {
    /// # Safety
    ///
    /// `ptr` is the base of the dense buffer `shape` describes, and every element
    /// `shape` selects in column `col`, coefficients `start..start + len`, is
    /// exclusively writable for `'a`.
    pub(crate) unsafe fn new(ptr: *mut i64, shape: VecZnxShape, col: usize, start: usize, len: usize) -> Self {
        Self {
            ptr,
            shape,
            col,
            start,
            len,
            marker: PhantomData,
        }
    }

    pub(crate) fn at_mut(&mut self, limb: usize) -> &mut [i64] {
        let offset = self.shape.scalar_offset(self.col, limb) + self.start;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), self.len) }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + I64NormalizeOps
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_k <= res.size() * res_base2k);
    let n = res.n();
    if res_base2k != a_base2k && n > 512 {
        for start in (0..n).step_by(512) {
            vec_znx_normalize_range::<BE>(
                res,
                res_base2k,
                res_k,
                res_offset,
                res_col,
                a,
                a_base2k,
                a_col,
                start,
                512.min(n - start),
                carry,
            );
        }
    } else {
        vec_znx_normalize_range::<BE>(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, 0, n, carry);
    }
}

/// [`vec_znx_normalize`] restricted to `[coeff_start, coeff_start + coeff_len)`;
/// `carry` needs `3 * coeff_len` elements private to the range.
#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_range<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + I64NormalizeOps
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    {
        assert_eq!(res.n(), a.n());
        assert!(coeff_start + coeff_len <= res.n());
        assert!(carry.len() >= 3 * coeff_len);
    }
    let res_shape = res.shape();
    let ptr = res.data_mut().as_mut().as_mut_ptr().cast::<i64>();
    unsafe {
        vec_znx_normalize_range_raw::<BE>(
            ptr,
            res_shape,
            res_base2k,
            res_k,
            res_offset,
            res_col,
            a,
            a_base2k,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        )
    }
}

/// Normalizes one coefficient range into a raw host output.
///
/// # Safety
///
/// `res_ptr` must be non-null, aligned for `i64` and the base of the dense buffer
/// `res_shape` describes; every element `res_shape` selects must be initialized.
/// Layout arithmetic is checked by [`VecZnxShape`].
/// The source must have valid initialized storage, degree `res_shape.n()`, and column `a_col`.
/// Require `res_col < res_shape.cols()`, `coeff_start + coeff_len <= res_shape.n()`,
/// and at least `3 * coeff_len` private scratch words in `carry`.
///
/// For every destination limb, the selected coefficient range must be exclusively
/// writable for this call and must not overlap source or scratch storage. Concurrent
/// calls may share the destination allocation only with disjoint coefficient ranges.
/// Source reads must remain immutable, and scratch must not overlap the source.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn vec_znx_normalize_range_raw<'a, BE>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + I64NormalizeOps
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufRef<'a>: HostDataRef,
{
    let (n, cols, size) = (res_shape.n(), res_shape.cols(), res_shape.size());
    {
        assert_eq!(n, a.n());
        assert!(res_col < cols);
        assert!(coeff_start + coeff_len <= n);
        assert!(carry.len() >= 3 * coeff_len);
        assert!(res_k <= size * res_base2k);
    }
    let mut res = unsafe { VecZnxRangeMut::new(res_ptr, res_shape, res_col, coeff_start, coeff_len) };
    if res_k == 0 {
        for limb in 0..size {
            res.at_mut(limb).fill(0);
        }
        return;
    }
    let active_size = res_k.div_ceil(res_base2k);
    let padding = (res_base2k - res_k % res_base2k) % res_base2k;
    let partial = res_k != size * res_base2k;
    if res_base2k == a_base2k
        && res_base2k <= 63
        && (partial || normalize_needs_exact(a.size(), a_base2k, active_size, res_base2k, res_offset))
    {
        let limb_offset = res_offset.div_euclid(res_base2k as i64);
        let boundary = (active_size as i64 - 1).saturating_add(limb_offset);
        if (0..a.size() as i64).contains(&boundary) {
            let lsh = res_offset.rem_euclid(res_base2k as i64) as usize;
            let carry = &mut carry[..coeff_len];
            let discarded = boundary as usize + 1 < a.size();
            for j in (boundary as usize + 1..a.size()).rev() {
                let source = &a.at(a_col, j)[coeff_start..coeff_start + coeff_len];
                match (j + 1 == a.size(), padding == 0 && j == boundary as usize + 1) {
                    (true, false) => BE::znx_normalize_floor::<false, false>(res_base2k, lsh, source, carry),
                    (true, true) => BE::znx_normalize_floor::<false, true>(res_base2k, lsh, source, carry),
                    (false, false) => BE::znx_normalize_floor::<true, false>(res_base2k, lsh, source, carry),
                    (false, true) => BE::znx_normalize_floor::<true, true>(res_base2k, lsh, source, carry),
                }
            }
            let source = &a.at(a_col, boundary as usize)[coeff_start..coeff_start + coeff_len];
            if discarded {
                BE::znx_normalize_round::<true, true>(res_base2k, lsh, padding, res.at_mut(active_size - 1), source, carry);
            } else {
                BE::znx_normalize_round::<false, true>(res_base2k, lsh, padding, res.at_mut(active_size - 1), source, carry);
            }
            for j in (0..active_size - 1).rev() {
                let source = j as i64 + limb_offset;
                if source >= 0 {
                    BE::znx_normalize_middle_step::<true>(
                        res_base2k,
                        lsh,
                        res.at_mut(j),
                        &a.at(a_col, source as usize)[coeff_start..coeff_start + coeff_len],
                        carry,
                    );
                } else {
                    BE::znx_extract_digit_mul(res_base2k, 0, res.at_mut(j), carry);
                }
            }
            for j in active_size..size {
                res.at_mut(j).fill(0);
            }
            return;
        }
    }
    let drops_precision = a.size() as i128 * a_base2k as i128 > res_k as i128 + res_offset as i128;
    let needs_exact = if res_base2k == a_base2k {
        normalize_needs_exact(a.size(), a_base2k, active_size, res_base2k, res_offset) || (partial && drops_precision)
    } else {
        normalize_cross_needs_exact(a.size(), a_base2k, res_base2k, res_k, res_offset)
    };
    if needs_exact {
        for i in 0..coeff_len {
            normalize_exact::<true, _, _>(
                |j| a.at(a_col, j)[coeff_start + i] as i128,
                a.size(),
                a_base2k,
                size,
                res_base2k,
                res_k,
                res_offset,
                |j, digit| res.at_mut(j)[i] = digit,
            );
        }
        return;
    }
    match res_base2k == a_base2k {
        true => vec_znx_normalize_inter_base2k::<BE>(
            res_base2k,
            &mut res,
            active_size,
            res_offset,
            a,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        ),
        false => vec_znx_normalize_cross_base2k::<BE>(
            &mut res,
            active_size,
            res_base2k,
            res_k,
            res_offset,
            a,
            a_base2k,
            a_col,
            coeff_start,
            coeff_len,
            carry,
        ),
    }
    for j in active_size..size {
        res.at_mut(j).fill(0);
    }
}

#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_inter_base2k<'r, 'a, BE>(
    base2k: usize,
    res: &mut VecZnxRangeMut<'r>,
    res_size: usize,
    res_offset: i64,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeMiddleStepAssign
        + I64NormalizeOps,
    BE::BufRef<'a>: HostDataRef,
{
    let (lo, hi) = (coeff_start, coeff_start + coeff_len);
    let a_size: usize = a.size();

    let (carry, _) = carry.split_at_mut(coeff_len);

    let mut lsh: i64 = res_offset % base2k as i64;
    let mut limbs_offset: i64 = res_offset / base2k as i64;

    // If res_offset is negative, makes it positive
    // and corrects by adding an additional offset
    // on the limbs.
    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + base2k as i64) % (base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    let res_end: usize = (-limbs_offset).clamp(0, res_size as i64) as usize;
    let res_start: usize = (a_size as i64 - limbs_offset).clamp(0, res_size as i64) as usize;
    let a_end: usize = limbs_offset.clamp(0, a_size as i64) as usize;
    let a_start: usize = (res_size as i64 + limbs_offset).clamp(0, a_size as i64) as usize;

    let a_out_range: usize = a_size.saturating_sub(a_start);

    // Computes the carry over the discarded limbs of a
    for j in 0..a_out_range {
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh_pos, &a.at(a_col, a_size - j - 1)[lo..hi], carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh_pos, &a.at(a_col, a_size - j - 1)[lo..hi], carry);
        }
    }

    // If no limbs were discarded, initialize carry to zero
    if a_out_range == 0 {
        BE::znx_zero(carry);
    }

    // Zeroes bottom limbs that will not be interacted with
    for j in res_start..res_size {
        BE::znx_zero(res.at_mut(j));
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    for j in 0..mid_range {
        let res_limb = res_start - j - 1;
        BE::znx_normalize_middle_step::<true>(
            base2k,
            lsh_pos,
            res.at_mut(res_limb),
            &a.at(a_col, a_start - j - 1)[lo..hi],
            carry,
        );
    }

    // Propagates the carry over the non-overlapping limbs between res and a
    for j in (0..res_end).rev() {
        BE::znx_extract_digit_mul(base2k, 0, res.at_mut(j), carry);
    }
}

#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_cross_base2k<'r, 'a, BE>(
    res: &mut VecZnxRangeMut<'r>,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + I64NormalizeOps
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufRef<'a>: HostDataRef,
{
    let (lo, hi) = (coeff_start, coeff_start + coeff_len);
    let a_size: usize = a.size();

    let (a_norm, carry) = carry.split_at_mut(coeff_len);
    let (res_carry, a_carry) = carry[..2 * coeff_len].split_at_mut(coeff_len);
    BE::znx_zero(res_carry);

    // Total precision (in bits) that `a` and `res` can represent.
    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_k;
    let drops_precision = a_tot_bits as i128 > res_k as i128 + res_offset as i128;

    // Derive intra-limb shift and cross-limb offset.
    let mut lsh: i64 = res_offset % a_base2k as i64;
    let mut limbs_offset: i64 = res_offset / a_base2k as i64;

    // If res_offset is negative, ensures it is positive
    // and corrects by incrementing the cross-limb offset.
    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + a_base2k as i64) % (a_base2k as i64);
        limbs_offset -= 1;
    }

    let lsh_pos: usize = lsh as usize;

    // Derive start/stop bit indexes of the overlap between `a` and `res` (after taking into account the offset)..
    let res_end_bit: usize = (-limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize; // Stop bit
    let res_start_bit: usize = (a_tot_bits as i64 - limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize; // Start bit
    let a_end_bit: usize = (limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize; // Stop bit
    let a_start_bit: usize = (res_tot_bits as i64 + limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize; // Start bit

    // Convert bits to limb indexes.
    let res_end: usize = res_end_bit / res_base2k;
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let a_end: usize = a_end_bit / a_base2k;
    let a_start: usize = a_start_bit.div_ceil(a_base2k);

    // Every overlapping limb is overwritten by its first extraction.
    for j in res_start..res_size {
        BE::znx_zero(res.at_mut(j));
    }

    // Case where offset is positive and greater or equal
    // to the precision of a.
    if res_start == 0 {
        return;
    }

    // Floor the discarded limbs; only the boundary contributes rounding.
    let a_out_range = a_size.saturating_sub(a_start);
    let take = (a_tot_bits - a_start_bit) % a_base2k;
    if a_out_range != 0 {
        for j in (a_start..a_size).rev() {
            let source = &a.at(a_col, j)[lo..hi];
            match (j + 1 == a_size, take == 0 && j == a_start) {
                (true, false) => BE::znx_normalize_floor::<false, false>(a_base2k, lsh_pos, source, a_carry),
                (true, true) => BE::znx_normalize_floor::<false, true>(a_base2k, lsh_pos, source, a_carry),
                (false, false) => BE::znx_normalize_floor::<true, false>(a_base2k, lsh_pos, source, a_carry),
                (false, true) => BE::znx_normalize_floor::<true, true>(a_base2k, lsh_pos, source, a_carry),
            }
        }
    } else if !drops_precision {
        BE::znx_zero(a_carry);
    }

    let mut res_acc_left = res_start_bit - (res_start - 1) * res_base2k;

    let mut res_limb: usize = res_start - 1;
    let mut initialized = false;

    // How many limbs of `a` overlap with `res` (after taking into account the offset).
    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;

        // Current res & a limbs
        let a_slice: &[i64] = &a.at(a_col, a_limb)[lo..hi];

        // Trackers: wow much of a_norm is left to
        // be flushed on res.
        let mut a_take_left: usize = a_base2k;

        if j == 0 && drops_precision {
            if a_out_range != 0 {
                BE::znx_normalize_round::<true, false>(a_base2k, lsh_pos, take, a_norm, a_slice, a_carry);
            } else {
                BE::znx_normalize_round::<false, false>(a_base2k, lsh_pos, take, a_norm, a_slice, a_carry);
            }
            a_take_left -= take;
        } else {
            BE::znx_normalize_middle_step::<true>(a_base2k, lsh_pos, a_norm, a_slice, a_carry);
            if j == 0 && take != 0 {
                BE::znx_mul_power_of_two_assign(-(take as i64), a_norm);
                a_take_left -= take;
            }
        }

        // Extract bits of `a_norm` and accumulates them on res[res_limb] until
        // res_base2k bits have been accumulated or until all bits of `a` are
        // extracted.
        'inner: loop {
            // Current limb of res
            let res_slice = res.at_mut(res_limb);

            // We can take at most a_base2k bits
            // but not more than what is left on a_norm or what is left to
            // fully populate the current limb of res.
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                // Extract `a_take` bits from a_norm and accumulates them on `res_slice`.
                let scale: usize = res_base2k - res_acc_left;
                if a_take == res_acc_left {
                    if initialized {
                        BE::znx_extract_digit_addmul_normalize::<false>(a_take, scale, res_base2k, res_slice, a_norm, res_carry);
                    } else {
                        BE::znx_extract_digit_addmul_normalize::<true>(a_take, scale, res_base2k, res_slice, a_norm, res_carry);
                    }
                } else if initialized {
                    BE::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);
                } else {
                    BE::znx_extract_digit_mul(a_take, scale, res_slice, a_norm);
                }
                initialized = true;
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            // If either:
            //  * At least `res_base2k` bits have been accumulated
            //  * We have reached the last limb of a
            // Then: Flushes them onto res
            if res_acc_left == 0 || a_limb == 0 {
                // This case happens only if `res_offset` is negative.
                // If `res_offset` is negative, we need to apply the offset BEFORE
                // the normalization to ensure the `res-offset` overflowing bits of `a`
                // are in the MSB of `res` instead of being discarded.
                if a_limb == 0 && a_take_left == 0 {
                    // Exhausting a normalized source leaves 0 or 1; its carry has room for that unit.
                    BE::znx_add_assign(a_carry, a_norm);

                    // Usual case where for example
                    // a:   [     overflow     ][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
                    // res:      [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
                    //
                    // where [overflow] are the overflowing bits of `a` (note that they are not a limb, but
                    // stored in a[0] & carry from a[1]) that are moved into the MSB of `res` due to the
                    // negative offset.
                    //
                    // In this case we populate what is left of `res_acc_left` using `a_carry`
                    //
                    // TODO: see if this can be simplified (e.g. just add).
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        BE::znx_extract_digit_addmul(res_acc_left, scale, res_slice, a_carry);
                    }

                    if res_acc_left != 0 {
                        BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                    }

                    // Previous step might not consume all bits of a_carry
                    // Extraction reduces a_carry before adding the bounded result carry.
                    BE::znx_add_assign(res_carry, a_carry);

                    // We are done, so breaks out of the loop (yes we are at a[0], but
                    // this avoids possible over/under flows of tracking variables)
                    break 'outer;
                }

                // If we reached the last limb of res
                if res_acc_left != 0 {
                    BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                }

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
                initialized = false;
            }

            // If a_norm is exhausted, breaks the inner loop.
            if a_take_left == 0 {
                BE::znx_add_assign(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    // This case will happen if offset is negative.
    if res_end != 0 {
        // If there are no overlapping limbs between `res` and `a`
        // (can happen if offset is negative), then we propagate the
        // carry of `a` on res. Note that the carry of `a` can be
        // greater than the precision of res.
        //
        // For example with offset = -8:
        //             a carry           a[0]     a[1]     a[2]     a[3]
        // a: [---------------------- ][x  x  x][x  x  x][x  x  x][x  x  x]
        // b: [x  x  x  x][x  x  x  x ]
        //        res[0]       res[1]
        //
        // If there are overlapping limbs between `res` and `a`,
        // we can use `res_carry`, which contains the carry of propagating
        // the shifted reconstruction of `a` in `res_base2k` along with
        // the carry of a[0].
        let carry_to_use = if a_start == a_end { a_carry } else { res_carry };

        for j in (0..res_end).rev() {
            BE::znx_extract_digit_mul(res_base2k, 0, res.at_mut(j), carry_to_use);
        }
    }
}

pub fn vec_znx_normalize_assign<'r, BE>(
    base2k: usize,
    res_k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + I64NormalizeOps
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign,
    BE::BufMut<'r>: HostDataMut,
{
    assert!(res_k <= res.size() * base2k);
    let n = res.n();
    vec_znx_normalize_assign_range::<BE>(base2k, res_k, res, res_col, 0, n, carry)
}

/// [`vec_znx_normalize_assign`] restricted to `[coeff_start, coeff_start + coeff_len)`.
fn vec_znx_normalize_assign_range<'r, BE>(
    base2k: usize,
    res_k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + I64NormalizeOps
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign,
    BE::BufMut<'r>: HostDataMut,
{
    {
        assert!(coeff_start + coeff_len <= res.n());
        assert!(carry.len() >= coeff_len);
    }

    let res_shape = res.shape();
    let ptr = res.data_mut().as_mut().as_mut_ptr().cast::<i64>();
    unsafe { vec_znx_normalize_assign_range_raw::<BE>(ptr, res_shape, base2k, res_k, res_col, coeff_start, coeff_len, carry) }
}

/// Normalizes one coefficient range in a raw host output.
///
/// # Safety
///
/// `res_ptr` must be non-null, aligned for `i64` and the base of the dense buffer
/// `res_shape` describes; every element `res_shape` selects must be initialized.
/// Layout arithmetic is checked by [`VecZnxShape`].
/// Require `res_col < res_shape.cols()`, `coeff_start + coeff_len <= res_shape.n()`,
/// and at least `coeff_len` private scratch words in `carry`.
///
/// For every limb, the selected coefficient range must be exclusively writable
/// and must not overlap scratch storage. Concurrent calls may share the allocation
/// only with disjoint coefficient ranges.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn vec_znx_normalize_assign_range_raw<BE>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    base2k: usize,
    res_k: usize,
    res_col: usize,
    coeff_start: usize,
    coeff_len: usize,
    carry: &mut [i64],
) where
    BE: Backend<ZnxWord = i64>
        + I64NormalizeOps
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let (cols, size) = (res_shape.cols(), res_shape.size());
    {
        assert!(res_col < cols);
        assert!(coeff_start + coeff_len <= res_shape.n());
        assert!(carry.len() >= coeff_len);
        assert!(res_k <= size * base2k);
    }
    let mut res = unsafe { VecZnxRangeMut::new(res_ptr, res_shape, res_col, coeff_start, coeff_len) };
    let carry = &mut carry[..coeff_len];
    if res_k == 0 {
        for j in 0..size {
            res.at_mut(j).fill(0);
        }
        return;
    }
    if base2k == 64 {
        for i in 0..coeff_len {
            normalize_exact::<true, _, _>(
                |j| unsafe { *res_ptr.add(res_shape.scalar_offset(res_col, j) + coeff_start + i) } as i128,
                size,
                base2k,
                size,
                base2k,
                res_k,
                0,
                |j, digit| res.at_mut(j)[i] = digit,
            );
        }
        return;
    }
    let active_size = res_k.div_ceil(base2k);
    if res_k != size * base2k {
        let padding = active_size * base2k - res_k;
        let discarded = active_size < size;
        for j in (active_size..size).rev() {
            match (j + 1 == size, padding == 0 && j == active_size) {
                (true, false) => BE::znx_normalize_floor::<false, false>(base2k, 0, res.at_mut(j), carry),
                (true, true) => BE::znx_normalize_floor::<false, true>(base2k, 0, res.at_mut(j), carry),
                (false, false) => BE::znx_normalize_floor::<true, false>(base2k, 0, res.at_mut(j), carry),
                (false, true) => BE::znx_normalize_floor::<true, true>(base2k, 0, res.at_mut(j), carry),
            }
        }
        if discarded {
            BE::znx_normalize_round_assign::<true>(base2k, 0, padding, res.at_mut(active_size - 1), carry);
        } else {
            BE::znx_normalize_round_assign::<false>(base2k, 0, padding, res.at_mut(active_size - 1), carry);
        }
        for j in (0..active_size - 1).rev() {
            BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(j), carry);
        }
        for j in active_size..size {
            res.at_mut(j).fill(0);
        }
        return;
    }
    for j in (0..size).rev() {
        if j == size - 1 {
            BE::znx_normalize_first_step_assign(base2k, 0, res.at_mut(j), carry);
        } else if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, 0, res.at_mut(j), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(j), carry);
        }
    }
}

#[test]
fn test_vec_znx_normalize_canonical_precision() {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };

    let (n, base2k, k) = (1, 4, 10);
    let mut a: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, 3);
    a.at_mut(0, 0)[0] = 3;
    a.at_mut(0, 1)[0] = 7;
    a.at_mut(0, 2)[0] = 7;

    let mut res: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, 4);
    let mut carry = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];
    vec_znx_normalize::<FFT64Ref>(
        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res),
        base2k,
        k,
        0,
        0,
        &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
        base2k,
        0,
        &mut carry,
    );

    assert_eq!(res.at(0, 0)[0], 4);
    assert_eq!(res.at(0, 1)[0], -8);
    assert_eq!(res.at(0, 2)[0], -8);
    assert_eq!(res.at(0, 3)[0], 0);

    vec_znx_normalize::<FFT64Ref>(
        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res),
        base2k,
        k,
        0,
        0,
        &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
        5,
        0,
        &mut carry,
    );
    assert_eq!(res.at(0, 0)[0], 2);
    assert_eq!(res.at(0, 1)[0], -6);
    assert_eq!(res.at(0, 2)[0], -4);
    assert_eq!(res.at(0, 3)[0], 0);

    vec_znx_normalize::<FFT64Ref>(
        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res),
        base2k,
        0,
        0,
        0,
        &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
        5,
        0,
        &mut carry,
    );
    assert!(res.data().iter().all(|&byte| byte == 0));
}

#[test]
fn test_vec_znx_normalize_cross_base2k() {
    use crate::{
        FFT64Ref,
        layouts::{FillUniform, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
        source::Source,
    };
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use dashu_float::{FBig, ops::Abs, round::mode::HalfEven};

    let prec: usize = 128;

    // Returns 2^exp as an FBig for any exp.
    let pow2 = |exp: u32| -> FBig<HalfEven> {
        let mut result = FBig::<HalfEven>::ONE;
        let chunk = FBig::<HalfEven>::from(1u64 << 63);
        let rem = exp % 63;
        let full = exp / 63;
        for _ in 0..full {
            result *= chunk.clone();
        }
        result * FBig::from(1u64 << rem)
    };

    // Reduces x modulo 1 toward zero (result in (-1, 1)), then adjusts to [-0.5, 0.5).
    // Using floor-based frac [0,1) + >= 0.5 subtract is equivalent to C fmod + adjust.
    let reduce = |x: FBig<HalfEven>| -> FBig<HalfEven> {
        let fl = x.floor();
        let mut r = x - fl; // now in [0, 1)
        if r >= FBig::<HalfEven>::from(1u64) / FBig::from(2u64) {
            r -= FBig::<HalfEven>::from(1u64);
        }
        r
    };

    for in_base2k in 1..=51 {
        for out_base2k in 1..=51 {
            for offset in [
                -(prec as i64),
                -(prec as i64 - 1),
                -(prec as i64 - in_base2k as i64),
                -(in_base2k as i64 + 1),
                in_base2k as i64,
                -(in_base2k as i64 - 1),
                0,
                (in_base2k as i64 - 1),
                in_base2k as i64,
                (in_base2k as i64 + 1),
                (prec as i64 - in_base2k as i64),
                (prec - 1) as i64,
                prec as i64,
            ] {
                let mut source: Source = Source::new([1u8; 32]);

                let in_size: usize = prec.div_ceil(in_base2k);
                let in_prec: u32 = (in_size * in_base2k) as u32;

                // Ensures no loss of precision (mostly for testing purpose)
                let out_size: usize = (in_prec as usize).div_ceil(out_base2k);

                let min_prec: u32 = (in_size * in_base2k).min(out_size * out_base2k) as u32;
                let mut want: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, in_size);
                want.fill_uniform(60, &mut source);

                let mut have: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, out_size);
                have.fill_uniform(60, &mut source);
                vec_znx_normalize::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut have),
                    out_base2k,
                    out_size * out_base2k,
                    offset,
                    0,
                    &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&want),
                    in_base2k,
                    0,
                    &mut carry,
                );

                let mut data_have: Vec<FBig<HalfEven>> = (0..n).map(|_| FBig::ZERO).collect();
                let mut data_want: Vec<FBig<HalfEven>> = (0..n).map(|_| FBig::ZERO).collect();

                have.decode_vec_float(out_base2k, 0, &mut data_have);
                want.decode_vec_float(in_base2k, 0, &mut data_want);

                let scale: FBig<HalfEven> = pow2(offset.unsigned_abs() as u32);

                if offset > 0 {
                    for x in &mut data_want {
                        *x = reduce(x.clone() * scale.clone());
                    }
                } else if offset < 0 {
                    for x in &mut data_want {
                        *x = reduce(x.clone() / scale.clone());
                    }
                } else {
                    for x in &mut data_want {
                        *x = reduce(x.clone());
                    }
                }

                // Adjust data_have to [-0.5, 0.5) (values already near torus domain)
                let half: FBig<HalfEven> = FBig::from(1u64) / FBig::from(2u64);
                let neg_half: FBig<HalfEven> = -FBig::from(1u64) / FBig::from(2u64);
                for x in &mut data_have {
                    if *x >= half {
                        *x = x.clone() - FBig::from(1u64);
                    } else if *x < neg_half {
                        *x = x.clone() + FBig::from(1u64);
                    }
                }

                for i in 0..n {
                    //println!("i:{i:02} {} {}", data_want[i], data_have[i]);

                    let err = (data_have[i].clone() - data_want[i].clone()).abs();
                    let err_log2: f64 = f64::try_from(err).unwrap_or(0.0).max(1e-60_f64).log2();

                    assert!(err_log2 <= -(min_prec as f64) + 1.0, "{} {}", err_log2, -(min_prec as f64))
                }
            }
        }
    }
}

#[test]
fn test_vec_znx_normalize_inter_base2k() {
    use crate::{
        FFT64Ref,
        layouts::{FillUniform, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
        source::Source,
    };
    let n: usize = 8;

    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];

    use dashu_float::{FBig, ops::Abs, round::mode::HalfEven};

    let mut source: Source = Source::new([1u8; 32]);

    let prec: usize = 128;
    let offset_range: i64 = prec as i64;

    // Returns 2^exp as an FBig for any exp.
    let pow2 = |exp: u32| -> FBig<HalfEven> {
        let mut result = FBig::<HalfEven>::ONE;
        let chunk = FBig::<HalfEven>::from(1u64 << 63);
        let rem = exp % 63;
        let full = exp / 63;
        for _ in 0..full {
            result *= chunk.clone();
        }
        result * FBig::from(1u64 << rem)
    };

    // Reduces x modulo 1 toward zero (result in (-1, 1)), then adjusts to [-0.5, 0.5).
    let reduce = |x: FBig<HalfEven>| -> FBig<HalfEven> {
        let fl = x.floor();
        let mut r = x - fl; // now in [0, 1)
        if r >= FBig::<HalfEven>::from(1u64) / FBig::from(2u64) {
            r -= FBig::<HalfEven>::from(1u64);
        }
        r
    };

    for base2k in 1..=51 {
        for offset in (-offset_range..=offset_range).step_by(base2k + 1) {
            let size: usize = prec.div_ceil(base2k);
            let out_prec: u32 = (size * base2k) as u32;

            // Fills "want" with uniform values
            let mut want: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, size);
            want.fill_uniform(60, &mut source);

            // Fills "have" with the shifted normalization of "want"
            let mut have: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, size);
            have.fill_uniform(60, &mut source);
            vec_znx_normalize::<FFT64Ref>(
                &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut have),
                base2k,
                size * base2k,
                offset,
                0,
                &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&want),
                base2k,
                0,
                &mut carry,
            );

            let mut data_have: Vec<FBig<HalfEven>> = (0..n).map(|_| FBig::ZERO).collect();
            let mut data_want: Vec<FBig<HalfEven>> = (0..n).map(|_| FBig::ZERO).collect();

            have.decode_vec_float(base2k, 0, &mut data_have);
            want.decode_vec_float(base2k, 0, &mut data_want);

            let scale: FBig<HalfEven> = pow2(offset.unsigned_abs() as u32);

            if offset > 0 {
                for x in &mut data_want {
                    *x = reduce(x.clone() * scale.clone());
                }
            } else if offset < 0 {
                for x in &mut data_want {
                    *x = reduce(x.clone() / scale.clone());
                }
            } else {
                for x in &mut data_want {
                    *x = reduce(x.clone());
                }
            }

            // Adjust data_have to [-0.5, 0.5) (values already near torus domain)
            let half: FBig<HalfEven> = FBig::from(1u64) / FBig::from(2u64);
            let neg_half: FBig<HalfEven> = -FBig::from(1u64) / FBig::from(2u64);
            for x in &mut data_have {
                if *x >= half {
                    *x = x.clone() - FBig::from(1u64);
                } else if *x < neg_half {
                    *x = x.clone() + FBig::from(1u64);
                }
            }

            for i in 0..n {
                //println!("i:{i:02} {} {}", data_want[i], data_have[i]);

                let err = (data_have[i].clone() - data_want[i].clone()).abs();
                let err_log2: f64 = f64::try_from(err).unwrap_or(0.0).max(1e-60_f64).log2();

                assert!(err_log2 <= -(out_prec as f64), "{} {}", err_log2, -(out_prec as f64))
            }
        }
    }
}

#[test]
fn test_vec_znx_normalize_limb_bounds() {
    use crate::{
        FFT64Ref,
        layouts::{FillUniform, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView},
        source::Source,
    };
    let n: usize = 8;
    let mut carry: Vec<i64> = vec![0i64; vec_znx_normalize_tmp_bytes(n) / size_of::<i64>()];
    let mut source: Source = Source::new([1u8; 32]);
    for in_base2k in 1..=51usize {
        for out_base2k in 1..=51usize {
            for offset in [-(in_base2k as i64), -3, -1, 0, 1, 3, in_base2k as i64] {
                for in_size in 1..=4usize {
                    for out_size in 1..=4usize {
                        let mut want: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, in_size);
                        want.fill_uniform(62, &mut source);
                        let mut have: VecZnx<Vec<u8>, i64> = alloc_host_vec_znx(n, 1, out_size);
                        vec_znx_normalize::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut have),
                            out_base2k,
                            out_size * out_base2k,
                            offset,
                            0,
                            &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&want),
                            in_base2k,
                            0,
                            &mut carry,
                        );
                        let bound: i64 = 1 << (out_base2k - 1);
                        for j in 0..out_size {
                            assert!(
                                have.at(0, j).iter().all(|x| (-bound..bound).contains(x)),
                                "in_base2k={in_base2k} out_base2k={out_base2k} offset={offset} in_size={in_size} out_size={out_size} limb={j}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Exact fixed-point quantization, with ties toward positive infinity.
#[cfg(test)]
pub(crate) fn normalize_integer_oracle(a: &[i128], a_base2k: usize, res_base2k: usize, res_size: usize, offset: i64) -> Vec<i64> {
    use dashu_int::IBig;
    let mut value = IBig::ZERO;
    for &limb in a {
        value = (value << a_base2k) + IBig::from(limb);
    }
    let shift = res_size as i64 * res_base2k as i64 + offset - a.len() as i64 * a_base2k as i64;
    if shift >= 0 {
        value <<= shift as usize;
    } else {
        let drop = (-shift) as usize;
        value = (value + (IBig::ONE << (drop - 1))) >> drop;
    }
    let radix = IBig::ONE << res_base2k;
    let half = IBig::ONE << (res_base2k - 1);
    let mut result = vec![0; res_size];
    for limb in result.iter_mut().rev() {
        let next = (&value + &half) >> res_base2k;
        *limb = i64::try_from(&value - &next * &radix).unwrap();
        value = next;
    }
    result
}

#[cfg(test)]
fn check_normalize_integer(a: &[i128], a_base2k: usize, res_base2k: usize, res_size: usize, offset: i64) {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let mut input = alloc_host_vec_znx(1, 1, a.len());
    for (j, &x) in a.iter().enumerate() {
        input.at_mut(0, j)[0] = x as i64;
    }
    let mut output = alloc_host_vec_znx(1, 1, res_size);
    for j in 0..res_size {
        output.at_mut(0, j)[0] = 12345;
    }
    vec_znx_normalize::<FFT64Ref>(
        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut output),
        res_base2k,
        res_size * res_base2k,
        offset,
        0,
        &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input),
        a_base2k,
        0,
        &mut [91; 3],
    );
    let have: Vec<i64> = (0..res_size).map(|j| output.at(0, j)[0]).collect();
    assert_eq!(
        have,
        normalize_integer_oracle(a, a_base2k, res_base2k, res_size, offset),
        "a={a:?}, a_base2k={a_base2k}, res_base2k={res_base2k}, res_size={res_size}, offset={offset}"
    );
}

#[test]
fn test_normalize_exact_regressions() {
    check_normalize_integer(&[], 2, 3, 1, -1);
    check_normalize_integer(&[1], 2, 2, 0, 0);
    check_normalize_integer(&[0, 1, 2], 2, 2, 1, 0);
    check_normalize_integer(&[9], 2, 2, 1, -4);
    check_normalize_integer(&[9], 2, 3, 1, -4);
    check_normalize_integer(&[-1], 3, 2, 1, 0);
    check_normalize_integer(&[1i128 << 62], 2, 3, 1, -5);
}

#[test]
fn test_normalize_integer_input_bound() {
    let mut state = 0x123456789abcdefu64;
    for a_base2k in 1..=62 {
        for res_base2k in 1..=62 {
            for a_size in 1..=8 {
                let a: Vec<i128> = (0..a_size)
                    .map(|j| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        match j % 4 {
                            0 => -(1i128 << 62),
                            1 => 1i128 << 62,
                            _ => (state as i64 >> 1) as i128,
                        }
                    })
                    .collect();
                for a in [a, vec![-(1i128 << 62); a_size], vec![1i128 << 62; a_size]] {
                    for res_size in [1, 3, 8] {
                        let bracket = (a_size * a_base2k + res_size * res_base2k) as i64;
                        let gap = (a_size * a_base2k) as i64 - (res_size * res_base2k) as i64;
                        for offset in [
                            -bracket,
                            -(a_base2k as i64) - 1,
                            -1,
                            0,
                            1,
                            a_base2k as i64,
                            bracket,
                            gap - a_base2k as i64 - 1,
                            gap - a_base2k as i64,
                            gap - 1,
                        ] {
                            check_normalize_integer(&a, a_base2k, res_base2k, res_size, offset);
                        }
                    }
                }
            }
        }
    }
}

/// Reads the two's-complement bits of the exact radix-weighted input integer.
pub(crate) struct NormalizationBits<F, const NARROW: bool> {
    source: F,
    left: usize,
    base2k: usize,
    carry: i128,
    digit: u64,
    available: usize,
}

impl<F: FnMut(usize) -> i128, const NARROW: bool> NormalizationBits<F, NARROW> {
    pub(crate) fn new(source: F, size: usize, base2k: usize) -> Self {
        Self {
            source,
            left: size,
            base2k,
            carry: 0,
            digit: 0,
            available: 0,
        }
    }

    fn refill(&mut self) {
        let mask = (1u64 << self.base2k) - 1;
        if self.left != 0 {
            self.left -= 1;
            let a = (self.source)(self.left);
            let low = (a as u64 & mask) + (self.carry as u64 & mask);
            self.carry = if NARROW {
                ((a as i64 >> self.base2k) + (self.carry as i64 >> self.base2k) + (low >> self.base2k) as i64) as i128
            } else {
                (a >> self.base2k) + (self.carry >> self.base2k) + (low >> self.base2k) as i128
            };
            self.digit = low & mask;
        } else {
            self.digit = self.carry as u64 & mask;
            self.carry = if NARROW {
                (self.carry as i64 >> self.base2k) as i128
            } else {
                self.carry >> self.base2k
            };
        }
        self.available = self.base2k;
    }

    fn read(&mut self, mut count: usize) -> u64 {
        let mut value = 0;
        let mut written = 0;
        while count != 0 {
            if self.available == 0 {
                self.refill();
            }
            let take = count.min(self.available);
            value |= (self.digit & ((1u64 << take) - 1)) << written;
            self.digit >>= take;
            self.available -= take;
            count -= take;
            written += take;
        }
        value
    }

    fn skip(&mut self, mut count: i128) {
        while count != 0 {
            if self.available == 0 {
                if self.left == 0 && (self.carry == 0 || self.carry == -1) {
                    return;
                }
                self.refill();
            }
            let take = count.min(self.available as i128) as usize;
            self.digit >>= take;
            self.available -= take;
            count -= take as i128;
        }
    }
}

/// Quantizes once at `res_k`, then emits centered digits with zero precision padding.
/// `NARROW` requires every source coefficient to fit in i64.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_exact<const NARROW: bool, F: FnMut(usize) -> i128, G: FnMut(usize, i64)>(
    source: F,
    a_size: usize,
    a_base2k: usize,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    mut output: G,
) {
    if a_base2k > 63 || res_base2k > 63 {
        normalize_exact_wide(source, a_size, a_base2k, res_size, res_base2k, res_k, offset, output);
        return;
    }
    let (active_size, precision_padding) = if res_k == res_size * res_base2k {
        (res_size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    let mut bits = NormalizationBits::<_, NARROW>::new(source, a_size, a_base2k);
    let drop = a_size as i128 * a_base2k as i128 - res_k as i128 - offset as i128;
    let mut carry = 0u64;
    let mut padding = 0i128;
    if drop > 0 {
        bits.skip(drop - 1);
        carry = bits.read(1);
    } else {
        padding = -drop;
    }
    let mut full_size = active_size;
    if precision_padding != 0 {
        full_size -= 1;
        let width = res_base2k - precision_padding;
        let zeroes = padding.min(width as i128) as usize;
        padding -= zeroes as i128;
        let raw = bits.read(width - zeroes) << zeroes;
        let value = raw + carry;
        carry = (value + (1u64 << (width - 1))) >> width;
        output(
            full_size,
            ((value as i128 - ((carry as i128) << width)) << precision_padding) as i64,
        );
    }
    let half = 1u64 << (res_base2k - 1);
    for j in (0..full_size).rev() {
        let zeroes = padding.min(res_base2k as i128) as usize;
        padding -= zeroes as i128;
        let raw = bits.read(res_base2k - zeroes) << zeroes;
        let value = raw + carry;
        carry = (value + half) >> res_base2k;
        output(j, (value as i128 - ((carry as i128) << res_base2k)) as i64);
    }
    for j in active_size..res_size {
        output(j, 0);
    }
}

// Preserves the wider source radices and base-64 output supported by NTT.
#[allow(clippy::too_many_arguments)]
fn normalize_exact_wide<F: FnMut(usize) -> i128, G: FnMut(usize, i64)>(
    mut source: F,
    a_size: usize,
    a_base2k: usize,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    mut output: G,
) {
    assert!((1..=127).contains(&a_base2k));
    assert!((1..=64).contains(&res_base2k));
    let (active_size, precision_padding) = if res_k == res_size * res_base2k {
        (res_size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    let mask = (1u128 << a_base2k) - 1;
    let mut left = a_size;
    let mut carry = 0i128;
    let mut pending = 0u128;
    let mut available = 0usize;
    let mut read = |mut count: i128, discard: bool| {
        let mut result = 0u128;
        let mut written = 0usize;
        while count != 0 {
            if available == 0 {
                if discard && left == 0 && (carry == 0 || carry == -1) {
                    break;
                }
                if left != 0 {
                    left -= 1;
                    let a = source(left);
                    let low = (a as u128 & mask) + (carry as u128 & mask);
                    carry = (a >> a_base2k) + (carry >> a_base2k) + (low >> a_base2k) as i128;
                    pending = low & mask;
                } else {
                    pending = carry as u128 & mask;
                    carry >>= a_base2k;
                }
                available = a_base2k;
            }
            let take = count.min(available as i128) as usize;
            if !discard {
                result |= (pending & ((1u128 << take) - 1)) << written;
                written += take;
            }
            pending >>= take;
            available -= take;
            count -= take as i128;
        }
        result
    };
    let drop = a_size as i128 * a_base2k as i128 - res_k as i128 - offset as i128;
    let mut rounding = 0u128;
    let mut padding = 0i128;
    if drop > 0 {
        read(drop - 1, true);
        rounding = read(1, false);
    } else {
        padding = -drop;
    }
    let mut full_size = active_size;
    if precision_padding != 0 {
        full_size -= 1;
        let width = res_base2k - precision_padding;
        let zeroes = padding.min(width as i128) as usize;
        padding -= zeroes as i128;
        let value = (read((width - zeroes) as i128, false) << zeroes) + rounding;
        rounding = (value + (1u128 << (width - 1))) >> width;
        output(
            full_size,
            ((value as i128 - ((rounding as i128) << width)) << precision_padding) as i64,
        );
    }
    let half = 1u128 << (res_base2k - 1);
    for j in (0..full_size).rev() {
        let zeroes = padding.min(res_base2k as i128) as usize;
        padding -= zeroes as i128;
        let value = (read((res_base2k - zeroes) as i128, false) << zeroes) + rounding;
        rounding = (value + half) >> res_base2k;
        output(j, (value as i128 - ((rounding as i128) << res_base2k)) as i64);
    }
    for j in active_size..res_size {
        output(j, 0);
    }
}

#[test]
fn test_normalize_inter_window_exhaustive() {
    for k in 1..=62i64 {
        for a_size in 1..=8i64 {
            for res_size in 1..=8i64 {
                let bracket = (a_size + res_size) * k;
                for offset in -bracket..=bracket {
                    let q = offset.div_euclid(k);
                    let lsh = offset.rem_euclid(k);
                    assert_eq!(offset, q * k + lsh);
                    let res_end = (-q).clamp(0, res_size);
                    let res_start = (a_size - q).clamp(0, res_size);
                    let a_end = q.clamp(0, a_size);
                    let a_start = (res_size + q).clamp(0, a_size);
                    assert_eq!(res_start - res_end, a_start - a_end);
                    for j in 0..a_size {
                        assert_eq!((a_end..a_start).contains(&j), (0..res_size).contains(&(j - q)));
                    }
                    for j in 0..res_size {
                        assert_eq!((res_end..res_start).contains(&j), (0..a_size).contains(&(j + q)));
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_centered_boundaries() {
    check_normalize_centered(false);
}

#[test]
#[ignore = "285 million cases; run explicitly with --release --ignored"]
fn test_normalize_exhaustive_centered_full() {
    check_normalize_centered(true);
}

#[cfg(test)]
fn check_normalize_centered(exhaustive: bool) {
    use dashu_int::IBig;

    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let bases = if exhaustive {
        (1..=6usize).collect::<Vec<_>>()
    } else {
        vec![1, 2, 17, 19, 21, 50, 51, 61, 62]
    };
    for &a_base2k in &bases {
        let half_input = 1i64 << (a_base2k - 1);
        let alphabet = if exhaustive {
            (-half_input..half_input).collect::<Vec<_>>()
        } else {
            let mut digits = vec![-half_input, -half_input + 1, -1, 0, 1, half_input - 1];
            digits.retain(|x| (-half_input..half_input).contains(x));
            digits.sort_unstable();
            digits.dedup();
            digits
        };
        for a_size in 1..=3usize {
            let mut input = alloc_host_vec_znx(1, 1, a_size);
            for &res_base2k in &bases {
                let half = IBig::ONE << (res_base2k - 1);
                for res_size in 1..=3usize {
                    let mut output = alloc_host_vec_znx(1, 1, res_size);
                    let bracket = (a_size * a_base2k + res_size * res_base2k + 1) as i64;
                    let gap = (a_size * a_base2k) as i64 - (res_size * res_base2k) as i64;
                    let offsets = if exhaustive {
                        (-bracket..=bracket).collect::<Vec<_>>()
                    } else {
                        let k = a_base2k as i64;
                        let mut offsets = vec![
                            -bracket,
                            -k,
                            -1,
                            0,
                            1,
                            k,
                            bracket,
                            gap - k - 1,
                            gap - k,
                            gap - 1,
                            gap,
                            gap + 1,
                        ];
                        offsets.sort_unstable();
                        offsets.dedup();
                        offsets
                    };
                    let offsets: Vec<_> = offsets
                        .into_iter()
                        .map(|offset| {
                            let shift = res_size as i64 * res_base2k as i64 + offset - a_size as i64 * a_base2k as i64;
                            let rounding = if shift < 0 {
                                IBig::ONE << (-shift - 1) as usize
                            } else {
                                IBig::ZERO
                            };
                            (offset, shift, rounding)
                        })
                        .collect();
                    for code in 0..alphabet.len().pow(a_size as u32) {
                        let mut digits = code;
                        let mut integer = IBig::ZERO;
                        for j in 0..a_size {
                            let digit = alphabet[digits % alphabet.len()];
                            digits /= alphabet.len();
                            input.at_mut(0, j)[0] = digit;
                            integer = (integer << a_base2k) + IBig::from(digit);
                        }
                        for (offset, shift, rounding) in &offsets {
                            vec_znx_normalize::<FFT64Ref>(
                                &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut output),
                                res_base2k,
                                res_size * res_base2k,
                                *offset,
                                0,
                                &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input),
                                a_base2k,
                                0,
                                &mut [37; 3],
                            );
                            let mut expected = if *shift >= 0 {
                                &integer << *shift as usize
                            } else {
                                (&integer + rounding) >> (-shift) as usize
                            };
                            for j in (0..res_size).rev() {
                                let next = (&expected + &half) >> res_base2k;
                                let digit = i64::try_from(&expected - (&next << res_base2k)).unwrap();
                                assert_eq!(
                                    output.at(0, j)[0],
                                    digit,
                                    "code={code}, ka={a_base2k}, kr={res_base2k}, sa={a_size}, sr={res_size}, offset={offset}, limb={j}"
                                );
                                expected = next;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_window_and_range_integer() {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let n = 35;
    let mut state = 0x58d719b3ce42a670u64;
    for a_base2k in 1..=62usize {
        for res_base2k in [1, 2, 17, 50, 62] {
            for size in [1, 2, 8] {
                let mut input = alloc_host_vec_znx(n, 2, size);
                for j in 0..size {
                    for x in input.at_mut(1, j) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *x = state as i64 >> 1;
                    }
                }
                for offset in [
                    i64::MIN,
                    -1001,
                    -(a_base2k as i64) - 1,
                    -(a_base2k as i64),
                    -1,
                    0,
                    1,
                    a_base2k as i64,
                    1001,
                    i64::MAX,
                ] {
                    let mut whole = alloc_host_vec_znx(n, 2, size);
                    let mut split = alloc_host_vec_znx(n, 2, size);
                    for j in 0..size {
                        whole.at_mut(0, j).fill(93);
                        split.at_mut(0, j).fill(93);
                    }
                    let a = <VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input);
                    vec_znx_normalize::<FFT64Ref>(
                        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut whole),
                        res_base2k,
                        size * res_base2k,
                        offset,
                        1,
                        &a,
                        a_base2k,
                        1,
                        &mut vec![73; 3 * n],
                    );
                    for (start, end) in [(17, 35), (1, 8), (0, 1), (8, 17)] {
                        let mut scratch = vec![83; 3 * (end - start) + 2];
                        vec_znx_normalize_range::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut split),
                            res_base2k,
                            size * res_base2k,
                            offset,
                            1,
                            &a,
                            a_base2k,
                            1,
                            start,
                            end - start,
                            &mut scratch[1..3 * (end - start) + 1],
                        );
                        assert_eq!(scratch[0], 83);
                        assert_eq!(*scratch.last().unwrap(), 83);
                    }
                    for i in 0..n {
                        let mut single = alloc_host_vec_znx(1, 1, size);
                        let a_i =
                            <VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input).window_coeffs(i, 1);
                        vec_znx_normalize::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut single),
                            res_base2k,
                            size * res_base2k,
                            offset,
                            0,
                            &a_i,
                            a_base2k,
                            1,
                            &mut [53; 3],
                        );
                        // Extreme offsets reduce to zero without enormous oracle allocations.
                        let expected = if offset == i64::MIN || offset == i64::MAX {
                            vec![0; size]
                        } else {
                            normalize_integer_oracle(
                                &(0..size).map(|j| input.at(1, j)[i] as i128).collect::<Vec<_>>(),
                                a_base2k,
                                res_base2k,
                                size,
                                offset,
                            )
                        };
                        for (j, &digit) in expected.iter().enumerate() {
                            assert_eq!(
                                whole.at(1, j)[i],
                                digit,
                                "ka={a_base2k}, kr={res_base2k}, size={size}, offset={offset}, i={i}, j={j}"
                            );
                            assert_eq!(split.at(1, j)[i], digit);
                            assert_eq!(single.at(0, j)[0], digit);
                            assert_eq!(whole.at(0, j)[i], 93);
                            assert_eq!(split.at(0, j)[i], 93);
                        }
                    }
                }
                let mut assigned = input.clone();
                vec_znx_normalize_assign::<FFT64Ref>(
                    a_base2k,
                    size * a_base2k,
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut assigned),
                    1,
                    &mut vec![37; n],
                );
                let mut coeff_assigned = input.clone();
                for i in 0..n {
                    vec_znx_normalize_assign::<FFT64Ref>(
                        a_base2k,
                        size * a_base2k,
                        &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut coeff_assigned)
                            .window_coeffs(i, 1),
                        1,
                        &mut [91],
                    );
                    let want = normalize_integer_oracle(
                        &(0..size).map(|j| input.at(1, j)[i] as i128).collect::<Vec<_>>(),
                        a_base2k,
                        a_base2k,
                        size,
                        0,
                    );
                    for (j, &digit) in want.iter().enumerate() {
                        assert_eq!(assigned.at(1, j)[i], digit);
                        assert_eq!(coeff_assigned.at(1, j)[i], digit);
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_blocked_matches_single_range() {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let n = 1031;
    for (a_base2k, res_base2k) in [(17, 19), (51, 50), (62, 1), (1, 62)] {
        for size in [3, 8] {
            let mut input = alloc_host_vec_znx(n, 1, size);
            for j in 0..size {
                for (i, x) in input.at_mut(0, j).iter_mut().enumerate() {
                    *x = (i as i64).wrapping_mul(0x31e5cd283792b601).wrapping_add(j as i64) >> 1;
                }
            }
            let a = <VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input);
            for offset in [-(a_base2k as i64), 0, 1, a_base2k as i64] {
                let mut whole = alloc_host_vec_znx(n, 1, size);
                let mut blocked = alloc_host_vec_znx(n, 1, size);
                let mut carry = vec![31; 3 * n];
                vec_znx_normalize_range::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut whole),
                    res_base2k,
                    size * res_base2k,
                    offset,
                    0,
                    &a,
                    a_base2k,
                    0,
                    0,
                    n,
                    &mut carry,
                );
                vec_znx_normalize::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut blocked),
                    res_base2k,
                    size * res_base2k,
                    offset,
                    0,
                    &a,
                    a_base2k,
                    0,
                    &mut carry,
                );
                for j in 0..size {
                    assert_eq!(whole.at(0, j), blocked.at(0, j));
                }
            }
        }
    }
}
