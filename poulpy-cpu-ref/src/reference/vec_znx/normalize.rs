use std::{hint::black_box, mem::size_of};

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxNormalize, VecZnxNormalizeAssignBackend,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{
        ZnxAddAssign, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
    },
    source::Source,
};

#[cfg(test)]
fn alloc_host_vec_znx(n: usize, cols: usize, size: usize) -> VecZnx<Vec<u8>> {
    crate::layouts::VecZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<i64>()
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize_coeff<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    a_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    match res_base2k == a_base2k {
        true => vec_znx_normalize_coeff_inter_base2k::<BE>(res_base2k, res, res_offset, res_col, a, a_col, a_coeff, carry),
        false => {
            vec_znx_normalize_coeff_cross_base2k::<BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, a_coeff, carry)
        }
    }
}

pub fn vec_znx_normalize_coeff_assign<'r, BE>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    res_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert!(res_coeff < res.n(), "res_coeff: {res_coeff} >= res.n(): {}", res.n());
    }

    let res_size: usize = res.size();
    let carry = &mut carry[..1];

    for j in (0..res_size).rev() {
        let dst = &mut res.at_mut(res_col, j)[res_coeff..res_coeff + 1];
        if j == res_size - 1 {
            BE::znx_normalize_first_step_assign(base2k, 0, dst, carry);
        } else if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, 0, dst, carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, 0, dst, carry);
        }
    }
}

fn vec_znx_normalize_coeff_inter_base2k<'r, 'a, BE>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeMiddleStepAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert_eq!(
            res.n(),
            1,
            "vec_znx_normalize_coeff expects a 1-coeff destination, got {}",
            res.n()
        );
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let carry = &mut carry[..1];

    let mut lsh: i64 = res_offset % base2k as i64;
    let mut limbs_offset: i64 = res_offset / base2k as i64;
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

    for j in 0..a_out_range {
        let src = [a.at(a_col, a_size - j - 1)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh_pos, &src, carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh_pos, &src, carry);
        }
    }

    if a_out_range == 0 {
        carry.fill(0);
    }

    for j in res_start..res_size {
        res.at_mut(res_col, j).fill(0);
    }

    let mid_range: usize = a_start.saturating_sub(a_end);
    for j in 0..mid_range {
        let src = [a.at(a_col, a_start - j - 1)[a_coeff]];
        BE::znx_normalize_middle_step::<true>(base2k, lsh_pos, res.at_mut(res_col, res_start - j - 1), &src, carry);
    }

    for j in 0..res_end {
        res.at_mut(res_col, res_end - j - 1).fill(0);
        if j == res_end - 1 {
            BE::znx_normalize_final_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_coeff_cross_base2k<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    a_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= 3);
        assert_eq!(
            res.n(),
            1,
            "vec_znx_normalize_coeff expects a 1-coeff destination, got {}",
            res.n()
        );
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (a_norm, carry) = carry.split_at_mut(1);
    let (res_carry, a_carry) = carry[..2].split_at_mut(1);
    res_carry[0] = 0;

    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_size * res_base2k;

    let mut lsh: i64 = res_offset % a_base2k as i64;
    let mut limbs_offset: i64 = res_offset / a_base2k as i64;
    if res_offset < 0 && lsh != 0 {
        lsh = (lsh + a_base2k as i64) % (a_base2k as i64);
        limbs_offset -= 1;
    }
    let lsh_pos: usize = lsh as usize;

    let res_end_bit: usize = (-limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let res_start_bit: usize = (a_tot_bits as i64 - limbs_offset * a_base2k as i64).clamp(0, res_tot_bits as i64) as usize;
    let a_end_bit: usize = (limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;
    let a_start_bit: usize = (res_tot_bits as i64 + limbs_offset * a_base2k as i64).clamp(0, a_tot_bits as i64) as usize;

    let res_end: usize = res_end_bit / res_base2k;
    let res_start: usize = res_start_bit.div_ceil(res_base2k);
    let a_end: usize = a_end_bit / a_base2k;
    let a_start: usize = a_start_bit.div_ceil(a_base2k);

    for j in 0..res_size {
        res.at_mut(res_col, j).fill(0);
    }

    if res_start == 0 {
        return;
    }

    let a_out_range: usize = a_size.saturating_sub(a_start);
    for j in 0..a_out_range {
        let src = [a.at(a_col, a_size - j - 1)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(a_base2k, lsh_pos, &src, a_carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(a_base2k, lsh_pos, &src, a_carry);
        }
    }

    if a_out_range == 0 {
        a_carry[0] = 0;
    }

    let mut res_acc_left: usize = res_base2k;
    let mut res_limb: usize = res_start - 1;
    let mid_range: usize = a_start.saturating_sub(a_end);

    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;
        let src = [a.at(a_col, a_limb)[a_coeff]];
        let mut a_take_left: usize = a_base2k;

        BE::znx_normalize_middle_step::<true>(a_base2k, lsh_pos, a_norm, &src, a_carry);

        if j == 0 {
            if !(a_tot_bits - a_start_bit).is_multiple_of(a_base2k) {
                let take: usize = (a_tot_bits - a_start_bit) % a_base2k;
                BE::znx_mul_power_of_two_assign(-(take as i64), a_norm);
                a_take_left -= take;
            } else if !(res_tot_bits - res_start_bit).is_multiple_of(res_base2k) {
                res_acc_left -= (res_tot_bits - res_start_bit) % res_base2k;
            }
        }

        'inner: loop {
            let res_slice = res.at_mut(res_col, res_limb);
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                let scale: usize = res_base2k - res_acc_left;
                BE::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);
                a_take_left -= a_take;
                res_acc_left -= a_take;
            }

            if res_acc_left == 0 || a_limb == 0 {
                if a_limb == 0 && a_take_left == 0 {
                    BE::znx_add_assign(a_carry, a_norm);
                    if res_acc_left != 0 {
                        let scale: usize = res_base2k - res_acc_left;
                        BE::znx_extract_digit_addmul(res_acc_left, scale, res_slice, a_carry);
                    }
                    BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);
                    BE::znx_add_assign(res_carry, a_carry);
                    break 'outer;
                }

                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
            }

            if a_take_left == 0 {
                BE::znx_add_assign(a_carry, a_norm);
                break 'inner;
            }
        }
    }

    if res_end != 0 {
        let carry_to_use = if a_start == a_end { a_carry } else { res_carry };
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            } else {
                BE::znx_normalize_middle_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    match res_base2k == a_base2k {
        true => vec_znx_normalize_inter_base2k::<BE>(res_base2k, res, res_offset, res_col, a, a_col, carry),
        false => vec_znx_normalize_cross_base2k::<BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry),
    }
}

fn vec_znx_normalize_inter_base2k<'r, 'a, BE>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeMiddleStepAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= vec_znx_normalize_tmp_bytes(res.n()) / size_of::<i64>());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (carry, _) = carry.split_at_mut(n);

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
            BE::znx_normalize_first_step_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh_pos, a.at(a_col, a_size - j - 1), carry);
        }
    }

    // If no limbs were discarded, initialize carry to zero
    if a_out_range == 0 {
        BE::znx_zero(carry);
    }

    // Zeroes bottom limbs that will not be interacted with
    for j in res_start..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }

    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    for j in 0..mid_range {
        BE::znx_normalize_middle_step::<true>(
            base2k,
            lsh_pos,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    // Propagates the carry over the non-overlapping limbs between res and a
    for j in 0..res_end {
        BE::znx_zero(res.at_mut(res_col, res_end - j - 1));
        if j == res_end - 1 {
            BE::znx_normalize_final_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, lsh_pos, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vec_znx_normalize_cross_base2k<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + ZnxExtractDigitAddMul
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign
        + ZnxNormalizeDigit,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= vec_znx_normalize_tmp_bytes(res.n()) / size_of::<i64>());
        assert_eq!(res.n(), a.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let (a_norm, carry) = carry.split_at_mut(n);
    let (res_carry, a_carry) = carry[..2 * n].split_at_mut(n);
    BE::znx_zero(res_carry);

    // Total precision (in bits) that `a` and `res` can represent.
    let a_tot_bits: usize = a_size * a_base2k;
    let res_tot_bits: usize = res_size * res_base2k;

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

    // Zero all limbs of `res`. Unlike the simple case
    // where `res_base2k` is equal to `a_base2k`, we also
    // need to ensure that the limbs starting from `res_end`
    // are zero.
    for j in 0..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }

    // Case where offset is positive and greater or equal
    // to the precision of a.
    if res_start == 0 {
        return;
    }

    // Limbs of `a` that have a greater precision than `res`.
    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(a_base2k, lsh_pos, a.at(a_col, a_size - j - 1), a_carry);
        }
    }

    // Zero carry if the above loop didn't trigger.
    if a_out_range == 0 {
        BE::znx_zero(a_carry);
    }

    // How much is left to accumulate to fill a limb of `res`.
    let mut res_acc_left: usize = res_base2k;

    // Starting limb of `res`.
    let mut res_limb: usize = res_start - 1;

    // How many limbs of `a` overlap with `res` (after taking into account the offset).
    let mid_range: usize = a_start.saturating_sub(a_end);

    // Regular normalization over the overlapping limbs of res and a.
    'outer: for j in 0..mid_range {
        let a_limb: usize = a_start - j - 1;

        // Current res & a limbs
        let a_slice: &[i64] = a.at(a_col, a_limb);

        // Trackers: wow much of a_norm is left to
        // be flushed on res.
        let mut a_take_left: usize = a_base2k;

        // Normalizes the j-th limb of a and store the results into `a_norm``.
        // This step is required to avoid overflow in the next step,
        // which assumes that |a| is bounded by 2^{a_base2k -1} (i.e. normalized).
        BE::znx_normalize_middle_step::<true>(a_base2k, lsh_pos, a_norm, a_slice, a_carry);

        // In the first iteration we need to match the precision `res` and `a`.
        if j == 0 {
            // Case where `a` has more precision than `res` (after taking into account the offset)
            //
            // For example:
            //
            // a:      [x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
            // res: [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
            if !(a_tot_bits - a_start_bit).is_multiple_of(a_base2k) {
                let take: usize = (a_tot_bits - a_start_bit) % a_base2k;
                BE::znx_mul_power_of_two_assign(-(take as i64), a_norm);
                a_take_left -= take;
            // Case where `res` has more precision than `a` (after taking into account the offset)
            //
            // For example:
            //
            // a:    [x  x  x  x  x][x  x  x  x  x][x  x  x  x  x][x  x  x  x  x]
            // res:           [x  x  x  x  x  x][x  x  x  x  x  x][x  x  x  x  x  x]
            } else if !(res_tot_bits - res_start_bit).is_multiple_of(res_base2k) {
                res_acc_left -= (res_tot_bits - res_start_bit) % res_base2k;
            }
        }

        // Extract bits of `a_norm` and accumulates them on res[res_limb] until
        // res_base2k bits have been accumulated or until all bits of `a` are
        // extracted.
        'inner: loop {
            // Current limb of res
            let res_slice: &mut [i64] = res.at_mut(res_col, res_limb);

            // We can take at most a_base2k bits
            // but not more than what is left on a_norm or what is left to
            // fully populate the current limb of res.
            let a_take: usize = a_base2k.min(a_take_left).min(res_acc_left);

            if a_take != 0 {
                // Extract `a_take` bits from a_norm and accumulates them on `res_slice`.
                let scale: usize = res_base2k - res_acc_left;
                BE::znx_extract_digit_addmul(a_take, scale, res_slice, a_norm);
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
                    // TODO: prove no overflow can happen here (should not intuitively)
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

                    BE::znx_normalize_middle_step_assign(res_base2k, 0, res_slice, res_carry);

                    // Previous step might not consume all bits of a_carry
                    // TODO: prove no overflow can happen here
                    BE::znx_add_assign(res_carry, a_carry);

                    // We are done, so breaks out of the loop (yes we are at a[0], but
                    // this avoids possible over/under flows of tracking variables)
                    break 'outer;
                }

                // If we reached the last limb of res
                if res_limb == 0 {
                    break 'outer;
                }

                res_acc_left += res_base2k;
                res_limb -= 1;
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

        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            } else {
                BE::znx_normalize_middle_step_assign(res_base2k, 0, res.at_mut(res_col, res_end - j - 1), carry_to_use);
            }
        }
    }
}

pub fn vec_znx_normalize_assign<'r, BE>(base2k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, carry: &mut [i64])
where
    BE: Backend + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            BE::znx_normalize_first_step_assign(base2k, 0, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, 0, res.at_mut(res_col, j), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(res_col, j), carry);
        }
    }
}

#[test]
fn test_vec_znx_normalize_cross_base2k() {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
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
                let mut want: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, 1, in_size);
                want.fill_uniform(60, &mut source);

                let mut have: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, 1, out_size);
                have.fill_uniform(60, &mut source);
                vec_znx_normalize_cross_base2k::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut have),
                    out_base2k,
                    offset,
                    0,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&want),
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
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
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
            let mut want: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, 1, size);
            want.fill_uniform(60, &mut source);

            // Fills "have" with the shifted normalization of "want"
            let mut have: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, 1, size);
            have.fill_uniform(60, &mut source);
            vec_znx_normalize_inter_base2k::<FFT64Ref>(
                base2k,
                &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut have),
                offset,
                0,
                &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&want),
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
pub fn bench_vec_znx_normalize<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>> + 'static,
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let group_name: String = format!("vec_znx_normalize::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>> + 'static,
        Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
        let res_offset: i64 = 0;
        move || {
            for i in 0..cols {
                module.vec_znx_normalize(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res),
                    base2k,
                    res_offset,
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    base2k,
                    i,
                    &mut scratch.borrow(),
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_normalize_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>>,
        Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_normalize_assign_backend(
                    base2k,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut a),
                    i,
                    &mut scratch.borrow(),
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
