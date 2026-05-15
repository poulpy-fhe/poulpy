use std::mem::size_of;

use dashu_float::ops::DivRemEuclid;

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxCopy, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep,
        ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
        ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxZero,
    },
};

pub fn vec_znx_lsh_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_lsh_coeff<'r, 'a, BE, const OVERWRITE: bool>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxCopy
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert_eq!(res.n(), 1, "vec_znx_lsh_coeff expects a 1-coeff destination, got {}", res.n());
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size = a.size();
    let (steps, k_rem) = k.div_rem_euclid(base2k);

    if steps >= res_size.max(a_size) {
        if OVERWRITE {
            for j in 0..res_size {
                res.at_mut(res_col, j).fill(0);
            }
        }
        return;
    }

    let min_size: usize = res_size.min(a_size.saturating_sub(steps));
    let carry_only_start: usize = (steps + min_size).min(a_size);
    let carry = &mut carry[..1];

    for j in (carry_only_start..a_size).rev() {
        let src = [a.at(a_col, j)[a_coeff]];
        if j == a_size - 1 {
            BE::znx_normalize_first_step_carry_only(base2k, k_rem, &src, carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, k_rem, &src, carry);
        }
    }

    if carry_only_start == a_size {
        carry[0] = 0;
    }

    for j in (0..min_size).rev() {
        let src = [a.at(a_col, j + steps)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_final_step::<OVERWRITE>(base2k, k_rem, res.at_mut(res_col, j), &src, carry);
        } else {
            BE::znx_normalize_middle_step::<OVERWRITE>(base2k, k_rem, res.at_mut(res_col, j), &src, carry);
        }
    }

    if OVERWRITE {
        for j in min_size..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }
}

pub fn vec_znx_lsh_assign<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, carry: &mut [i64])
where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let n: usize = res.n();
    let cols: usize = res.cols();
    let size: usize = res.size();
    let (steps, k_rem) = k.div_rem_euclid(base2k);

    if steps >= size {
        for j in 0..size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
        return;
    }

    // Assign shift of limbs by a k/base2k
    if steps > 0 {
        let start: usize = n * res_col;
        let end: usize = start + n;
        let slice_size: usize = n * cols;
        let res_raw: &mut [i64] = res.raw_mut();

        (0..size - steps).for_each(|j| {
            let (lhs, rhs) = res_raw.split_at_mut(slice_size * (j + steps));
            BE::znx_copy(&mut lhs[start + j * slice_size..end + j * slice_size], &rhs[start..end]);
        });

        for j in size - steps..size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }

    for j in (0..size - steps).rev() {
        if j == size - steps - 1 {
            BE::znx_normalize_first_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        }
    }
}

pub fn vec_znx_lsh<'r, 'a, BE, const OVERWRITE: bool>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxCopy
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly,
{
    let res_size: usize = res.size();
    let a_size = a.size();
    let (steps, k_rem) = k.div_rem_euclid(base2k);

    if steps >= res_size.max(a_size) {
        if OVERWRITE {
            for j in 0..res_size {
                BE::znx_zero(res.at_mut(res_col, j));
            }
        }

        return;
    }

    let min_size: usize = res_size.min(a_size.saturating_sub(steps));
    let carry_only_start: usize = (steps + min_size).min(a_size);

    for j in (carry_only_start..a_size).rev() {
        if j == a_size - 1 {
            BE::znx_normalize_first_step_carry_only(base2k, k_rem, a.at(a_col, j), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, k_rem, a.at(a_col, j), carry);
        }
    }

    if carry_only_start == a_size {
        BE::znx_zero(carry);
    }

    // Simply a left shifted normalization of limbs
    // by k/base2k and intra-limb by base2k - k%base2k
    for j in (0..min_size).rev() {
        if j == 0 {
            BE::znx_normalize_final_step::<OVERWRITE>(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        } else {
            BE::znx_normalize_middle_step::<OVERWRITE>(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        }
    }

    if OVERWRITE {
        // Zeroes bottom
        for j in min_size..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_lsh_sub<'r, 'a, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepSub
        + ZnxNormalizeFinalStepSub
        + ZnxNormalizeMiddleStepCarryOnly,
{
    let res_size: usize = res.size();
    let a_size = a.size();
    let (steps, k_rem) = k.div_rem_euclid(base2k);

    if steps >= res_size.max(a_size) {
        return;
    }

    let min_size: usize = res_size.min(a_size.saturating_sub(steps));
    let carry_only_start: usize = (steps + min_size).min(a_size);

    for j in (carry_only_start..a_size).rev() {
        if j == a_size - 1 {
            BE::znx_normalize_first_step_carry_only(base2k, k_rem, a.at(a_col, j), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, k_rem, a.at(a_col, j), carry);
        }
    }

    if carry_only_start == a_size {
        BE::znx_zero(carry);
    }

    for j in (0..min_size).rev() {
        if j == 0 {
            BE::znx_normalize_final_step_sub(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        } else {
            BE::znx_normalize_middle_step_sub(base2k, k_rem, res.at_mut(res_col, j), a.at(a_col, j + steps), carry);
        }
    }
}

pub fn vec_znx_rsh_tmp_bytes(n: usize) -> usize {
    2 * n * size_of::<i64>()
}

pub fn vec_znx_rsh_coeff<'r, 'a, BE, const OVERWRITE: bool>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert_eq!(res.n(), 1, "vec_znx_rsh_coeff expects a 1-coeff destination, got {}", res.n());
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;
    if !k.is_multiple_of(base2k) {
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k;
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));
    let a_out_range: usize = a_size.saturating_sub(a_start);
    let carry = &mut carry[..1];

    for j in 0..a_out_range {
        let src = [a.at(a_col, a_size - j - 1)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, &src, carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, &src, carry);
        }
    }

    if a_out_range == 0 {
        carry[0] = 0;
    }

    if OVERWRITE {
        for j in 0..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }

    let mid_range: usize = res_start.saturating_sub(res_end);
    for j in 0..mid_range {
        let src = [a.at(a_col, a_start - j - 1)[a_coeff]];
        BE::znx_normalize_middle_step::<OVERWRITE>(base2k, lsh, res.at_mut(res_col, res_start - j - 1), &src, carry);
    }

    if OVERWRITE {
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                BE::znx_normalize_middle_step_assign(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
    } else {
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
    }
}

pub fn vec_znx_rsh_add_coeff_into<'r, 'a, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
    res_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert!(res_coeff < res.n(), "res_coeff: {res_coeff} >= res.n(): {}", res.n());
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;
    if !k.is_multiple_of(base2k) {
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k;
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));
    let a_out_range: usize = a_size.saturating_sub(a_start);
    let carry = &mut carry[..1];

    for j in 0..a_out_range {
        let src = [a.at(a_col, a_size - j - 1)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, &src, carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, &src, carry);
        }
    }

    if a_out_range == 0 {
        carry[0] = 0;
    }

    let mid_range: usize = res_start.saturating_sub(res_end);
    for j in 0..mid_range {
        let src = [a.at(a_col, a_start - j - 1)[a_coeff]];
        let dst = &mut res.at_mut(res_col, res_start - j - 1)[res_coeff..res_coeff + 1];
        BE::znx_normalize_middle_step::<false>(base2k, lsh, dst, &src, carry);
    }

    for j in 0..res_end {
        let dst = &mut res.at_mut(res_col, res_end - j - 1)[res_coeff..res_coeff + 1];
        if j == res_end - 1 {
            BE::znx_normalize_final_step_assign(base2k, 0, dst, carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, 0, dst, carry);
        }
    }
}

pub fn vec_znx_rsh_sub_coeff_into<'r, 'a, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
    res_coeff: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepSub
        + ZnxNormalizeFinalStepSub,
{
    #[cfg(debug_assertions)]
    {
        assert!(!carry.is_empty());
        assert!(res_coeff < res.n(), "res_coeff: {res_coeff} >= res.n(): {}", res.n());
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;
    if !k.is_multiple_of(base2k) {
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k;
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));
    let a_out_range: usize = a_size.saturating_sub(a_start);
    let carry = &mut carry[..1];

    for j in 0..a_out_range {
        let src = [a.at(a_col, a_size - j - 1)[a_coeff]];
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, &src, carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, &src, carry);
        }
    }

    if a_out_range == 0 {
        carry[0] = 0;
    }

    let mid_range: usize = res_start.saturating_sub(res_end);
    for j in 0..mid_range {
        let src = [a.at(a_col, a_start - j - 1)[a_coeff]];
        let dst = &mut res.at_mut(res_col, res_start - j - 1)[res_coeff..res_coeff + 1];
        BE::znx_normalize_middle_step_sub(base2k, lsh, dst, &src, carry);
    }

    for j in 0..res_end {
        let dst = &mut res.at_mut(res_col, res_end - j - 1)[res_coeff..res_coeff + 1];
        let zero = [0i64];
        if j == res_end - 1 {
            BE::znx_normalize_final_step_sub(base2k, 0, dst, &zero, carry);
        } else {
            BE::znx_normalize_middle_step_sub(base2k, 0, dst, &zero, carry);
        }
    }
}

pub fn vec_znx_rsh_assign<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let n: usize = res.n();

    let size: usize = res.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if !k.is_multiple_of(base2k) {
        // We rsh by an additional base2k and then lsh by base2k-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;
    }

    let (carry, tmp) = tmp[..2 * n].split_at_mut(n);

    let lsh: usize = (base2k - k_rem) % base2k;

    // All limbs of a that would fall outside of the limbs of res are discarded,
    // but the carry still need to be computed.
    for j in 0..steps {
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, res.at(res_col, size - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, res.at(res_col, size - j - 1), carry);
        }
    }

    // Continues with shifted normalization
    for j in 0..size - steps {
        BE::znx_copy(tmp, res.at(res_col, size - steps - j - 1));
        BE::znx_normalize_middle_step_assign(base2k, lsh, tmp, carry);
        BE::znx_copy(res.at_mut(res_col, size - j - 1), tmp);
    }

    // Propagates carry on the rest of the limbs of res
    for j in 0..steps {
        BE::znx_zero(res.at_mut(res_col, j));
        if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, lsh, res.at_mut(res_col, steps - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, lsh, res.at_mut(res_col, steps - j - 1), carry);
        }
    }
}

pub fn vec_znx_rsh<'r, 'a, BE, const OVERWRITE: bool>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if !k.is_multiple_of(base2k) {
        // We rsh by an additional base2k and then lsh by base2k-k
        // Allows to re-use efficient normalization code, avoids
        // avoids overflows & produce output that is normalized
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k; // 0 if k | base2k
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));

    // All limbs of a that are moved outside of the limbs of res are discarded,
    // but the carry still need to be computed.
    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        }
    }

    if a_out_range == 0 {
        BE::znx_zero(carry);
    }

    if OVERWRITE {
        // Zeroes lower limbs of res if a_size + steps < res_size
        for j in 0..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }

    // Continues with shifted normalization
    let mid_range: usize = res_start.saturating_sub(res_end);

    for j in 0..mid_range {
        BE::znx_normalize_middle_step::<OVERWRITE>(
            base2k,
            lsh,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    if OVERWRITE {
        // Propagates carry on the rest of the limbs of res
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                BE::znx_normalize_middle_step_assign(base2k, lsh, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
    } else {
        // Propagates carry on the rest of the limbs of res
        for j in 0..res_end {
            if j == res_end - 1 {
                BE::znx_normalize_final_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            } else {
                BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
            }
        }
    }
}

pub fn vec_znx_rsh_sub<'r, 'a, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    carry: &mut [i64],
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStepSub
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let mut steps: usize = k / base2k;
    let k_rem: usize = k % base2k;

    if !k.is_multiple_of(base2k) {
        steps += 1;
    }

    let lsh: usize = (base2k - k_rem) % base2k;
    let res_end: usize = res_size.min(steps);
    let res_start: usize = res_size.min(a_size + steps);
    let a_start: usize = a_size.min(res_size.saturating_sub(steps));

    let a_out_range: usize = a_size.saturating_sub(a_start);

    for j in 0..a_out_range {
        if j == 0 {
            BE::znx_normalize_first_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_carry_only(base2k, lsh, a.at(a_col, a_size - j - 1), carry);
        }
    }

    if a_out_range == 0 {
        BE::znx_zero(carry);
    }

    let mid_range: usize = res_start.saturating_sub(res_end);

    for j in 0..mid_range {
        BE::znx_normalize_middle_step_sub(
            base2k,
            lsh,
            res.at_mut(res_col, res_start - j - 1),
            a.at(a_col, a_start - j - 1),
            carry,
        );
    }

    // Negate carry before propagation: the carry from normalizing rsh(a)
    // must be subtracted from the lower limbs of res.
    carry.iter_mut().for_each(|c| *c = -*c);

    for j in 0..res_end {
        if j == res_end - 1 {
            BE::znx_normalize_final_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, 0, res.at_mut(res_col, res_end - j - 1), carry);
        }
    }
}
