use std::{hint::black_box, mem::size_of};

use criterion::{BenchmarkId, Criterion};

use dashu_float::ops::DivRemEuclid;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxLshAssignBackend, VecZnxLshBackend,
        VecZnxRshAssignBackend, VecZnxRshBackend,
    },
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{
        ZnxCopy, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep,
        ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
        ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxZero,
    },
    source::Source,
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

pub fn bench_vec_znx_lsh_inplace<B: Backend<OwnedBuf = Vec<u8>>>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxLshAssignBackend<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh_inplace_backend::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxLshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_lsh_assign_backend(
                    base2k,
                    base2k - 1,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b),
                    i,
                    &mut scratch.borrow(),
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_lsh<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxLshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>>,
        Module<B>: VecZnxLshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            let a_backend = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut res_backend = <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_lsh_backend(base2k, base2k - 1, &mut res_backend, i, &a_backend, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_rsh_inplace<B: Backend<OwnedBuf = Vec<u8>>>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh_inplace_backend::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rsh_assign_backend(
                    base2k,
                    base2k - 1,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b),
                    i,
                    &mut scratch.borrow(),
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_rsh<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxRshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>>,
        Module<B>: VecZnxRshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            let a_backend = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut res_backend = <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_rsh_backend(base2k, base2k - 1, &mut res_backend, i, &a_backend, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

#[cfg(test)]
mod tests {
    use crate::{
        FFT64Ref,
        layouts::{Backend, FillUniform, HostBytesBackend, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView},
        reference::vec_znx::{
            vec_znx_copy, vec_znx_lsh, vec_znx_lsh_assign, vec_znx_lsh_tmp_bytes, vec_znx_normalize_assign, vec_znx_rsh,
            vec_znx_rsh_assign, vec_znx_rsh_tmp_bytes, vec_znx_sub_assign,
        },
        source::Source,
    };

    fn alloc_host_vec_znx(n: usize, cols: usize, size: usize) -> VecZnx<Vec<u8>> {
        VecZnx::from_data(
            HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n, cols, size)),
            n,
            cols,
            size,
        )
    }

    #[test]
    fn test_vec_znx_lsh() {
        let n: usize = 8;
        let cols: usize = 2;
        let size: usize = 7;

        let mut a: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, size);
        let mut res_ref: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, size);
        let mut res_test: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, size);

        let mut source: Source = Source::new([0u8; 32]);

        let mut carry: Vec<i64> = vec![0i64; vec_znx_lsh_tmp_bytes(n) / size_of::<i64>()];

        let base2k: usize = 50;

        for k in 0..256 {
            a.fill_uniform(50, &mut source);

            for i in 0..cols {
                vec_znx_normalize_assign::<FFT64Ref>(
                    base2k,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut a),
                    i,
                    &mut carry,
                );
                vec_znx_copy::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                    i,
                );
            }

            for i in 0..cols {
                vec_znx_lsh_assign::<FFT64Ref>(
                    base2k,
                    k,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                    i,
                    &mut carry,
                );
                vec_znx_lsh::<FFT64Ref, true>(
                    base2k,
                    k,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                    i,
                    &mut carry,
                );
                vec_znx_normalize_assign::<FFT64Ref>(
                    base2k,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                    i,
                    &mut carry,
                );
            }

            assert_eq!(res_ref, res_test);
        }
    }

    #[test]
    fn test_vec_znx_lsh_steps_past_source_yields_zero() {
        let n: usize = 8;
        let cols: usize = 2;
        let base2k: usize = 50;
        let mut source: Source = Source::new([0u8; 32]);
        let mut carry: Vec<i64> = vec![0i64; vec_znx_lsh_tmp_bytes(n) / size_of::<i64>()];

        let a_size: usize = 1;
        let res_size: usize = 4;
        let k: usize = 2 * base2k;

        let mut a: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, a_size);
        let mut res_test: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, res_size);
        let zero: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, res_size);

        a.fill_uniform(base2k, &mut source);
        res_test.fill_uniform(base2k, &mut source);

        for i in 0..cols {
            vec_znx_normalize_assign::<FFT64Ref>(
                base2k,
                &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut a),
                i,
                &mut carry,
            );
            vec_znx_lsh::<FFT64Ref, true>(
                base2k,
                k,
                &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                i,
                &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                i,
                &mut carry,
            );
            vec_znx_normalize_assign::<FFT64Ref>(
                base2k,
                &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                i,
                &mut carry,
            );
        }

        assert_eq!(res_test, zero);
    }

    #[test]
    fn test_vec_znx_rsh() {
        let n: usize = 8;
        let cols: usize = 2;

        let res_size: usize = 7;

        let mut res_ref: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, res_size);

        let mut carry: Vec<i64> = vec![0i64; vec_znx_rsh_tmp_bytes(n) / size_of::<i64>()];

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let zero: Vec<i64> = vec![0i64; n];

        for a_size in [res_size - 1, res_size, res_size + 1] {
            let mut a: VecZnx<Vec<u8>> = alloc_host_vec_znx(n, cols, a_size);

            for k in 0..res_size * base2k {
                a.fill_uniform(50, &mut source);

                for i in 0..cols {
                    vec_znx_normalize_assign::<FFT64Ref>(
                        base2k,
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut a),
                        i,
                        &mut carry,
                    );
                    vec_znx_copy::<FFT64Ref>(
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                        i,
                        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                        i,
                    );
                }

                res_test.fill_uniform(50, &mut source);

                for j in 0..cols {
                    vec_znx_rsh_assign::<FFT64Ref>(
                        base2k,
                        k,
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                        j,
                        &mut carry,
                    );
                    vec_znx_rsh::<FFT64Ref, true>(
                        base2k,
                        k,
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                        j,
                        &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                        j,
                        &mut carry,
                    );
                }

                for j in 0..cols {
                    vec_znx_lsh_assign::<FFT64Ref>(
                        base2k,
                        k,
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                        j,
                        &mut carry,
                    );
                    vec_znx_lsh_assign::<FFT64Ref>(
                        base2k,
                        k,
                        &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                        j,
                        &mut carry,
                    );
                }

                // Case where res has enough to fully store a right shifted without any loss
                // In this case we can check exact equality.
                if a_size + k.div_ceil(base2k) <= res_size {
                    assert_eq!(res_ref, res_test);

                    for i in 0..cols {
                        for j in 0..a_size {
                            assert_eq!(res_ref.at(i, j), a.at(i, j), "r0 {} {}", i, j);
                            assert_eq!(res_test.at(i, j), a.at(i, j), "r1 {} {}", i, j);
                        }

                        for j in a_size..res_size {
                            assert_eq!(res_ref.at(i, j), zero, "r0 {} {}", i, j);
                            assert_eq!(res_test.at(i, j), zero, "r1 {} {}", i, j);
                        }
                    }
                // Some loss occures, either because a initially has more precision than res
                // or because the storage of the right shift of a requires more precision than
                // res.
                } else {
                    for j in 0..cols {
                        vec_znx_sub_assign::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                            j,
                            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                            j,
                        );
                        vec_znx_sub_assign::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                            j,
                            &<VecZnx<Vec<u8>> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&a),
                            j,
                        );

                        vec_znx_normalize_assign::<FFT64Ref>(
                            base2k,
                            &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_ref),
                            j,
                            &mut carry,
                        );
                        vec_znx_normalize_assign::<FFT64Ref>(
                            base2k,
                            &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut res_test),
                            j,
                            &mut carry,
                        );

                        assert!(res_ref.stats(base2k, j).std().log2() - (k as f64) <= (k * base2k) as f64);
                        assert!(res_test.stats(base2k, j).std().log2() - (k as f64) <= (k * base2k) as f64);
                    }
                }
            }
        }
    }
}
