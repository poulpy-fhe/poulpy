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

pub fn vec_znx_lsh_assign<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, carry: &mut [i64])
where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE: ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeFinalStepAssign,
{
    let n: usize = res.n();
    let size: usize = res.size();
    let (steps, k_rem) = k.div_rem_euclid(base2k);

    if steps >= size {
        for j in 0..size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
        return;
    }

    // Assign shift of limbs by a k/base2k. The limbs are moved one at a time
    // through `carry`: the normalization below starts with a first step, which
    // overwrites the carry instead of reading it.
    if steps > 0 {
        let bounce = &mut carry[..n];
        for j in 0..size - steps {
            BE::znx_copy(bounce, res.at(res_col, j + steps));
            BE::znx_copy(res.at_mut(res_col, j), bounce);
        }

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
    BE: Backend<ZnxWord = i64>,
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
    BE: Backend<ZnxWord = i64>,
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

pub fn vec_znx_rsh_assign<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ZnxWord = i64>,
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
    // Shifting past the top limb discards every limb; the rounding carry still
    // lands in res, as in the out-of-place [`vec_znx_rsh`].
    steps = steps.min(size);

    let (carry, tmp) = tmp[..2 * n].split_at_mut(n);

    let lsh: usize = (base2k - k_rem) % base2k;

    // All limbs of a that would fall outside of the limbs of res are discarded,
    // but the carry still need to be computed. With nothing discarded (k == 0)
    // the incoming carry is zero and the loop below must not read scratch.
    if steps == 0 {
        carry.fill(0);
    }
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
    }
    for j in 0..steps {
        if j == steps - 1 {
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
    BE: Backend<ZnxWord = i64>,
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
    BE: Backend<ZnxWord = i64>,
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

/// `vec_znx_rsh_assign` agrees with the out-of-place [`vec_znx_rsh`] for every
/// shift, including `k == 0` (nothing is discarded, so there is no incoming
/// carry to read) and shifts past the top limb, and never depends on what the
/// scratch buffer happened to hold.
#[test]
fn test_rsh_assign_matches_rsh() {
    use crate::{
        FFT64Ref,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    type Host = VecZnx<Vec<u8>, i64>;
    let (n, base2k) = (8usize, 12usize);
    let mut state = 0x1234_5678_9abc_def0u64;
    for size in [1usize, 2, 4] {
        let mut input = poulpy_hal::test_suite::alloc_host_vec_znx::<FFT64Ref>(n, 1, size);
        for j in 0..size {
            for x in input.at_mut(0, j) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *x = (state as i64) >> 40;
            }
        }
        for k in [0, 1, base2k, base2k + 2, size * base2k, size * base2k + 5] {
            let mut want = poulpy_hal::test_suite::alloc_host_vec_znx::<FFT64Ref>(n, 1, size);
            vec_znx_rsh::<FFT64Ref, true>(
                base2k,
                k,
                &mut <Host as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut want),
                0,
                &<Host as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input),
                0,
                &mut vec![0i64; n],
            );
            for fill in [0i64, 7] {
                let mut got = input.clone();
                vec_znx_rsh_assign::<FFT64Ref>(
                    base2k,
                    k,
                    &mut <Host as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut got),
                    0,
                    &mut vec![fill; 2 * n],
                );
                for j in 0..size {
                    assert_eq!(got.at(0, j), want.at(0, j), "size {size} k {k} fill {fill} limb {j}");
                }
            }
        }
    }
}
