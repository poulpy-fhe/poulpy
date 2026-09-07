//! Bounded coefficient tests shared by the normalization backends.

use crate::reference::znx::*;

type Step = fn(usize, usize, &mut [i64], &[i64], &mut [i64]);
type AssignStep = fn(usize, usize, &mut [i64], &mut [i64]);

/// Compare every normalization kernel, including scalar tails, for every base and shift.
pub fn test_normalization_kernels<B>()
where
    B: ZnxNormalizeFirstStep
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeMiddleStepSub
        + ZnxNormalizeFinalStepSub
        + ZnxNormalizeFinalStepAssign
        + ZnxExtractDigitAddMul
        + ZnxNormalizeDigit,
{
    let steps: [(Step, Step); 8] = [
        (znx_normalize_first_step_ref::<true>, B::znx_normalize_first_step::<true>),
        (znx_normalize_first_step_ref::<false>, B::znx_normalize_first_step::<false>),
        (znx_normalize_middle_step_ref::<true>, B::znx_normalize_middle_step::<true>),
        (znx_normalize_middle_step_ref::<false>, B::znx_normalize_middle_step::<false>),
        (znx_normalize_middle_step_sub_ref, B::znx_normalize_middle_step_sub),
        (znx_normalize_final_step_ref::<true>, B::znx_normalize_final_step::<true>),
        (znx_normalize_final_step_ref::<false>, B::znx_normalize_final_step::<false>),
        (znx_normalize_final_step_sub_ref, B::znx_normalize_final_step_sub),
    ];
    let assign_steps: [(AssignStep, AssignStep); 3] = [
        (znx_normalize_first_step_assign_ref, B::znx_normalize_first_step_assign),
        (znx_normalize_middle_step_assign_ref, B::znx_normalize_middle_step_assign),
        (znx_normalize_final_step_assign_ref, B::znx_normalize_final_step_assign),
    ];
    let mut state = 0xc105_ca77_1234_5678u64;
    for k in 1..=62 {
        let half = 1i64 << (k - 1);
        let edge = [
            -(1i64 << 62),
            1i64 << 62,
            -half,
            half,
            -half - 1,
            half - 1,
            -half + 1,
            half + 1,
            -1,
            0,
            1,
        ];
        for lsh in 0..k {
            let shifted_half = 1i64 << (k - lsh - 1);
            let mut edge = edge.to_vec();
            edge.extend([
                -shifted_half,
                shifted_half,
                -shifted_half - 1,
                shifted_half - 1,
                -shifted_half + 1,
                shifted_half + 1,
            ]);
            for n in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let a: Vec<_> = (0..n)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if i < edge.len() {
                            edge[(i + lsh) % edge.len()]
                        } else {
                            state as i64 >> 1
                        }
                    })
                    .collect();
                let mut carry = vec![0; n + 3];
                znx_normalize_first_step_carry_only_ref(k, lsh, &a, &mut carry);
                carry[..n].rotate_left(n / 2);
                for (c, endpoint) in carry[..n].iter_mut().zip([-(1i64 << 62), 1i64 << 62]) {
                    *c = endpoint;
                }
                for (index, (reference, backend)) in steps.iter().enumerate() {
                    let mut r = vec![7; n];
                    let mut b = r.clone();
                    let mut rc = carry.clone();
                    let mut bc = carry.clone();
                    reference(k, lsh, &mut r, &a, &mut rc);
                    backend(k, lsh, &mut b, &a, &mut bc);
                    assert_eq!((&r, &rc), (&b, &bc), "step={index}, k={k}, lsh={lsh}, n={n}");
                    let base = 1i128 << k;
                    for i in 0..n {
                        let total = ((a[i] as i128) << lsh) + if index < 2 { 0 } else { carry[i] as i128 };
                        let q = (total + base / 2).div_euclid(base);
                        let d = (total - q * base) as i64;
                        let expected = match index {
                            0 | 2 | 5 => d,
                            4 | 7 => 7 - d,
                            _ => 7 + d,
                        };
                        assert_eq!(r[i], expected, "oracle step={index}, k={k}, lsh={lsh}");
                        assert_eq!(rc[i], if index < 5 { q as i64 } else { carry[i] });
                    }
                }
                for (reference, backend) in &assign_steps {
                    let mut r = a.clone();
                    let mut b = a.clone();
                    let mut rc = carry.clone();
                    let mut bc = carry.clone();
                    reference(k, lsh, &mut r, &mut rc);
                    backend(k, lsh, &mut b, &mut bc);
                    assert_eq!((&r, &rc), (&b, &bc), "assign k={k}, lsh={lsh}, n={n}");
                }
                for first in [true, false] {
                    let mut rc = carry.clone();
                    let mut bc = carry.clone();
                    if first {
                        znx_normalize_first_step_carry_only_ref(k, lsh, &a, &mut rc);
                        B::znx_normalize_first_step_carry_only(k, lsh, &a, &mut bc);
                    } else {
                        znx_normalize_middle_step_carry_only_ref(k, lsh, &a, &mut rc);
                        B::znx_normalize_middle_step_carry_only(k, lsh, &a, &mut bc);
                    }
                    assert_eq!(rc, bc, "carry only k={k}, lsh={lsh}, n={n}");
                }
                let mut r = vec![0; n];
                let mut b = r.clone();
                let mut rs = a.clone();
                let mut bs = a.clone();
                znx_extract_digit_addmul_ref(k - lsh, lsh, &mut r, &mut rs);
                B::znx_extract_digit_addmul(k - lsh, lsh, &mut b, &mut bs);
                assert_eq!((&r, &rs), (&b, &bs), "extract k={k}, lsh={lsh}, n={n}");
                r.fill(0);
                b.fill(1i64 << 62);
                rs.copy_from_slice(&a);
                bs.copy_from_slice(&a);
                znx_extract_digit_addmul_ref(k - lsh, lsh, &mut r, &mut rs);
                B::znx_extract_digit_mul(k - lsh, lsh, &mut b, &mut bs);
                assert_eq!((&r, &rs), (&b, &bs), "extract overwrite k={k}, lsh={lsh}, n={n}");
                for overwrite in [false, true] {
                    for (i, value) in r.iter_mut().enumerate() {
                        *value = if overwrite || k == 1 { 0 } else { get_digit_i64(k - 1, a[i]) };
                    }
                    b.copy_from_slice(&r);
                    if overwrite {
                        b.fill(1i64 << 62);
                    }
                    rs.copy_from_slice(&a);
                    bs.copy_from_slice(&a);
                    let mut rc: Vec<_> = (0..n + 3).map(|i| (i % 3) as i64 - 1).collect();
                    let mut bc = rc.clone();
                    znx_extract_digit_addmul_ref(k - lsh, lsh, &mut r, &mut rs);
                    znx_normalize_middle_step_assign_ref(k, 0, &mut r, &mut rc);
                    if overwrite {
                        B::znx_extract_digit_addmul_normalize::<true>(k - lsh, lsh, k, &mut b, &mut bs, &mut bc);
                    } else {
                        B::znx_extract_digit_addmul_normalize::<false>(k - lsh, lsh, k, &mut b, &mut bs, &mut bc);
                    }
                    assert_eq!(
                        (&r, &rs, &rc),
                        (&b, &bs, &bc),
                        "fused overwrite={overwrite}, k={k}, lsh={lsh}, n={n}"
                    );
                }
                r.copy_from_slice(&a);
                b.copy_from_slice(&a);
                rs.fill(0);
                bs.fill(0);
                znx_normalize_digit_ref(k, &mut r, &mut rs);
                B::znx_normalize_digit(k, &mut b, &mut bs);
                assert_eq!((&r, &rs), (&b, &bs), "digit k={k}, lsh={lsh}, n={n}");
            }
        }
    }
}
