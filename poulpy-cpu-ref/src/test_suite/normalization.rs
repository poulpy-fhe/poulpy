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
        + I64NormalizeOps
        + ZnxNormalizeDigit,
{
    test_boundary_kernels::<B>();
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

fn test_boundary_kernels<B: I64NormalizeOps>() {
    fn check<B: I64NormalizeOps, const INPUT: bool, const MODE: bool>(k: usize, lsh: usize, padding: usize, n: usize) {
        use crate::reference::normalization::{znx_normalize_floor_ref, znx_normalize_round_assign_ref, znx_normalize_round_ref};
        let mut state = 0x1234_5678_abcd_ef90u64;
        let half = 1i64 << (k - 1);
        let edge = [-(1i64 << 62), 1i64 << 62, -half, half, half - 1, -half + 1, -1, 0, 1];
        let a: Vec<_> = (0..n + 2)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if i > 0 && i <= edge.len() {
                    edge[(i - 1 + lsh) % edge.len()]
                } else {
                    state as i64 >> 1
                }
            })
            .collect();
        let carry: Vec<_> = (0..n + 2).map(|i| edge[(i + padding) % edge.len()]).collect();
        let (mut got, mut want) = (carry.clone(), carry.clone());
        znx_normalize_floor_ref::<INPUT, MODE>(k, lsh, &a[1..], &mut want[1..n + 1]);
        B::znx_normalize_floor::<INPUT, MODE>(k, lsh, &a[1..], &mut got[1..n + 1]);
        assert_eq!(got, want, "floor k={k} lsh={lsh} n={n} input={INPUT} guard={MODE}");
        for i in 1..n + 1 {
            let value = ((a[i] as i128) << lsh) + if INPUT { carry[i] as i128 } else { 0 };
            assert_eq!(got[i] as i128, (value + if MODE { 1i128 << (k - 1) } else { 0 }) >> k);
        }
        let (mut got_c, mut want_c) = (carry.clone(), carry.clone());
        let (mut got_r, mut want_r) = (a.clone(), a.clone());
        znx_normalize_round_ref::<INPUT, MODE>(k, lsh, padding, &mut want_r[1..n + 1], &a[1..], &mut want_c[1..]);
        B::znx_normalize_round::<INPUT, MODE>(k, lsh, padding, &mut got_r[1..n + 1], &a[1..], &mut got_c[1..]);
        assert_eq!(
            (&got_r, &got_c),
            (&want_r, &want_c),
            "round k={k} lsh={lsh} padding={padding} n={n} input={INPUT} pad={MODE}"
        );
        for i in 1..n + 1 {
            let value = ((a[i] as i128) << lsh) + if INPUT { carry[i] as i128 } else { 0 };
            let rounded = (value + if padding == 0 { 0 } else { 1i128 << (padding - 1) }) >> padding;
            let width = k - padding;
            let high = (rounded + (1i128 << (width - 1))) >> width;
            let digit = rounded - (high << width);
            assert_eq!(got_c[i] as i128, high);
            assert_eq!(got_r[i] as i128, digit << if MODE { padding } else { 0 });
        }
        if MODE {
            let (mut got_c, mut want_c) = (carry.clone(), carry);
            let (mut got_r, mut want_r) = (a.clone(), a);
            znx_normalize_round_assign_ref::<INPUT>(k, lsh, padding, &mut want_r[1..n + 1], &mut want_c[1..]);
            B::znx_normalize_round_assign::<INPUT>(k, lsh, padding, &mut got_r[1..n + 1], &mut got_c[1..]);
            assert_eq!(
                (got_r, got_c),
                (want_r, want_c),
                "round assign k={k} lsh={lsh} padding={padding} n={n}"
            );
        }
    }
    for k in 1..=63 {
        for lsh in 0..k {
            let mut paddings = vec![0, k / 2, k - 1];
            paddings.dedup();
            for padding in paddings {
                for n in [0, 1, 3, 7, 8, 9, 17] {
                    check::<B, false, false>(k, lsh, padding, n);
                    check::<B, false, true>(k, lsh, padding, n);
                    check::<B, true, false>(k, lsh, padding, n);
                    check::<B, true, true>(k, lsh, padding, n);
                }
            }
        }
    }
}
