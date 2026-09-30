//! Wide fused normalization parity across backend implementations.

use crate::reference::ntt4x30::I128NormalizeOps;

use crate::reference::znx::{get_digit_i64, znx_extract_digit_addmul_normalize_i128_ref};

pub fn test_i128_normalize_fused<BE: I128NormalizeOps>() {
    test_boundary_kernels::<BE>();
    let idft_bound = (1_073_479_681i128 * 1_071_513_601 * 1_070_727_169 * 1_068_236_801 - 1) / 2;
    for res_base2k in 1..=63 {
        for base2k in 1..=res_base2k {
            let half = 1i128 << (base2k - 1);
            let mut input = vec![
                -(1i128 << 126),
                1i128 << 126,
                -idft_bound,
                idft_bound,
                -half - 1,
                -half,
                half - 1,
                half,
                -1,
                0,
                1,
            ];
            let mut state = 0x123456789abcdef0u128;
            for _ in 0..12 {
                state ^= state << 17;
                state ^= state >> 29;
                state ^= state << 43;
                input.push(state as i128 >> 1);
            }
            let mut res: Vec<_> = input
                .iter()
                .map(|&v| {
                    if res_base2k == 1 {
                        0
                    } else {
                        get_digit_i64(res_base2k - 1, v as i64)
                    }
                })
                .collect();
            let mut carry: Vec<_> = (0..input.len()).map(|i| -(i as i128 & 1)).collect();
            let original_input = input.clone();
            let original_res = res.clone();
            let original_carry = carry.clone();
            let mut want_input = input.clone();
            let mut want_res = res.clone();
            let mut want_carry = carry.clone();
            let scale = res_base2k - base2k;
            znx_extract_digit_addmul_normalize_i128_ref::<false>(
                base2k,
                scale,
                res_base2k,
                &mut want_res,
                &mut want_input,
                &mut want_carry,
            );
            BE::znx_extract_digit_addmul_normalize_i128::<false>(base2k, scale, res_base2k, &mut res, &mut input, &mut carry);
            assert_eq!(
                (&res, &input, &carry),
                (&want_res, &want_input, &want_carry),
                "take={base2k} res_base2k={res_base2k}"
            );
            input.clone_from(&original_input);
            want_input.clone_from(&original_input);
            carry.clone_from(&original_carry);
            want_carry.clone_from(&original_carry);
            res.clone_from(&original_res);
            want_res.clone_from(&original_res);
            znx_extract_digit_addmul_normalize_i128_ref::<true>(
                base2k,
                scale,
                res_base2k,
                &mut want_res,
                &mut want_input,
                &mut want_carry,
            );
            BE::znx_extract_digit_addmul_normalize_i128::<true>(base2k, scale, res_base2k, &mut res, &mut input, &mut carry);
            assert_eq!((&res, &input, &carry), (&want_res, &want_input, &want_carry));
            input.clone_from(&original_input);
            want_input.clone_from(&original_input);
            crate::reference::znx::znx_extract_digit_mul_i128_ref(base2k, scale, &mut want_res, &mut want_input);
            BE::znx_extract_digit_mul_i128(base2k, scale, &mut res, &mut input);
            assert_eq!((&res, &input), (&want_res, &want_input));
        }
    }
}

fn test_boundary_kernels<B: I128NormalizeOps>() {
    fn check<B: I128NormalizeOps, const INPUT: bool, const MODE: bool>(k: usize, lsh: usize, padding: usize, n: usize) {
        use crate::reference::normalization::{
            nfc_normalize_floor_ref, nfc_normalize_round_ref, znx_extract_digit_addmul_i128_ref,
        };
        let idft_bound = (1_073_479_681i128 * 1_071_513_601 * 1_070_727_169 * 1_068_236_801 - 1) / 2;
        let half = 1i128 << (k - 1);
        let edge = [
            -(1i128 << 126),
            1i128 << 126,
            -idft_bound,
            idft_bound,
            -half,
            half,
            half - 1,
            -half + 1,
            -1,
            0,
            1,
        ];
        let mut state = 0x1234_5678_abcd_ef90u128;
        let a: Vec<_> = (0..n + 2)
            .map(|i| {
                state ^= state << 17;
                state ^= state >> 29;
                state ^= state << 43;
                if i > 0 && i <= edge.len() {
                    edge[(i - 1 + lsh) % edge.len()]
                } else {
                    state as i128 >> 1
                }
            })
            .collect();
        let carry: Vec<_> = (0..n + 2).map(|i| edge[(i + padding) % edge.len()]).collect();
        let (mut got, mut want) = (carry.clone(), carry.clone());
        nfc_normalize_floor_ref::<INPUT, MODE>(k, lsh, &a[1..], &mut want[1..n + 1]);
        B::nfc_normalize_floor::<INPUT, MODE>(k, lsh, &a[1..], &mut got[1..n + 1]);
        assert_eq!(got, want, "wide floor k={k} lsh={lsh} n={n} input={INPUT} guard={MODE}");
        let output: Vec<_> = a.iter().map(|&v| v as i64).collect();
        let (mut got_c, mut want_c) = (carry.clone(), carry.clone());
        let (mut got_r, mut want_r) = (output.clone(), output.clone());
        nfc_normalize_round_ref::<INPUT, MODE>(k, lsh, padding, &mut want_r[1..n + 1], &a[1..], &mut want_c[1..]);
        B::nfc_normalize_round::<INPUT, MODE>(k, lsh, padding, &mut got_r[1..n + 1], &a[1..], &mut got_c[1..]);
        assert_eq!(
            (got_r, got_c),
            (want_r, want_c),
            "wide round k={k} lsh={lsh} padding={padding} n={n} input={INPUT} pad={MODE}"
        );
        if INPUT && MODE {
            let (mut got, mut want) = (carry.clone(), carry);
            for (out, &v) in want[1..n + 1].iter_mut().zip(&output[1..]) {
                *out += v as i128;
            }
            B::nfc_add_small_carry(&mut got[1..n + 1], &output[1..]);
            assert_eq!(got, want);
            let (mut got_r, mut want_r) = (output.clone(), output);
            let (mut got_a, mut want_a) = (a.clone(), a);
            znx_extract_digit_addmul_i128_ref(k, 63 - k, &mut want_r[1..n + 1], &mut want_a[1..]);
            B::znx_extract_digit_addmul_i128(k, 63 - k, &mut got_r[1..n + 1], &mut got_a[1..]);
            assert_eq!((got_r, got_a), (want_r, want_a), "wide tail k={k} n={n}");
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
