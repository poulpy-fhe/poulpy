//! Wide fused normalization parity across backend implementations.

use crate::reference::znx::{ZnxExtractDigitAddMulI128, get_digit_i64, znx_extract_digit_addmul_normalize_i128_ref};

pub fn test_i128_normalize_fused<BE: ZnxExtractDigitAddMulI128>() {
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
