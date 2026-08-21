use itertools::izip;

#[inline(always)]
pub fn get_digit_i64(base2k: usize, x: i64) -> i64 {
    (x << (u64::BITS - base2k as u32)) >> (u64::BITS - base2k as u32)
}

#[inline(always)]
pub fn get_carry_i64(base2k: usize, x: i64, digit: i64) -> i64 {
    (x.wrapping_sub(digit)) >> base2k
}

#[inline(always)]
pub fn get_digit_i128(base2k: usize, x: i128) -> i128 {
    (x << (u128::BITS - base2k as u32)) >> (u128::BITS - base2k as u32)
}

#[inline(always)]
pub fn get_carry_i128(base2k: usize, x: i128, digit: i128) -> i128 {
    (x.wrapping_sub(digit)) >> base2k
}

#[inline(always)]
pub fn znx_normalize_first_step_carry_only_ref(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            *c = get_carry_i64(base2k, *x, get_digit_i64(base2k, *x));
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            *c = get_carry_i64(base2k_lsh, *x, get_digit_i64(base2k_lsh, *x));
        });
    }
}

#[inline(always)]
pub fn znx_normalize_first_step_assign_ref(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k, *x);
            *c = get_carry_i64(base2k, *x, digit);
            *x = digit;
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *x);
            *c = get_carry_i64(base2k_lsh, *x, digit);
            *x = digit << lsh;
        });
    }
}

#[inline(always)]
pub fn znx_normalize_first_step_ref<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), a.len());
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k, *a);
            *c = get_carry_i64(base2k, *a, digit);
            if OVERWRITE {
                *x = digit;
            } else {
                *x += digit;
            }
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *a);
            *c = get_carry_i64(base2k_lsh, *a, digit);
            if OVERWRITE {
                *x = digit << lsh;
            } else {
                *x += digit << lsh;
            }
        });
    }
}

#[inline(always)]
pub fn znx_normalize_middle_step_carry_only_ref(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }
    if lsh == 0 {
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k, *x);
            let carry: i64 = get_carry_i64(base2k, *x, digit);
            let digit_plus_c: i64 = digit + *c;
            *c = carry + get_carry_i64(base2k, digit_plus_c, get_digit_i64(base2k, digit_plus_c));
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *x);
            let carry: i64 = get_carry_i64(base2k_lsh, *x, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *c = carry + get_carry_i64(base2k, digit_plus_c, get_digit_i64(base2k, digit_plus_c));
        });
    }
}

#[inline(always)]
pub fn znx_normalize_middle_step_assign_ref(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k, *x);
            let carry: i64 = get_carry_i64(base2k, *x, digit);
            let digit_plus_c: i64 = digit + *c;
            *x = get_digit_i64(base2k, digit_plus_c);
            *c = carry + get_carry_i64(base2k, digit_plus_c, *x);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *x);
            let carry: i64 = get_carry_i64(base2k_lsh, *x, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *x = get_digit_i64(base2k, digit_plus_c);
            *c = carry + get_carry_i64(base2k, digit_plus_c, *x);
        });
    }
}

#[inline(always)]
pub fn znx_extract_digit_addmul_ref(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let digit: i64 = get_digit_i64(base2k, *s);
        *s = get_carry_i64(base2k, *s, digit);
        *r += digit << lsh;
    }
}

#[inline(always)]
pub fn znx_normalize_digit_ref(base2k: usize, res: &mut [i64], src: &mut [i64]) {
    for (r, s) in res.iter_mut().zip(src.iter_mut()) {
        let ri_digit: i64 = get_digit_i64(base2k, *r);
        let ri_carry: i64 = get_carry_i64(base2k, *r, ri_digit);
        *r = ri_digit;
        *s += ri_carry;
    }
}

#[inline(always)]
pub fn znx_normalize_middle_step_ref<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), a.len());
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k, *a);
            let carry: i64 = get_carry_i64(base2k, *a, digit);
            let digit_plus_c: i64 = digit + *c;
            let x1: i64 = get_digit_i64(base2k, digit_plus_c);
            if OVERWRITE {
                *x = x1;
            } else {
                *x += x1;
            }

            *c = carry + get_carry_i64(base2k, digit_plus_c, x1);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *a);
            let carry: i64 = get_carry_i64(base2k_lsh, *a, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            let x1: i64 = get_digit_i64(base2k, digit_plus_c);
            if OVERWRITE {
                *x = x1;
            } else {
                *x += x1;
            }
            *c = carry + get_carry_i64(base2k, digit_plus_c, x1);
        });
    }
}

#[inline(always)]
pub fn znx_normalize_middle_step_sub_ref(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(x.len(), a.len());
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k, *a);
            let carry: i64 = get_carry_i64(base2k, *a, digit);
            let digit_plus_c: i64 = digit + *c;
            let x1: i64 = get_digit_i64(base2k, digit_plus_c);
            *x -= x1;
            *c = carry + get_carry_i64(base2k, digit_plus_c, x1);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *a);
            let carry: i64 = get_carry_i64(base2k_lsh, *a, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            let x1: i64 = get_digit_i64(base2k, digit_plus_c);
            *x -= x1;
            *c = carry + get_carry_i64(base2k, digit_plus_c, x1);
        });
    }
}

#[inline(always)]
pub fn znx_normalize_final_step_assign_ref(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit_i64(base2k, get_digit_i64(base2k, *x) + *c);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit_i64(base2k, (get_digit_i64(base2k_lsh, *x) << lsh) + *c);
        });
    }
}

#[inline(always)]
pub fn znx_normalize_final_step_ref<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            if OVERWRITE {
                *x = get_digit_i64(base2k, get_digit_i64(base2k, *a) + *c);
            } else {
                *x += get_digit_i64(base2k, get_digit_i64(base2k, *a) + *c);
            }
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            if OVERWRITE {
                *x = get_digit_i64(base2k, (get_digit_i64(base2k_lsh, *a) << lsh) + *c);
            } else {
                *x += get_digit_i64(base2k, (get_digit_i64(base2k_lsh, *a) << lsh) + *c);
            }
        });
    }
}

#[inline(always)]
pub fn znx_normalize_final_step_sub_ref(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }
    if lsh == 0 {
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            *x -= get_digit_i64(base2k, get_digit_i64(base2k, *a) + *c);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        izip!(x.iter_mut(), a.iter(), carry.iter_mut()).for_each(|(x, a, c)| {
            *x -= get_digit_i64(base2k, (get_digit_i64(base2k_lsh, *a) << lsh) + *c);
        });
    }
}
