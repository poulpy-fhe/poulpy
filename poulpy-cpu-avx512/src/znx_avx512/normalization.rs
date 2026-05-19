#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::__m512i;

/// Vector forms of normalisation constants (broadcast to all 8 lanes).
///
/// Returns `(mask_k, sign_k, base2k_vec)`.
///
/// Unlike the AVX2 helper, we do **not** need a `top_mask` because AVX-512F
/// provides `_mm512_srav_epi64` — a native variable arithmetic right shift.
#[target_feature(enable = "avx512f")]
unsafe fn normalize_consts_avx512(base2k: usize) -> (__m512i, __m512i, __m512i) {
    use core::arch::x86_64::_mm512_set1_epi64;

    assert!((1..=63).contains(&base2k));
    let mask_k: i64 = ((1u64 << base2k) - 1) as i64; // bits 0..k-1 set
    let sign_k: i64 = (1u64 << (base2k - 1)) as i64; // bit k-1
    (
        _mm512_set1_epi64(mask_k),        // mask_k_vec
        _mm512_set1_epi64(sign_k),        // sign_k_vec
        _mm512_set1_epi64(base2k as i64), // base2k_vec (shift amount)
    )
}

/// AVX-512 `get_digit`:  `digit = ((x & mask_k) ^ sign_k) - sign_k`.
#[target_feature(enable = "avx512f")]
unsafe fn get_digit_avx512(x: __m512i, mask_k: __m512i, sign_k: __m512i) -> __m512i {
    use core::arch::x86_64::{_mm512_and_si512, _mm512_sub_epi64, _mm512_xor_si512};
    let low: __m512i = _mm512_and_si512(x, mask_k);
    let t: __m512i = _mm512_xor_si512(low, sign_k);
    _mm512_sub_epi64(t, sign_k)
}

/// AVX-512 `get_carry`:  `carry = (x - digit) >>_arith k`.
///
/// Uses `_mm512_srav_epi64` for a native variable arithmetic right shift,
/// replacing the 4-instruction workaround needed by AVX2.
#[target_feature(enable = "avx512f")]
unsafe fn get_carry_avx512(x: __m512i, digit: __m512i, base2k: __m512i) -> __m512i {
    use core::arch::x86_64::{_mm512_srav_epi64, _mm512_sub_epi64};
    let diff: __m512i = _mm512_sub_epi64(x, digit);
    _mm512_srav_epi64(diff, base2k)
}

// ---------------------------------------------------------------------------
// Public normalization functions
// ---------------------------------------------------------------------------

/// `res += digit(src) << lsh;  src = carry(src)`
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_extract_digit_addmul_avx512(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
    debug_assert_eq!(res.len(), src.len());

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_set1_epi64, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = res.len();
    let span: usize = n >> 3;

    let mut rr: *mut __m512i = res.as_mut_ptr() as *mut __m512i;
    let mut ss: *mut __m512i = src.as_mut_ptr() as *mut __m512i;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);
    let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

    for _ in 0..span {
        let sv: __m512i = _mm512_loadu_si512(ss as *const _);
        let digit_512: __m512i = get_digit_avx512(sv, mask, sign);
        let carry_512: __m512i = get_carry_avx512(sv, digit_512, base2k_vec);

        let rv: __m512i = _mm512_loadu_si512(rr as *const _);
        let madd: __m512i = _mm512_sllv_epi64(digit_512, lsh_v);
        let sum: __m512i = _mm512_add_epi64(rv, madd);

        _mm512_storeu_si512(rr, sum);
        _mm512_storeu_si512(ss, carry_512);

        rr = rr.add(1);
        ss = ss.add(1);
    }

    // scalar tail
    if !n.is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_extract_digit_addmul_ref;

        let off: usize = span << 3;
        znx_extract_digit_addmul_ref(base2k, lsh, &mut res[off..], &mut src[off..]);
    }
}

/// `res = digit(res);  src += carry(res)`
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_digit_avx512(base2k: usize, res: &mut [i64], src: &mut [i64]) {
    debug_assert_eq!(res.len(), src.len());

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512};

    let n: usize = res.len();
    let span: usize = n >> 3;

    let mut rr: *mut __m512i = res.as_mut_ptr() as *mut __m512i;
    let mut ss: *mut __m512i = src.as_mut_ptr() as *mut __m512i;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

    for _ in 0..span {
        let rv: __m512i = _mm512_loadu_si512(rr as *const _);

        let digit_512: __m512i = get_digit_avx512(rv, mask, sign);
        let carry_512: __m512i = get_carry_avx512(rv, digit_512, base2k_vec);

        let sv: __m512i = _mm512_loadu_si512(ss as *const _);
        let sum: __m512i = _mm512_add_epi64(sv, carry_512);

        _mm512_storeu_si512(ss, sum);
        _mm512_storeu_si512(rr, digit_512);

        rr = rr.add(1);
        ss = ss.add(1);
    }

    // scalar tail
    if !n.is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_digit_ref;

        let off: usize = span << 3;
        znx_normalize_digit_ref(base2k, &mut res[off..], &mut src[off..]);
    }
}

/// `carry = carry_of(x)` (with lsh adjustment).
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_first_step_carry_only_avx512(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let mut xx: *const __m512i = x.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    let (mask, sign, base2k_vec) = if lsh == 0 {
        normalize_consts_avx512(base2k)
    } else {
        normalize_consts_avx512(base2k - lsh)
    };

    for _ in 0..span {
        let xv: __m512i = _mm512_loadu_si512(xx as *const _);

        let digit_512: __m512i = get_digit_avx512(xv, mask, sign);
        let carry_512: __m512i = get_carry_avx512(xv, digit_512, base2k_vec);

        _mm512_storeu_si512(cc, carry_512);

        xx = xx.add(1);
        cc = cc.add(1);
    }

    // tail
    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_first_step_carry_only_ref;

        znx_normalize_first_step_carry_only_ref(base2k, lsh, &x[span << 3..], &mut carry[span << 3..]);
    }
}

/// `x = digit(x) << lsh;  carry = carry(x)`
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_first_step_assign_avx512(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_set1_epi64, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);

            let digit_512: __m512i = get_digit_avx512(xv, mask, sign);
            let carry_512: __m512i = get_carry_avx512(xv, digit_512, base2k_vec);

            _mm512_storeu_si512(xx, digit_512);
            _mm512_storeu_si512(cc, carry_512);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    } else {
        let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);

            let digit_512: __m512i = get_digit_avx512(xv, mask, sign);
            let carry_512: __m512i = get_carry_avx512(xv, digit_512, base2k_vec);

            _mm512_storeu_si512(xx, _mm512_sllv_epi64(digit_512, lsh_v));
            _mm512_storeu_si512(cc, carry_512);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    // tail
    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_first_step_assign_ref;

        znx_normalize_first_step_assign_ref(base2k, lsh, &mut x[span << 3..], &mut carry[span << 3..]);
    }
}

/// `x = digit(a) << lsh;  carry = carry(a)` if `OVERWRITE`,
/// else `x += digit(a) << lsh;  carry = carry(a)`.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_first_step_avx512<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut aa: *const __m512i = a.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);

            let digit_512: __m512i = get_digit_avx512(av, mask, sign);
            let carry_512: __m512i = get_carry_avx512(av, digit_512, base2k_vec);

            if OVERWRITE {
                _mm512_storeu_si512(xx, digit_512);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, digit_512));
            }
            _mm512_storeu_si512(cc, carry_512);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);

            let digit_512: __m512i = get_digit_avx512(av, mask, sign);
            let carry_512: __m512i = get_carry_avx512(av, digit_512, base2k_vec);
            let shifted: __m512i = _mm512_sllv_epi64(digit_512, lsh_v);

            if OVERWRITE {
                _mm512_storeu_si512(xx, shifted);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, shifted));
            }
            _mm512_storeu_si512(cc, carry_512);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    // tail
    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_first_step_ref;

        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, &mut x[span << 3..], &a[span << 3..], &mut carry[span << 3..]);
    }
}

/// Two-step middle normalization (carry-only variant).
///
/// Step 1: extract digit0/carry0 from input (base2k or base2k-lsh).
/// Step 2: sum = digit0 (<<lsh if lsh!=0) + carry_in, extract digit1/carry1.
/// Output: carry_out = carry0 + carry1.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_middle_step_carry_only_avx512(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

    let mut xx: *const __m512i = x.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask, sign);
            let c0: __m512i = get_carry_avx512(xv, d0, base2k_vec);

            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, base2k_vec_lsh) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask_lsh, sign_lsh);
            let c0: __m512i = get_carry_avx512(xv, d0, base2k_vec_lsh);

            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_middle_step_carry_only_ref;

        znx_normalize_middle_step_carry_only_ref(base2k, lsh, &x[span << 3..], &mut carry[span << 3..]);
    }
}

/// Two-step middle normalization (in-place variant).
///
/// Step 1: extract digit0/carry0 from x (base2k or base2k-lsh).
/// Step 2: sum = digit0 (<<lsh if lsh!=0) + carry_in, extract digit1/carry1.
/// Output: x = digit1, carry_out = carry0 + carry1.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_middle_step_assign_avx512(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask, sign);
            let c0: __m512i = get_carry_avx512(xv, d0, base2k_vec);

            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            _mm512_storeu_si512(xx, x1);
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, base2k_vec_lsh) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask_lsh, sign_lsh);
            let c0: __m512i = get_carry_avx512(xv, d0, base2k_vec_lsh);

            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            _mm512_storeu_si512(xx, x1);
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_middle_step_assign_ref;

        znx_normalize_middle_step_assign_ref(base2k, lsh, &mut x[span << 3..], &mut carry[span << 3..]);
    }
}

/// Two-step middle normalization (out-of-place: reads from `a`, writes to `x`).
///
/// Step 1: extract digit0/carry0 from a (base2k or base2k-lsh).
/// Step 2: sum = digit0 (<<lsh if lsh!=0) + carry_in, extract digit1/carry1.
/// Output: x = digit1 (or x += digit1 if !OVERWRITE), carry_out = carry0 + carry1.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_middle_step_avx512<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut aa: *const __m512i = a.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask, sign);
            let c0: __m512i = get_carry_avx512(av, d0, base2k_vec);

            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            if OVERWRITE {
                _mm512_storeu_si512(xx, x1);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, x1));
            }
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, base2k_vec_lsh) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask_lsh, sign_lsh);
            let c0: __m512i = get_carry_avx512(av, d0, base2k_vec_lsh);

            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            if OVERWRITE {
                _mm512_storeu_si512(xx, x1);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, x1));
            }
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_middle_step_ref;

        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, &mut x[span << 3..], &a[span << 3..], &mut carry[span << 3..]);
    }
}

/// Subtractive variant of `znx_normalize_middle_step_avx512`: `x -= digit1`, carry as usual.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_middle_step_sub_avx512(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512, _mm512_sub_epi64};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, base2k_vec) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut aa: *const __m512i = a.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask, sign);
            let c0: __m512i = get_carry_avx512(av, d0, base2k_vec);

            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            _mm512_storeu_si512(xx, _mm512_sub_epi64(xv, x1));
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, base2k_vec_lsh) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask_lsh, sign_lsh);
            let c0: __m512i = get_carry_avx512(av, d0, base2k_vec_lsh);

            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);
            let c1: __m512i = get_carry_avx512(s, x1, base2k_vec);
            let cout: __m512i = _mm512_add_epi64(c0, c1);

            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            _mm512_storeu_si512(xx, _mm512_sub_epi64(xv, x1));
            _mm512_storeu_si512(cc, cout);

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_middle_step_sub_ref;

        znx_normalize_middle_step_sub_ref(base2k, lsh, &mut x[span << 3..], &a[span << 3..], &mut carry[span << 3..]);
    }
}

/// Final step normalization (in-place).
///
/// `x = digit( (digit(x, base2k_eff) << lsh) + carry )`   where base2k_eff = base2k when lsh==0
/// or base2k-lsh otherwise.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_final_step_assign_avx512(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, _) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask, sign);
            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            _mm512_storeu_si512(xx, x1);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, _) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(xv, mask_lsh, sign_lsh);
            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            _mm512_storeu_si512(xx, x1);

            xx = xx.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_final_step_assign_ref;

        znx_normalize_final_step_assign_ref(base2k, lsh, &mut x[span << 3..], &mut carry[span << 3..]);
    }
}

/// Final step normalization (out-of-place: reads from `a`, writes to `x`).
///
/// `x = digit( (digit(a, base2k_eff) << lsh) + carry )` if `OVERWRITE`, else `x += …`.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_final_step_avx512<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, _) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut aa: *const __m512i = a.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask, sign);
            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            if OVERWRITE {
                _mm512_storeu_si512(xx, x1);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, x1));
            }

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, _) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask_lsh, sign_lsh);
            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            if OVERWRITE {
                _mm512_storeu_si512(xx, x1);
            } else {
                let xv: __m512i = _mm512_loadu_si512(xx as *const _);
                _mm512_storeu_si512(xx, _mm512_add_epi64(xv, x1));
            }

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_final_step_ref;

        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, &mut x[span << 3..], &a[span << 3..], &mut carry[span << 3..]);
    }
}

/// Subtractive variant of `znx_normalize_final_step_avx512`: `x -= digit1`.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_normalize_final_step_sub_avx512(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);

    use core::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_si512, _mm512_sllv_epi64, _mm512_storeu_si512, _mm512_sub_epi64};

    let n: usize = x.len();
    let span: usize = n >> 3;

    let (mask, sign, _) = normalize_consts_avx512(base2k);

    let mut xx: *mut __m512i = x.as_mut_ptr() as *mut __m512i;
    let mut aa: *const __m512i = a.as_ptr() as *const __m512i;
    let mut cc: *mut __m512i = carry.as_mut_ptr() as *mut __m512i;

    if lsh == 0 {
        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask, sign);
            let s: __m512i = _mm512_add_epi64(d0, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            _mm512_storeu_si512(xx, _mm512_sub_epi64(xv, x1));

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    } else {
        use core::arch::x86_64::_mm512_set1_epi64;

        let (mask_lsh, sign_lsh, _) = normalize_consts_avx512(base2k - lsh);
        let lsh_v: __m512i = _mm512_set1_epi64(lsh as i64);

        for _ in 0..span {
            let av: __m512i = _mm512_loadu_si512(aa as *const _);
            let cv: __m512i = _mm512_loadu_si512(cc as *const _);

            let d0: __m512i = get_digit_avx512(av, mask_lsh, sign_lsh);
            let d0_lsh: __m512i = _mm512_sllv_epi64(d0, lsh_v);

            let s: __m512i = _mm512_add_epi64(d0_lsh, cv);
            let x1: __m512i = get_digit_avx512(s, mask, sign);

            let xv: __m512i = _mm512_loadu_si512(xx as *const _);
            _mm512_storeu_si512(xx, _mm512_sub_epi64(xv, x1));

            xx = xx.add(1);
            aa = aa.add(1);
            cc = cc.add(1);
        }
    }

    if !x.len().is_multiple_of(8) {
        use poulpy_cpu_ref::reference::znx::znx_normalize_final_step_sub_ref;

        znx_normalize_final_step_sub_ref(base2k, lsh, &mut x[span << 3..], &a[span << 3..], &mut carry[span << 3..]);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::{
        get_carry_i64, get_digit_i64, znx_extract_digit_addmul_ref, znx_normalize_digit_ref, znx_normalize_final_step_assign_ref,
        znx_normalize_final_step_ref, znx_normalize_first_step_assign_ref, znx_normalize_first_step_ref,
        znx_normalize_middle_step_assign_ref, znx_normalize_middle_step_ref,
    };

    use super::*;

    use core::arch::x86_64::{_mm512_loadu_si512, _mm512_storeu_si512};

    // Test data — 8 lanes instead of 4.
    const X_DATA: [i64; 8] = [
        7638646372408325293,
        -61440197422348985,
        6835891051541717957,
        -4835376105455195188,
        1234567890123456789,
        -9_223_372_036_854_775_000,
        4503599627370496,
        -7205759403792793600,
    ];

    const C_DATA: [i64; 8] = [
        621182201135793202,
        9000856573317006236,
        5542252755421113668,
        -6036847263131690631,
        3141592653589793238,
        -2718281828459045235,
        1618033988749894848,
        -1414213562373095048,
    ];

    #[target_feature(enable = "avx512f")]
    unsafe fn test_get_digit_ifma_internal() {
        let base2k: usize = 12;
        let y0: Vec<i64> = X_DATA.iter().map(|&v| get_digit_i64(base2k, v)).collect();
        let mut y1: Vec<i64> = vec![0i64; 8];
        let x_512: __m512i = _mm512_loadu_si512(X_DATA.as_ptr() as *const _);
        let (mask, sign, _) = normalize_consts_avx512(base2k);
        let digit: __m512i = get_digit_avx512(x_512, mask, sign);
        _mm512_storeu_si512(y1.as_mut_ptr() as *mut _, digit);
        assert_eq!(y0, y1);
    }

    #[test]
    fn test_get_digit_avx512() {
        unsafe {
            test_get_digit_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_get_carry_ifma_internal() {
        let base2k: usize = 12;
        let digits: Vec<i64> = X_DATA.iter().map(|&v| get_digit_i64(base2k, v)).collect();
        let y0: Vec<i64> = X_DATA
            .iter()
            .zip(digits.iter())
            .map(|(&x, &d)| get_carry_i64(base2k, x, d))
            .collect();
        let mut y1: Vec<i64> = vec![0i64; 8];
        let x_512: __m512i = _mm512_loadu_si512(X_DATA.as_ptr() as *const _);
        let d_512: __m512i = _mm512_loadu_si512(digits.as_ptr() as *const _);
        let (_, _, base2k_vec) = normalize_consts_avx512(base2k);
        let carry: __m512i = get_carry_avx512(x_512, d_512, base2k_vec);
        _mm512_storeu_si512(y1.as_mut_ptr() as *mut _, carry);
        assert_eq!(y0, y1);
    }

    #[test]
    fn test_get_carry_avx512() {
        unsafe {
            test_get_carry_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_first_step_assign_ifma_internal() {
        let mut y0: [i64; 8] = X_DATA;
        let mut y1: [i64; 8] = X_DATA;
        let mut c0: [i64; 8] = C_DATA;
        let mut c1: [i64; 8] = C_DATA;
        let base2k = 12;

        znx_normalize_first_step_assign_ref(base2k, 0, &mut y0, &mut c0);
        znx_normalize_first_step_assign_avx512(base2k, 0, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_first_step_assign_ref(base2k, base2k - 1, &mut y0, &mut c0);
        znx_normalize_first_step_assign_avx512(base2k, base2k - 1, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_first_step_assign_avx512() {
        unsafe {
            test_znx_normalize_first_step_assign_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_first_step_ifma_internal() {
        let a: [i64; 8] = X_DATA;
        let base2k = 12;

        // OVERWRITE = true
        {
            let mut y0: [i64; 8] = X_DATA;
            let mut y1: [i64; 8] = X_DATA;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_first_step_ref::<true>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_first_step_avx512::<true>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_first_step_ref::<true>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_first_step_avx512::<true>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }

        // OVERWRITE = false (accumulate into existing destination)
        {
            let mut y0: [i64; 8] = [11, -22, 33, -44, 55, -66, 77, -88];
            let mut y1: [i64; 8] = y0;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_first_step_ref::<false>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_first_step_avx512::<false>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_first_step_ref::<false>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_first_step_avx512::<false>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }
    }

    #[test]
    fn test_znx_normalize_first_step_avx512() {
        unsafe {
            test_znx_normalize_first_step_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_middle_step_assign_ifma_internal() {
        let mut y0: [i64; 8] = X_DATA;
        let mut y1: [i64; 8] = X_DATA;
        let mut c0: [i64; 8] = C_DATA;
        let mut c1: [i64; 8] = C_DATA;
        let base2k = 12;

        znx_normalize_middle_step_assign_ref(base2k, 0, &mut y0, &mut c0);
        znx_normalize_middle_step_assign_avx512(base2k, 0, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_middle_step_assign_ref(base2k, base2k - 1, &mut y0, &mut c0);
        znx_normalize_middle_step_assign_avx512(base2k, base2k - 1, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_middle_step_assign_avx512() {
        unsafe {
            test_znx_normalize_middle_step_assign_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_middle_step_ifma_internal() {
        use poulpy_cpu_ref::reference::znx::znx_normalize_middle_step_sub_ref;

        let a: [i64; 8] = X_DATA;
        let base2k = 12;

        // OVERWRITE = true
        {
            let mut y0: [i64; 8] = X_DATA;
            let mut y1: [i64; 8] = X_DATA;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_middle_step_ref::<true>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_avx512::<true>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_middle_step_ref::<true>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_avx512::<true>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }

        // OVERWRITE = false (accumulate)
        {
            let mut y0: [i64; 8] = [11, -22, 33, -44, 55, -66, 77, -88];
            let mut y1: [i64; 8] = y0;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_middle_step_ref::<false>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_avx512::<false>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_middle_step_ref::<false>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_avx512::<false>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }

        // sub variant
        {
            let mut y0: [i64; 8] = [11, -22, 33, -44, 55, -66, 77, -88];
            let mut y1: [i64; 8] = y0;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_middle_step_sub_ref(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_sub_avx512(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_middle_step_sub_ref(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_middle_step_sub_avx512(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }
    }

    #[test]
    fn test_znx_normalize_middle_step_avx512() {
        unsafe {
            test_znx_normalize_middle_step_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_final_step_assign_ifma_internal() {
        let mut y0: [i64; 8] = X_DATA;
        let mut y1: [i64; 8] = X_DATA;
        let mut c0: [i64; 8] = C_DATA;
        let mut c1: [i64; 8] = C_DATA;
        let base2k = 12;

        znx_normalize_final_step_assign_ref(base2k, 0, &mut y0, &mut c0);
        znx_normalize_final_step_assign_avx512(base2k, 0, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_normalize_final_step_assign_ref(base2k, base2k - 1, &mut y0, &mut c0);
        znx_normalize_final_step_assign_avx512(base2k, base2k - 1, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_final_step_assign_avx512() {
        unsafe {
            test_znx_normalize_final_step_assign_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_normalize_final_step_ifma_internal() {
        use poulpy_cpu_ref::reference::znx::znx_normalize_final_step_sub_ref;

        let a: [i64; 8] = X_DATA;
        let base2k = 12;

        // OVERWRITE = true
        {
            let mut y0: [i64; 8] = X_DATA;
            let mut y1: [i64; 8] = X_DATA;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_final_step_ref::<true>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_final_step_avx512::<true>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_final_step_ref::<true>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_final_step_avx512::<true>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }

        // OVERWRITE = false (accumulate)
        {
            let mut y0: [i64; 8] = [11, -22, 33, -44, 55, -66, 77, -88];
            let mut y1: [i64; 8] = y0;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_final_step_ref::<false>(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_final_step_avx512::<false>(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_final_step_ref::<false>(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_final_step_avx512::<false>(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }

        // sub variant
        {
            let mut y0: [i64; 8] = [11, -22, 33, -44, 55, -66, 77, -88];
            let mut y1: [i64; 8] = y0;
            let mut c0: [i64; 8] = C_DATA;
            let mut c1: [i64; 8] = C_DATA;

            znx_normalize_final_step_sub_ref(base2k, 0, &mut y0, &a, &mut c0);
            znx_normalize_final_step_sub_avx512(base2k, 0, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);

            znx_normalize_final_step_sub_ref(base2k, base2k - 1, &mut y0, &a, &mut c0);
            znx_normalize_final_step_sub_avx512(base2k, base2k - 1, &mut y1, &a, &mut c1);
            assert_eq!(y0, y1);
            assert_eq!(c0, c1);
        }
    }

    #[test]
    fn test_znx_normalize_final_step_avx512() {
        unsafe {
            test_znx_normalize_final_step_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn znx_extract_digit_addmul_ifma_internal() {
        let mut y0: [i64; 8] = X_DATA;
        let mut y1: [i64; 8] = X_DATA;
        let mut c0: [i64; 8] = C_DATA;
        let mut c1: [i64; 8] = C_DATA;
        let base2k: usize = 12;

        znx_extract_digit_addmul_ref(base2k, 0, &mut y0, &mut c0);
        znx_extract_digit_addmul_avx512(base2k, 0, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);

        znx_extract_digit_addmul_ref(base2k, base2k - 1, &mut y0, &mut c0);
        znx_extract_digit_addmul_avx512(base2k, base2k - 1, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_extract_digit_addmul_avx512() {
        unsafe {
            znx_extract_digit_addmul_ifma_internal();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn znx_normalize_digit_ifma_internal() {
        let mut y0: [i64; 8] = X_DATA;
        let mut y1: [i64; 8] = X_DATA;
        let mut c0: [i64; 8] = C_DATA;
        let mut c1: [i64; 8] = C_DATA;
        let base2k: usize = 12;

        znx_normalize_digit_ref(base2k, &mut y0, &mut c0);
        znx_normalize_digit_avx512(base2k, &mut y1, &mut c1);
        assert_eq!(y0, y1);
        assert_eq!(c0, c1);
    }

    #[test]
    fn test_znx_normalize_digit_internal_avx512() {
        unsafe {
            znx_normalize_digit_ifma_internal();
        }
    }
}
