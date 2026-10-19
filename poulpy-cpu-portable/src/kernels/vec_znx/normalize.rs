//! Shared normalization loops. Public inputs follow [`crate::api::VecZnxNormalize`]:
//! i64 coefficients in `[-2^62, 2^62]` and radix widths in `1..=62`.
//! Coefficient, in-place and disjoint-range entrypoints share these bounds.

use crate::{
    kernels::znx_normalize::{Limbs, LimbsMut, ZnxNormalize},
    layouts::{HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxShape, ZnxView, ZnxViewMut},
};

pub(crate) use crate::kernels::znx_normalize::normalize_exact;

#[cfg(test)]
use poulpy_hal::{AlignedBuf, alloc_aligned};

#[cfg(test)]
fn alloc_host_vec_znx(n: usize, cols: usize, size: usize) -> crate::layouts::VecZnx<AlignedBuf, i64> {
    use crate::layouts::VecZnx;

    crate::layouts::VecZnx::from_data(
        alloc_aligned::<u8>(VecZnx::<AlignedBuf, i64>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

pub fn vec_znx_normalize_tmp_bytes_portable(_n: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_normalize_portable<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
) where
    BE: ZnxNormalize,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let (n, shape) = (res.n(), res.shape());
    let ptr = res.base_mut_ptr();
    unsafe {
        vec_znx_normalize_range_raw_portable::<BE>(ptr, shape, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, 0, n)
    }
}

/// Normalizes coefficients `coeff_start..coeff_start + coeff_len` of one column into a raw host output.
///
/// # Safety
///
/// `res_ptr` is the base of the dense buffer `res_shape` describes, every element it selects is
/// initialized, and the selected range of every limb of `res_col` is exclusively writable for this
/// call and disjoint from the source. Concurrent calls share the destination only with disjoint
/// coefficient ranges.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn vec_znx_normalize_range_raw_portable<'a, BE>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
) where
    BE: ZnxNormalize,
    BE::BufRef<'a>: HostDataRef,
{
    assert_eq!(res_shape.n(), a.n(), "normalize: degree mismatch");
    assert!(
        res_col < res_shape.cols() && a_col < a.cols(),
        "normalize: column out of range"
    );
    assert!(
        coeff_start + coeff_len <= res_shape.n(),
        "normalize: coefficient range out of bounds"
    );
    if a.size() > 0 {
        // Bounds-checks the last limb, the farthest one the kernel reads.
        let _ = a.at_ptr(a_col, a.size() - 1);
    }
    unsafe {
        BE::znx_normalize(
            LimbsMut::new(res_ptr, res_shape, res_col, coeff_start),
            res_base2k,
            res_k,
            res_offset,
            Limbs::new(a.base_ptr(), a.shape(), a_col, coeff_start),
            a_base2k,
            coeff_len,
        )
    }
}

pub fn vec_znx_normalize_assign_portable<'r, BE>(
    base2k: usize,
    res_k: usize,
    res_offset: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
) where
    BE: ZnxNormalize,
    BE::BufMut<'r>: HostDataMut,
{
    let (n, shape) = (res.n(), res.shape());
    let ptr = res.base_mut_ptr();
    unsafe { vec_znx_normalize_assign_range_raw_portable::<BE>(ptr, shape, base2k, res_k, res_offset, res_col, 0, n) }
}

/// In-place [`vec_znx_normalize_range_raw_portable`].
///
/// # Safety
///
/// As [`vec_znx_normalize_range_raw_portable`], the source being the destination column.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn vec_znx_normalize_assign_range_raw_portable<BE: ZnxNormalize>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    coeff_start: usize,
    coeff_len: usize,
) {
    assert!(res_col < res_shape.cols(), "normalize_assign: column out of range");
    assert!(
        coeff_start + coeff_len <= res_shape.n(),
        "normalize_assign: coefficient range out of bounds"
    );
    assert!(
        res_k <= res_shape.size() * base2k,
        "normalize_assign: precision exceeds the destination"
    );
    if res_offset != 0 {
        assert_eq!(
            res_k,
            res_shape.size() * base2k,
            "vec_znx_normalize_assign: a shifting offset needs the full destination precision"
        );
        assert!(
            res_offset < 0,
            "vec_znx_normalize_assign: only a negative offset is in place, a positive one would write a limb before reading it"
        );
    }
    let res = LimbsMut::new(res_ptr, res_shape, res_col, coeff_start);
    let a = Limbs {
        ptr: res.ptr.cast_const(),
        stride: res.stride,
        size: res.size,
    };
    unsafe { BE::znx_normalize(res, base2k, res_k, res_offset, a, base2k, coeff_len) }
}

#[test]
fn test_vec_znx_normalize_canonical_precision() {
    use crate::{
        FFT64Portable,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };

    let (n, base2k, k) = (1, 4, 10);
    let mut a: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, 3);
    a.at_mut(0, 0)[0] = 3;
    a.at_mut(0, 1)[0] = 7;
    a.at_mut(0, 2)[0] = 7;

    let mut res: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, 4);
    vec_znx_normalize_portable::<FFT64Portable>(
        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut res),
        base2k,
        k,
        0,
        0,
        &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&a),
        base2k,
        0,
    );

    assert_eq!(res.at(0, 0)[0], 4);
    assert_eq!(res.at(0, 1)[0], -8);
    assert_eq!(res.at(0, 2)[0], -8);
    assert_eq!(res.at(0, 3)[0], 0);

    vec_znx_normalize_portable::<FFT64Portable>(
        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut res),
        base2k,
        k,
        0,
        0,
        &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&a),
        5,
        0,
    );
    assert_eq!(res.at(0, 0)[0], 2);
    assert_eq!(res.at(0, 1)[0], -6);
    assert_eq!(res.at(0, 2)[0], -4);
    assert_eq!(res.at(0, 3)[0], 0);

    vec_znx_normalize_portable::<FFT64Portable>(
        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut res),
        base2k,
        0,
        0,
        0,
        &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&a),
        5,
        0,
    );
    assert!(res.data().iter().all(|&byte| byte == 0));
}

#[test]
fn test_vec_znx_normalize_cross_base2k() {
    use crate::{
        FFT64Portable,
        api::VecZnxFillUniformSource,
        layouts::{Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
        source::Source,
    };
    let n: usize = 8;
    let module: Module<FFT64Portable> = Module::new(n as u64);

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
                let mut want: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, in_size);
                module.vec_znx_fill_uniform_source(
                    60,
                    in_size * 60,
                    &mut VecZnxToBackendMut::<FFT64Portable>::to_backend_mut(&mut want),
                    0,
                    &mut source,
                );

                let mut have: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, out_size);
                module.vec_znx_fill_uniform_source(
                    60,
                    out_size * 60,
                    &mut VecZnxToBackendMut::<FFT64Portable>::to_backend_mut(&mut have),
                    0,
                    &mut source,
                );
                vec_znx_normalize_portable::<FFT64Portable>(
                    &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut have),
                    out_base2k,
                    out_size * out_base2k,
                    offset,
                    0,
                    &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&want),
                    in_base2k,
                    0,
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
fn test_vec_znx_normalize_limb_bounds() {
    use crate::{
        FFT64Portable,
        api::VecZnxFillUniformSource,
        layouts::{Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView},
        source::Source,
    };
    let n: usize = 8;
    let module: Module<FFT64Portable> = Module::new(n as u64);
    let mut source: Source = Source::new([1u8; 32]);
    for in_base2k in 1..=51usize {
        for out_base2k in 1..=51usize {
            for offset in [-(in_base2k as i64), -3, -1, 0, 1, 3, in_base2k as i64] {
                for in_size in 1..=4usize {
                    for out_size in 1..=4usize {
                        let mut want: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, in_size);
                        module.vec_znx_fill_uniform_source(
                            62,
                            in_size * 62,
                            &mut VecZnxToBackendMut::<FFT64Portable>::to_backend_mut(&mut want),
                            0,
                            &mut source,
                        );
                        let mut have: VecZnx<AlignedBuf, i64> = alloc_host_vec_znx(n, 1, out_size);
                        vec_znx_normalize_portable::<FFT64Portable>(
                            &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut have),
                            out_base2k,
                            out_size * out_base2k,
                            offset,
                            0,
                            &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&want),
                            in_base2k,
                            0,
                        );
                        let bound: i64 = 1 << (out_base2k - 1);
                        for j in 0..out_size {
                            assert!(
                                have.at(0, j).iter().all(|x| (-bound..bound).contains(x)),
                                "in_base2k={in_base2k} out_base2k={out_base2k} offset={offset} in_size={in_size} out_size={out_size} limb={j}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Exact fixed-point quantization, with ties to even.
#[cfg(test)]
pub(crate) fn normalize_integer_oracle(a: &[i128], a_base2k: usize, res_base2k: usize, res_size: usize, offset: i64) -> Vec<i64> {
    use dashu_int::IBig;
    let mut value = IBig::ZERO;
    for &limb in a {
        value = (value << a_base2k) + IBig::from(limb);
    }
    let shift = res_size as i64 * res_base2k as i64 + offset - a.len() as i64 * a_base2k as i64;
    if shift >= 0 {
        value <<= shift as usize;
    } else {
        let drop = (-shift) as usize;
        value = crate::test_suite::znx_normalize::round_ties_even(value, drop);
    }
    let radix = IBig::ONE << res_base2k;
    let half = IBig::ONE << (res_base2k - 1);
    let mut result = vec![0; res_size];
    for limb in result.iter_mut().rev() {
        let next = (&value + &half) >> res_base2k;
        *limb = i64::try_from(&value - &next * &radix).unwrap();
        value = next;
    }
    result
}

#[cfg(test)]
fn check_normalize_integer(a: &[i128], a_base2k: usize, res_base2k: usize, res_size: usize, offset: i64) {
    use crate::{
        FFT64Portable,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let mut input = alloc_host_vec_znx(1, 1, a.len());
    for (j, &x) in a.iter().enumerate() {
        input.at_mut(0, j)[0] = x as i64;
    }
    let mut output = alloc_host_vec_znx(1, 1, res_size);
    for j in 0..res_size {
        output.at_mut(0, j)[0] = 12345;
    }
    vec_znx_normalize_portable::<FFT64Portable>(
        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut output),
        res_base2k,
        res_size * res_base2k,
        offset,
        0,
        &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&input),
        a_base2k,
        0,
    );
    let have: Vec<i64> = (0..res_size).map(|j| output.at(0, j)[0]).collect();
    assert_eq!(
        have,
        normalize_integer_oracle(a, a_base2k, res_base2k, res_size, offset),
        "a={a:?}, a_base2k={a_base2k}, res_base2k={res_base2k}, res_size={res_size}, offset={offset}"
    );
}

#[test]
fn test_normalize_exact_regressions() {
    check_normalize_integer(&[], 2, 3, 1, -1);
    check_normalize_integer(&[1], 2, 2, 0, 0);
    check_normalize_integer(&[0, 1, 2], 2, 2, 1, 0);
    check_normalize_integer(&[9], 2, 2, 1, -4);
    check_normalize_integer(&[9], 2, 3, 1, -4);
    check_normalize_integer(&[-1], 3, 2, 1, 0);
    check_normalize_integer(&[1i128 << 62], 2, 3, 1, -5);
}

#[test]
fn test_normalize_integer_input_bound() {
    let mut state = 0x123456789abcdefu64;
    for a_base2k in 1..=62 {
        for res_base2k in 1..=62 {
            for a_size in 1..=8 {
                let a: Vec<i128> = (0..a_size)
                    .map(|j| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        match j % 4 {
                            0 => -(1i128 << 62),
                            1 => 1i128 << 62,
                            _ => (state as i64 >> 1) as i128,
                        }
                    })
                    .collect();
                for a in [a, vec![-(1i128 << 62); a_size], vec![1i128 << 62; a_size]] {
                    for res_size in [1, 3, 8] {
                        let bracket = (a_size * a_base2k + res_size * res_base2k) as i64;
                        let gap = (a_size * a_base2k) as i64 - (res_size * res_base2k) as i64;
                        for offset in [
                            -bracket,
                            -(a_base2k as i64) - 1,
                            -1,
                            0,
                            1,
                            a_base2k as i64,
                            bracket,
                            gap - a_base2k as i64 - 1,
                            gap - a_base2k as i64,
                            gap - 1,
                        ] {
                            check_normalize_integer(&a, a_base2k, res_base2k, res_size, offset);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_centered_boundaries() {
    check_normalize_centered(false);
}

#[test]
#[ignore = "285 million cases; run explicitly with --release --ignored"]
fn test_normalize_exhaustive_centered_full() {
    check_normalize_centered(true);
}

#[cfg(test)]
fn check_normalize_centered(exhaustive: bool) {
    use dashu_int::IBig;

    use crate::{
        FFT64Portable,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let bases = if exhaustive {
        (1..=6usize).collect::<Vec<_>>()
    } else {
        vec![1, 2, 17, 19, 21, 50, 51, 61, 62]
    };
    for &a_base2k in &bases {
        let half_input = 1i64 << (a_base2k - 1);
        let alphabet = if exhaustive {
            (-half_input..half_input).collect::<Vec<_>>()
        } else {
            let mut digits = vec![-half_input, -half_input + 1, -1, 0, 1, half_input - 1];
            digits.retain(|x| (-half_input..half_input).contains(x));
            digits.sort_unstable();
            digits.dedup();
            digits
        };
        for a_size in 1..=3usize {
            let mut input = alloc_host_vec_znx(1, 1, a_size);
            for &res_base2k in &bases {
                let half = IBig::ONE << (res_base2k - 1);
                for res_size in 1..=3usize {
                    let mut output = alloc_host_vec_znx(1, 1, res_size);
                    let bracket = (a_size * a_base2k + res_size * res_base2k + 1) as i64;
                    let gap = (a_size * a_base2k) as i64 - (res_size * res_base2k) as i64;
                    let offsets = if exhaustive {
                        (-bracket..=bracket).collect::<Vec<_>>()
                    } else {
                        let k = a_base2k as i64;
                        let mut offsets = vec![
                            -bracket,
                            -k,
                            -1,
                            0,
                            1,
                            k,
                            bracket,
                            gap - k - 1,
                            gap - k,
                            gap - 1,
                            gap,
                            gap + 1,
                        ];
                        offsets.sort_unstable();
                        offsets.dedup();
                        offsets
                    };
                    let offsets: Vec<_> = offsets
                        .into_iter()
                        .map(|offset| {
                            let shift = res_size as i64 * res_base2k as i64 + offset - a_size as i64 * a_base2k as i64;
                            (offset, shift)
                        })
                        .collect();
                    for code in 0..alphabet.len().pow(a_size as u32) {
                        let mut digits = code;
                        let mut integer = IBig::ZERO;
                        for j in 0..a_size {
                            let digit = alphabet[digits % alphabet.len()];
                            digits /= alphabet.len();
                            input.at_mut(0, j)[0] = digit;
                            integer = (integer << a_base2k) + IBig::from(digit);
                        }
                        for (offset, shift) in &offsets {
                            vec_znx_normalize_portable::<FFT64Portable>(
                                &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut output),
                                res_base2k,
                                res_size * res_base2k,
                                *offset,
                                0,
                                &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&input),
                                a_base2k,
                                0,
                            );
                            let mut expected = if *shift >= 0 {
                                &integer << *shift as usize
                            } else {
                                crate::test_suite::znx_normalize::round_ties_even(integer.clone(), (-shift) as usize)
                            };
                            for j in (0..res_size).rev() {
                                let next = (&expected + &half) >> res_base2k;
                                let digit = i64::try_from(&expected - (&next << res_base2k)).unwrap();
                                assert_eq!(
                                    output.at(0, j)[0],
                                    digit,
                                    "code={code}, ka={a_base2k}, kr={res_base2k}, sa={a_size}, sr={res_size}, offset={offset}, limb={j}"
                                );
                                expected = next;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_window_and_range_integer() {
    use crate::{
        FFT64Portable,
        layouts::{VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    };
    let n = 35;
    let mut state = 0x58d719b3ce42a670u64;
    for a_base2k in 1..=62usize {
        for res_base2k in [1, 2, 17, 50, 62] {
            for size in [1, 2, 8] {
                let mut input = alloc_host_vec_znx(n, 2, size);
                for j in 0..size {
                    for x in input.at_mut(1, j) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        *x = state as i64 >> 1;
                    }
                }
                for offset in [
                    i64::MIN,
                    -1001,
                    -(a_base2k as i64) - 1,
                    -(a_base2k as i64),
                    -1,
                    0,
                    1,
                    a_base2k as i64,
                    1001,
                    i64::MAX,
                ] {
                    let mut whole = alloc_host_vec_znx(n, 2, size);
                    let mut split = alloc_host_vec_znx(n, 2, size);
                    for j in 0..size {
                        whole.at_mut(0, j).fill(93);
                        split.at_mut(0, j).fill(93);
                    }
                    let a = <VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&input);
                    vec_znx_normalize_portable::<FFT64Portable>(
                        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut whole),
                        res_base2k,
                        size * res_base2k,
                        offset,
                        1,
                        &a,
                        a_base2k,
                        1,
                    );
                    let (split_shape, split_ptr) = (split.shape(), split.base_mut_ptr());
                    for (start, end) in [(17, 35), (1, 8), (0, 1), (8, 17)] {
                        unsafe {
                            vec_znx_normalize_range_raw_portable::<FFT64Portable>(
                                split_ptr,
                                split_shape,
                                res_base2k,
                                size * res_base2k,
                                offset,
                                1,
                                &a,
                                a_base2k,
                                1,
                                start,
                                end - start,
                            )
                        };
                    }
                    for i in 0..n {
                        let mut single = alloc_host_vec_znx(1, 1, size);
                        let a_i = <VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&input)
                            .window_coeffs(i, 1);
                        vec_znx_normalize_portable::<FFT64Portable>(
                            &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut single),
                            res_base2k,
                            size * res_base2k,
                            offset,
                            0,
                            &a_i,
                            a_base2k,
                            1,
                        );
                        // Extreme offsets reduce to zero without enormous oracle allocations.
                        let expected = if offset == i64::MIN || offset == i64::MAX {
                            vec![0; size]
                        } else {
                            normalize_integer_oracle(
                                &(0..size).map(|j| input.at(1, j)[i] as i128).collect::<Vec<_>>(),
                                a_base2k,
                                res_base2k,
                                size,
                                offset,
                            )
                        };
                        for (j, &digit) in expected.iter().enumerate() {
                            assert_eq!(
                                whole.at(1, j)[i],
                                digit,
                                "ka={a_base2k}, kr={res_base2k}, size={size}, offset={offset}, i={i}, j={j}"
                            );
                            assert_eq!(split.at(1, j)[i], digit);
                            assert_eq!(single.at(0, j)[0], digit);
                            assert_eq!(whole.at(0, j)[i], 93);
                            assert_eq!(split.at(0, j)[i], 93);
                        }
                    }
                }
                let mut assigned = input.clone();
                vec_znx_normalize_assign_portable::<FFT64Portable>(
                    a_base2k,
                    size * a_base2k,
                    0,
                    &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut assigned),
                    1,
                );
                let mut coeff_assigned = input.clone();
                for i in 0..n {
                    vec_znx_normalize_assign_portable::<FFT64Portable>(
                        a_base2k,
                        size * a_base2k,
                        0,
                        &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut coeff_assigned)
                            .window_coeffs(i, 1),
                        1,
                    );
                    let want = normalize_integer_oracle(
                        &(0..size).map(|j| input.at(1, j)[i] as i128).collect::<Vec<_>>(),
                        a_base2k,
                        a_base2k,
                        size,
                        0,
                    );
                    for (j, &digit) in want.iter().enumerate() {
                        assert_eq!(assigned.at(1, j)[i], digit);
                        assert_eq!(coeff_assigned.at(1, j)[i], digit);
                    }
                }
            }
        }
    }
}
